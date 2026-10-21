# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situ_glu
from megatron.core.fusions.fused_bias_swiglu import (
    bias_clamped_swiglu,
    bias_swiglu_impl,
    clamped_swiglu,
    clamped_swiglu_back,
    clamped_weighted_swiglu,
)
from megatron.core.fusions.fused_bias_swiglu import situ_glu as fused_situ_glu
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.transformer.transformer_config import TransformerConfig


def _clamped_swiglu_config(**kwargs):
    defaults = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
    )
    return TransformerConfig(**(defaults | kwargs))


def test_clamped_swiglu_config_accepts_positive_moe_clamp():
    assert _clamped_swiglu_config().activation_func_clamp_value == 10.0


@pytest.mark.parametrize("clamp_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_clamped_swiglu_config_requires_positive_clamp(clamp_value):
    with pytest.raises(ValueError, match="greater than zero"):
        _clamped_swiglu_config(activation_func_clamp_value=clamp_value)


def test_clamped_swiglu_config_rejects_linear_offset():
    with pytest.raises(ValueError, match="glu_linear_offset must be zero"):
        _clamped_swiglu_config(glu_linear_offset=1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_moe_experts": None}, "only supported with MoE"),
        ({"use_te_activation_func": True}, "use_te_activation_func must be False"),
    ],
)
def test_clamped_swiglu_config_rejects_unsupported_paths(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _clamped_swiglu_config(**kwargs)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("linear_clamp_scale", [None, 3.0], ids=("gate_only", "gate_and_linear"))
@pytest.mark.parametrize("with_bias", [False, True])
def test_situ_glu_fusion_matches_unfused(input_dtype, linear_clamp_scale, with_bias):
    """Fused SiTU-GLU must match the eager path taken when bias_activation_fusion is False."""
    torch.manual_seed(0)
    gate_clamp_scale = 5.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=2.0e-1)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    # Scaled up so a good share of the inputs land in the saturating region of the clamp.
    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = situ_glu(x + bias if with_bias else x, gate_clamp_scale, linear_clamp_scale)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_fused = bias_swiglu_impl(
        x_fused,
        bias_fused,
        gate_clamp_scale=gate_clamp_scale,
        linear_clamp_scale=linear_clamp_scale,
    )
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    if with_bias:
        assert bias_fused.grad.dtype == bias.grad.dtype
        assert torch.allclose(bias.grad, bias_fused.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_clamped_weighted_bias_swiglu(input_dtype):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    wgrad_tols = dict(rtol=1.0e-6, atol=2.0e-4)

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: clamp and activate in FP32, then restore the input dtype.
    y_1, y_2 = torch.chunk(x.to(torch.float32), 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value))
        * y_2.clamp(min=-clamp_value, max=clamp_value)
        * weights
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    y_fused = weighted_bias_swiglu_impl(x_fused, None, weights_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    assert weights_fused.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_fused.grad, **wgrad_tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamped_bias_swiglu_impl(input_dtype, with_bias):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    x_fp32 = x.to(torch.float32)
    x_effective = x_fp32 + bias.to(torch.float32) if with_bias else x_fp32
    y_1, y_2 = torch.chunk(x_effective, 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value)) * y_2.clamp(min=-clamp_value, max=clamp_value)
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_fused = bias_swiglu_impl(x_fused, bias_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    if with_bias:
        assert bias_fused.grad.dtype == bias.grad.dtype
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0), bias_fused.grad.flatten().float().unsqueeze(0)
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_impl_clamp_none_matches_unclamped(input_dtype, with_bias):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    else:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda").requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, bias)
    y.backward(bwd_input)

    x_explicit = x.detach().clone().requires_grad_(True)
    bias_explicit = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_explicit = bias_swiglu_impl(x_explicit, bias_explicit, clamp_value=None)
    y_explicit.backward(bwd_input.detach().clone())

    assert torch.allclose(y, y_explicit, **tols)
    assert torch.allclose(x.grad, x_explicit.grad, **tols)
    if with_bias:
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0),
            bias_explicit.grad.flatten().float().unsqueeze(0),
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


# --- the clamp scalars must not be baked into the compiled kernels ------------------------
#
# torch.compile specializes a Python float argument until it sees a second distinct value and
# traces one shared graph from then on; a bound baked into that graph at trace time shows up
# on the third distinct value. These run on CPU as well as GPU.


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reference_clamped_swiglu(x, clamp_value):
    y_1, y_2 = torch.chunk(x.float(), 2, -1)
    return (F.silu(y_1.clamp(max=clamp_value)) * y_2.clamp(-clamp_value, clamp_value)).to(x.dtype)


def _reference_situ_glu(x, gate_clamp_scale, linear_clamp_scale):
    x_gate, x_linear = torch.chunk(x.float(), 2, -1)
    gate = gate_clamp_scale * torch.tanh(x_gate / gate_clamp_scale) * torch.sigmoid(x_gate)
    if linear_clamp_scale is not None:
        x_linear = linear_clamp_scale * torch.tanh(x_linear / linear_clamp_scale)
    return (gate * x_linear).to(x.dtype)


def _assert_matches_reference(fused_fn, reference_fn, scalars, scale=8.0):
    torch.manual_seed(0)
    tols = dict(rtol=1.0e-5, atol=1.0e-5)
    for scalar in scalars:
        x = (torch.randn(16, 64, dtype=torch.float32, device=_device()) * scale).requires_grad_(
            True
        )
        grad_output = torch.randn(16, 32, dtype=torch.float32, device=_device())

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = reference_fn(x_ref, scalar)
        y_ref.backward(grad_output)

        y = fused_fn(x, scalar)
        y.backward(grad_output)

        assert torch.allclose(y, y_ref, **tols), scalar
        assert torch.allclose(x.grad, x_ref.grad, **tols), scalar


@pytest.mark.parametrize(
    "entry_point",
    ["clamped_swiglu", "bias_clamped_swiglu", "clamped_weighted_swiglu", "bias_swiglu_impl"],
)
def test_clamped_swiglu_follows_each_clamp_value(entry_point):
    """Three distinct clamp values in a row must each clamp at their own value, through the
    compiled helpers called directly (autograd through the compiled graph) and through the
    autograd function."""
    device = _device()
    if entry_point == "clamped_swiglu":
        fused = lambda x, cv: clamped_swiglu(x, cv)
        reference = _reference_clamped_swiglu
    elif entry_point == "bias_clamped_swiglu":
        bias = torch.randn(64, dtype=torch.float32, device=device)
        fused = lambda x, cv: bias_clamped_swiglu(x, bias, cv)
        reference = lambda x, cv: _reference_clamped_swiglu(x + bias, cv)
    elif entry_point == "clamped_weighted_swiglu":
        weights = torch.rand(16, 1, dtype=torch.float32, device=device) + 0.5
        fused = lambda x, cv: clamped_weighted_swiglu(x, weights, cv)
        reference = lambda x, cv: _reference_clamped_swiglu(x, cv) * weights
    else:
        fused = lambda x, cv: bias_swiglu_impl(x, None, clamp_value=cv)
        reference = _reference_clamped_swiglu
    _assert_matches_reference(fused, reference, (5.0, 10.0, 2.0))


def test_clamped_swiglu_back_follows_each_clamp_value():
    """The explicit backward helper, called directly as the autograd functions do."""
    torch.manual_seed(0)
    for clamp_value in (5.0, 10.0, 2.0):
        x = (torch.randn(16, 64, dtype=torch.float32, device=_device()) * 8.0).requires_grad_(True)
        grad_output = torch.randn(16, 32, dtype=torch.float32, device=_device())
        (reference,) = torch.autograd.grad(
            _reference_clamped_swiglu(x, clamp_value), x, grad_output
        )
        fused = clamped_swiglu_back(grad_output, x.detach(), clamp_value)
        assert torch.allclose(fused, reference, rtol=1.0e-5, atol=1.0e-5), clamp_value


@pytest.mark.parametrize("entry_point", ["situ_glu", "bias_swiglu_impl"])
def test_situ_glu_follows_each_clamp_scale(entry_point):
    """Same contract for the soft-clamp scales."""
    if entry_point == "situ_glu":
        fused = lambda x, s: fused_situ_glu(x, s, s / 2)
    else:
        fused = lambda x, s: bias_swiglu_impl(x, None, gate_clamp_scale=s, linear_clamp_scale=s / 2)
    _assert_matches_reference(
        fused, lambda x, s: _reference_situ_glu(x, s, s / 2), (5.0, 10.0, 2.0), scale=5.0
    )
