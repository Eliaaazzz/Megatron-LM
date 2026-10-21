# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser, jit_scalar


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


# The scales reach the compiled bodies as 0-d tensors (see jit_scalar); the public
# functions keep taking Python floats.


@jit_fuser
def _tanh_soft_clamp(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def tanh_soft_clamp(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Tanh Soft Clamp to precondition activation inputs."""
    return _tanh_soft_clamp(x, jit_scalar(scale, x.device))


@jit_fuser
def _situ(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _tanh_soft_clamp(x, scale) * torch.sigmoid(x)


def situ(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Sigmoid-Tanh Unit from Kimi K3: ``s * tanh(x / s) * sigmoid(x)``"""
    return _situ(x, jit_scalar(scale, x.device))


@jit_fuser
def _situ_glu(
    x: torch.Tensor,
    gate_scale: torch.Tensor,
    linear_scale: Optional[torch.Tensor],
    linear_offset: torch.Tensor,
) -> torch.Tensor:
    x_gate, x_linear = torch.chunk(x, 2, dim=-1)
    if linear_scale is not None:
        x_linear = _tanh_soft_clamp(x_linear, linear_scale)
    return _situ(x_gate, gate_scale) * (x_linear + linear_offset)


def situ_glu(
    x: torch.Tensor,
    gate_scale: float,
    linear_scale: Optional[float] = None,
    linear_offset: float = 0.0,
) -> torch.Tensor:
    """SiTU-GLU: ``situ(x_gate, gate_scale) * (x_linear + linear_offset)``."""
    return _situ_glu(
        x,
        jit_scalar(gate_scale, x.device),
        None if linear_scale is None else jit_scalar(linear_scale, x.device),
        jit_scalar(linear_offset, x.device),
    )


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
