# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Tuple

import torch

from megatron.core.utils import is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2


def noop_decorator(func):
    '''No-op decorator'''
    return func


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser
    try:
        if is_torch_min_version("2.2.0a0"):
            jit_fuser = torch.compile
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


_SCALAR_TENSORS: Dict[Tuple[float, torch.device], torch.Tensor] = {}


def jit_scalar(value: float, device: torch.device) -> torch.Tensor:
    """Return a cached fp32 0-d tensor holding ``value`` on ``device``.

    Pass Python scalars to ``jit_fuser`` functions through this. ``torch.compile`` specializes a
    float argument only until it sees a second distinct value; from then on the float is traced
    as a symbol, and some ops still bake the first value into the generated code
    (pytorch/pytorch#194976). A tensor argument is read at run time by every compiled graph.
    """
    key = (float(value), device)
    tensor = _SCALAR_TENSORS.get(key)
    if tensor is None:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
        _SCALAR_TENSORS[key] = tensor
    return tensor


enable_jit_fuser()
