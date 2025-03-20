from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

class Attend(nn.Module):
    """
    Efficient attention mechanism with support for flash attention.

    This module implements an attention mechanism that can utilize optimized
    attention computation using flash attention when available. It also supports
    dropout for regularization.

    Args:
        dropout (float, optional): Dropout probability. Defaults to 0.
        flash (bool, optional): Whether to use flash attention (if supported). Defaults to False.
        scale (float | None, optional): Scaling factor for queries. Defaults to None.

    Attributes:
        dropout (float): Dropout probability.
        scale (float | None): Scaling factor for queries.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
        flash (bool): Flag indicating whether to use flash attention.
        cpu_config (AttentionConfig): Configuration for CPU attention.
        cuda_config (AttentionConfig | None): Configuration for CUDA attention.

    Example Usage:
    ```python
    attend = Attend(dropout=0.1, flash=True)
    q = torch.randn(8, 4, 32, 64)  # Batch of 8, 4 heads, 32 queries, 64 features
    k = torch.randn(8, 4, 32, 64)  # Batch of 8, 4 heads, 32 keys, 64 features
    v = torch.randn(8, 4, 32, 64)  # Batch of 8, 4 heads, 32 values, 64 features
    out = attend(q, k, v)  # Output shape: (8, 4, 32, 64)
    ```
    """

    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), \
            'Flash attention requires PyTorch 2.0 or above'

        # Determine efficient attention configurations for CPU and CUDA
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on CUDA')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or memory-efficient attention if input tensor is on CUDA')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        """
        Compute attention using flash attention.

        Args:
            q (Tensor): Query tensor of shape `(batch_size, heads, query_length, head_dim)`.
            k (Tensor): Key tensor of shape `(batch_size, heads, key_length, head_dim)`.
            v (Tensor): Value tensor of shape `(batch_size, heads, value_length, head_dim)`.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)  # Scale queries if scale is defined

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Determine the configuration for efficient attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # Apply scaled dot-product attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        Forward pass for the attention mechanism.

        Args:
            q (Tensor): Query tensor of shape `(batch_size, heads, query_length, head_dim)`.
            k (Tensor): Key tensor of shape `(batch_size, heads, key_length, head_dim)`.
            v (Tensor): Value tensor of shape `(batch_size, heads, value_length, head_dim)`.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)  # Use flash attention if enabled

        # Default scale for queries
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # Compute similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale  # Calculate attention scores

        # Apply softmax to obtain attention weights
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)  # Apply dropout to attention weights

        # Aggregate values based on attention weights
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
