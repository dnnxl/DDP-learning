from torch import nn, einsum
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torch
import math
import torch.nn as nn

from .utils import divisible_by

def Downsample(dim, dim_out=None):
    """
    Downsampling module that reduces the spatial resolution of feature maps.

    This function performs downsampling using:
    1. Rearranging feature maps to group neighboring pixels into channel space.
    2. A 1x1 convolution to adjust the number of output channels.

    Args:
        dim (int): Number of input channels.
        dim_out (int, optional): Number of output channels. If None, it defaults to dim.

    Returns:
        nn.Sequential: A PyTorch module performing downsampling followed by 1x1 convolution.
    """
    dim_out = dim_out if dim_out is not None else dim  # Default dim_out to dim if not provided

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),  # Spatial to channel rearrangement
        nn.Conv2d(dim * 4, dim_out, kernel_size=1)  # 1x1 convolution to adjust channel count
    )

def Upsample(dim, dim_out=None):
    """
    Upsampling module that increases the spatial resolution of feature maps.

    This function creates a sequential module that:
    1. Doubles the spatial resolution using nearest-neighbor interpolation.
    2. Applies a 2D convolution to refine the upsampled features.

    Args:
        dim (int): Number of input channels.
        dim_out (int, optional): Number of output channels. If None, it defaults to dim.

    Returns:
        nn.Sequential: A PyTorch module performing upsampling followed by convolution.
    """
    dim_out = dim_out if dim_out is not None else dim  # Default dim_out to dim if not provided

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample by a factor of 2
        nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)  # 3x3 convolution for feature refinement
    )

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    RMSNorm is a type of normalization technique that scales the input tensor
    based on the root mean square (RMS) of its activations. Unlike batch normalization,
    it does not depend on batch statistics and is computationally efficient.

    Attributes:
        scale (float): A scaling factor based on the input dimension.
        g (torch.nn.Parameter): A learnable parameter to scale the normalized output.

    Args:
        dim (int): The number of channels (or features) in the input tensor.
    
    Forward Pass:
        x (torch.Tensor): Input tensor of shape (batch_size, dim, height, width).
        
    Returns:
        torch.Tensor: The RMS-normalized tensor with the same shape as input.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5  # Scaling factor based on feature dimension
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))  # Learnable gain parameter

    def forward(self, x):
        """
        Applies RMS normalization to the input tensor.

        The input tensor is normalized along the feature dimension (dim=1)
        using L2 normalization, and then scaled by `g` and `scale`.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim, height, width).

        Returns:
            torch.Tensor: Normalized and scaled tensor of the same shape as input.
        """
        return F.normalize(x, dim = 1) * self.g * self.scale
    
class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding Module.

    This module computes sinusoidal positional embeddings, commonly used in transformers and diffusion models.
    The embeddings help encode positional information in a way that preserves relative distances between elements.

    Attributes:
        dim (int): The dimension of the embedding vector.
        theta (float): A scaling factor (default: 10000) used in sinusoidal computations.

    Args:
        dim (int): The embedding dimension.
        theta (float, optional): Scaling factor for frequency calculation (default: 10000).

    Forward Pass:
        x (torch.Tensor): A 1D tensor representing positions (e.g., time steps in a diffusion model).
        
    Returns:
        torch.Tensor: A tensor containing the sinusoidal positional embeddings.
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim  # Embedding dimension
        self.theta = theta  # Scaling factor for frequency calculation

    def forward(self, x):
        """
        Computes sinusoidal embeddings for the given input tensor.

        The embeddings consist of sine and cosine functions with frequencies
        determined by the scaling factor `theta`. The embeddings are concatenated
        along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, dim).
        """
        device = x.device
        half_dim = self.dim // 2  # Half of the embedding dimension

        # Compute the scaling factor for different frequencies
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # Compute frequency factors

        # Compute sinusoidal embeddings
        emb = x[:, None] * emb[None, :]  # Expand dimensions to align shapes
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Concatenate sine and cosine components

        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding module with an option for either random (fixed) or learned embeddings.
    
    This module generates sinusoidal positional embeddings using sine and cosine functions, inspired by
    @crowsonkb's implementation in v-diffusion-jax.
    
    Reference:
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    
    Parameters:
    -----------
    dim : int
        The total dimension of the positional embedding. Must be an even number.
    is_random : bool, optional (default=False)
        If True, the embedding frequencies are randomly initialized and fixed.
        If False, the embedding frequencies are learned during training.
    """

    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        
        # Ensure dimension is even for proper sine-cosine pairing
        assert divisible_by(dim, 2), "Embedding dimension must be divisible by 2."
        
        half_dim = dim // 2  # Half the dimension for sine-cosine components
        
        # Create frequency parameters, learnable if is_random=False
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal positional embeddings for the input tensor.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size], typically containing scalar values (e.g., timesteps).
        
        Returns:
        --------
        torch.Tensor
            Positional embedding tensor of shape [batch_size, dim].
        """
        
        # Reshape input tensor to [batch_size, 1] for broadcasting
        x = rearrange(x, 'b -> b 1')
        
        # Compute frequency values scaled by 2π for periodicity
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        
        # Apply sine and cosine transformations and concatenate
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        
        # Append original input x to retain absolute positional information
        fouriered = torch.cat((x, fouriered), dim=-1)
        
        return fouriered

class Block(nn.Module):
    """
    A convolutional residual block with normalization, activation, and optional scale-shift conditioning.

    This block consists of:
    1. A 3x3 convolution (`proj`) for feature transformation.
    2. RMS normalization (`norm`) for stable training.
    3. SiLU activation (`act`) for non-linearity.
    4. Optional scale-shift conditioning.
    5. Dropout (`dropout`) for regularization.

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        dropout (float, optional): Dropout probability. Defaults to 0.

    Attributes:
        proj (nn.Conv2d): 3x3 convolutional layer with padding=1 (preserves spatial dimensions).
        norm (RMSNorm): Root Mean Square Normalization layer.
        act (nn.SiLU): SiLU (Swish) activation function.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.

    Forward Input:
        x (Tensor): Input feature map of shape `(batch_size, dim, height, width)`.
        scale_shift (tuple[Tensor, Tensor] | None, optional):
            - If provided, contains `(scale, shift)`, both of shape `(batch_size, dim_out, 1, 1)`.
            - Used to scale and shift feature maps (common in diffusion models).

    Forward Output:
        Tensor: Transformed feature map of shape `(batch_size, dim_out, height, width)`.

    Example Usage:
    ```python
    block = Block(dim=64, dim_out=128, dropout=0.1)
    x = torch.randn(8, 64, 32, 32)  # Batch of 8 images, 64 channels, 32x32 resolution
    out = block(x)  # Output shape: (8, 128, 32, 32)
    ```
    """

    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)  # 3x3 convolution
        self.norm = RMSNorm(dim_out)  # Root Mean Square Normalization
        self.act = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x, scale_shift=None):
        """
        Forward pass of the Block.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.
            scale_shift (tuple[Tensor, Tensor] | None, optional):
                - If provided, contains `(scale, shift)`, both of shape `(batch_size, dim_out, 1, 1)`.
                - Used for adaptive normalization or conditioning (e.g., in diffusion models).

        Returns:
            Tensor: Transformed output of shape `(batch_size, dim_out, height, width)`.
        """
        x = self.proj(x)  # Apply 3x3 convolution
        x = self.norm(x)  # Apply RMS normalization

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # Apply scale and shift conditioning

        x = self.act(x)  # Apply SiLU activation
        return self.dropout(x)  # Apply dropout
    
class ResnetBlock(nn.Module):
    """
    A ResNet-inspired block with optional time-based conditioning, commonly used in diffusion models.

    This block consists of:
    1. Two convolutional blocks (`Block`), each with normalization, activation, and dropout.
    2. Optional time-based conditioning using a Multi-Layer Perceptron (MLP).
    3. A residual connection with a `1x1` convolution (`res_conv`) to match input and output dimensions.

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        time_emb_dim (int, optional): Dimension of the time embedding (if used).
        dropout (float, optional): Dropout probability. Defaults to `0.0`.

    Attributes:
        mlp (nn.Sequential | None): Optional MLP for time-based conditioning.
        block1 (Block): First convolutional block.
        block2 (Block): Second convolutional block.
        res_conv (nn.Conv2d | nn.Identity): Residual connection.

    Forward Input:
        x (Tensor): Input feature map of shape `(batch_size, dim, height, width)`.
        time_emb (Tensor | None, optional): Time-based embedding of shape `(batch_size, time_emb_dim)`.

    Forward Output:
        Tensor: Output feature map of shape `(batch_size, dim_out, height, width)`.

    Example Usage:
    ```python
    resnet_block = ResnetBlock(dim=64, dim_out=128, time_emb_dim=32, dropout=0.1)
    x = torch.randn(8, 64, 32, 32)  # Batch of 8 images, 64 channels, 32x32 resolution
    time_emb = torch.randn(8, 32)   # Time embeddings for conditioning
    out = resnet_block(x, time_emb)  # Output shape: (8, 128, 32, 32)
    ```
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()

        # Optional MLP for time-based conditioning
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        # Two convolutional blocks
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)

        # Residual connection: identity if `dim == dim_out`, else `1x1` convolution
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Forward pass of the ResnetBlock.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.
            time_emb (Tensor | None, optional): Time-based embedding of shape `(batch_size, time_emb_dim)`.
        
        Returns:
            Tensor: Transformed output of shape `(batch_size, dim_out, height, width)`.
        """
        scale_shift = None  # Default: No conditioning

        # Apply time embedding if available
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)  # MLP projection
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')  # Reshape for broadcasting
            scale_shift = time_emb.chunk(2, dim=1)  # Split into (scale, shift)

        # First convolutional block with optional scale-shift conditioning
        h = self.block1(x, scale_shift=scale_shift)

        # Second convolutional block
        h = self.block2(h)

        # Residual connection
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    Linear attention mechanism with memory keys and values.

    This module implements a linear attention mechanism that computes attention 
    using kernelized operations to improve efficiency over standard softmax attention.

    Args:
        dim (int): The number of input channels.
        heads (int, optional): Number of attention heads. Defaults to 4.
        dim_head (int, optional): Dimensionality of each attention head. Defaults to 32.
        num_mem_kv (int, optional): Number of memory key-value pairs. Defaults to 4.

    Attributes:
        scale (float): Scaling factor for queries.
        heads (int): Number of attention heads.
        norm (RMSNorm): RMS normalization layer.
        mem_kv (nn.Parameter): Memory key-value pairs.
        to_qkv (nn.Conv2d): Convolutional layer to produce queries, keys, and values.
        to_out (nn.Sequential): Output layer consisting of a convolution and normalization.

    Forward Input:
        x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.

    Forward Output:
        Tensor: Output tensor of shape `(batch_size, dim, height, width)`.

    Example Usage:
    ```python
    linear_attention = LinearAttention(dim=64, heads=4, dim_head=32, num_mem_kv=4)
    x = torch.randn(8, 64, 32, 32)  # Batch of 8 images, 64 channels, 32x32 resolution
    out = linear_attention(x)  # Output shape: (8, 64, 32, 32)
    ```
    """

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head ** -0.5  # Scaling factor for queries
        self.heads = heads
        hidden_dim = dim_head * heads  # Total hidden dimension for all heads

        self.norm = RMSNorm(dim)  # RMS normalization layer

        # Memory key-value pairs
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))

        # Convolution to generate queries, keys, and values
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Output layer: convolution followed by normalization
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Forward pass of the LinearAttention.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, dim, height, width)`.
        """
        b, c, h, w = x.shape  # Get batch size, channels, height, width

        x = self.norm(x)  # Normalize input

        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)  # Split into (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # Expand memory key-value pairs to match batch size
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        
        # Concatenate memory with current keys and values
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        # Softmax normalization for queries and keys
        q = q.softmax(dim=-2)  # Normalize queries
        k = k.softmax(dim=-1)  # Normalize keys

        q = q * self.scale  # Scale queries

        # Compute context via einsum operation
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Compute output via einsum operation
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)  # Rearrange for output
        return self.to_out(out)  # Final output

class Attention(nn.Module):
    """
    Attention module that implements an efficient attention mechanism with optional memory keys and values.

    This module utilizes attention heads and supports kernelized operations to improve efficiency.
    It can also incorporate memory key-value pairs for improved context retrieval.

    Args:
        dim (int): The number of input channels.
        heads (int, optional): Number of attention heads. Defaults to 4.
        dim_head (int, optional): Dimensionality of each attention head. Defaults to 32.
        num_mem_kv (int, optional): Number of memory key-value pairs. Defaults to 4.
        flash (bool, optional): Whether to use flash attention (if supported). Defaults to False.

    Attributes:
        heads (int): Number of attention heads.
        norm (RMSNorm): RMS normalization layer.
        attend (Attend): Attention mechanism, optionally using flash attention.
        mem_kv (nn.Parameter): Memory key-value pairs.
        to_qkv (nn.Conv2d): Convolutional layer to produce queries, keys, and values.
        to_out (nn.Conv2d): Output layer to project back to the original dimension.

    Forward Input:
        x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.

    Forward Output:
        Tensor: Output tensor of shape `(batch_size, dim, height, width)`.

    Example Usage:
    ```python
    attention = Attention(dim=64, heads=4, dim_head=32, num_mem_kv=4, flash=False)
    x = torch.randn(8, 64, 32, 32)  # Batch of 8 images, 64 channels, 32x32 resolution
    out = attention(x)  # Output shape: (8, 64, 32, 32)
    ```
    """

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads  # Total hidden dimension for all heads

        self.norm = RMSNorm(dim)  # RMS normalization layer
        self.attend = Attend(flash=flash)  # Attention mechanism (possibly using flash attention)

        # Memory key-value pairs for improved attention context
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        # Convolution to generate queries, keys, and values
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Output layer: convolution to project back to the original dimension
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, dim, height, width)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, dim, height, width)`.
        """
        b, c, h, w = x.shape  # Get batch size, channels, height, and width

        x = self.norm(x)  # Normalize input

        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)  # Split into (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        # Expand memory key-value pairs to match batch size
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)

        # Concatenate memory with current keys and values
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        # Apply attention mechanism
        out = self.attend(q, k, v)

        # Rearrange output to match expected shape
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)  # Final output

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Unet(nn.Module):
    """
    U-Net architecture for diffusion models.

    This implementation supports various enhancements, including:
    - Learned sinusoidal embeddings for time conditioning.
    - Attention mechanisms (full and linear attention).
    - Configurable feature scaling using `dim_mults`.

    Args:
        dim (int): Base channel size for the network.
        init_dim (int, optional): Initial convolutional layer dimension.
        out_dim (int, optional): Output channel size.
        dim_mults (tuple, optional): Multipliers for channel dimensions across U-Net layers.
        channels (int, optional): Number of input channels (e.g., RGB = 3).
        self_condition (bool, optional): If True, includes a self-conditioning input.
        learned_variance (bool, optional): If True, outputs variance prediction for diffusion models.
        learned_sinusoidal_cond (bool, optional): If True, uses learned sinusoidal time embeddings.
        random_fourier_features (bool, optional): If True, applies random Fourier features for embeddings.
        learned_sinusoidal_dim (int, optional): Feature dimension for learned sinusoidal embeddings.
        sinusoidal_pos_emb_theta (int, optional): Scale factor for sinusoidal embeddings.
        dropout (float, optional): Dropout rate in ResNet blocks.
        attn_dim_head (int, optional): Head dimension for attention layers.
        attn_heads (int, optional): Number of attention heads.
        full_attn (bool or tuple, optional): Whether to apply full attention at each U-Net level.
        flash_attn (bool, optional): Whether to use Flash Attention for efficiency.

    """

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,  # Full attention at the deepest layer by default
        flash_attn=False
    ):
        super().__init__()

        # Initialize input properties
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        # Initial convolutional layer
        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)

        # Define channel sizes at different U-Net levels
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding network
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Configure attention mechanisms
        if full_attn is None:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)  # Full attention at the last layer
        num_stages = len(dim_mults)
        full_attn = tuple(full_attn) * num_stages if isinstance(full_attn, bool) else full_attn
        attn_heads = (attn_heads,) * num_stages if isinstance(attn_heads, int) else attn_heads
        attn_dim_head = (attn_dim_head,) * num_stages if isinstance(attn_dim_head, int) else attn_dim_head

        # Ensure all parameter lists match the number of U-Net stages
        assert len(full_attn) == len(dim_mults)

        # Partial functions for different blocks
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # Define U-Net encoder (downsampling path)
        self.downs = nn.ModuleList([])
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
            ]))

        # Bottleneck layers
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # Define U-Net decoder (upsampling path)
        self.ups = nn.ModuleList([])
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, kernel_size=3, padding=1)
            ]))

        # Define final layers
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, kernel_size=1)

    @property
    def downsample_factor(self):
        """ Returns the total downsampling factor of the U-Net. """
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        """ 
        Forward pass of the U-Net.
        
        Args:
            x (torch.Tensor): Input image tensor.
            time (torch.Tensor): Time step embeddings for conditioning.
            x_self_cond (torch.Tensor, optional): Self-conditioning tensor.

        Returns:
            torch.Tensor: Output image tensor.
        """
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


