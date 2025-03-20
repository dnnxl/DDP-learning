import math
import torch
import torch.nn.functional as F

from torch.nn import Module
from functools import partial
from tqdm import tqdm

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extracts values from tensor `a` using indices from tensor `t` and reshapes the output.
    
    Parameters:
    a (torch.Tensor): The source tensor from which values are gathered.
    t (torch.Tensor): Indices for gathering values from `a`.
    x_shape (tuple): The target shape to reshape the extracted values.
    
    Returns:
    torch.Tensor: The gathered and reshaped tensor.
    """
    b, *_ = t.shape  # Extract batch size
    out = a.gather(-1, t)  # Gather values along the last dimension using indices from `t`
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # Reshape to match the desired dimensions

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Generates a linear beta schedule, as proposed in the original DDPM (Denoising Diffusion Probabilistic Models) paper.
    
    Parameters:
    timesteps (int): The number of diffusion steps.
    
    Returns:
    torch.Tensor: Linearly spaced beta values.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001  # Smallest beta value
    beta_end = scale * 0.02  # Largest beta value
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generates a cosine-based beta schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
    
    Reference: https://openreview.net/forum?id=-NEXDKk8gZ
    
    Parameters:
    timesteps (int): The number of diffusion steps.
    s (float, optional): Small offset to prevent extreme values. Default is 0.008.
    
    Returns:
    torch.Tensor: Cosine-based beta values.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    
    # Compute the cumulative alpha values using a cosine function
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize by the first value
    
    # Compute beta values as the difference between successive alpha values
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)  # Clip values to ensure numerical stability

def sigmoid_beta_schedule(timesteps: int, start: float = -3, end: float = 3, tau: float = 1, clamp_min: float = 1e-5) -> torch.Tensor:
    """
    Generates a sigmoid-based beta schedule, as proposed in "EDM: Elucidating the Design Space of Diffusion-Based Generative Models".
    
    Reference: https://arxiv.org/abs/2212.11972 - Figure 8
    
    This schedule is particularly beneficial for training high-resolution images (> 64x64).
    
    Parameters:
    timesteps (int): The number of diffusion steps.
    start (float, optional): Starting value for sigmoid transformation. Default is -3.
    end (float, optional): Ending value for sigmoid transformation. Default is 3.
    tau (float, optional): Controls the steepness of the sigmoid function. Default is 1.
    clamp_min (float, optional): Minimum value to avoid numerical instability. Default is 1e-5.
    
    Returns:
    torch.Tensor: Sigmoid-based beta values.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    
    # Compute cumulative alpha values using a sigmoid function
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize by the first value
    
    # Compute beta values as the difference between successive alpha values
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)  # Clip values for stability

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t

class GaussianDiffusion(Module):
    """
    A class implementing a Gaussian diffusion model for generative modeling.
    
    This model applies a noise schedule over timesteps to transform an image into noise and then learns to reverse the process.
    
    Attributes:
        model (Module): The underlying neural network used for denoising.
        image_size (tuple): The spatial dimensions of the input image.
        timesteps (int): Number of diffusion steps.
        sampling_timesteps (int): Number of timesteps used for sampling (DDIM sampling if less than timesteps).
        objective (str): The type of prediction the model is trained for ('pred_noise', 'pred_x0', 'pred_v').
        beta_schedule (str): Type of noise variance schedule ('linear', 'cosine', 'sigmoid').
        ddim_sampling_eta (float): Hyperparameter for DDIM sampling controlling stochasticity.
        offset_noise_strength (float): Strength of noise offset used in diffusion (helps stabilize training).
        min_snr_loss_weight (bool): Whether to use minimum signal-to-noise ratio (SNR) loss weighting.
        min_snr_gamma (float): Clipping threshold for SNR loss weight.
        immiscible (bool): Whether to apply immiscible diffusion (an advanced diffusion method).
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_v',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        offset_noise_strength=0.,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        immiscible=False
    ):
        super().__init__()
        
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim), "Model channels and output dimensions must match."
        
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        
        # Ensure image_size is a tuple of (height, width)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, "Image size must be an integer or a tuple/list of two integers."
        self.image_size = image_size
        
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, "Objective must be 'pred_noise', 'pred_x0', or 'pred_v'."
        
        # Define beta schedule function
        beta_schedule_fn = {
            'linear': linear_beta_schedule,
            'cosine': cosine_beta_schedule,
            'sigmoid': sigmoid_beta_schedule
        }.get(beta_schedule, None)
        
        if beta_schedule_fn is None:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")
        
        # Compute beta values and other diffusion-related parameters
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = sampling_timesteps or timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
        # Helper function to register buffers
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute useful values for diffusion process
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Compute posterior variance for sampling
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # Enable immiscible diffusion if needed
        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength
        
        # Compute loss weighting using signal-to-noise ratio (SNR)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if min_snr_loss_weight:
            snr = snr.clone().clamp_(max=min_snr_gamma)
        
        # Define loss weight depending on the objective
        if objective == 'pred_noise':
            register_buffer('loss_weight', snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', snr / (snr + 1))
        
        # Normalize images between [-1, 1] (can be disabled with auto_normalize=False)
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        
    @property
    def device(self):
        return self.betas.device
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predicts the original image x_0 from noisy input x_t."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        """Predicts the noise from the original image x_0 and noisy input x_t."""
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def q_posterior(self, x_start, x_t, t):
        """Computes posterior mean and variance for the reverse diffusion process."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
