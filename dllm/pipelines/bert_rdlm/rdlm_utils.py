"""
RDLM-specific utilities for Riemannian Diffusion Language Modeling.

This module provides:
- Prior distributions (masked, mixture)
- Noise schedules (geometric)
- Riemannian normal approximation
- Bridge process drift computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from tqdm import tqdm

# Reuse geodesic utilities from bert_sfm
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map,
    log_map,
    make_tangent,
    simplex_to_sphere,
    sphere_to_simplex,
    uniform_prior,
    geodesic_interpolant,
)


# =============================================================================
# Prior Types
# =============================================================================

class RDLMPriorType:
    """RDLM prior distribution types."""
    UNIFORM = "uniform"
    MASKED = "masked"
    MIXTURE = "mixture"


def masked_prior(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mask_idx: int = -1
) -> torch.Tensor:
    """
    Sample from masked prior: all probability mass on mask token.

    On the hypersphere, this is the one-hot vector e_m.

    Args:
        shape: (batch_size, seq_len, vocab_size)
        device: torch device
        dtype: tensor dtype
        mask_idx: index of mask token (default: -1, i.e., last token)

    Returns:
        x0: [B, L, D] one-hot at mask position
    """
    x0 = torch.zeros(shape, device=device, dtype=dtype)
    x0[..., mask_idx] = 1.0
    return x0


def mixture_prior(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mixing_prob: float = 0.5,
    mask_idx: int = -1
) -> torch.Tensor:
    """
    Sample from mixture prior: Bernoulli mixture of masked and uniform.

    With probability λ, sample masked state e_m.
    With probability (1-λ), sample uniform on positive orthant.

    Args:
        shape: (batch_size, seq_len, vocab_size)
        device: torch device
        dtype: tensor dtype
        mixing_prob: probability of selecting masked state (λ)
        mask_idx: index of mask token

    Returns:
        x0: [B, L, D] samples from mixture distribution
    """
    B, L, D = shape

    # Bernoulli mask: True = use masked state
    use_mask = torch.rand(B, L, 1, device=device) < mixing_prob

    # Masked samples (one-hot at mask_idx)
    masked = torch.zeros(shape, device=device, dtype=dtype)
    masked[..., mask_idx] = 1.0

    # Uniform samples on positive orthant of sphere
    uniform = uniform_prior(shape, device, dtype)

    # Mix according to Bernoulli
    x0 = torch.where(use_mask, masked, uniform)

    return x0


def get_rdlm_prior(
    prior_type: str,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    t: Optional[torch.Tensor] = None,
    mask_idx: int = -1,
    mixing_prob: float = 0.5,
    lambda_fn: Optional[Callable] = None
) -> torch.Tensor:
    """
    Get prior samples based on RDLM prior type.

    Args:
        prior_type: "uniform", "masked", or "mixture"
        shape: (batch_size, seq_len, vocab_size)
        device: torch device
        dtype: tensor dtype
        t: [B] time values (for time-dependent mixture)
        mask_idx: index of mask token
        mixing_prob: mixing probability for mixture prior
        lambda_fn: function t -> λ_t for time-dependent mixture

    Returns:
        x0: [B, L, D] prior samples on hypersphere
    """
    if prior_type == RDLMPriorType.UNIFORM:
        return uniform_prior(shape, device, dtype)

    elif prior_type == RDLMPriorType.MASKED:
        return masked_prior(shape, device, dtype, mask_idx)

    elif prior_type == RDLMPriorType.MIXTURE:
        if t is not None and lambda_fn is not None:
            # Time-dependent mixing probability
            B, L, D = shape
            lambda_t = lambda_fn(t)  # [B]
            use_mask = torch.rand(B, L, 1, device=device) < lambda_t.view(B, 1, 1)
            masked = torch.zeros(shape, device=device, dtype=dtype)
            masked[..., mask_idx] = 1.0
            uniform = uniform_prior(shape, device, dtype)
            return torch.where(use_mask, masked, uniform)
        else:
            return mixture_prior(shape, device, dtype, mixing_prob, mask_idx)

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


# =============================================================================
# Noise Schedules
# =============================================================================

def geometric_schedule(
    t: torch.Tensor,
    sigma_0: float = 0.001,
    sigma_T: float = 1.0
) -> torch.Tensor:
    """
    Geometric noise schedule: σ_t = σ_0^{1-t} * σ_T^t

    RDLM default schedule for smooth interpolation between endpoints.

    Args:
        t: [B] or scalar time in [0, 1]
        sigma_0: initial noise level
        sigma_T: final noise level

    Returns:
        sigma_t: noise level at time t
    """
    return (sigma_0 ** (1 - t)) * (sigma_T ** t)


def linear_schedule(
    t: torch.Tensor,
    sigma_0: float = 0.001,
    sigma_T: float = 1.0
) -> torch.Tensor:
    """Linear noise schedule: σ_t = σ_0 + (σ_T - σ_0) * t"""
    return sigma_0 + (sigma_T - sigma_0) * t


def cosine_schedule(
    t: torch.Tensor,
    sigma_0: float = 0.001,
    sigma_T: float = 1.0
) -> torch.Tensor:
    """Cosine noise schedule."""
    import math
    return sigma_0 + (sigma_T - sigma_0) * (1 - torch.cos(t * math.pi / 2))


def bridge_gamma(
    t: torch.Tensor,
    sigma_t: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Bridge drift coefficient: γ_t = σ_t² / (1 - t)

    Args:
        t: [B] time in [0, 1)
        sigma_t: [B] noise level at time t
        eps: small value to prevent division by zero

    Returns:
        gamma_t: [B] drift coefficient
    """
    return sigma_t ** 2 / (1 - t).clamp(min=eps)


# =============================================================================
# Precomputation of α_t and ρ_t
# =============================================================================

def precompute_alpha_rho(
    sigma_fn: Callable,
    n_time_steps: int = 1000,
    n_simulations: int = 10000,
    device: torch.device = torch.device('cpu'),
    prior_type: str = "uniform"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute α_t and ρ_t via Monte Carlo simulation of 1D projected processes.

    RDLM Equations 18-19:
    - z^T_t: projection onto target direction e_k
    - z^0_t: magnitude of orthogonal component

    The bridge SDE in 1D:
    dz^T_t = γ_t * (1 - (z^T_t)²) / z^T_t * dt + σ_t * √(1 - (z^T_t)²) * dB_t

    Args:
        sigma_fn: function mapping t -> σ_t
        n_time_steps: number of discretization steps
        n_simulations: number of Monte Carlo samples
        device: torch device
        prior_type: "uniform" or "masked" (affects initial z^T_0)

    Returns:
        t_grid: [n_time_steps] time points
        alpha_t: [n_time_steps] mean of z^T_t (expected projection onto target)
        rho_t: [n_time_steps] std of orthogonal components
    """
    dt = 1.0 / n_time_steps
    t_grid = torch.linspace(0, 1 - dt, n_time_steps, device=device)

    # Initialize z^T_0 based on prior
    if prior_type == "masked":
        # Masked prior: z^T_0 = 0 (orthogonal to any non-mask target)
        z_T = torch.zeros(n_simulations, device=device)
    else:
        # Uniform prior: z^T_0 ~ uniform on [0, 1]
        # (projection of uniform sphere point onto random basis vector)
        z_T = torch.rand(n_simulations, device=device)

    alpha_t = torch.zeros(n_time_steps, device=device)
    rho_t = torch.zeros(n_time_steps, device=device)

    for i, t in enumerate(tqdm(t_grid, desc="Precomputing α_t, ρ_t", leave=False)):
        # Record statistics
        alpha_t[i] = z_T.mean()
        z_0_sq = (1 - z_T ** 2).clamp(min=0)
        rho_t[i] = z_0_sq.mean().sqrt()  # RMS of orthogonal component

        # Get noise schedule at current time
        sigma_t = sigma_fn(t)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.item()
        gamma_t = sigma_t ** 2 / max(1 - t.item(), 1e-4)

        # Brownian increment
        dB = torch.randn(n_simulations, device=device) * (dt ** 0.5)

        # SDE for z^T (Eq. 18)
        # Avoid division by zero when z_T is near 0
        z_T_safe = z_T.clamp(min=1e-6)
        one_minus_zT_sq = (1 - z_T ** 2).clamp(min=1e-8)

        drift = gamma_t * one_minus_zT_sq / z_T_safe
        diffusion = sigma_t * one_minus_zT_sq.sqrt()

        z_T = z_T + drift * dt + diffusion * dB
        z_T = z_T.clamp(1e-6, 1 - 1e-6)  # Keep in valid range

    return t_grid, alpha_t, rho_t


# =============================================================================
# RDLM Schedule Class
# =============================================================================

@dataclass
class RDLMScheduleConfig:
    """Configuration for RDLM noise schedule."""
    schedule_type: str = "geometric"
    sigma_0: float = 0.001
    sigma_T: float = 1.0
    n_time_steps: int = 1000
    prior_type: str = "uniform"


class RDLMSchedule:
    """
    RDLM noise schedule with precomputed α_t and ρ_t.

    Supports geometric, linear, and cosine schedules.
    """

    def __init__(
        self,
        config: Optional[RDLMScheduleConfig] = None,
        schedule_type: str = "geometric",
        sigma_0: float = 0.001,
        sigma_T: float = 1.0,
        n_time_steps: int = 1000,
        prior_type: str = "uniform",
        device: torch.device = torch.device('cpu'),
        precompute: bool = True
    ):
        if config is not None:
            schedule_type = config.schedule_type
            sigma_0 = config.sigma_0
            sigma_T = config.sigma_T
            n_time_steps = config.n_time_steps
            prior_type = config.prior_type

        self.schedule_type = schedule_type
        self.sigma_0 = sigma_0
        self.sigma_T = sigma_T
        self.n_time_steps = n_time_steps
        self.prior_type = prior_type
        self.device = device

        # Define sigma function
        if schedule_type == "geometric":
            self.sigma_fn = lambda t: geometric_schedule(t, sigma_0, sigma_T)
        elif schedule_type == "linear":
            self.sigma_fn = lambda t: linear_schedule(t, sigma_0, sigma_T)
        elif schedule_type == "cosine":
            self.sigma_fn = lambda t: cosine_schedule(t, sigma_0, sigma_T)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Precompute α_t and ρ_t
        if precompute:
            self.t_grid, self.alpha_t, self.rho_t = precompute_alpha_rho(
                self.sigma_fn, n_time_steps, device=device, prior_type=prior_type
            )
        else:
            self.t_grid = None
            self.alpha_t = None
            self.rho_t = None

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get σ_t at time t."""
        return self.sigma_fn(t)

    def get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Get γ_t = σ_t² / (1-t) at time t."""
        sigma_t = self.sigma_fn(t)
        return bridge_gamma(t, sigma_t)

    def get_alpha_rho(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up precomputed α_t and ρ_t.

        Args:
            t: [B] time values in [0, 1]

        Returns:
            alpha_t: [B] expected projection onto target
            rho_t: [B] std of orthogonal component
        """
        if self.alpha_t is None:
            raise RuntimeError("Schedule was created with precompute=False")

        t_idx = (t * (self.n_time_steps - 1)).long().clamp(0, self.n_time_steps - 1)
        return self.alpha_t[t_idx], self.rho_t[t_idx]

    def to(self, device: torch.device) -> "RDLMSchedule":
        """Move precomputed tensors to device."""
        self.device = device
        if self.t_grid is not None:
            self.t_grid = self.t_grid.to(device)
            self.alpha_t = self.alpha_t.to(device)
            self.rho_t = self.rho_t.to(device)
        return self


# =============================================================================
# Riemannian Normal Sampling
# =============================================================================

def sample_riemannian_normal(
    mean_direction: torch.Tensor,
    scale: torch.Tensor,
    base_point: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sample from Riemannian normal distribution on hypersphere.

    N_R(μ, σ²I) where μ is on the sphere.

    Args:
        mean_direction: [B, L, D] mean direction on sphere
        scale: [B] or [B, 1, 1] standard deviation ρ_t
        base_point: optional base point for tangent space

    Returns:
        samples: [B, L, D] samples on hypersphere
    """
    if base_point is None:
        base_point = mean_direction

    # Sample in tangent space
    tangent_noise = torch.randn_like(mean_direction)
    tangent_noise = make_tangent(base_point, tangent_noise)

    # Scale by ρ_t
    if scale.dim() == 1:
        scale = scale.view(-1, 1, 1)
    tangent_vec = tangent_noise * scale

    # Map to sphere via exponential map
    samples = exp_map(mean_direction, tangent_vec)

    return samples


# =============================================================================
# RDLM Interpolation
# =============================================================================

def rdlm_interpolant(
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    t: torch.Tensor,
    alpha_t: torch.Tensor,
    rho_t: torch.Tensor,
    vocab_size: int,
    add_noise: bool = True
) -> torch.Tensor:
    """
    RDLM interpolation using Riemannian normal approximation.

    The interpolant approximates the bridge marginal X_t | X_1 = e_k
    using a Riemannian normal centered at α_t * e_k with variance ρ_t².

    Args:
        x0: [B, L, D] prior samples on sphere
        target_indices: [B, L] target token indices
        t: [B] time values in [0, 1]
        alpha_t: [B] precomputed mean projection (from schedule)
        rho_t: [B] precomputed std (from schedule)
        vocab_size: vocabulary size
        add_noise: whether to add Riemannian normal noise

    Returns:
        xt: [B, L, D] interpolated samples on sphere
    """
    B, L, D = x0.shape
    device = x0.device
    dtype = x0.dtype

    # Create one-hot targets (on sphere, these are basis vectors e_k)
    x1 = torch.zeros(B, L, vocab_size, device=device, dtype=dtype)
    x1.scatter_(-1, target_indices.unsqueeze(-1), 1.0)

    # Mean direction: scaled toward target
    # At t=0: mostly x0 direction, at t=1: mostly x1 direction
    alpha_t_expanded = alpha_t.view(B, 1, 1)

    # Project x0 orthogonal to x1 to get orthogonal direction
    x0_proj_x1 = x1 * (x0 * x1).sum(-1, keepdim=True)  # Component of x0 along x1
    x0_orthogonal = x0 - x0_proj_x1
    x0_orth_norm = x0_orthogonal.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x0_orthogonal = x0_orthogonal / x0_orth_norm  # Normalized orthogonal component

    # Construct mean on sphere: α_t * e_k + √(1-α_t²) * orthogonal_direction
    sqrt_one_minus_alpha_sq = torch.sqrt((1 - alpha_t ** 2).clamp(min=0)).view(B, 1, 1)
    mean_on_sphere = alpha_t_expanded * x1 + sqrt_one_minus_alpha_sq * x0_orthogonal
    mean_on_sphere = mean_on_sphere / mean_on_sphere.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    if add_noise:
        # Sample from Riemannian normal around mean
        xt = sample_riemannian_normal(mean_on_sphere, rho_t)
    else:
        xt = mean_on_sphere

    return xt


# =============================================================================
# Bridge Process Drift
# =============================================================================

def compute_target_drift(
    xt: torch.Tensor,
    target_indices: torch.Tensor,
    gamma_t: torch.Tensor,
    vocab_size: int
) -> torch.Tensor:
    """
    Compute bridge drift toward target: γ_t * log_{X_t}(e_k)

    RDLM Equation 12: b(x,t) = γ_t * log_x(e_k)

    Args:
        xt: [B, L, D] current position on sphere
        target_indices: [B, L] target token indices
        gamma_t: [B] or scalar drift coefficient
        vocab_size: vocabulary size

    Returns:
        drift: [B, L, D] tangent vector at xt pointing toward target
    """
    B, L, D = xt.shape
    device = xt.device
    dtype = xt.dtype

    # Create one-hot targets
    x1 = torch.zeros(B, L, vocab_size, device=device, dtype=dtype)
    x1.scatter_(-1, target_indices.unsqueeze(-1), 1.0)

    # Compute log map (direction from xt to x1)
    log_vec = log_map(xt, x1)

    # Scale by γ_t
    if isinstance(gamma_t, torch.Tensor) and gamma_t.dim() >= 1:
        gamma_t = gamma_t.view(-1, 1, 1)

    drift = gamma_t * log_vec

    return drift


def expected_drift_from_probs(
    xt: torch.Tensor,
    probs: torch.Tensor,
    gamma_t: torch.Tensor
) -> torch.Tensor:
    """
    Compute expected drift from predicted probabilities.

    E[γ_t * log_{X_t}(e_k)] = γ_t * Σ_k p_k * log_{xt}(e_k)

    This is used when the model outputs probabilities rather than drift directly.

    Args:
        xt: [B, L, D] current position on sphere
        probs: [B, L, D] predicted probabilities
        gamma_t: [B] drift coefficient

    Returns:
        expected_drift: [B, L, D] expected tangent vector
    """
    B, L, D = xt.shape
    device = xt.device

    # For efficiency, we compute the weighted sum of log maps
    # log_xt(e_k) = angle_k * (e_k - xt * cos(angle_k)) / |e_k - xt * cos(angle_k)|

    # Inner product <xt, e_k> for each k
    # xt: [B, L, D], e_k is k-th column of identity
    # <xt, e_k> = xt[..., k]
    inner = xt  # [B, L, D] - inner product with each basis vector
    inner = inner.clamp(-1 + 1e-6, 1 - 1e-6)

    # Angles to each basis vector
    angles = torch.arccos(inner)  # [B, L, D]

    # Direction vectors: e_k - xt * <xt, e_k>
    # e_k has 1 at position k, so e_k - xt * inner_k
    eye = torch.eye(D, device=device)  # [D, D]
    directions = eye.unsqueeze(0).unsqueeze(0) - xt.unsqueeze(-1) * inner.unsqueeze(-2)  # [B, L, D, D]

    # Normalize directions
    dir_norms = directions.norm(dim=-2, keepdim=True).clamp(min=1e-8)  # [B, L, 1, D]
    directions = directions / dir_norms  # [B, L, D, D]

    # Log vectors: angle * direction
    log_vecs = directions * angles.unsqueeze(-2)  # [B, L, D, D]

    # Weight by probabilities and sum over vocabulary
    probs_expanded = probs.unsqueeze(-2)  # [B, L, 1, D]
    expected_log = (log_vecs * probs_expanded).sum(dim=-1)  # [B, L, D]

    # Scale by γ_t
    if isinstance(gamma_t, torch.Tensor) and gamma_t.dim() >= 1:
        gamma_t = gamma_t.view(-1, 1, 1)

    return gamma_t * expected_log
