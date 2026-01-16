"""
RDLM-specific utilities for Riemannian Diffusion Language Modeling.

This module provides:
- Prior distributions (masked, mixture)
- Noise schedules (geometric)
- Riemannian normal approximation
- Bridge process drift computation
"""

import abc
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from tqdm import tqdm


def _is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    # Check various environment variables used by different launchers
    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", "0")
    return local_rank == "0" and rank == "0"

# Reuse geodesic utilities from bert_sfm
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map_inplace,
    log_map_inplace,
    make_tangent,
    simplex_to_sphere,
    sphere_to_simplex,
    uniform_prior,
    geodesic_interpolant,
    expected_logmap_to_onehots,
)


# =============================================================================
# Weighting utilities (matches rdlm/utils/weight_utils.py)
# =============================================================================

_WEIGHT_FN = {}


def register_weight_fn(cls=None, *, name=None):
    """Decorator for registering weight functions."""
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _WEIGHT_FN:
            raise ValueError(f"Already registered weight fn with name: {local_name}")
        _WEIGHT_FN[local_name] = cls
        return cls

    return _register(cls) if cls is not None else _register


def get_weight_fn(name):
    return _WEIGHT_FN[name]


class SchedulerWeight(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, t):
        pass

    @abc.abstractmethod
    def cum_weight_fn(self, t):
        pass

    @property
    def norm_const(self):
        return self.cum_weight_fn(1)


@register_weight_fn(name="default")
class DefaultWeight(SchedulerWeight):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, t):
        return torch.ones_like(t)

    def cum_weight_fn(self, t):
        return t


@register_weight_fn(name="step")
class StepWeight(SchedulerWeight):
    def __init__(self, **kwargs):
        super().__init__()
        self.left = kwargs.get("left", 0.3)
        self.right = kwargs.get("right", 0.6)
        self.ub = kwargs.get("ub", 1.0)
        self.lb = kwargs.get("lb", 1e-4)

    def forward(self, t):
        return torch.where(
            (t > self.left) & (t < self.right),
            torch.ones_like(t) * self.ub,
            torch.ones_like(t) * self.lb,
        )

    def cum_weight_fn(self, t):
        if isinstance(t, torch.Tensor):
            return torch.where(
                t < self.left,
                t * self.lb,
                torch.where(
                    t > self.right,
                    self.left * self.lb + (self.right - self.left) * self.ub + (t - self.right) * self.lb,
                    self.left * self.lb + (t - self.left) * self.ub,
                ),
            )
        if t < self.left:
            return t * self.lb
        if t > self.right:
            return self.left * self.lb + (self.right - self.left) * self.ub + (t - self.right) * self.lb
        return self.left * self.lb + (t - self.left) * self.ub


# =============================================================================
# Prior Types
# =============================================================================

class RDLMPriorType:
    """RDLM prior distribution types."""
    UNIFORM = "uniform"
    MASKED = "masked"
    MIXTURE = "mixture"


def _resolve_mask_idx(mask_idx: int, vocab_dim: int) -> int:
    return mask_idx if mask_idx >= 0 else vocab_dim + mask_idx


def mixture_prob_schedule(t: torch.Tensor, mix_type: str, step_thr: float = 0.0) -> torch.Tensor:
    """RDLM mixture schedule: returns probability of uniform state."""
    if mix_type == "linear":
        return 1 - t
    if mix_type == "sqrt":
        return 1 - t.sqrt()
    if "step" in mix_type:
        return torch.where(t < step_thr, torch.ones_like(t), torch.zeros_like(t))
    raise ValueError(f"Invalid mix_type: {mix_type}")


def uniform_barycenter_prior(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic barycenter on the sphere: 1 / sqrt(D) in each dimension."""
    scale = 1.0 / math.sqrt(shape[-1])
    return torch.full(shape, scale, device=device, dtype=dtype)


def init_uniform_prior(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mask_idx: int = -1,
) -> torch.Tensor:
    """Uniform prior for Init path: uniform over token dims, zero on mask dim."""
    vocab_dim = shape[-1]
    mask_idx = _resolve_mask_idx(mask_idx, vocab_dim)
    x0 = torch.ones(shape, device=device, dtype=dtype)
    x0[..., mask_idx] = 0.0
    return x0 / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def initial_prior(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rlambda: float = 1.0,
    mask_idx: int = -1,
) -> torch.Tensor:
    """Initial distribution for LogBridge_Init (distribution.Initial)."""
    vocab_dim = shape[-1]
    mask_idx = _resolve_mask_idx(mask_idx, vocab_dim)
    token_size = vocab_dim - 1
    if token_size <= 0:
        return uniform_barycenter_prior(shape, device, dtype)
    init_val = math.sqrt(rlambda / token_size) if rlambda > 0 else 0.0
    x0 = torch.full(shape, init_val, device=device, dtype=dtype)
    x0[..., mask_idx] = math.sqrt(max(1.0 - rlambda, 0.0))
    return x0


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

    With probability mixing_prob, sample uniform barycenter.
    With probability (1-mixing_prob), sample masked state e_m.

    Args:
        shape: (batch_size, seq_len, vocab_size)
        device: torch device
        dtype: tensor dtype
        mixing_prob: probability of selecting uniform state
        mask_idx: index of mask token

    Returns:
        x0: [B, L, D] samples from mixture distribution
    """
    B, L, D = shape

    # Bernoulli mask: True = use uniform state (per batch, broadcast across tokens)
    use_uniform = torch.rand(B, 1, 1, device=device) < mixing_prob

    # Masked samples (one-hot at mask_idx)
    masked = torch.zeros(shape, device=device, dtype=dtype)
    masked[..., mask_idx] = 1.0

    # Uniform barycenter on sphere
    uniform = uniform_barycenter_prior(shape, device, dtype)

    # Mix according to Bernoulli
    x0 = torch.where(use_uniform, uniform, masked)

    return x0


def get_rdlm_prior(
    prior_type: str,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    t: Optional[torch.Tensor] = None,
    mask_idx: int = -1,
    mixing_prob: float = 0.5,
    lambda_fn: Optional[Callable] = None,
    mix_type: Optional[str] = None,
    mix_step_thr: float = 0.0,
    init_lambda: Optional[float] = None,
    add_mask_token: bool = False,
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
        mixing_prob: probability of selecting uniform state in mixture
        lambda_fn: function t -> lambda_t for time-dependent mixture
        mix_type: mix schedule name (linear/sqrt/step)
        mix_step_thr: step threshold for step schedule
        init_lambda: rlambda for LogBridge_Init (None -> infer from prior_type)
        add_mask_token: whether to treat mask as extra dimension

    Returns:
        x0: [B, L, D] prior samples on hypersphere
    """
    if prior_type == RDLMPriorType.UNIFORM:
        if add_mask_token:
            if init_lambda is None:
                init_lambda = 1.0
            return initial_prior(shape, device, dtype, rlambda=init_lambda, mask_idx=mask_idx)
        return uniform_barycenter_prior(shape, device, dtype)

    elif prior_type == RDLMPriorType.MASKED:
        if add_mask_token and init_lambda is not None and init_lambda != 0.0:
            return initial_prior(shape, device, dtype, rlambda=init_lambda, mask_idx=mask_idx)
        return masked_prior(shape, device, dtype, mask_idx)

    elif prior_type == RDLMPriorType.MIXTURE:
        if t is None:
            t = torch.zeros(shape[0], device=device)
        if mix_type is not None:
            uniform_prob = mixture_prob_schedule(t, mix_type, step_thr=mix_step_thr)
        elif lambda_fn is not None:
            uniform_prob = lambda_fn(t)
        else:
            uniform_prob = torch.full_like(t, mixing_prob)
        # Sample mixture (uniform vs mask) per batch (broadcast across tokens)
        B, L, _ = shape
        uniform_prob = uniform_prob.view(B, 1, 1)
        use_uniform = torch.rand(B, 1, 1, device=device) < uniform_prob
        masked = torch.zeros(shape, device=device, dtype=dtype)
        masked[..., mask_idx] = 1.0
        uniform = uniform_barycenter_prior(shape, device, dtype)
        return torch.where(use_uniform, uniform, masked)

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
    """Cosine noise schedule (matches rdlm/scheduler_lib.py)."""
    cos_t = torch.cos(math.pi / 2 * (1 - t))
    return sigma_0 * (1 - cos_t) + sigma_T * cos_t


def int_beta(
    t: torch.Tensor,
    schedule_type: str,
    sigma_0: float,
    sigma_T: float,
) -> torch.Tensor:
    """Integral of beta from t to T for the chosen schedule."""
    if schedule_type == "geometric":
        if sigma_0 == sigma_T:
            return sigma_0 * (1 - t)
        r = sigma_T / sigma_0
        return sigma_0 * (r - r ** t) / math.log(r)
    if schedule_type == "linear":
        beta = sigma_T - sigma_0
        return (1 - t) * (sigma_0 + 0.5 * (1 + t) * beta)
    if schedule_type == "cosine":
        beta = sigma_T - sigma_0
        return sigma_0 * (1 - t) + 2 / math.pi * beta * torch.sin(math.pi / 2 * (1 - t))
    raise ValueError(f"Unknown schedule type: {schedule_type}")


def drift_coeff(
    t: torch.Tensor,
    schedule_type: str,
    sigma_0: float,
    sigma_T: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bridge drift coefficient gamma_t = beta_t / int_beta(t)."""
    if schedule_type == "geometric":
        if sigma_0 == sigma_T:
            return 1.0 / (1 - t).clamp(min=eps)
        r = sigma_T / sigma_0
        return math.log(r) / (r ** (1 - t) - 1).clamp(min=eps)
    if schedule_type == "linear":
        rbeta = sigma_0 / sigma_T if sigma_T > 0 else sigma_0 / max(sigma_0, eps)
        numer = (1 - t) * rbeta + t
        denom = (1 - t) * (rbeta + 0.5 * (1 + t) * (1 - rbeta))
        return numer / denom.clamp(min=eps)
    if schedule_type == "cosine":
        beta_t = cosine_schedule(t, sigma_0, sigma_T)
        return beta_t / int_beta(t, schedule_type, sigma_0, sigma_T).clamp(min=eps)
    raise ValueError(f"Unknown schedule type: {schedule_type}")


def bridge_gamma(
    t: torch.Tensor,
    sigma_t: Optional[torch.Tensor] = None,
    sigma_0: Optional[float] = None,
    sigma_T: Optional[float] = None,
    schedule_type: str = "geometric",
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Bridge drift coefficient helper.

    If sigma_0 and sigma_T are provided, returns drift_coeff for the schedule.
    Otherwise falls back to the legacy sigma_t^2 / (1 - t) form.
    """
    if sigma_0 is not None and sigma_T is not None:
        return drift_coeff(t, schedule_type, sigma_0, sigma_T, eps=eps)
    if sigma_t is None:
        raise ValueError("bridge_gamma requires sigma_t or (sigma_0, sigma_T).")
    return sigma_t ** 2 / (1 - t).clamp(min=eps)


# =============================================================================
# Precomputation of alpha_t and rho_t
# =============================================================================

def _coord_laplacian(x: torch.Tensor, t: torch.Tensor, scheduler, manifold_dim: int) -> torch.Tensor:
    """Laplacian term in radial process."""
    laplacian_coeff = -0.5 * manifold_dim * scheduler.beta(t)
    if len(x.shape) == len(t.shape):
        return laplacian_coeff * x
    if len(x.shape) == len(t.shape) + 1:
        return torch.einsum("...,...i->...i", laplacian_coeff, x)
    return torch.einsum("...,...ij->...ij", laplacian_coeff, x)


def _solve_rho(cos_norm: float, manifold_dim: int, init: float = 0.0) -> float:
    """
    Solve rho from Kummer function inversion (rdlm/sde.py).

    For a Riemannian normal distribution on S^{D-1} with concentration parameter
    related to rho, the expected cosine with the mean direction satisfies:
        E[cos(theta)] = exp(-rho^2/2) * 1F1(D/2, 1/2, -rho^2/2)

    For high-dimensional manifolds (D > 50), the hyp1f1 function becomes
    numerically unstable. We use the von Mises-Fisher approximation instead:
        E[cos(theta)] ≈ 1 - (D-1)*rho^2/2 + (D-1)*(D-3)*rho^4/24 - ...

    This gives us a polynomial equation to solve for rho.
    """
    import numpy as np

    # Clamp cos_norm to valid range
    cos_norm = max(min(cos_norm, 1.0 - 1e-10), -1.0 + 1e-10)

    # Edge cases
    if cos_norm >= 1.0 - 1e-8:
        return 0.0  # Very concentrated, rho ≈ 0

    if cos_norm <= 0.0:
        # For spread-out distributions, use the asymptotic limit
        # When rho is large, E[cos] → 0 on high-dim sphere
        # Approximate: E[cos] ≈ exp(-D*rho^2/4) for large rho
        # Solving: cos_norm = exp(-D*rho^2/4) → rho = sqrt(-4*log(max(cos_norm,1e-10))/D)
        if cos_norm <= 1e-10:
            return math.sqrt(4.0 * 10.0 / max(manifold_dim, 1))  # Cap at reasonable value
        return math.sqrt(-4.0 * math.log(cos_norm) / max(manifold_dim, 1))

    # For high dimensions, use first-order approximation from Taylor expansion
    # E[cos] = exp(-rho^2/2) * 1F1(D/2, 1/2, -rho^2/2) ≈ 1 - (D-1)*rho^2/2 + O(rho^4)
    # Solving for rho: rho = sqrt(2*(1-E[cos])/(D-1))
    if manifold_dim > 50:
        D = manifold_dim
        rho_sq = 2.0 * (1.0 - cos_norm) / max(D - 1, 1)
        if rho_sq <= 0:
            return 0.0
        return math.sqrt(rho_sq)

    import scipy.special as sp
    from scipy.optimize import brentq

    def f(rho):
        if rho <= 1e-10:
            return 1.0 - cos_norm
        log_exp_term = -rho ** 2 / 2.0
        hyp_val = sp.hyp1f1(manifold_dim / 2.0, 0.5, -rho ** 2 / 2.0)

        # Check for numerical issues - use approximation if hyp1f1 fails
        if not np.isfinite(hyp_val) or hyp_val <= 0:
            return (1.0 - (manifold_dim - 1) * rho ** 2 / 2.0) - cos_norm

        lhs = math.exp(log_exp_term) * hyp_val
        return lhs - cos_norm

    # Use first-order approximation as initial guess
    init_guess = math.sqrt(2.0 * (1.0 - cos_norm) / max(manifold_dim - 1, 1))
    init_guess = max(init_guess, 1e-6)

    # Find upper bound where f changes sign
    upper = max(init_guess * 3, 0.5)
    for _ in range(15):
        if f(upper) < 0:
            break
        upper *= 2
        if upper > 10:
            return init_guess  # Couldn't bracket, use approximation

    if f(upper) >= 0:
        return init_guess  # Couldn't bracket, use approximation

    # Use Brent's method - guaranteed to converge if properly bracketed
    rho = brentq(f, 1e-10, upper, xtol=1e-8)
    return float(rho)


def _precompute_alpha_rho_init(
    scheduler,
    preprocess_steps: int,
    dims: int,
    manifold_dim: int,
    device: torch.device,
    init_lambda: float,
    rho_scale: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Precompute alpha/rho for LogBridge_Init (rdlm/sde.py)."""
    inner_prod = math.sqrt(init_lambda / manifold_dim) if manifold_dim > 0 else 0.0
    proj_norm = math.sqrt(max(1.0 - inner_prod ** 2, 0.0))

    proj_f = torch.ones(preprocess_steps, device=device) * inner_prod
    proj_f[-1] = 1.0
    proj_0 = torch.ones(preprocess_steps, device=device) * inner_prod
    proj_0[0] = 1.0

    x = torch.stack(
        [
            torch.ones(dims, device=device) * inner_prod,
            torch.ones(dims, device=device),
        ],
        dim=-1,
    )

    timesteps = torch.linspace(0.0, 1.0, preprocess_steps, device=device)
    dt = timesteps[1] - timesteps[0]

    iterator = tqdm(range(0, timesteps.shape[0] - 2), desc="Precomputing init", leave=False, disable=not _is_main_process())
    for i in iterator:
        t = torch.ones(dims, device=device) * timesteps[i]
        z = torch.randn_like(x)

        laplacian_term = _coord_laplacian(x, t, scheduler, manifold_dim)
        coeff = scheduler.drift_coeff(t)
        arccos_x0 = x[..., 0].clamp(min=-1.0 + eps, max=1.0 - eps).arccos()
        sin_arccos_x0 = (1 - x[..., 0] ** 2).clamp(min=0).sqrt()
        drift = torch.stack(
            [
                coeff * arccos_x0 * sin_arccos_x0,
                coeff
                * (inner_prod - x[..., 0] * x[..., 1])
                * arccos_x0.clamp(min=eps)
                / sin_arccos_x0.clamp(min=eps),
            ],
            dim=-1,
        ) + laplacian_term

        diffusion = torch.einsum("...,...i->...i", scheduler.beta(t), 1 - x ** 2).clamp(min=0).sqrt()
        x = x + drift * dt + diffusion * z * dt.abs().sqrt()

        proj_f[i + 1] = x[..., 0].mean()
        proj_0[i + 1] = x[..., 1].mean()

    # Moment matching to get alphas (Eq. 25 in RDLM paper)
    # Clamp proj_0 to avoid division by zero
    rtheta = proj_f / proj_0.clamp(min=eps)
    rtheta = (rtheta - inner_prod) ** 2
    # Ensure denominator is positive and clamp rtheta to be non-negative
    rtheta = rtheta.clamp(min=0)
    denom = (1 - inner_prod ** 2 + rtheta).clamp(min=eps)
    alphas = (rtheta / denom).clamp(min=0, max=1).sqrt()

    # Compute cos_norm for rho calculation
    one_minus_alpha_sq = (1 - alphas ** 2).clamp(min=eps)
    cos_norm_start = proj_0.clamp(min=eps) / one_minus_alpha_sq.sqrt().clamp(min=eps)
    cos_norm_end = proj_f / (proj_norm * alphas + inner_prod * one_minus_alpha_sq.sqrt()).clamp(min=eps)
    cos_norm = torch.cat([cos_norm_start[: len(alphas) // 2], cos_norm_end[len(alphas) // 2 :]], dim=0)
    # Clamp cos_norm to valid range for _solve_rho
    cos_norm = cos_norm.clamp(min=-1.0 + eps, max=1.0 - eps)

    rhos = [0.0]
    for i in tqdm(range(1, len(cos_norm)), leave=False, disable=not _is_main_process()):
        init = 1e-4 if i == 1 else rhos[-1]
        rhos.append(_solve_rho(cos_norm[i].item(), manifold_dim, init))
    rhos = torch.tensor(rhos, device=device) * rho_scale

    return alphas, rhos, inner_prod, proj_norm


def _precompute_alpha_rho_mixture(
    scheduler,
    preprocess_steps: int,
    dims: int,
    manifold_dim: int,
    device: torch.device,
    rho_scale: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Precompute alpha/rho for LogBridge_Mixture (rdlm/sde.py)."""
    inner_prod = 1.0 / math.sqrt(manifold_dim + 1)
    proj_norm = math.sqrt(max(1.0 - inner_prod ** 2, 0.0))

    proj_f = torch.zeros((2, preprocess_steps), device=device)
    proj_f[1, ...] = inner_prod
    proj_f[..., -1] = 1.0
    proj_0 = torch.zeros((2, preprocess_steps), device=device)
    proj_0[1, ...] = inner_prod
    proj_0[..., 0] = 1.0

    x = torch.stack(
        [
            torch.zeros(dims, device=device),
            torch.ones(dims, device=device),
            torch.ones(dims, device=device) * inner_prod,
            torch.ones(dims, device=device),
        ],
        dim=-1,
    )
    timesteps = torch.linspace(0.0, 1.0, preprocess_steps, device=device)
    dt = timesteps[1] - timesteps[0]

    iterator = tqdm(range(0, timesteps.shape[0] - 2), desc="Precomputing mixture", leave=False, disable=not _is_main_process())
    for i in iterator:
        t = torch.ones(dims, device=device) * timesteps[i]
        z = torch.randn_like(x)

        coeff = scheduler.drift_coeff(t)
        arccos_x0_mask = x[..., 0].clamp(min=-1.0 + eps, max=1.0 - eps).arccos()
        sin_arccos_x0_mask = (1 - x[..., 0] ** 2).clamp(min=0).sqrt()
        arccos_x0_unif = x[..., 2].clamp(min=-1.0 + eps, max=1.0 - eps).arccos()
        sin_arccos_x0_unif = (1 - x[..., 2] ** 2).clamp(min=0).sqrt()

        drift = torch.stack(
            [
                coeff * arccos_x0_mask * sin_arccos_x0_mask,
                -coeff
                * x[..., 0]
                * x[..., 1]
                * arccos_x0_mask.clamp(min=eps)
                / sin_arccos_x0_mask.clamp(min=eps),
                coeff * arccos_x0_unif * sin_arccos_x0_unif,
                coeff
                * (inner_prod - x[..., 2] * x[..., 3])
                * arccos_x0_unif.clamp(min=eps)
                / sin_arccos_x0_unif.clamp(min=eps),
            ],
            dim=-1,
        ) + _coord_laplacian(x, t, scheduler, manifold_dim)

        diffusion = torch.einsum("...,...i->...i", scheduler.beta(t), 1 - x ** 2).clamp(min=0).sqrt()
        x = x + drift * dt + diffusion * z * dt.abs().sqrt()

        proj_f[0, i + 1] = x[..., 0].mean()
        proj_0[0, i + 1] = x[..., 1].mean()
        proj_f[1, i + 1] = x[..., 2].mean()
        proj_0[1, i + 1] = x[..., 3].mean()

    # Moment matching for mask path
    rtheta = proj_0[0] / proj_f[0].clamp(min=eps)
    alphas_mask = 1 / (1 + rtheta ** 2).clamp(min=eps).sqrt()
    alphas_mask = alphas_mask.clamp(min=0, max=1)
    cos_norm_mask = proj_f[0].clamp(min=eps) / alphas_mask.clamp(min=eps)
    cos_norm_mask = cos_norm_mask.clamp(min=-1.0 + eps, max=1.0 - eps)

    # Moment matching for uniform path
    rtheta_unif = proj_f[1] / proj_0[1].clamp(min=eps)
    rtheta_unif = (rtheta_unif - inner_prod) ** 2
    rtheta_unif = rtheta_unif.clamp(min=0)
    denom_unif = (1 - inner_prod ** 2 + rtheta_unif).clamp(min=eps)
    alphas_unif = (rtheta_unif / denom_unif).clamp(min=0, max=1).sqrt()

    one_minus_alpha_unif_sq = (1 - alphas_unif ** 2).clamp(min=eps)
    cos_norm_start = proj_0[1].clamp(min=eps) / one_minus_alpha_unif_sq.sqrt().clamp(min=eps)
    cos_norm_end = proj_f[1] / (proj_norm * alphas_unif + inner_prod * one_minus_alpha_unif_sq.sqrt()).clamp(min=eps)
    cos_norm_unif = torch.cat(
        [cos_norm_start[: len(alphas_unif) // 2], cos_norm_end[len(alphas_unif) // 2 :]], dim=0
    )
    cos_norm_unif = cos_norm_unif.clamp(min=-1.0 + eps, max=1.0 - eps)

    rhos_mask = [0.0]
    for i in tqdm(range(1, len(cos_norm_mask)), leave=False, disable=not _is_main_process()):
        init = 1e-4 if i == 1 else rhos_mask[-1]
        rhos_mask.append(_solve_rho(cos_norm_mask[i].item(), manifold_dim, init))
    rhos_mask = torch.tensor(rhos_mask, device=device)

    rhos_unif = [0.0]
    for i in tqdm(range(1, len(cos_norm_unif)), leave=False, disable=not _is_main_process()):
        init = 1e-4 if i == 1 else rhos_unif[-1]
        rhos_unif.append(_solve_rho(cos_norm_unif[i].item(), manifold_dim, init))
    rhos_unif = torch.tensor(rhos_unif, device=device)

    alphas = torch.stack([alphas_mask, alphas_unif], dim=0)
    rhos = torch.stack([rhos_mask, rhos_unif], dim=0) * rho_scale

    return alphas, rhos, inner_prod, proj_norm


def precompute_alpha_rho(
    sigma_fn: Optional[Callable] = None,
    n_time_steps: int = 1000,
    n_simulations: int = 10000,
    device: torch.device = torch.device("cpu"),
    prior_type: str = "uniform",
    schedule=None,
    manifold_dim: Optional[int] = None,
    mix_type: Optional[str] = None,
    mix_step_thr: float = 0.0,
    rho_scale: float = 1.0,
    init_lambda: Optional[float] = None,
    preprocess_dims: int = 2 ** 14,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute alpha_t and rho_t.

    If schedule and manifold_dim are provided, uses the reference RDLM precompute
    (LogBridge_Init/LogBridge_Mixture). Otherwise falls back to the legacy
    Monte Carlo approximation.
    """
    if schedule is not None and manifold_dim is not None:
        preprocess_steps = n_time_steps
        if prior_type == RDLMPriorType.MIXTURE:
            alphas, rhos, _, _ = _precompute_alpha_rho_mixture(
                scheduler=schedule,
                preprocess_steps=preprocess_steps,
                dims=preprocess_dims,
                manifold_dim=manifold_dim,
                device=device,
                rho_scale=rho_scale,
            )
            t_grid = torch.linspace(0.0, 1.0, preprocess_steps, device=device)
            return t_grid, alphas, rhos
        if init_lambda is None:
            init_lambda = 1.0 if prior_type == RDLMPriorType.UNIFORM else 0.0
        alphas, rhos, _, _ = _precompute_alpha_rho_init(
            scheduler=schedule,
            preprocess_steps=preprocess_steps,
            dims=preprocess_dims,
            manifold_dim=manifold_dim,
            device=device,
            init_lambda=init_lambda,
            rho_scale=rho_scale,
        )
        t_grid = torch.linspace(0.0, 1.0, preprocess_steps, device=device)
        return t_grid, alphas, rhos

    if sigma_fn is None:
        raise ValueError("precompute_alpha_rho requires schedule or sigma_fn.")

    dt = 1.0 / n_time_steps
    t_grid = torch.linspace(0, 1 - dt, n_time_steps, device=device)

    if prior_type == "masked":
        z_T = torch.zeros(n_simulations, device=device)
    else:
        z_T = torch.rand(n_simulations, device=device)

    alpha_t = torch.zeros(n_time_steps, device=device)
    rho_t = torch.zeros(n_time_steps, device=device)

    iterator = tqdm(t_grid, desc="Precomputing alpha/rho (legacy)", leave=False, disable=not _is_main_process())
    for i, t in enumerate(iterator):
        alpha_t[i] = z_T.mean()
        z_0_sq = (1 - z_T ** 2).clamp(min=0)
        rho_t[i] = z_0_sq.mean().sqrt()

        sigma_t = sigma_fn(t)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.item()
        gamma_t = sigma_t ** 2 / max(1 - t.item(), 1e-4)

        dB = torch.randn(n_simulations, device=device) * (dt ** 0.5)

        z_T_safe = z_T.clamp(min=1e-6)
        one_minus_zT_sq = (1 - z_T ** 2).clamp(min=1e-8)

        drift = gamma_t * one_minus_zT_sq / z_T_safe
        diffusion = sigma_t * one_minus_zT_sq.sqrt()

        z_T = z_T + drift * dt + diffusion * dB
        z_T = z_T.clamp(1e-6, 1 - 1e-6)

    return t_grid, alpha_t, rho_t


# =============================================================================
# RDLM Schedule Class
# =============================================================================

@dataclass
class RDLMScheduleConfig:
    """Configuration for RDLM noise schedule (defaults match RDLM paper)."""
    schedule_type: str = "geometric"
    sigma_0: float = 0.001   # beta_0 in RDLM
    sigma_T: float = 0.2     # beta_f in RDLM (paper uses 0.2)
    n_time_steps: int = 10000  # preprocess_steps in RDLM paper
    prior_type: str = "uniform"
    add_mask_token: bool = True
    init_lambda: Optional[float] = None
    mix_type: str = "step"
    mix_step_thr: float = 0.0
    rho_scale: float = 1.0
    preprocess_dims: int = 2 ** 14
    weight_type: str = "step"
    weight_left: float = 0.3
    weight_right: float = 0.75
    weight_lb: float = 1e-4
    weight_ub: float = 1.0
    eps: float = 1e-6


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
        sigma_T: float = 0.2,  # RDLM paper default
        n_time_steps: int = 10000,  # RDLM paper default
        prior_type: str = "uniform",
        add_mask_token: bool = True,
        init_lambda: Optional[float] = None,
        mix_type: str = "step",
        mix_step_thr: float = 0.0,
        rho_scale: float = 1.0,
        preprocess_dims: int = 2 ** 14,
        weight_type: str = "step",
        weight_left: float = 0.3,
        weight_right: float = 0.75,
        weight_lb: float = 1e-4,
        weight_ub: float = 1.0,
        eps: float = 1e-6,
        manifold_dim: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        precompute: bool = True
    ):
        if config is not None:
            schedule_type = config.schedule_type
            sigma_0 = config.sigma_0
            sigma_T = config.sigma_T
            n_time_steps = config.n_time_steps
            prior_type = config.prior_type
            add_mask_token = config.add_mask_token
            init_lambda = config.init_lambda
            mix_type = config.mix_type
            mix_step_thr = config.mix_step_thr
            rho_scale = config.rho_scale
            preprocess_dims = config.preprocess_dims
            weight_type = config.weight_type
            weight_left = config.weight_left
            weight_right = config.weight_right
            weight_lb = config.weight_lb
            weight_ub = config.weight_ub
            eps = config.eps

        self.schedule_type = schedule_type
        self.sigma_0 = sigma_0
        self.sigma_T = sigma_T
        self.n_time_steps = n_time_steps
        self.prior_type = prior_type
        self.add_mask_token = add_mask_token
        self.init_lambda = init_lambda
        self.mix_type = mix_type
        self.mix_step_thr = mix_step_thr
        self.rho_scale = rho_scale
        self.preprocess_dims = preprocess_dims
        self.weight_type = weight_type
        self.weight_left = weight_left
        self.weight_right = weight_right
        self.weight_lb = weight_lb
        self.weight_ub = weight_ub
        self.eps = eps
        self.manifold_dim = manifold_dim
        self.device = device

        # Define beta function
        if schedule_type == "geometric":
            self.sigma_fn = lambda t: geometric_schedule(t, sigma_0, sigma_T)
        elif schedule_type == "linear":
            self.sigma_fn = lambda t: linear_schedule(t, sigma_0, sigma_T)
        elif schedule_type == "cosine":
            self.sigma_fn = lambda t: cosine_schedule(t, sigma_0, sigma_T)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.weight_fn = get_weight_fn(self.weight_type)(
            left=self.weight_left,
            right=self.weight_right,
            lb=self.weight_lb,
            ub=self.weight_ub,
        )

        if self.manifold_dim is None and precompute:
            raise ValueError("manifold_dim is required for RDLM schedule precomputation.")

        self.preprocess_steps = n_time_steps + 1
        if self.manifold_dim is not None:
            if self.prior_type == RDLMPriorType.MIXTURE:
                self.inner_prod = 1.0 / math.sqrt(self.manifold_dim + 1)
            else:
                if self.init_lambda is None:
                    self.init_lambda = 1.0 if self.prior_type == RDLMPriorType.UNIFORM else 0.0
                self.inner_prod = math.sqrt(self.init_lambda / self.manifold_dim) if self.manifold_dim > 0 else 0.0
            self.proj_norm = math.sqrt(max(1.0 - self.inner_prod ** 2, 0.0))
        else:
            self.inner_prod = None
            self.proj_norm = None

        # Precompute α_t and ρ_t
        if precompute:
            self.t_grid, self.alpha_t, self.rho_t = precompute_alpha_rho(
                n_time_steps=self.preprocess_steps,
                device=device,
                prior_type=prior_type,
                schedule=self,
                manifold_dim=self.manifold_dim,
                mix_type=self.mix_type,
                mix_step_thr=self.mix_step_thr,
                rho_scale=self.rho_scale,
                init_lambda=self.init_lambda,
                preprocess_dims=self.preprocess_dims,
            )
        else:
            self.t_grid = None
            self.alpha_t = None
            self.rho_t = None

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get σ_t at time t."""
        return self.sigma_fn(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta schedule (same as sigma_t in RDLM notation)."""
        return self.sigma_fn(t)

    def drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma_t = beta_t / int_beta(t)."""
        return drift_coeff(t, self.schedule_type, self.sigma_0, self.sigma_T, eps=self.eps)

    def get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Alias for drift_coeff."""
        return self.drift_coeff(t)

    def importance_weight(self, t: torch.Tensor, train: bool) -> torch.Tensor:
        if train:
            return self.weight_fn.norm_const / self.weight_fn(t)
        return torch.ones_like(t)

    def importance_weighted_time(self, shape, device, steps: int = 100) -> torch.Tensor:
        quantile = torch.rand(shape, device=device) * self.weight_fn.norm_const
        lb = torch.zeros_like(quantile)
        ub = torch.ones_like(quantile) * (1 - self.eps)

        for _ in range(steps):
            mid = (lb + ub) / 2.0
            value = self.weight_fn.cum_weight_fn(mid)
            lb = torch.where(value <= quantile, mid, lb)
            ub = torch.where(value <= quantile, ub, mid)

        return (lb + ub) / 2.0

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

        if self.t_grid is None:
            raise RuntimeError("Schedule has no t_grid for interpolation.")

        t = t.clamp(min=0.0, max=1.0)
        idx = torch.searchsorted(self.t_grid, t) - 1
        idx = idx.clamp(min=0, max=len(self.t_grid) - 2)
        r = (t - self.t_grid[idx]) / (self.t_grid[idx + 1] - self.t_grid[idx])

        if self.alpha_t.dim() == 1:
            alpha = self.alpha_t[idx] * (1 - r) + self.alpha_t[idx + 1] * r
            rho = self.rho_t[idx] * (1 - r) + self.rho_t[idx + 1] * r
            return alpha, rho

        alpha = self.alpha_t[:, idx] * (1 - r) + self.alpha_t[:, idx + 1] * r
        rho = self.rho_t[:, idx] * (1 - r) + self.rho_t[:, idx + 1] * r
        return alpha, rho

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

    Memory-optimized version using in-place operations.

    Args:
        mean_direction: [B, L, D] mean direction on sphere
        scale: [B] or [B, 1, 1] standard deviation ρ_t
        base_point: optional base point for tangent space

    Returns:
        samples: [B, L, D] samples on hypersphere
    """
    if base_point is None:
        base_point = mean_direction

    # Sample in tangent space (in-place projection)
    tangent_vec = torch.randn_like(mean_direction)
    # Project to tangent space: v - <v, p> * p
    dot = (tangent_vec * base_point).sum(dim=-1, keepdim=True)
    tangent_vec.sub_(dot * base_point)
    del dot

    # Scale by ρ_t (in-place)
    if scale.dim() == 1:
        scale = scale.view(-1, 1, 1)
    tangent_vec.mul_(scale)

    # Map to sphere via exponential map (in-place where possible)
    v_norm = tangent_vec.norm(dim=-1, keepdim=True).clamp_(min=1e-8)
    cos_v = torch.cos(v_norm)
    sin_v = torch.sin(v_norm)
    sin_v.div_(v_norm)

    # samples = mean * cos(|v|) + v * sin(|v|)/|v|
    samples = mean_direction * cos_v
    samples.addcmul_(tangent_vec, sin_v)

    del tangent_vec, v_norm, cos_v, sin_v

    return samples


# =============================================================================
# RDLM Interpolation
# =============================================================================

def rdlm_interpolant(
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    t: torch.Tensor,
    alpha_t: Optional[torch.Tensor] = None,
    rho_t: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
    schedule: Optional[RDLMSchedule] = None,
    mask_idx: Optional[int] = None,
    add_noise: bool = True,
) -> torch.Tensor:
    """
    RDLM interpolant (memory-optimized).

    If schedule is provided, uses LogBridge_Init/LogBridge_Mixture formulas from
    the reference implementation. Otherwise falls back to the legacy interpolant
    using alpha_t and rho_t.
    """
    B, L, D = x0.shape
    device = x0.device

    if schedule is not None:
        if mask_idx is None:
            mask_idx = D - 1

        alpha_t, rho_t = schedule.get_alpha_rho(t)

        if schedule.prior_type == RDLMPriorType.MIXTURE:
            alphas_mask = alpha_t[0].view(-1, 1, 1)
            rhos_mask = rho_t[0].view(-1, 1, 1)
            alphas_unif = alpha_t[1].view(-1, 1, 1)
            rhos_unif = rho_t[1].view(-1, 1, 1)

            # Determine which samples use mask vs uniform prior
            mask_flag = (x0[..., mask_idx] == 1).unsqueeze(-1)  # [B, L, 1]

            # Compute coefficients for x0 term (in-place where possible)
            sqrt_one_minus_alpha_mask_sq = (1 - alphas_mask.squeeze(-1) ** 2).sqrt().view(-1, 1, 1)
            sqrt_one_minus_alpha_unif_sq = (1 - alphas_unif.squeeze(-1) ** 2).sqrt().view(-1, 1, 1)
            unif_x0_coeff = sqrt_one_minus_alpha_unif_sq - alphas_unif * schedule.inner_prod / schedule.proj_norm

            # Select coefficients based on mask_flag
            alpha_coeff = torch.where(mask_flag, alphas_mask, alphas_unif / schedule.proj_norm)
            x0_coeff = torch.where(mask_flag, sqrt_one_minus_alpha_mask_sq, unif_x0_coeff)

            # Build mu in-place: mu = alpha_coeff * end + x0_coeff * x0
            # end is one-hot at target_indices
            mu = x0 * x0_coeff
            mu.scatter_add_(-1, target_indices.unsqueeze(-1), alpha_coeff.expand(B, L, 1))

            del alpha_coeff, x0_coeff, sqrt_one_minus_alpha_mask_sq, sqrt_one_minus_alpha_unif_sq, unif_x0_coeff

            if not add_noise:
                return mu

            # Add noise with coefficient based on mask_flag
            rho_coeff = torch.where(mask_flag, rhos_mask, rhos_unif)
            z = torch.randn_like(x0)
            z.mul_(rho_coeff)
            normal = mu + z
            del z, rho_coeff, mask_flag

            tangent = make_tangent(mu, normal)
            del normal
            result = exp_map_inplace(mu, tangent)  # tangent now contains result
            del mu
            return result

        # Init path (non-mixture)
        alpha_t = alpha_t.view(-1, 1, 1)
        rho_t = rho_t.view(-1, 1, 1)

        # Compute coefficients
        alpha_over_proj = alpha_t / schedule.proj_norm
        sqrt_one_minus_alpha_sq = (1 - alpha_t ** 2).sqrt()
        x0_coeff = sqrt_one_minus_alpha_sq - alpha_t * schedule.inner_prod / schedule.proj_norm

        # Build mu: mu = (alpha_t / proj_norm) * end + x0_coeff * x0
        mu = x0 * x0_coeff
        mu.scatter_add_(-1, target_indices.unsqueeze(-1), alpha_over_proj.expand(B, L, 1))

        del alpha_over_proj, sqrt_one_minus_alpha_sq, x0_coeff

        if not add_noise:
            return mu

        z = torch.randn_like(x0)
        z.mul_(rho_t)
        normal = mu + z
        del z

        tangent = make_tangent(mu, normal)
        del normal
        result = exp_map_inplace(mu, tangent)  # tangent now contains result
        del mu
        return result

    if alpha_t is None or rho_t is None or vocab_size is None:
        raise ValueError("rdlm_interpolant requires alpha_t, rho_t, and vocab_size when schedule is None.")

    # Legacy interpolant (kept for compatibility/tests)
    alpha_t_expanded = alpha_t.view(B, 1, 1)
    sqrt_one_minus_alpha_sq = torch.sqrt((1 - alpha_t ** 2).clamp(min=0)).view(B, 1, 1)

    x0_at_target = x0.gather(dim=-1, index=target_indices.unsqueeze(-1))
    x0_orth_sq = (1 - x0_at_target ** 2).clamp(min=1e-12)
    x0_orth_norm = torch.sqrt(x0_orth_sq)

    mean_on_sphere = x0 * (sqrt_one_minus_alpha_sq / x0_orth_norm)
    mean_on_sphere.scatter_(
        dim=-1,
        index=target_indices.unsqueeze(-1),
        src=alpha_t_expanded.expand(B, L, 1),
    )
    mean_norm = mean_on_sphere.norm(dim=-1, keepdim=True).clamp_(min=1e-8)
    mean_on_sphere.div_(mean_norm)
    del mean_norm, x0_at_target, x0_orth_sq, x0_orth_norm

    if add_noise:
        xt = sample_riemannian_normal(mean_on_sphere, rho_t)
        del mean_on_sphere
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

    Memory-optimized version using in-place operations.

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

    # Compute log map in-place (direction from xt to x1)
    # x1 will be modified to contain the result
    drift = log_map_inplace(xt, x1)

    # Scale by γ_t (in-place)
    if isinstance(gamma_t, torch.Tensor) and gamma_t.dim() >= 1:
        gamma_t = gamma_t.view(-1, 1, 1)
    drift.mul_(gamma_t)

    return drift


def expected_drift_from_probs(
    xt: torch.Tensor,
    probs: torch.Tensor,
    gamma_t: torch.Tensor
) -> torch.Tensor:
    """
    Compute expected drift from predicted probabilities (memory-efficient).

    E[γ_t * log_{X_t}(e_k)] = γ_t * Σ_k p_k * log_{xt}(e_k)

    This is used when the model outputs probabilities rather than drift directly.

    Memory-optimized: avoids creating [B, L, D, D] tensors by computing the
    weighted sum analytically. The log map to basis vector e_k is:
        log_xt(e_k) = θ_k * (e_k - xt * cos(θ_k)) / sin(θ_k)
    where θ_k = arccos(<xt, e_k>) = arccos(xt[k]).

    The expected log map is:
        Σ_k p_k * log_xt(e_k) = Σ_k p_k * θ_k * (e_k - xt * cos(θ_k)) / sin(θ_k)
                              = Σ_k p_k * θ_k / sin(θ_k) * e_k - xt * Σ_k p_k * θ_k * cos(θ_k) / sin(θ_k)

    Args:
        xt: [B, L, D] current position on sphere
        probs: [B, L, D] predicted probabilities
        gamma_t: [B] drift coefficient

    Returns:
        expected_drift: [B, L, D] expected tangent vector
    """
    # Inner product <xt, e_k> = xt[k] for each k
    inner = xt.clamp(-1 + 1e-6, 1 - 1e-6)  # [B, L, D]

    # Angles to each basis vector: θ_k = arccos(xt[k])
    angles = torch.arccos(inner)  # [B, L, D]

    # sin(θ_k) = sqrt(1 - xt[k]^2)
    sin_angles = (1 - inner ** 2).sqrt().clamp(min=1e-8)  # [B, L, D]

    # Compute θ_k / sin(θ_k) - this is the scaling factor for each log map
    # Handle θ ≈ 0 case where θ/sin(θ) → 1
    angle_scale = angles / sin_angles  # [B, L, D]

    # Expected log map = Σ_k p_k * θ_k / sin(θ_k) * e_k - xt * Σ_k p_k * θ_k * cos(θ_k) / sin(θ_k)
    # First term: coefficient for each basis vector e_k
    first_term = probs * angle_scale  # [B, L, D] - this IS the result (weighted sum of e_k)

    # Second term: scalar coefficient for xt
    second_term_coeff = (probs * angle_scale * inner).sum(dim=-1, keepdim=True)  # [B, L, 1]

    expected_log = first_term - xt * second_term_coeff  # [B, L, D]

    # Scale by γ_t
    if isinstance(gamma_t, torch.Tensor) and gamma_t.dim() >= 1:
        gamma_t = gamma_t.view(-1, 1, 1)

    return gamma_t * expected_log
