"""
Geodesic utilities for Fisher-Rao Flow Matching on the sphere.

This module provides manifold operations on the positive orthant of the unit
hypersphere, which is the Fisher-Rao geometry for categorical distributions.

References:
- Fisher-Rao geometry: sqrt of probability simplex maps to sphere
- Geodesics are great circle arcs on the sphere
"""

import torch


# ============== Manifold Operations ==============


def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on the sphere.

    Move from point p in direction v (tangent vector) on the sphere.

    Args:
        p: Point on the sphere, shape (..., D)
        v: Tangent vector at p, shape (..., D)

    Returns:
        New point on the sphere, shape (..., D)
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    return p * torch.cos(v_norm) + v * torch.sin(v_norm) / v_norm


def exp_map_inplace(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient exponential map that reuses the v tensor.

    Computes exp_map(p, v) but stores the result in v to save memory.
    WARNING: This destroys the contents of v.

    Args:
        p: Point on the sphere, shape (..., D)
        v: Tangent vector at p, shape (..., D) - WILL BE MODIFIED

    Returns:
        v tensor now containing the result (same memory location)
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    sin_norm = torch.sin(v_norm)
    cos_norm = torch.cos(v_norm)
    # v = v * sin(||v||) / ||v|| + p * cos(||v||)
    v.mul_(sin_norm).div_(v_norm).add_(p * cos_norm)
    return v


def log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map on the sphere.

    Compute the tangent vector at p pointing toward q.

    Args:
        p: Source point on the sphere, shape (..., D)
        q: Target point on the sphere, shape (..., D)

    Returns:
        Tangent vector at p, shape (..., D)
    """
    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    q_proj = q - dot_pq * p
    q_proj_norm = torch.norm(q_proj, dim=-1, keepdim=True).clamp(min=1e-8)
    dot_clamped = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
    dist = torch.acos(dot_clamped)
    return q_proj / q_proj_norm * dist


def log_map_inplace(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient logarithmic map that reuses the q tensor.

    Computes log_map(p, q) but stores the result in q to save memory.
    WARNING: This destroys the contents of q.

    Args:
        p: Source point on the sphere, shape (..., D)
        q: Target point on the sphere, shape (..., D) - WILL BE MODIFIED

    Returns:
        q tensor now containing the tangent vector (same memory location)
    """
    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    # q_proj = q - dot_pq * p  (in-place)
    q.sub_(dot_pq * p)
    q_proj_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    dot_clamped = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
    dist = torch.acos(dot_clamped)
    # q = q / ||q|| * dist  (in-place)
    q.div_(q_proj_norm).mul_(dist)
    return q


def parallel_transport(p: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Parallel transport on the sphere.

    Transport tangent vector v from tangent space at p to tangent space at q.

    Args:
        p: Source point on the sphere, shape (..., D)
        q: Target point on the sphere, shape (..., D)
        v: Tangent vector at p to transport, shape (..., D)

    Returns:
        Transported tangent vector at q, shape (..., D)
    """
    # Use the formula: v - <v, q> * (p + q) / (1 + <p, q>)
    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    dot_vq = (v * q).sum(dim=-1, keepdim=True)
    denom = 1.0 + dot_pq
    return v - dot_vq * (p + q) / denom.clamp(min=1e-8)


def make_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto the tangent space at point p on the sphere.

    Args:
        p: Point on the sphere, shape (..., D)
        v: Vector to project, shape (..., D)

    Returns:
        Tangent vector at p, shape (..., D)
    """
    # Project: v - <p, v> * p
    dot_pv = (p * v).sum(dim=-1, keepdim=True)
    return v - dot_pv * p


def project_to_sphere(x: torch.Tensor) -> torch.Tensor:
    """Project to unit sphere for numerical stability."""
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)


def geodesic_interpolant(
    x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Interpolate along geodesic from x_0 to x_1.

    Args:
        x_0: Start point on sphere, shape (B, L, V) or (B, V)
        x_1: End point on sphere, shape (B, L, V) or (B, V)
        t: Interpolation parameter in [0, 1], shape (B,) or (B, 1) or (B, 1, 1)

    Returns:
        Interpolated point on sphere, same shape as x_0
    """
    # Expand t to match x_0 dimensions
    while t.dim() < x_0.dim():
        t = t.unsqueeze(-1)
    return exp_map(x_0, t * log_map(x_0, x_1))


# ============== Simplex-Sphere Mappings ==============


def simplex_to_sphere(p: torch.Tensor) -> torch.Tensor:
    """Map probability simplex to sphere via square root."""
    return torch.sqrt(p.clamp(min=1e-8))


def sphere_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """Map sphere point back to simplex via squaring."""
    return x**2


# ============== Prior Sampling ==============


def uniform_prior(
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample uniformly from positive orthant of the sphere.

    Args:
        shape: Shape of output tensor, last dim is the manifold dimension
        device: Device to create tensor on
        dtype: Data type for the tensor

    Returns:
        Points on positive orthant of unit sphere
    """
    # Sample from standard normal and take absolute value for positive orthant
    x = torch.randn(shape, device=device, dtype=dtype).abs()
    # Project to sphere
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)


# ============== Interpolation Schedules ==============


def linear_schedule(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Linear interpolation schedule.

    Returns:
        (alpha_t, alpha_t_prime): Interpolation value and its derivative
    """
    return t, torch.ones_like(t)


def cosine_schedule(
    t: torch.Tensor, nu: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine interpolation schedule.

    Args:
        t: Time values in [0, 1]
        nu: Schedule parameter (default 1.0)

    Returns:
        (alpha_t, alpha_t_prime): Interpolation value and its derivative
    """
    import math

    t_safe = t.clamp(min=1e-9)
    alpha_t = 1 - torch.cos(math.pi / 2 * t_safe.pow(nu)).square()
    alpha_t_prime = (
        math.pi / 2 * torch.sin(math.pi * t_safe.pow(nu)) * nu * t_safe.pow(nu - 1)
    )
    return alpha_t, alpha_t_prime
