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


# ============== Memory-Efficient MSE Loss ==============


def mse_velocity_loss_to_onehot(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    target_indices: torch.Tensor,
    predicted_velocity: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-efficient MSE velocity loss when x_1 is one-hot.

    Instead of materializing full (b, l, V) tensors for x_1, log_map, and
    parallel transport, we compute the loss more efficiently by exploiting
    the one-hot structure.

    The target velocity is: parallel_transport(x_0, x_t, log_map(x_0, x_1))
    where x_1 is one-hot at target_indices.

    Args:
        x_0: Noise starting point, shape (B, L, V)
        x_t: Current point on geodesic, shape (B, L, V)
        target_indices: Token indices for one-hot x_1, shape (B, L)
        predicted_velocity: Model's predicted velocity, shape (B, L, V)

    Returns:
        MSE loss per token, shape (B, L)
    """
    # Get x_0 and x_t values at target index (the only non-zero component of x_1)
    # x_1[b, l, :] = one-hot at target_indices[b, l]
    idx = target_indices.unsqueeze(-1)  # (B, L, 1)
    x_0_at_target = x_0.gather(dim=-1, index=idx).squeeze(-1)  # (B, L)
    x_t_at_target = x_t.gather(dim=-1, index=idx).squeeze(-1)  # (B, L)

    # ===== Compute log_map(x_0, x_1) =====
    # dot_pq = <x_0, x_1> = x_0[target_idx] since x_1 is one-hot
    dot_x0_x1 = x_0_at_target  # (B, L)

    # q_proj = x_1 - dot_x0_x1 * x_0
    # Since x_1 is one-hot, q_proj has:
    #   - At target idx: 1 - dot_x0_x1 * x_0[target_idx] = 1 - x_0[target_idx]^2
    #   - At other idx j: 0 - dot_x0_x1 * x_0[j] = -x_0[target_idx] * x_0[j]
    # So q_proj = -x_0[target_idx] * x_0, except at target_idx add 1
    # q_proj = -dot_x0_x1.unsqueeze(-1) * x_0  + one_hot

    # ||q_proj||^2 = sum_j (q_proj[j])^2
    #   = (1 - x_0[t]^2)^2 + sum_{j!=t} (x_0[t] * x_0[j])^2
    #   = (1 - x_0[t]^2)^2 + x_0[t]^2 * (sum_{j!=t} x_0[j]^2)
    #   = (1 - x_0[t]^2)^2 + x_0[t]^2 * (1 - x_0[t]^2)   [since ||x_0||=1]
    #   = (1 - x_0[t]^2) * [(1 - x_0[t]^2) + x_0[t]^2]
    #   = (1 - x_0[t]^2)
    q_proj_norm_sq = (1 - dot_x0_x1.square()).clamp(min=1e-16)  # (B, L)
    q_proj_norm = q_proj_norm_sq.sqrt()  # (B, L)

    # dist = arccos(<x_0, x_1>) = arccos(x_0[target_idx])
    dist = torch.acos(dot_x0_x1.clamp(-1 + 1e-7, 1 - 1e-7))  # (B, L)

    # log_map = q_proj / ||q_proj|| * dist
    # Since q_proj = -dot_x0_x1 * x_0 + one_hot:
    #   log_map[j] = (-dot_x0_x1 * x_0[j] / ||q_proj||) * dist  for j != target
    #   log_map[target] = ((1 - dot_x0_x1^2) / ||q_proj||) * dist
    #                   = (||q_proj||^2 / ||q_proj||) * dist = ||q_proj|| * dist

    # Scale factor for non-target entries
    scale = (-dot_x0_x1 / q_proj_norm * dist).unsqueeze(-1)  # (B, L, 1)

    # log_map = scale * x_0, then add correction at target index
    # Correction at target: ||q_proj|| * dist - scale * x_0[target]
    #                     = ||q_proj|| * dist - (-dot_x0_x1 / ||q_proj|| * dist) * x_0[target]
    #                     = dist * (||q_proj|| + dot_x0_x1 * x_0[target] / ||q_proj||)
    #                     = dist * (||q_proj||^2 + dot_x0_x1^2) / ||q_proj||
    #                     = dist * (1 - dot_x0_x1^2 + dot_x0_x1^2) / ||q_proj||
    #                     = dist / ||q_proj||
    correction_at_target = (dist / q_proj_norm).unsqueeze(-1)  # (B, L, 1)

    # ===== Parallel transport from x_0 to x_t =====
    # PT formula: v - <v, q> * (p + q) / (1 + <p, q>)
    # where p = x_0, q = x_t, v = log_map(x_0, x_1)

    dot_x0_xt = (x_0 * x_t).sum(dim=-1, keepdim=True)  # (B, L, 1)
    denom = (1.0 + dot_x0_xt).clamp(min=1e-8)  # (B, L, 1)

    # <log_map, x_t> = scale * <x_0, x_t> + correction * x_t[target]
    # where we need to add the correction only at target index
    log_map_dot_xt = (scale * dot_x0_xt).squeeze(-1) + correction_at_target.squeeze(-1) * x_t_at_target  # (B, L)

    # target_velocity = log_map - log_map_dot_xt * (x_0 + x_t) / denom
    # Compute: scale * x_0 + correction * one_hot - log_map_dot_xt.unsqueeze(-1) * x0_plus_xt / denom

    # Group terms:
    # = scale * x_0 - log_map_dot_xt.unsqueeze(-1) * (x_0 + x_t) / denom + correction * one_hot
    # = x_0 * (scale - log_map_dot_xt.unsqueeze(-1) / denom) - log_map_dot_xt.unsqueeze(-1) * x_t / denom + correction * one_hot

    pt_factor = log_map_dot_xt.unsqueeze(-1) / denom  # (B, L, 1)

    # Coefficients for x_0 and x_t in target_velocity
    coeff_x0 = scale - pt_factor  # (B, L, 1)
    coeff_xt = -pt_factor  # (B, L, 1)

    # Now compute MSE = ||predicted_velocity - target_velocity||^2
    # target_velocity = coeff_x0 * x_0 + coeff_xt * x_t + correction * one_hot

    # diff = predicted_velocity - target_velocity
    # ||diff||^2 = ||predicted||^2 - 2<predicted, target> + ||target||^2

    # ||predicted||^2
    pred_norm_sq = predicted_velocity.square().sum(dim=-1)  # (B, L)

    # ||target||^2 = coeff_x0^2 * ||x_0||^2 + coeff_xt^2 * ||x_t||^2 + correction^2
    #                + 2 * coeff_x0 * coeff_xt * <x_0, x_t>
    #                + 2 * coeff_x0 * correction * x_0[target]
    #                + 2 * coeff_xt * correction * x_t[target]
    # ||x_0||^2 = ||x_t||^2 = 1
    target_norm_sq = (
        coeff_x0.square().squeeze(-1)
        + coeff_xt.square().squeeze(-1)
        + correction_at_target.square().squeeze(-1)
        + 2 * coeff_x0.squeeze(-1) * coeff_xt.squeeze(-1) * dot_x0_xt.squeeze(-1)
        + 2 * coeff_x0.squeeze(-1) * correction_at_target.squeeze(-1) * x_0_at_target
        + 2 * coeff_xt.squeeze(-1) * correction_at_target.squeeze(-1) * x_t_at_target
    )  # (B, L)

    # <predicted, target> = coeff_x0 * <predicted, x_0> + coeff_xt * <predicted, x_t>
    #                       + correction * predicted[target]
    pred_dot_x0 = (predicted_velocity * x_0).sum(dim=-1)  # (B, L)
    pred_dot_xt = (predicted_velocity * x_t).sum(dim=-1)  # (B, L)
    pred_at_target = predicted_velocity.gather(dim=-1, index=idx).squeeze(-1)  # (B, L)

    pred_dot_target = (
        coeff_x0.squeeze(-1) * pred_dot_x0
        + coeff_xt.squeeze(-1) * pred_dot_xt
        + correction_at_target.squeeze(-1) * pred_at_target
    )  # (B, L)

    # MSE = ||predicted||^2 - 2<predicted, target> + ||target||^2
    mse = pred_norm_sq - 2 * pred_dot_target + target_norm_sq  # (B, L)

    return mse.clamp(min=0)  # Numerical stability
