from __future__ import annotations

import torch


def normalize_quaternion_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a quaternion stored as ``(w, x, y, z)``."""
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)


def quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert ``(w, x, y, z)`` quaternions to rotation matrices."""
    q = normalize_quaternion_wxyz(q)
    w, x, y, z = torch.unbind(q, dim=-1)

    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    one = torch.as_tensor(1.0, dtype=q.dtype, device=q.device)

    r00 = one - two * (y * y + z * z)
    r01 = two * (x * y - z * w)
    r02 = two * (x * z + y * w)
    r10 = two * (x * y + z * w)
    r11 = one - two * (x * x + z * z)
    r12 = two * (y * z - x * w)
    r20 = two * (x * z - y * w)
    r21 = two * (y * z + x * w)
    r22 = one - two * (x * x + y * y)

    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions in ``(w, x, y, z)`` order."""
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    return normalize_quaternion_wxyz(torch.stack((qw, qx, qy, qz), dim=-1))


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert fixed-axis URDF roll-pitch-yaw angles to rotation matrices."""
    roll, pitch, yaw = torch.unbind(rpy, dim=-1)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def make_transform(position: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Build ``w_H_b`` from base position and base-to-world orientation."""
    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    batch_shape = position.shape[:-1]
    transform = torch.zeros(*batch_shape, 4, 4, dtype=position.dtype, device=position.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = position
    transform[..., 3, 3] = 1.0
    return transform


def quaternion_derivative_from_world_angular_velocity(
    quaternion_wxyz: torch.Tensor, angular_velocity_world: torch.Tensor
) -> torch.Tensor:
    """Return ``q_dot`` for ``q`` mapping base coordinates to world coordinates.

    The angular velocity is expressed in the world frame. For this convention,
    ``q_dot = 0.5 * [0, omega_W] * q``.
    """
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    w, x, y, z = torch.unbind(q, dim=-1)
    wx, wy, wz = torch.unbind(angular_velocity_world, dim=-1)

    return 0.5 * torch.stack(
        (
            -wx * x - wy * y - wz * z,
            wx * w + wy * z - wz * y,
            -wx * z + wy * w + wz * x,
            wx * y - wy * x + wz * w,
        ),
        dim=-1,
    )


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew matrix ``[vector]x`` for the last dimension."""
    x, y, z = torch.unbind(vector, dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure a tensor has a leading batch dimension."""
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    return tensor, False
