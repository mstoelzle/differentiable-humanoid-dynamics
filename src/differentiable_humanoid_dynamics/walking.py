from __future__ import annotations

import torch

from .dynamics import HumanoidDynamics


def simple_walking_sequence(
    model: HumanoidDynamics,
    *,
    frames: int = 120,
    dt: float = 1.0 / 30.0,
    stride: float = 0.22,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a lightweight kinematic walking-like sequence for visualization.

    This is not a dynamically feasible trajectory. It is a deterministic joint
    animation that moves the G1 legs out of phase so contact-point FK can be
    inspected as configuration changes.
    """
    times = torch.arange(frames, dtype=model.dtype, device=model.device) * dt
    phase = 2.0 * torch.pi * times / (frames * dt / 2.0)
    states = model.neutral_state().repeat(frames, 1)
    states[:, 0] = torch.linspace(0.0, stride, frames, dtype=model.dtype, device=model.device)
    states[:, 2] = 0.78 + 0.015 * torch.sin(2.0 * phase)

    q = states[:, 7 : 7 + model.n_joints]
    names = model.joint_names
    _set_joint(q, names, "left_hip_pitch_joint", 0.22 * torch.sin(phase))
    _set_joint(q, names, "right_hip_pitch_joint", 0.22 * torch.sin(phase + torch.pi))
    _set_joint(q, names, "left_knee_joint", 0.42 * torch.clamp(torch.sin(phase), min=0.0))
    _set_joint(q, names, "right_knee_joint", 0.42 * torch.clamp(torch.sin(phase + torch.pi), min=0.0))
    _set_joint(q, names, "left_ankle_pitch_joint", -0.18 * torch.sin(phase))
    _set_joint(q, names, "right_ankle_pitch_joint", -0.18 * torch.sin(phase + torch.pi))
    _set_joint(q, names, "left_hip_roll_joint", 0.04 * torch.sin(phase + torch.pi / 2.0))
    _set_joint(q, names, "right_hip_roll_joint", -0.04 * torch.sin(phase + torch.pi / 2.0))
    _set_joint(q, names, "left_shoulder_pitch_joint", -0.18 * torch.sin(phase))
    _set_joint(q, names, "right_shoulder_pitch_joint", -0.18 * torch.sin(phase + torch.pi))

    velocity_start = model.nq
    states[:, velocity_start] = stride / max((frames - 1) * dt, dt)
    if frames > 1:
        joint_velocity = torch.gradient(q, spacing=(times,), dim=0)[0]
        states[:, velocity_start + 6 :] = joint_velocity
    return states, times


def _set_joint(
    q: torch.Tensor, joint_names: tuple[str, ...], joint_name: str, values: torch.Tensor
) -> None:
    if joint_name in joint_names:
        q[:, joint_names.index(joint_name)] = values
