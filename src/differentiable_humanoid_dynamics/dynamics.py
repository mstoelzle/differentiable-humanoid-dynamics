from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import NamedTuple

import torch

from ._torch import (
    ensure_batch,
    make_transform,
    normalize_quaternion_wxyz,
    quaternion_derivative_from_world_angular_velocity,
)
from .assets import HumanoidAsset, load_asset
from .contacts import HumanoidContactModel


class SplitState(NamedTuple):
    base_position: torch.Tensor
    base_quaternion_wxyz: torch.Tensor
    joint_positions: torch.Tensor
    base_velocity: torch.Tensor
    joint_velocities: torch.Tensor
    was_single: bool


@dataclass(frozen=True)
class DynamicsTerms:
    mass_matrix: torch.Tensor
    coriolis: torch.Tensor
    gravity: torch.Tensor
    bias: torch.Tensor


class HumanoidDynamics(torch.nn.Module):
    r"""Control-affine differentiable humanoid dynamics.

    State convention
    ----------------
    The state is ``x = (q, nu)`` with
    ``q = (p_WB, quat_WB, s)`` and
    ``nu = (v_WB, omega_WB, s_dot)``.

    ``p_WB`` is the world-frame position of the floating-base/root link
    origin. ``quat_WB`` is a unit quaternion in ``(w, x, y, z)`` order and maps
    vectors from the base/root frame ``B`` to the world frame ``W``. The joint
    vector ``s`` is ordered exactly as ``self.joint_names``; this list is parsed
    from the URDF and passed to Adam.

    The floating-base velocity uses Adam's mixed representation:
    ``v_WB`` and ``omega_WB`` are both expressed in the inertial/world frame.
    Therefore the configuration derivative is
    ``p_dot = v_WB`` and ``quat_dot = 0.5 * [0, omega_WB] * quat_WB``.

    Generalized acceleration is
    ``nu_dot = (v_dot_WB, omega_dot_WB, s_ddot)``. Adam provides terms in
    ``M(q) nu_dot + h(q, nu) = S^T tau + J_c(q)^T lambda``. The drift ``f(x)``
    uses zero joint torques and zero contact forces. The input matrix ``g(x)``
    maps joint torques through ``S^T`` and, when enabled, stacked contact
    forces through translational contact Jacobians. Contact forces can be
    represented in world coordinates or in each contact frame, controlled by
    ``contact_force_frame``.
    """

    def __init__(
        self,
        asset_name: str = "unitree_g1",
        *,
        include_contact_forces: bool = False,
        contact_mode: str = "feet_corners",
        contact_force_frame: Literal["world", "contact"] = "world",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.asset: HumanoidAsset = load_asset(asset_name)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.include_contact_forces = include_contact_forces
        self.contact_force_frame = contact_force_frame
        if contact_force_frame not in {"world", "contact"}:
            raise ValueError("contact_force_frame must be 'world' or 'contact'.")

        self.joint_names = self.asset.joint_names
        self.root_link = self.asset.root_link
        self.n_joints = len(self.joint_names)
        self.nv = 6 + self.n_joints
        self.nq = 7 + self.n_joints
        self.state_dim = self.nq + self.nv

        self.kindyn = _build_adam_kindyn(self.asset, self.dtype, self.device)
        self.contact_model: HumanoidContactModel | None = None
        if include_contact_forces:
            self.contact_model = HumanoidContactModel(
                self.asset,
                kin_dyn=self.kindyn,
                mode=contact_mode,
                dtype=dtype,
                device=self.device,
            )
        self.input_dim = self.n_joints + (
            self.contact_model.force_dim if self.contact_model is not None else 0
        )

    def neutral_state(self, *, base_height: float = 0.78) -> torch.Tensor:
        """Return a neutral floating-base state for smoke tests and examples."""
        state = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        state[2] = base_height
        state[3] = 1.0
        return state

    def split_state(self, x: torch.Tensor) -> SplitState:
        x, was_single = ensure_batch(x.to(dtype=self.dtype, device=self.device))
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {x.shape[-1]}")

        joint_start = 7
        velocity_start = 7 + self.n_joints
        base_position = x[..., :3]
        base_quaternion = normalize_quaternion_wxyz(x[..., 3:7])
        joint_positions = x[..., joint_start:velocity_start]
        base_velocity = x[..., velocity_start : velocity_start + 6]
        joint_velocities = x[..., velocity_start + 6 :]
        return SplitState(
            base_position=base_position,
            base_quaternion_wxyz=base_quaternion,
            joint_positions=joint_positions,
            base_velocity=base_velocity,
            joint_velocities=joint_velocities,
            was_single=was_single,
        )

    def base_transform(self, x: torch.Tensor) -> torch.Tensor:
        split = self.split_state(x)
        transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        return transform.squeeze(0) if split.was_single else transform

    def dynamics_terms(self, x: torch.Tensor) -> DynamicsTerms:
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        coriolis = self.kindyn.coriolis_term(
            base_transform,
            split.joint_positions,
            split.base_velocity,
            split.joint_velocities,
        )
        gravity = self.kindyn.gravity_term(base_transform, split.joint_positions)
        if hasattr(self.kindyn, "bias_force"):
            bias = self.kindyn.bias_force(
                base_transform,
                split.joint_positions,
                split.base_velocity,
                split.joint_velocities,
            )
        else:
            bias = coriolis + gravity
        if split.was_single:
            return DynamicsTerms(
                mass_matrix=mass.squeeze(0),
                coriolis=coriolis.squeeze(0),
                gravity=gravity.squeeze(0),
                bias=bias.squeeze(0),
            )
        return DynamicsTerms(mass_matrix=mass, coriolis=coriolis, gravity=gravity, bias=bias)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Return the autonomous drift dynamics ``x_dot = f(x)``."""
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        if hasattr(self.kindyn, "bias_force"):
            bias = self.kindyn.bias_force(
                base_transform,
                split.joint_positions,
                split.base_velocity,
                split.joint_velocities,
            )
        else:
            bias = self.kindyn.coriolis_term(
                base_transform,
                split.joint_positions,
                split.base_velocity,
                split.joint_velocities,
            ) + self.kindyn.gravity_term(base_transform, split.joint_positions)

        acceleration = torch.linalg.solve(mass, -bias.unsqueeze(-1)).squeeze(-1)
        q_dot = self._configuration_derivative(split)
        x_dot = torch.cat((q_dot, acceleration), dim=-1)
        return x_dot.squeeze(0) if split.was_single else x_dot

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Return the control matrix in ``x_dot = f(x) + g(x) u``."""
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        generalized_input = self._generalized_input_matrix(base_transform, split.joint_positions)
        acceleration_map = torch.linalg.solve(mass, generalized_input)

        batch = split.base_position.shape[0]
        control_matrix = torch.zeros(
            batch,
            self.state_dim,
            self.input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        control_matrix[..., self.nq :, :] = acceleration_map
        return control_matrix.squeeze(0) if split.was_single else control_matrix

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate ``f(x)`` or ``f(x) + g(x) u``."""
        drift = self.f(x)
        if u is None:
            return drift
        control = self.g(x)
        return drift + torch.matmul(control, u.unsqueeze(-1)).squeeze(-1)

    def selection_matrix_transpose(self, *, batch_size: int | None = None) -> torch.Tensor:
        """Return ``S^T`` mapping joint torques to generalized forces."""
        selection = torch.zeros(self.nv, self.n_joints, dtype=self.dtype, device=self.device)
        selection[6:, :] = torch.eye(self.n_joints, dtype=self.dtype, device=self.device)
        if batch_size is None:
            return selection
        return selection.expand(batch_size, self.nv, self.n_joints)

    def _configuration_derivative(self, split: SplitState) -> torch.Tensor:
        p_dot = split.base_velocity[..., :3]
        omega_world = split.base_velocity[..., 3:6]
        quat_dot = quaternion_derivative_from_world_angular_velocity(
            split.base_quaternion_wxyz, omega_world
        )
        return torch.cat((p_dot, quat_dot, split.joint_velocities), dim=-1)

    def _generalized_input_matrix(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        batch = base_transform.shape[0]
        torque_map = self.selection_matrix_transpose(batch_size=batch)
        if self.contact_model is None:
            return torque_map

        contact_jacobian = self.contact_model.contact_jacobian(base_transform, joint_positions)
        force_transform = self.contact_model.contact_force_transform(
            base_transform,
            joint_positions,
            force_frame=self.contact_force_frame,
        )
        contact_map = torch.matmul(contact_jacobian.transpose(-1, -2), force_transform)
        return torch.cat((torque_map, contact_map), dim=-1)


def _build_adam_kindyn(asset: HumanoidAsset, dtype: torch.dtype, device: torch.device):
    try:
        import adam
        from adam.pytorch import KinDynComputations
    except ImportError as exc:
        raise ImportError(
            "Adam is required for HumanoidDynamics. Install with "
            "`uv sync` or `pip install 'adam-robotics[pytorch]'`."
        ) from exc

    gravity = torch.as_tensor([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    try:
        kindyn = KinDynComputations(
            str(asset.adam_urdf_path),
            list(asset.joint_names),
            device=device,
            dtype=dtype,
            gravity=gravity,
        )
    except TypeError:
        kindyn = KinDynComputations(
            str(asset.adam_urdf_path),
            list(asset.joint_names),
            gravity=gravity,
        )
    kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    return kindyn
