from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from ._torch import matrix_to_quaternion_wxyz, rpy_to_matrix, skew
from .assets import HumanoidAsset, load_asset


@dataclass(frozen=True)
class ContactPointSpec:
    name: str
    link_name: str
    offset: tuple[float, float, float]
    rpy: tuple[float, float, float]


class ContactPoses(NamedTuple):
    positions: torch.Tensor
    quaternions_wxyz: torch.Tensor
    transforms: torch.Tensor


class HumanoidContactModel(torch.nn.Module):
    """Differentiable contact-pose kinematics for humanoid feet.

    Contact frames are initialized from collision geometry in the URDF. For the
    Unitree G1, the ankle-roll links contain four small sphere collision origins
    per foot, which are used as foot-corner contact candidates.

    The returned pose convention follows the Adam homogeneous-transform
    convention: ``W_H_C`` maps contact-frame coordinates into world coordinates.
    Quaternions are stored as ``(w, x, y, z)`` and represent the same
    contact-to-world orientation. The contact normal is the contact frame's
    positive z-axis expressed in world coordinates.
    """

    def __init__(
        self,
        asset: str | HumanoidAsset = "unitree_g1",
        *,
        kin_dyn=None,
        mode: str = "feet_corners",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.asset = load_asset(asset) if isinstance(asset, str) else asset
        self.mode = mode
        self.kindyn = kin_dyn
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        specs = _contact_specs_from_asset(self.asset, mode)
        self.contact_specs = tuple(specs)
        self.contact_names = tuple(spec.name for spec in specs)
        self.contact_link_names = tuple(spec.link_name for spec in specs)
        offsets = torch.as_tensor([spec.offset for spec in specs], dtype=dtype, device=self.device)
        local_rpy = torch.as_tensor([spec.rpy for spec in specs], dtype=dtype, device=self.device)
        self.register_buffer("local_offsets", offsets, persistent=False)
        self.register_buffer("local_rotations", rpy_to_matrix(local_rpy), persistent=False)

    @property
    def num_contacts(self) -> int:
        return len(self.contact_specs)

    @property
    def force_dim(self) -> int:
        return 3 * self.num_contacts

    def contact_positions(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return world-frame contact point positions.

        Shape is ``(num_contacts, 3)`` for a single state or
        ``(batch, num_contacts, 3)`` for batched inputs.
        """
        return self.contact_transforms(base_transform, joint_positions)[..., :3, 3]

    def contact_quaternions(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return contact-to-world quaternions in Adam/scalar-first order."""
        rotations = self.contact_transforms(base_transform, joint_positions)[..., :3, :3]
        return matrix_to_quaternion_wxyz(rotations)

    def contact_poses(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> ContactPoses:
        """Return world-frame SE(3) contact poses as positions, quaternions, and matrices."""
        transforms = self.contact_transforms(base_transform, joint_positions)
        return ContactPoses(
            positions=transforms[..., :3, 3],
            quaternions_wxyz=matrix_to_quaternion_wxyz(transforms[..., :3, :3]),
            transforms=transforms,
        )

    def contact_normals(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return each contact frame's positive z-axis in world coordinates."""
        transforms = self.contact_transforms(base_transform, joint_positions)
        return transforms[..., :3, 2]

    def contact_transforms(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return ``W_H_C`` contact transforms.

        Shape is ``(num_contacts, 4, 4)`` for a single state or
        ``(batch, num_contacts, 4, 4)`` for batched inputs.
        """
        transforms = []
        for link_name, offset, local_rotation in zip(
            self.contact_link_names, self.local_offsets, self.local_rotations
        ):
            transform = self._fk(link_name, base_transform, joint_positions)
            rotation = transform[..., :3, :3]
            translation = transform[..., :3, 3]
            contact_rotation = torch.matmul(rotation, local_rotation.to(rotation))
            contact_translation = translation + torch.matmul(
                rotation, offset.to(rotation).unsqueeze(-1)
            ).squeeze(-1)

            contact_transform = torch.zeros_like(transform)
            contact_transform[..., :3, :3] = contact_rotation
            contact_transform[..., :3, 3] = contact_translation
            contact_transform[..., 3, 3] = 1.0
            transforms.append(contact_transform)
        return torch.stack(transforms, dim=-3)

    def contact_jacobian(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return the translational contact Jacobian in world coordinates.

        The output maps Adam mixed generalized velocity
        ``nu = (v_WB, omega_WB, s_dot)`` to stacked contact point velocities.
        Shape is ``(3 * num_contacts, 6 + n_joints)`` or batched as
        ``(batch, 3 * num_contacts, 6 + n_joints)``.
        """
        jacobians = []
        for link_name, offset in zip(self.contact_link_names, self.local_offsets):
            transform = self._fk(link_name, base_transform, joint_positions)
            link_jacobian = self._jacobian(link_name, base_transform, joint_positions)
            rotation = transform[..., :3, :3]
            r_world = torch.matmul(rotation, offset.to(rotation).unsqueeze(-1)).squeeze(-1)
            linear = link_jacobian[..., :3, :]
            angular = link_jacobian[..., 3:6, :]
            point_jacobian = linear - torch.matmul(skew(r_world), angular)
            jacobians.append(point_jacobian)
        return torch.cat(jacobians, dim=-2)

    def contact_spatial_jacobian(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return stacked Adam-order spatial Jacobians at the contact frames.

        Each 6-row block is ``(linear_velocity, angular_velocity)`` in world
        coordinates, matching the mixed-representation convention used for
        contact poses and contact force mapping.
        """
        spatial_blocks = []
        for link_name, offset in zip(self.contact_link_names, self.local_offsets):
            transform = self._fk(link_name, base_transform, joint_positions)
            link_jacobian = self._jacobian(link_name, base_transform, joint_positions)
            rotation = transform[..., :3, :3]
            r_world = torch.matmul(rotation, offset.to(rotation).unsqueeze(-1)).squeeze(-1)
            linear = link_jacobian[..., :3, :] - torch.matmul(
                skew(r_world), link_jacobian[..., 3:6, :]
            )
            spatial_blocks.append(torch.cat((linear, link_jacobian[..., 3:6, :]), dim=-2))
        return torch.cat(spatial_blocks, dim=-2)

    def contact_force_transform(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        *,
        force_frame: str,
    ) -> torch.Tensor:
        """Return the block map from contact-force coordinates to world forces."""
        if force_frame == "world":
            dim = self.force_dim
            batch_shape = base_transform.shape[:-2]
            eye = torch.eye(dim, dtype=base_transform.dtype, device=base_transform.device)
            return eye.expand(*batch_shape, dim, dim)
        if force_frame != "contact":
            raise ValueError("force_frame must be 'world' or 'contact'.")

        rotations = self.contact_transforms(base_transform, joint_positions)[..., :3, :3]
        return _block_diag_rotations(rotations)

    def _fk(
        self, link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        if self.kindyn is None:
            raise RuntimeError("HumanoidContactModel requires an Adam KinDynComputations instance.")
        return self.kindyn.forward_kinematics(link_name, base_transform, joint_positions)

    def _jacobian(
        self, link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        if self.kindyn is None:
            raise RuntimeError("HumanoidContactModel requires an Adam KinDynComputations instance.")
        return self.kindyn.jacobian(link_name, base_transform, joint_positions)


def _contact_specs_from_asset(asset: HumanoidAsset, mode: str) -> list[ContactPointSpec]:
    if not asset.default_contact_links:
        raise ValueError(f"Asset {asset.name!r} has no default foot contact links.")

    foot_collisions = {
        link: [
            collision
            for collision in asset.urdf.collisions
            if collision.link_name == link and collision.geometry_type == "sphere"
        ]
        for link in asset.default_contact_links
    }
    missing = [link for link, collisions in foot_collisions.items() if not collisions]
    if missing:
        raise ValueError(f"No sphere collision contact candidates found for {missing}.")

    specs: list[ContactPointSpec] = []
    if mode == "feet_corners":
        for link_name, collisions in foot_collisions.items():
            for index, collision in enumerate(_sort_foot_collisions(collisions)):
                specs.append(
                    ContactPointSpec(
                        name=f"{link_name}:{index}",
                        link_name=link_name,
                        offset=tuple(float(value) for value in collision.xyz),
                        rpy=tuple(float(value) for value in collision.rpy),
                    )
                )
        return specs

    if mode == "feet_centers":
        for link_name, collisions in foot_collisions.items():
            points = np.asarray([collision.xyz for collision in collisions], dtype=np.float64)
            center = np.mean(points, axis=0)
            specs.append(
                ContactPointSpec(
                    name=f"{link_name}:center",
                    link_name=link_name,
                    offset=tuple(float(value) for value in center),
                    rpy=(0.0, 0.0, 0.0),
                )
            )
        return specs

    raise ValueError("Unknown contact mode. Expected 'feet_corners' or 'feet_centers'.")


def _sort_foot_collisions(collisions):
    # Stable order: heel/toe by x, then left/right by y.
    points = np.asarray([collision.xyz for collision in collisions], dtype=np.float64)
    order = np.lexsort((points[:, 1], points[:, 0]))
    return [collisions[index] for index in order]


def _block_diag_rotations(rotations: torch.Tensor) -> torch.Tensor:
    batch_shape = rotations.shape[:-3]
    num_contacts = rotations.shape[-3]
    output = torch.zeros(
        *batch_shape,
        3 * num_contacts,
        3 * num_contacts,
        dtype=rotations.dtype,
        device=rotations.device,
    )
    for index in range(num_contacts):
        start = 3 * index
        output[..., start : start + 3, start : start + 3] = rotations[..., index, :, :]
    return output
