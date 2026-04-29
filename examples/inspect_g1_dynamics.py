from __future__ import annotations

import torch

from differentiable_humanoid_dynamics import HumanoidDynamics


def main() -> None:
    """Inspect Adam-backed Unitree G1 dynamics tensors.

    Args:
        None.

    Returns:
        None. Prints tensor shapes and finiteness checks for a neutral state.
    """
    model = HumanoidDynamics(
        "unitree_g1",
        include_contact_forces=True,
        contact_mode="feet_corners",
        dtype=torch.float64,
    )
    x = model.neutral_state()
    terms = model.dynamics_terms(x)
    contact_poses = model.contact_model.contact_poses(  # type: ignore[union-attr]
        model.base_transform(x),
        model.split_state(x).joint_positions.squeeze(0),
    )

    print(f"asset: {model.asset.name}")
    print(f"state_dim: {model.state_dim}")
    print(f"input_dim: {model.input_dim}")
    print(f"joint_names: {model.joint_names}")
    print(f"mass_matrix: {tuple(terms.mass_matrix.shape)}")
    print(f"bias: {tuple(terms.bias.shape)}")
    print(f"contact_positions: {tuple(contact_poses.positions.shape)}")
    print(f"contact_quaternions_wxyz: {tuple(contact_poses.quaternions_wxyz.shape)}")
    print(f"all finite: {torch.isfinite(model.f(x)).all().item() and torch.isfinite(model.g(x)).all().item()}")


if __name__ == "__main__":
    main()
