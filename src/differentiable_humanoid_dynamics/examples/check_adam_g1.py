from __future__ import annotations

import torch

from differentiable_humanoid_dynamics import HumanoidDynamics


def main() -> None:
    model = HumanoidDynamics(
        "unitree_g1",
        include_contact_forces=True,
        contact_mode="feet_corners",
        dtype=torch.float64,
    )
    x = model.neutral_state()
    terms = model.dynamics_terms(x)
    drift = model.f(x)
    control = model.g(x)

    print(f"asset: {model.asset.name}")
    print(f"Adam URDF: {model.asset.adam_urdf_path.name}")
    print(f"root link: {model.root_link}")
    print(f"joints: {model.n_joints}")
    print(f"M: {tuple(terms.mass_matrix.shape)} finite={torch.isfinite(terms.mass_matrix).all().item()}")
    print(f"C: {tuple(terms.coriolis.shape)} finite={torch.isfinite(terms.coriolis).all().item()}")
    print(f"G: {tuple(terms.gravity.shape)} finite={torch.isfinite(terms.gravity).all().item()}")
    print(f"f(x): {tuple(drift.shape)} finite={torch.isfinite(drift).all().item()}")
    print(f"g(x): {tuple(control.shape)} finite={torch.isfinite(control).all().item()}")


if __name__ == "__main__":
    main()
