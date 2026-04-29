from __future__ import annotations

import pytest
import torch

from differentiable_humanoid_dynamics import HumanoidDynamics


@pytest.fixture(scope="module")
def model() -> HumanoidDynamics:
    pytest.importorskip("adam")
    return HumanoidDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)


def test_contact_fk_and_jacobian_shapes(model: HumanoidDynamics) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    split = model.split_state(x)
    base_transform = model.base_transform(x)
    positions = model.contact_model.contact_positions(base_transform, split.joint_positions.squeeze(0))
    poses = model.contact_model.contact_poses(base_transform, split.joint_positions.squeeze(0))
    normals = model.contact_model.contact_normals(base_transform, split.joint_positions.squeeze(0))
    jacobian = model.contact_model.contact_jacobian(base_transform, split.joint_positions.squeeze(0))
    spatial_jacobian = model.contact_model.contact_spatial_jacobian(
        base_transform, split.joint_positions.squeeze(0)
    )
    assert positions.shape == (8, 3)
    assert poses.positions.shape == (8, 3)
    assert poses.quaternions_wxyz.shape == (8, 4)
    assert poses.transforms.shape == (8, 4, 4)
    assert normals.shape == (8, 3)
    assert jacobian.shape == (24, model.nv)
    assert spatial_jacobian.shape == (48, model.nv)
    assert torch.isfinite(positions).all()
    assert torch.isfinite(poses.quaternions_wxyz).all()
    assert torch.isfinite(normals).all()
    assert torch.isfinite(jacobian).all()
    unit = torch.ones(8, dtype=model.dtype)
    assert torch.allclose(torch.linalg.norm(poses.quaternions_wxyz, dim=-1), unit)
    assert torch.allclose(torch.linalg.norm(normals, dim=-1), unit)


def test_contact_jacobian_joint_directional_derivative(model: HumanoidDynamics) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    split = model.split_state(x)
    q = split.joint_positions.squeeze(0)
    direction = torch.linspace(-1.0, 1.0, model.n_joints, dtype=model.dtype)
    direction = direction / torch.linalg.norm(direction)
    eps = torch.as_tensor(1e-6, dtype=model.dtype)
    base_transform = model.base_transform(x)

    p_plus = model.contact_model.contact_positions(base_transform, q + eps * direction).reshape(-1)
    p_minus = model.contact_model.contact_positions(base_transform, q - eps * direction).reshape(-1)
    finite_difference = (p_plus - p_minus) / (2.0 * eps)
    jacobian = model.contact_model.contact_jacobian(base_transform, q)
    predicted = jacobian[:, 6:].matmul(direction)
    assert torch.allclose(predicted, finite_difference, atol=2e-4, rtol=2e-4)


def test_contact_frame_force_mapping_is_supported() -> None:
    pytest.importorskip("adam")
    model = HumanoidDynamics(
        "unitree_g1",
        include_contact_forces=True,
        contact_force_frame="contact",
        dtype=torch.float64,
    )
    x = model.neutral_state()
    control = model.g(x)
    assert control.shape == (model.state_dim, model.input_dim)
    assert torch.isfinite(control).all()
