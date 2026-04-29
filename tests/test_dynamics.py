from __future__ import annotations

import pytest
import torch

from differentiable_humanoid_dynamics import HumanoidDynamics


@pytest.fixture(scope="module")
def model() -> HumanoidDynamics:
    pytest.importorskip("adam")
    return HumanoidDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)


def test_adam_computes_g1_dynamics_terms(model: HumanoidDynamics) -> None:
    x = model.neutral_state()
    terms = model.dynamics_terms(x)
    nv = model.nv
    assert terms.mass_matrix.shape == (nv, nv)
    assert terms.coriolis.shape == (nv,)
    assert terms.gravity.shape == (nv,)
    assert terms.bias.shape == (nv,)
    assert torch.isfinite(terms.mass_matrix).all()
    assert torch.isfinite(terms.coriolis).all()
    assert torch.isfinite(terms.gravity).all()
    assert torch.allclose(terms.mass_matrix, terms.mass_matrix.T, atol=1e-7, rtol=1e-7)


def test_f_and_g_shapes(model: HumanoidDynamics) -> None:
    x = model.neutral_state()
    drift = model.f(x)
    control = model.g(x)
    assert drift.shape == (model.state_dim,)
    assert control.shape == (model.state_dim, model.input_dim)
    assert torch.isfinite(drift).all()
    assert torch.isfinite(control).all()
    assert torch.count_nonzero(control[: model.nq]) == 0
    assert model.input_dim == model.n_joints + 24


def test_forward_matches_control_affine_formula(model: HumanoidDynamics) -> None:
    x = model.neutral_state()
    u = torch.linspace(-0.05, 0.05, model.input_dim, dtype=model.dtype)
    expected = model.f(x) + model.g(x).matmul(u)
    assert torch.allclose(model(x, u), expected)


def test_drift_is_differentiable(model: HumanoidDynamics) -> None:
    x = model.neutral_state().requires_grad_(True)
    y = model.f(x)[model.nq :].sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
