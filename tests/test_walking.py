from __future__ import annotations

import pytest
import torch

from focodyn import FloatingBaseDynamics, simple_walking_sequence


def test_simple_walking_sequence_changes_contact_positions() -> None:
    pytest.importorskip("adam")
    model = FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)
    assert model.contact_model is not None
    states, times = simple_walking_sequence(model, frames=12)
    assert states.shape == (12, model.state_dim)
    assert times.shape == (12,)

    split0 = model.split_state(states[0])
    split1 = model.split_state(states[5])
    contacts0 = model.contact_model.contact_positions(
        model.base_transform(states[0]), split0.joint_positions.squeeze(0)
    )
    contacts1 = model.contact_model.contact_positions(
        model.base_transform(states[5]), split1.joint_positions.squeeze(0)
    )
    assert not torch.allclose(contacts0, contacts1)
