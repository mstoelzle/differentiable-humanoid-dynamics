from __future__ import annotations

import pytest
import torch

from differentiable_humanoid_dynamics.visualization import _floor_geometry_from_states


def test_floor_geometry_covers_motion_extent() -> None:
    states = torch.zeros(4, 10, dtype=torch.float64)
    states[:, 0] = torch.tensor([-0.5, 0.0, 2.0, 4.5], dtype=torch.float64)
    states[:, 1] = torch.tensor([-1.0, 0.25, 0.5, 1.0], dtype=torch.float64)

    width, height, position = _floor_geometry_from_states(
        states,
        margin=1.0,
        min_width=2.0,
        min_height=2.0,
    )

    assert width == pytest.approx(7.0)
    assert height == pytest.approx(4.0)
    assert position == pytest.approx((2.0, 0.0, 0.0))


def test_floor_geometry_uses_minimum_size_for_short_clips() -> None:
    states = torch.zeros(2, 10, dtype=torch.float64)

    width, height, position = _floor_geometry_from_states(states)

    assert width == pytest.approx(8.0)
    assert height == pytest.approx(4.0)
    assert position == pytest.approx((0.0, 0.0, 0.0))


def test_floor_geometry_rejects_bad_state_shape() -> None:
    with pytest.raises(ValueError, match="Expected states"):
        _floor_geometry_from_states(torch.zeros(3))
