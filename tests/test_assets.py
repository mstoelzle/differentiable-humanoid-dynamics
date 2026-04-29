from __future__ import annotations

from differentiable_humanoid_dynamics import HumanoidContactModel, available_assets, load_asset


def test_unitree_g1_asset_metadata() -> None:
    assert "unitree_g1" in available_assets()
    asset = load_asset("unitree_g1")
    assert asset.urdf_path.name == "g1_29dof_rev_1_0.urdf"
    assert asset.adam_urdf_path.name == "g1_29dof_rev_1_0.adam.urdf"
    assert asset.root_link == "pelvis"
    assert len(asset.joint_names) == 29
    assert "left_ankle_roll_link" in asset.default_contact_links
    assert "right_ankle_roll_link" in asset.default_contact_links


def test_deprecated_unitree_g1_files_are_not_vendored() -> None:
    asset = load_asset("unitree_g1")
    folder = asset.urdf_path.parent
    deprecated = {
        "g1_23dof.urdf",
        "g1_23dof.xml",
        "g1_29dof.urdf",
        "g1_29dof.xml",
        "g1_29dof_with_hand.urdf",
        "g1_29dof_with_hand.xml",
        "g1_29dof_lock_waist.urdf",
        "g1_29dof_lock_waist.xml",
    }
    assert not deprecated.intersection(path.name for path in folder.iterdir())


def test_contact_modes_from_collision_geometry() -> None:
    corners = HumanoidContactModel("unitree_g1", mode="feet_corners")
    centers = HumanoidContactModel("unitree_g1", mode="feet_centers")
    assert corners.num_contacts == 8
    assert centers.num_contacts == 2
    assert corners.local_offsets.shape == (8, 3)
    assert centers.local_offsets.shape == (2, 3)
