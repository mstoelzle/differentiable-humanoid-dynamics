from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .urdf import UrdfInfo, parse_urdf


@dataclass(frozen=True)
class HumanoidAsset:
    name: str
    urdf_path: Path
    adam_urdf_path: Path
    root_link: str
    joint_names: tuple[str, ...]
    default_contact_links: tuple[str, ...]
    urdf: UrdfInfo


_ASSET_VARIANTS = {
    "unitree_g1": "g1_29dof_rev_1_0.urdf",
    "g1": "g1_29dof_rev_1_0.urdf",
    "g1_29dof_rev_1_0": "g1_29dof_rev_1_0.urdf",
    "g1_29dof_mode_11": "g1_29dof_mode_11.urdf",
    "g1_29dof_mode_12": "g1_29dof_mode_12.urdf",
    "g1_29dof_mode_13": "g1_29dof_mode_13.urdf",
    "g1_29dof_mode_14": "g1_29dof_mode_14.urdf",
    "g1_29dof_mode_15": "g1_29dof_mode_15.urdf",
    "g1_29dof_mode_16": "g1_29dof_mode_16.urdf",
}

_UNITREE_G1_CONTACT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def available_assets() -> tuple[str, ...]:
    return tuple(sorted(_ASSET_VARIANTS))


@lru_cache(maxsize=None)
def load_asset(asset_name: str = "unitree_g1") -> HumanoidAsset:
    """Resolve a supported asset name or direct URDF path."""
    candidate = Path(asset_name).expanduser()
    if candidate.exists():
        urdf_path = candidate.resolve()
        canonical_name = urdf_path.stem
    else:
        if asset_name not in _ASSET_VARIANTS:
            raise KeyError(
                f"Unknown asset {asset_name!r}. Available assets: {', '.join(available_assets())}"
            )
        canonical_name = asset_name
        urdf_path = (
            resources.files("differentiable_humanoid_dynamics")
            / "assets"
            / "robots"
            / "unitree_g1"
            / _ASSET_VARIANTS[asset_name]
        )
        urdf_path = Path(str(urdf_path))

    info = parse_urdf(urdf_path)
    contact_links = tuple(
        link for link in _UNITREE_G1_CONTACT_LINKS if any(c.link_name == link for c in info.collisions)
    )
    return HumanoidAsset(
        name=canonical_name,
        urdf_path=urdf_path,
        adam_urdf_path=_adam_compatible_path(urdf_path),
        root_link=info.root_link,
        joint_names=info.joint_names,
        default_contact_links=contact_links,
        urdf=info,
    )


def _adam_compatible_path(urdf_path: Path) -> Path:
    candidate = urdf_path.with_name(f"{urdf_path.stem}.adam.urdf")
    return candidate if candidate.exists() else urdf_path
