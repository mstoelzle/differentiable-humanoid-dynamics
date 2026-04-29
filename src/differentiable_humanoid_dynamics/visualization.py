from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .dynamics import HumanoidDynamics
from .motion import default_g1_motion_reference, load_kinematic_motion_reference
from .walking import simple_walking_sequence


def run_contact_viewer(
    *,
    asset_name: str = "unitree_g1",
    contact_mode: str = "feet_corners",
    fps: float = 30.0,
    port: int = 8080,
    load_meshes: bool = True,
    max_frames: int | None = None,
    motion_reference: str | Path | None = None,
    synthetic_motion: bool = False,
) -> None:
    """Run a Viser URDF viewer with Adam FK contact frames overlaid.

    Args:
        asset_name: Asset alias or URDF path passed to
            :class:`HumanoidDynamics`.
        contact_mode: Contact extraction mode passed to
            :class:`HumanoidDynamics`, such as ``"feet_corners"`` or
            ``"feet_centers"``.
        fps: Playback frequency for the synthetic walking sequence.
        port: TCP port used by the local Viser server.
        load_meshes: Whether to load visual meshes from the URDF.
        max_frames: Optional number of frames to render before exiting. This
            is intended for smoke tests; ``None`` runs the viewer indefinitely.
        motion_reference: Optional path to a retargeted G1 motion ``.npz`` or
            raw ``.npy`` file. If ``None``, the bundled retargeted AMASS/G1
            sample is used unless ``synthetic_motion`` is ``True``.
        synthetic_motion: Whether to use the old deterministic walking-like
            fallback instead of a retargeted motion reference.

    Returns:
        None. The function starts a local Viser server and updates scene nodes
        until interrupted or until ``max_frames`` is reached.
    """
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise ImportError("Install visualization extras with `uv sync --extra viz`.") from exc

    model = HumanoidDynamics(
        asset_name,
        include_contact_forces=True,
        contact_mode=contact_mode,
        dtype=torch.float64,
    )
    if model.contact_model is None:
        raise RuntimeError("Contact model was not initialized.")

    if synthetic_motion:
        states, _ = simple_walking_sequence(model, frames=180, dt=1.0 / fps)
    elif motion_reference is None:
        states = default_g1_motion_reference(model).states
    else:
        states = load_kinematic_motion_reference(motion_reference, model).states

    floor_width, floor_height, floor_position = _floor_geometry_from_states(states)

    server = viser.ViserServer(port=port)
    floor_thickness = 0.018
    server.scene.add_box(
        "/floor_plane",
        dimensions=(floor_width, floor_height, floor_thickness),
        color=(238, 241, 245),
        material="standard",
        position=(floor_position[0], floor_position[1], -0.5 * floor_thickness),
        cast_shadow=False,
        receive_shadow=True,
    )
    server.scene.add_grid(
        "/ground_grid",
        width=floor_width,
        height=floor_height,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        cell_color=(198, 206, 216),
        section_color=(126, 140, 156),
        cell_thickness=0.8,
        section_thickness=1.2,
        shadow_opacity=0.25,
        fade_distance=max(floor_width, floor_height) * 1.5,
        fade_strength=0.35,
        position=(floor_position[0], floor_position[1], 0.002),
    )
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(
        server,
        urdf_or_path=Path(model.asset.urdf_path),
        root_node_name="/robot",
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )

    contact_handles = [
        server.scene.add_icosphere(
            f"/contacts/{name.replace(':', '_')}",
            radius=0.018,
            color=(255, 80, 20),
        )
        for name in model.contact_model.contact_names
    ]
    contact_frame_handles = [
        server.scene.add_frame(
            f"/contact_frames/{name.replace(':', '_')}",
            axes_length=0.06,
            axes_radius=0.004,
        )
        for name in model.contact_model.contact_names
    ]

    frame = 0
    frames_rendered = 0
    period = 1.0 / fps
    while True:
        state = states[frame]
        split = model.split_state(state)
        joint_positions = split.joint_positions.squeeze(0).detach().cpu().numpy()
        base_position = split.base_position.squeeze(0).detach().cpu().numpy()
        base_quaternion = split.base_quaternion_wxyz.squeeze(0).detach().cpu().numpy()
        urdf_vis.update_cfg(joint_positions)
        robot_root.position = tuple(float(value) for value in base_position)
        robot_root.wxyz = tuple(float(value) for value in base_quaternion)

        base_transform = model.base_transform(state)
        contact_poses = model.contact_model.contact_poses(
            base_transform,
            split.joint_positions.squeeze(0),
        )
        contact_positions = (
            contact_poses.positions
            .detach()
            .cpu()
            .numpy()
        )
        contact_quaternions = (
            contact_poses.quaternions_wxyz
            .detach()
            .cpu()
            .numpy()
        )
        for handle, frame_handle, point, quat in zip(
            contact_handles,
            contact_frame_handles,
            np.asarray(contact_positions),
            np.asarray(contact_quaternions),
        ):
            handle.position = tuple(float(value) for value in point)
            frame_handle.position = tuple(float(value) for value in point)
            frame_handle.wxyz = tuple(float(value) for value in quat)

        frame = (frame + 1) % len(states)
        frames_rendered += 1
        if max_frames is not None and frames_rendered >= max_frames:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        time.sleep(period)


def main() -> None:
    """Parse CLI arguments and launch the contact viewer.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 contact-point FK.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--contact-mode", default="feet_corners", choices=("feet_corners", "feet_centers"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--motion-reference", type=Path, default=None)
    parser.add_argument("--synthetic-motion", action="store_true")
    args = parser.parse_args()
    run_contact_viewer(
        asset_name=args.asset,
        contact_mode=args.contact_mode,
        fps=args.fps,
        port=args.port,
        load_meshes=not args.no_meshes,
        max_frames=args.max_frames,
        motion_reference=args.motion_reference,
        synthetic_motion=args.synthetic_motion,
    )


def _floor_geometry_from_states(
    states: torch.Tensor,
    *,
    margin: float = 1.5,
    min_width: float = 8.0,
    min_height: float = 4.0,
) -> tuple[float, float, tuple[float, float, float]]:
    """Choose a ground-plane footprint that covers a walking sequence.

    Args:
        states: Motion state tensor with shape ``(num_frames, state_dim)``.
            The first two state entries are interpreted as the floating-base
            ``x`` and ``y`` world positions.
        margin: Extra meters added to both sides of the motion extent.
        min_width: Minimum ground size in meters along world ``x``.
        min_height: Minimum ground size in meters along world ``y``.

    Returns:
        Tuple ``(width, height, position)`` where ``width`` and ``height`` are
        meter dimensions for an ``xy`` Viser grid, and ``position`` is the
        grid center with shape ``(3,)``.

    Raises:
        ValueError: If ``states`` does not have shape ``(num_frames, state_dim)``
            with at least two state entries.
    """
    if states.ndim != 2 or states.shape[-1] < 2:
        raise ValueError("Expected states with shape (num_frames, state_dim).")

    xy = states[..., :2].detach().cpu()
    lower = torch.amin(xy, dim=0)
    upper = torch.amax(xy, dim=0)
    center = 0.5 * (lower + upper)
    span = upper - lower
    width = max(min_width, float(span[0]) + 2.0 * margin)
    height = max(min_height, float(span[1]) + 2.0 * margin)
    position = (float(center[0]), float(center[1]), 0.0)
    return width, height, position
