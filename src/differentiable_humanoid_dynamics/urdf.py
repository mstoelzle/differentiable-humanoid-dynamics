from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


MOVABLE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class JointInfo:
    """Movable URDF joint metadata.

    Attributes:
        name: Joint name.
        joint_type: URDF joint type.
        parent: Parent link name.
        child: Child link name.
        lower: Lower position limit, when present.
        upper: Upper position limit, when present.
        effort: Effort limit, when present.
        velocity: Velocity limit, when present.
    """

    name: str
    joint_type: str
    parent: str
    child: str
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass(frozen=True)
class CollisionInfo:
    """URDF collision geometry metadata.

    Attributes:
        link_name: Link that owns this collision geometry.
        xyz: Collision origin translation in the link frame with shape
            ``(3,)``.
        rpy: Collision origin roll-pitch-yaw orientation in radians with shape
            ``(3,)``.
        geometry_type: One of ``"sphere"``, ``"box"``, ``"cylinder"``,
            ``"mesh"``, or ``"unknown"``.
        size: Geometry parameters. Sphere uses ``(radius,)``; box uses
            ``(x, y, z)``; cylinder uses ``(radius, length)``; mesh stores a
            placeholder scalar.
    """

    link_name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    geometry_type: str
    size: tuple[float, ...]


@dataclass(frozen=True)
class UrdfInfo:
    """Parsed subset of a URDF needed by this package.

    Attributes:
        name: Robot name from the URDF root.
        root_link: Link with no parent joint.
        joints: Ordered movable joint metadata.
        collisions: Collision geometry metadata used for contact extraction.
    """

    name: str
    root_link: str
    joints: tuple[JointInfo, ...]
    collisions: tuple[CollisionInfo, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return ordered movable joint names.

        Args:
            None.

        Returns:
            Tuple of joint names. Tensor joint coordinates with shape
            ``(..., n_joints)`` follow this order.
        """
        return tuple(joint.name for joint in self.joints)


def parse_urdf(path: str | Path) -> UrdfInfo:
    """Parse the URDF metadata needed by dynamics and contact setup.

    Args:
        path: Path to a URDF file.

    Returns:
        Parsed :class:`UrdfInfo` containing the robot name, root link, ordered
        movable joints, and collision geometries.

    Raises:
        ValueError: If the URDF has no root link or required joint tags are
            missing.
        xml.etree.ElementTree.ParseError: If the XML is invalid.
    """
    urdf_path = Path(path)
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    child_links: set[str] = set()
    joints: list[JointInfo] = []

    for joint in root.findall("joint"):
        joint_type = joint.attrib["type"]
        parent = _required_link(joint, "parent")
        child = _required_link(joint, "child")
        child_links.add(child)
        if joint_type not in MOVABLE_JOINT_TYPES:
            continue

        limit = joint.find("limit")
        joints.append(
            JointInfo(
                name=joint.attrib["name"],
                joint_type=joint_type,
                parent=parent,
                child=child,
                lower=_optional_float(limit, "lower"),
                upper=_optional_float(limit, "upper"),
                effort=_optional_float(limit, "effort"),
                velocity=_optional_float(limit, "velocity"),
            )
        )

    root_candidates = sorted(links - child_links)
    if not root_candidates:
        raise ValueError(f"URDF has no root link: {urdf_path}")
    root_link = root_candidates[0]

    collisions: list[CollisionInfo] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz = _vector_attr(origin, "xyz", default=(0.0, 0.0, 0.0))
            rpy = _vector_attr(origin, "rpy", default=(0.0, 0.0, 0.0))
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            geometry_type, size = _parse_geometry(geometry)
            collisions.append(
                CollisionInfo(
                    link_name=link_name,
                    xyz=xyz,
                    rpy=rpy,
                    geometry_type=geometry_type,
                    size=size,
                )
            )

    return UrdfInfo(
        name=root.attrib.get("name", urdf_path.stem),
        root_link=root_link,
        joints=tuple(joints),
        collisions=tuple(collisions),
    )


def foot_collision_points(
    info: UrdfInfo,
    link_names: tuple[str, ...],
    *,
    geometry_type: str = "sphere",
) -> dict[str, np.ndarray]:
    """Return collision-origin contact candidates grouped by foot link.

    Args:
        info: Parsed URDF metadata.
        link_names: Link names whose collision origins should be extracted.
        geometry_type: Collision geometry type to include.

    Returns:
        Dictionary mapping each requested link name to an array with shape
        ``(num_points_for_link, 3)`` containing collision-origin translations in
        that link frame.

    Raises:
        ValueError: If no matching collision geometry is found for any
            requested link.
    """
    points: dict[str, list[tuple[float, float, float]]] = {link: [] for link in link_names}
    for collision in info.collisions:
        if collision.link_name not in points:
            continue
        if collision.geometry_type != geometry_type:
            continue
        points[collision.link_name].append(collision.xyz)

    missing = [link for link, link_points in points.items() if not link_points]
    if missing:
        raise ValueError(
            "No collision-origin contact points found for "
            f"{missing}; available collision links are "
            f"{sorted({collision.link_name for collision in info.collisions})}"
        )

    return {link: np.asarray(link_points, dtype=np.float64) for link, link_points in points.items()}


def _required_link(joint: ET.Element, tag: str) -> str:
    """Read a required ``<parent>`` or ``<child>`` link from a joint element.

    Args:
        joint: XML joint element.
        tag: Either ``"parent"`` or ``"child"``.

    Returns:
        Link name stored in the required child element.

    Raises:
        ValueError: If the required child element or ``link`` attribute is
            missing.
    """
    child = joint.find(tag)
    if child is None or "link" not in child.attrib:
        raise ValueError(f"Joint {joint.attrib.get('name', '<unnamed>')} is missing <{tag} link=...>")
    return child.attrib["link"]


def _optional_float(element: ET.Element | None, key: str) -> float | None:
    """Read an optional float attribute from an XML element.

    Args:
        element: XML element or ``None``.
        key: Attribute name to read.

    Returns:
        Parsed float value, or ``None`` when the element or attribute is absent.
    """
    if element is None or key not in element.attrib:
        return None
    return float(element.attrib[key])


def _vector_attr(
    element: ET.Element | None,
    key: str,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Read a three-vector attribute from an XML element.

    Args:
        element: XML element or ``None``.
        key: Attribute name containing three whitespace-separated numbers.
        default: Value returned when the element or attribute is absent.

    Returns:
        Tuple with shape ``(3,)``.

    Raises:
        ValueError: If the attribute exists but does not contain exactly three
            values.
    """
    if element is None or key not in element.attrib:
        return default
    values = tuple(float(value) for value in element.attrib[key].split())
    if len(values) != 3:
        raise ValueError(f"Expected three values for {key}, got {values}")
    return values


def _parse_geometry(geometry: ET.Element) -> tuple[str, tuple[float, ...]]:
    """Parse a URDF ``<geometry>`` element.

    Args:
        geometry: XML geometry element.

    Returns:
        Tuple ``(geometry_type, size)``. ``size`` contains the geometry
        parameters described by :class:`CollisionInfo`.
    """
    sphere = geometry.find("sphere")
    if sphere is not None:
        return "sphere", (float(sphere.attrib["radius"]),)

    box = geometry.find("box")
    if box is not None:
        return "box", tuple(float(value) for value in box.attrib["size"].split())

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return "cylinder", (
            float(cylinder.attrib["radius"]),
            float(cylinder.attrib["length"]),
        )

    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.attrib.get("filename", "")
        return "mesh", (float(len(filename)),)

    return "unknown", ()
