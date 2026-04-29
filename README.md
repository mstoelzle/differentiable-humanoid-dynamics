# differentiable-humanoid-dynamics

Differentiable humanoid dynamics utilities for CBF/CLF and optimization
workflows that need gradients through rigid-body dynamics and contact-point
kinematics.

The package wraps [Adam](https://github.com/gbionics/adam) through its PyTorch
backend. Adam provides the floating-base mass matrix, Coriolis/centrifugal
terms, gravity terms, Jacobians, and forward kinematics. This repository adds a
small humanoid-facing layer around those quantities:

- an installable Python package with `uv` and Python 3.12,
- vendored Unitree G1 URDF/MJCF assets,
- a shared `torch.nn.Module` exposing control-affine `f(x)` and `g(x)`,
- differentiable foot contact-point forward kinematics and Jacobians, and
- a Viser-based viewer that overlays contact points on a walking-like sequence.

## Install

```bash
uv sync --extra dev
```

For the optional browser visualization:

```bash
uv sync --extra dev --extra viz
```

Run the Adam/G1 smoke check:

```bash
uv run dhd-check-adam-g1
```

Run the test suite:

```bash
uv run pytest
```

## Dynamics API

```python
import torch
from differentiable_humanoid_dynamics import HumanoidDynamics

model = HumanoidDynamics(
    "unitree_g1",
    include_contact_forces=True,
    contact_mode="feet_corners",
    dtype=torch.float64,
)

x = model.neutral_state()
xdot_drift = model.f(x)
control_map = model.g(x)
```

The dynamics are represented as:

```text
x_dot = f(x) + g(x) u
M(q) nu_dot + h(q, nu) = S^T tau + J_c(q)^T lambda
```

`u` is stacked as joint torques first, then optional contact forces
`lambda = (f_c0, f_c1, ...)` with three force components per contact. By
default contact-force inputs are world-frame vectors. Pass
`contact_force_frame="contact"` to express each force in its contact frame; the
module rotates those forces into world coordinates before applying `J_c^T`.

## State Convention

The state is `x = (q, nu)`.

`q = (p_WB, quat_WB, s)`:

- `p_WB` is the world-frame position of the floating-base/root link origin.
- `quat_WB` is a unit quaternion in `(w, x, y, z)` order and maps base-frame
  vectors into the world frame.
- `s` is ordered exactly as `model.joint_names`, which is parsed from the URDF
  and passed to Adam.

`nu = (v_WB, omega_WB, s_dot)` uses Adam's mixed representation:

- `v_WB` is the base linear velocity in the world frame.
- `omega_WB` is the base angular velocity in the world frame.
- `s_dot` follows the same joint order as `s`.

The generalized acceleration is `nu_dot = (v_dot_WB, omega_dot_WB, s_ddot)`.

## Contacts

`HumanoidContactModel` initializes contact candidates from URDF collision
geometry. For the Unitree G1, each ankle-roll link has four small sphere
collisions; `feet_corners` uses those eight sphere origins, and `feet_centers`
uses one averaged point per foot.

Contacts are full `SE(3)` frames, not only points. `contact_poses(...)` returns
world positions, `(w, x, y, z)` contact-to-world quaternions, and homogeneous
`W_H_C` matrices. This matches the Adam transform convention used elsewhere in
the package. The contact normal is the contact frame's positive z-axis
expressed in world coordinates and is available through `contact_normals(...)`.

The translational contact Jacobian maps Adam mixed generalized velocity to
stacked world-frame contact point velocities. The contact force contribution to
the equations of motion is `J_c(q)^T lambda` for world-frame forces, or
`J_c(q)^T R_WC lambda_C` for contact-frame forces.

## Visualization

Start the Viser viewer with:

```bash
uv run --extra viz dhd-visualize-g1
```

The viewer loads the local G1 URDF, plays a simple kinematic walking-like
sequence, and overlays the differentiable contact-point FK estimates. The
sequence is intentionally a visualization fixture, not a dynamically feasible
walking trajectory.

## Sources And Attribution

Rigid-body dynamics are computed by
[Adam](https://github.com/gbionics/adam), distributed on PyPI as
`adam-robotics`. This package depends on `adam-robotics[pytorch]` and does not
reimplement CRBA, RNEA, ABA, frame Jacobians, or forward kinematics.

Robot assets under
`src/differentiable_humanoid_dynamics/assets/robots/unitree_g1` come from
[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
The deprecated G1 model files identified in Unitree's README were not vendored.
The upstream BSD 3-Clause license is included at
`src/differentiable_humanoid_dynamics/assets/robots/unitree_g1/LICENSE.unitree_ros`.

Files ending in `.adam.urdf` are generated compatibility copies of the upstream
URDFs. They add identity joint origins where the upstream URDF omits them and
remove Unitree/MuJoCo-specific XML that `urdf_parser_py` warns about. The
original upstream URDFs remain vendored unchanged and are used for source
inspection and visualization.

The visualization uses [Viser](https://viser.studio/) when the optional
`viz` extra is installed.
