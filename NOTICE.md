# Notices

## Unitree G1 Robot Assets

The Unitree G1 URDF, MJCF, mesh, and image assets vendored in
`src/differentiable_humanoid_dynamics/assets/robots/unitree_g1` were downloaded
from:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description

Only the non-deprecated G1 model files listed as up-to-date in the upstream
README are included. The upstream BSD 3-Clause license is preserved as:

`src/differentiable_humanoid_dynamics/assets/robots/unitree_g1/LICENSE.unitree_ros`

The `.adam.urdf` files in the same directory are generated compatibility copies
used by Adam. They are derived from the upstream Unitree URDFs by adding
identity origins to joints that omit `<origin>` and removing parser-irrelevant
Unitree/MuJoCo XML extensions.

## Adam

This package relies on Adam for rigid-body dynamics and kinematics:

https://github.com/gbionics/adam

The Python dependency is declared as `adam-robotics[pytorch]`.
