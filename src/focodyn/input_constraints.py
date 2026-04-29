from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


class AffineConstraintTerms(NamedTuple):
    """Affine inequality terms for input constraints.

    Attributes:
        matrix: Constraint matrix ``A`` with shape ``(num_constraints, input_dim)``.
        upper_bound: Constraint upper bound ``b`` with shape ``(num_constraints,)``.
    """

    matrix: torch.Tensor
    upper_bound: torch.Tensor


class AffineInputConstraint(torch.nn.Module):
    r"""Base class for affine input constraints.

    Constraints are represented as ``A u <= b``. The module call returns the
    residual ``A u - b`` so feasible inputs have non-positive residuals.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the common input dimension.

        Args:
            input_dim: Dimension of the control input vector ``u``.

        Returns:
            None.

        Raises:
            ValueError: If ``input_dim`` is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)

    @property
    def num_constraints(self) -> int:
        """Return the number of scalar inequalities.

        Args:
            None.

        Returns:
            Number of scalar rows in ``A u <= b``.
        """
        return int(self.affine_terms().upper_bound.shape[-1])

    def affine_terms(self) -> AffineConstraintTerms:
        """Return the affine inequality terms ``A`` and ``b``.

        Args:
            None.

        Returns:
            :class:`AffineConstraintTerms` for ``A u <= b``.

        Raises:
            NotImplementedError: Always for the abstract base class.
        """
        raise NotImplementedError

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate the affine inequality residual ``A u - b``.

        Args:
            u: Input tensor with shape ``(input_dim,)`` or
                ``(..., input_dim)``.

        Returns:
            Residual tensor with shape ``(num_constraints,)`` or
            ``(..., num_constraints)``. Feasible inputs have residuals less
            than or equal to zero.

        Raises:
            ValueError: If the last dimension of ``u`` does not match
                ``input_dim``.
        """
        terms = self.affine_terms()
        inputs = u.to(dtype=terms.matrix.dtype, device=terms.matrix.device)
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}.")
        return torch.matmul(inputs, terms.matrix.transpose(-1, -2)) - terms.upper_bound

    def violation(self, u: torch.Tensor) -> torch.Tensor:
        """Return non-negative inequality violations.

        Args:
            u: Input tensor with shape ``(input_dim,)`` or
                ``(..., input_dim)``.

        Returns:
            Tensor with the same shape as :meth:`forward`, where satisfied
            inequalities are zero and violated inequalities are positive.
        """
        return torch.clamp(self(u), min=0.0)

    def is_satisfied(self, u: torch.Tensor, *, atol: float = 0.0) -> torch.Tensor:
        """Return whether every inequality is satisfied.

        Args:
            u: Input tensor with shape ``(input_dim,)`` or
                ``(..., input_dim)``.
            atol: Absolute tolerance on the residual. A residual of ``atol``
                or less is treated as feasible.

        Returns:
            Boolean tensor with shape ``()`` for unbatched inputs or the
            leading batch shape of ``u`` for batched inputs.
        """
        return torch.all(self(u) <= atol, dim=-1)

    def compose(self, *constraints: "AffineInputConstraint") -> "InputConstraintSet":
        """Stack this constraint with additional constraints.

        Args:
            constraints: Additional constraints with the same ``input_dim``.

        Returns:
            :class:`InputConstraintSet` containing this constraint followed by
            ``constraints``.
        """
        return InputConstraintSet(self, *constraints)


class StaticAffineInputConstraint(AffineInputConstraint):
    """Input constraint backed by constant affine terms."""

    def __init__(self, matrix: torch.Tensor, upper_bound: torch.Tensor) -> None:
        """Initialize constant affine inequality terms.

        Args:
            matrix: Constraint matrix ``A`` with shape
                ``(num_constraints, input_dim)``.
            upper_bound: Constraint upper bound ``b`` with shape
                ``(num_constraints,)``.

        Returns:
            None.

        Raises:
            ValueError: If the matrix or bound shapes are invalid.
        """
        if matrix.ndim != 2:
            raise ValueError("matrix must have shape (num_constraints, input_dim).")
        if upper_bound.ndim != 1:
            raise ValueError("upper_bound must have shape (num_constraints,).")
        if matrix.shape[0] != upper_bound.shape[0]:
            raise ValueError("matrix and upper_bound must have the same number of constraints.")
        super().__init__(int(matrix.shape[1]))
        self.register_buffer("matrix", matrix, persistent=True)
        self.register_buffer("upper_bound", upper_bound, persistent=True)

    def affine_terms(self) -> AffineConstraintTerms:
        """Return the stored affine inequality terms.

        Args:
            None.

        Returns:
            :class:`AffineConstraintTerms` for ``A u <= b``.
        """
        return AffineConstraintTerms(self.matrix, self.upper_bound)


class InputConstraintSet(AffineInputConstraint):
    """Composition of multiple affine input constraints."""

    def __init__(self, *constraints: AffineInputConstraint) -> None:
        """Initialize a stacked constraint set.

        Args:
            constraints: One or more constraints with the same ``input_dim``.

        Returns:
            None.

        Raises:
            ValueError: If no constraints are provided or their input
                dimensions differ.
        """
        if not constraints:
            raise ValueError("At least one constraint is required.")
        input_dim = constraints[0].input_dim
        for constraint in constraints:
            if constraint.input_dim != input_dim:
                raise ValueError("All constraints must have the same input_dim.")
        super().__init__(input_dim)
        self.constraints = torch.nn.ModuleList(constraints)

    @property
    def num_constraints(self) -> int:
        """Return the total number of stacked scalar inequalities.

        Args:
            None.

        Returns:
            Sum of ``num_constraints`` across all child constraints.
        """
        return sum(constraint.num_constraints for constraint in self.constraints)

    def affine_terms(self) -> AffineConstraintTerms:
        """Return concatenated affine inequality terms.

        Args:
            None.

        Returns:
            :class:`AffineConstraintTerms` with matrices and bounds stacked in
            the same order as ``self.constraints``.
        """
        terms = [constraint.affine_terms() for constraint in self.constraints]
        matrix = torch.cat([term.matrix for term in terms], dim=0)
        upper_bound = torch.cat([term.upper_bound for term in terms], dim=0)
        return AffineConstraintTerms(matrix, upper_bound)


class JointTorqueLimits(StaticAffineInputConstraint):
    """Box constraints on the joint-torque block of ``u``."""

    def __init__(
        self,
        lower: torch.Tensor | float | Sequence[float],
        upper: torch.Tensor | float | Sequence[float],
        *,
        n_joints: int | None = None,
        input_dim: int | None = None,
        torque_start: int = 0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        r"""Initialize joint torque limits.

        The generated inequalities are ``tau <= upper`` and
        ``-tau <= -lower`` for the selected torque block.

        Args:
            lower: Lower torque limits. A scalar is expanded to ``n_joints``.
            upper: Upper torque limits. A scalar is expanded to ``n_joints``.
            n_joints: Number of actuated joints. Required when both ``lower``
                and ``upper`` are scalars.
            input_dim: Full input dimension. Defaults to
                ``torque_start + n_joints``.
            torque_start: Start index of the joint-torque block in ``u``.
            dtype: Torch dtype used for the affine terms.
            device: Torch device used for the affine terms. ``None`` selects
                CPU.

        Returns:
            None.

        Raises:
            ValueError: If limits are inconsistent or the torque block does
                not fit in ``input_dim``.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        n_joints = _infer_vector_size(lower, upper, n_values=n_joints, name="joint limits")
        lower_tensor = _as_vector(
            lower,
            n_values=n_joints,
            name="lower",
            dtype=dtype,
            device=device,
        )
        upper_tensor = _as_vector(
            upper,
            n_values=n_joints,
            name="upper",
            dtype=dtype,
            device=device,
        )
        if torch.any(lower_tensor > upper_tensor):
            raise ValueError("lower torque limits must be less than or equal to upper limits.")
        input_dim = _resolve_input_dim(
            requested=input_dim,
            required_end=torque_start + n_joints,
        )
        _validate_block(name="torque", input_dim=input_dim, start=torque_start, width=n_joints)

        matrix = torch.zeros(2 * n_joints, input_dim, dtype=dtype, device=device)
        columns = torch.arange(torque_start, torque_start + n_joints, device=device)
        rows = torch.arange(n_joints, device=device)
        matrix[rows, columns] = 1.0
        matrix[rows + n_joints, columns] = -1.0
        upper_bound = torch.cat((upper_tensor, -lower_tensor), dim=0)

        super().__init__(matrix, upper_bound)
        self.n_joints = int(n_joints)
        self.torque_start = int(torque_start)


class PositiveNormalContactForces(StaticAffineInputConstraint):
    """Non-negative normal-force constraints for local contact-frame forces."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_contacts: int,
        contact_force_start: int,
        minimum_normal_force: torch.Tensor | float | Sequence[float] = 0.0,
        normal_axis: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        r"""Initialize positive normal contact-force constraints.

        The input is assumed to contain stacked local contact-frame force
        vectors ``(f_t0, f_t1, f_n)`` by default. The generated inequalities
        are ``-f_n <= -minimum_normal_force`` for each contact.

        Args:
            input_dim: Full input dimension.
            num_contacts: Number of local contact-force vectors in ``u``.
            contact_force_start: Start index of the stacked contact-force
                block in ``u``.
            minimum_normal_force: Minimum normal force per contact. A scalar
                is expanded to ``num_contacts``.
            normal_axis: Normal-force component within each 3D local force
                vector.
            dtype: Torch dtype used for the affine terms.
            device: Torch device used for the affine terms. ``None`` selects
                CPU.

        Returns:
            None.

        Raises:
            ValueError: If the contact-force block is invalid.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        _validate_contact_force_block(
            input_dim=input_dim,
            num_contacts=num_contacts,
            contact_force_start=contact_force_start,
            normal_axis=normal_axis,
        )
        minimum_normal_force = _as_vector(
            minimum_normal_force,
            n_values=num_contacts,
            name="minimum_normal_force",
            dtype=dtype,
            device=device,
        )
        matrix = torch.zeros(num_contacts, input_dim, dtype=dtype, device=device)
        columns = contact_force_start + 3 * torch.arange(num_contacts, device=device) + normal_axis
        rows = torch.arange(num_contacts, device=device)
        matrix[rows, columns] = -1.0
        upper_bound = -minimum_normal_force

        super().__init__(matrix, upper_bound)
        self.num_contacts = int(num_contacts)
        self.contact_force_start = int(contact_force_start)
        self.normal_axis = int(normal_axis)


class LinearizedFrictionCone(StaticAffineInputConstraint):
    """Regular polyhedral friction-cone constraints for local contact forces."""

    def __init__(
        self,
        friction_coefficient: torch.Tensor | float | Sequence[float],
        *,
        input_dim: int,
        num_contacts: int,
        contact_force_start: int,
        num_facets: int = 4,
        facet_phase: float = 0.0,
        conservative: bool = True,
        normal_axis: int = 2,
        tangent_axes: tuple[int, int] = (0, 1),
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        r"""Initialize linearized friction-cone constraints.

        The input is assumed to contain stacked local contact-frame force
        vectors. Each contact gets ``num_facets`` inequalities
        ``d_i.dot(f_tangent) <= mu * scale * f_n`` with evenly spaced unit
        directions ``d_i`` in the tangent plane. ``scale`` is
        ``cos(pi / num_facets)`` for the default conservative inner
        approximation and ``1`` otherwise.

        Args:
            friction_coefficient: Friction coefficient ``mu``. A scalar is
                expanded to ``num_contacts``.
            input_dim: Full input dimension.
            num_contacts: Number of local contact-force vectors in ``u``.
            contact_force_start: Start index of the stacked contact-force
                block in ``u``.
            num_facets: Number of linear cone facets per contact.
            facet_phase: Angular offset in radians for the regular polygon's
                tangent-plane directions.
            conservative: Whether to scale the normal-force bound so the
                polyhedral cone is contained inside the circular Coulomb cone.
            normal_axis: Normal-force component within each 3D local force
                vector.
            tangent_axes: Tangential-force components within each 3D local
                force vector.
            dtype: Torch dtype used for the affine terms.
            device: Torch device used for the affine terms. ``None`` selects
                CPU.

        Returns:
            None.

        Raises:
            ValueError: If the contact-force block, friction coefficients, or
                facet count are invalid.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        _validate_contact_force_block(
            input_dim=input_dim,
            num_contacts=num_contacts,
            contact_force_start=contact_force_start,
            normal_axis=normal_axis,
            tangent_axes=tangent_axes,
        )
        friction = _as_vector(
            friction_coefficient,
            n_values=num_contacts,
            name="friction_coefficient",
            dtype=dtype,
            device=device,
        )
        if torch.any(friction < 0):
            raise ValueError("friction_coefficient must be non-negative.")
        directions = _regular_tangent_directions(
            num_facets,
            facet_phase=facet_phase,
            dtype=dtype,
            device=device,
        )
        friction_scale = _friction_bound_scale(
            num_facets,
            conservative=conservative,
            dtype=dtype,
            device=device,
        )

        num_facets = directions.shape[0]
        matrix = torch.zeros(num_contacts * num_facets, input_dim, dtype=dtype, device=device)
        row = 0
        for contact_index in range(num_contacts):
            base = contact_force_start + 3 * contact_index
            for direction in directions:
                matrix[row, base + tangent_axes[0]] = direction[0]
                matrix[row, base + tangent_axes[1]] = direction[1]
                matrix[row, base + normal_axis] = -friction[contact_index] * friction_scale
                row += 1
        upper_bound = torch.zeros(num_contacts * num_facets, dtype=dtype, device=device)

        super().__init__(matrix, upper_bound)
        self.num_contacts = int(num_contacts)
        self.contact_force_start = int(contact_force_start)
        self.normal_axis = int(normal_axis)
        self.tangent_axes = tuple(int(axis) for axis in tangent_axes)
        self.num_facets = int(num_facets)
        self.facet_phase = float(facet_phase)
        self.conservative = bool(conservative)


def _infer_vector_size(
    *values: torch.Tensor | float | Sequence[float],
    n_values: int | None,
    name: str,
) -> int:
    """Infer a shared vector length from scalars or one-dimensional values."""
    if n_values is not None:
        if n_values <= 0:
            raise ValueError(f"{name} length must be positive.")
        return int(n_values)

    lengths = []
    for value in values:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1:
            lengths.append(int(tensor.numel()))
        elif tensor.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional.")

    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
        raise ValueError(f"{name} vectors must have matching lengths.")
    if unique_lengths:
        inferred = unique_lengths.pop()
        if inferred <= 0:
            raise ValueError(f"{name} length must be positive.")
        return inferred
    raise ValueError(f"n_values is required when all {name} are scalars.")


def _as_vector(
    value: torch.Tensor | float | Sequence[float],
    *,
    n_values: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert a scalar or vector value to a one-dimensional tensor."""
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        return tensor.expand(n_values).clone()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    if tensor.numel() != n_values:
        raise ValueError(f"{name} must have length {n_values}, got {tensor.numel()}.")
    return tensor


def _resolve_input_dim(*, requested: int | None, required_end: int) -> int:
    """Resolve and validate the full input dimension."""
    if requested is None:
        return int(required_end)
    if requested < required_end:
        raise ValueError(f"input_dim must be at least {required_end}, got {requested}.")
    return int(requested)


def _validate_block(*, name: str, input_dim: int, start: int, width: int) -> None:
    """Validate a contiguous block inside the input vector."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if start < 0:
        raise ValueError(f"{name} block start must be non-negative.")
    if width <= 0:
        raise ValueError(f"{name} block width must be positive.")
    if start + width > input_dim:
        raise ValueError(f"{name} block exceeds input_dim.")


def _validate_contact_force_block(
    *,
    input_dim: int,
    num_contacts: int,
    contact_force_start: int,
    normal_axis: int,
    tangent_axes: tuple[int, int] | None = None,
) -> None:
    """Validate a stacked 3D contact-force block."""
    _validate_block(
        name="contact-force",
        input_dim=input_dim,
        start=contact_force_start,
        width=3 * num_contacts,
    )
    if normal_axis not in {0, 1, 2}:
        raise ValueError("normal_axis must be 0, 1, or 2.")
    if tangent_axes is None:
        return
    if len(tangent_axes) != 2:
        raise ValueError("tangent_axes must contain two axes.")
    axes = {int(axis) for axis in (*tangent_axes, normal_axis)}
    if axes != {0, 1, 2}:
        raise ValueError("tangent_axes and normal_axis must be a permutation of 0, 1, 2.")


def _regular_tangent_directions(
    num_facets: int,
    *,
    facet_phase: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return evenly spaced tangent direction rows for a friction polygon."""
    if num_facets < 3:
        raise ValueError("num_facets must be at least 3.")
    phase = torch.as_tensor(facet_phase, dtype=dtype, device=device)
    angles = phase + (2.0 * torch.pi / num_facets) * torch.arange(
        num_facets,
        dtype=dtype,
        device=device,
    )
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def _friction_bound_scale(
    num_facets: int,
    *,
    conservative: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the normal-force scale for the friction polygon bound."""
    if not conservative:
        return torch.ones((), dtype=dtype, device=device)
    return torch.cos(torch.as_tensor(torch.pi / num_facets, dtype=dtype, device=device))
