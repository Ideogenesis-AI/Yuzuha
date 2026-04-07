# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Yuzuha library.
#
# Yuzuha is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Yuzuha is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Yuzuha. If not, see <https://www.gnu.org/licenses/>.

"""Type stubs for the yuzuha Rust extension module (yuzuha.abi3.so).

These stubs allow static analysis tools (mypy, pyright) and mkdocstrings
to introspect the PyO3-defined classes and functions without reading the
compiled extension.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Spin
# ---------------------------------------------------------------------------

class Spin:
    """SU(2) irreducible representation label, stored as doubled integer 2j.

    Parameters
    ----------
    j_doubled : int
        The doubled spin value ``2j``. Must be a non-negative integer.

    Raises
    ------
    ValueError
        If ``j_doubled`` is negative.

    Examples
    --------
    >>> import yuzuha
    >>> j_half = yuzuha.Spin(1)   # j = 1/2
    >>> j1     = yuzuha.Spin(2)   # j = 1
    >>> print(j_half.value())
    0.5
    >>> print(j1.dimension())
    3
    """

    def __init__(self, j_doubled: int) -> None: ...

    def value(self) -> float:
        """Return the spin as a float ``j = j_doubled / 2``.

        Returns
        -------
        float
            The spin value j.
        """
        ...

    def twice(self) -> int:
        """Return the internal doubled integer ``2j``.

        Returns
        -------
        int
            The doubled spin value.
        """
        ...

    def dimension(self) -> int:
        """Return the representation dimension ``2j + 1``.

        Returns
        -------
        int
            The number of magnetic quantum number states.
        """
        ...

    def is_integer(self) -> bool:
        """Return ``True`` if ``j`` is a non-negative integer.

        Returns
        -------
        bool
            True for integer spin, False for half-integer spin.
        """
        ...

    def is_half_integer(self) -> bool:
        """Return ``True`` if ``j`` is a strict half-integer.

        Returns
        -------
        bool
            True for half-integer spin, False for integer spin.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: Spin) -> bool: ...


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

class Direction:
    """Arrow direction for a tensor network edge: incoming (+1) or outgoing (-1).

    Use the static constructors ``incoming()`` and ``outgoing()`` rather than
    instantiating directly.

    Examples
    --------
    >>> import yuzuha
    >>> d = yuzuha.Direction.incoming()
    >>> print(d.sign())
    1
    >>> print(d.flip())
    Direction.outgoing
    """

    @staticmethod
    def incoming() -> Direction:
        """Create an incoming Direction (sign = +1).

        Returns
        -------
        Direction
            An incoming direction.
        """
        ...

    @staticmethod
    def outgoing() -> Direction:
        """Create an outgoing Direction (sign = -1).

        Returns
        -------
        Direction
            An outgoing direction.
        """
        ...

    @staticmethod
    def from_sign(sign: int) -> Direction:
        """Create a Direction from its sign value.

        Parameters
        ----------
        sign : int
            ``+1`` for incoming, ``-1`` for outgoing.

        Returns
        -------
        Direction
            The corresponding Direction.

        Raises
        ------
        ValueError
            If ``sign`` is not ``+1`` or ``-1``.
        """
        ...

    def sign(self) -> int:
        """Return the numerical sign of this direction.

        Returns
        -------
        int
            ``+1`` for incoming, ``-1`` for outgoing.
        """
        ...

    def flip(self) -> Direction:
        """Return the opposite Direction.

        Returns
        -------
        Direction
            The flipped direction.
        """
        ...

    def is_incoming(self) -> bool:
        """Return ``True`` if this direction is incoming.

        Returns
        -------
        bool
            True if incoming.
        """
        ...

    def is_outgoing(self) -> bool:
        """Return ``True`` if this direction is outgoing.

        Returns
        -------
        bool
            True if outgoing.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class Edge:
    """One external edge of a CG tensor: a spin paired with a direction.

    Use the static constructors ``incoming()`` and ``outgoing()`` rather than
    instantiating directly.

    Examples
    --------
    >>> import yuzuha
    >>> j = yuzuha.Spin(1)
    >>> e_in  = yuzuha.Edge.incoming(j)
    >>> e_out = yuzuha.Edge.outgoing(j)
    """

    @staticmethod
    def incoming(spin: Spin) -> Edge:
        """Create an incoming edge with the given spin.

        Parameters
        ----------
        spin : Spin
            The spin quantum number for this edge.

        Returns
        -------
        Edge
            An incoming edge.
        """
        ...

    @staticmethod
    def outgoing(spin: Spin) -> Edge:
        """Create an outgoing edge with the given spin.

        Parameters
        ----------
        spin : Spin
            The spin quantum number for this edge.

        Returns
        -------
        Edge
            An outgoing edge.
        """
        ...

    @property
    def j(self) -> Spin:
        """The spin quantum number of this edge.

        Returns
        -------
        Spin
            The spin label.
        """
        ...

    @property
    def dir(self) -> Direction:
        """The arrow direction of this edge.

        Returns
        -------
        Direction
            Incoming or outgoing.
        """
        ...

    def is_incoming(self) -> bool:
        """Return ``True`` if this edge is incoming.

        Returns
        -------
        bool
            True if incoming, False if outgoing.
        """
        ...

    def is_outgoing(self) -> bool:
        """Return ``True`` if this edge is outgoing.

        Returns
        -------
        bool
            True if outgoing, False if incoming.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# CGSpec
# ---------------------------------------------------------------------------

class CGSpec:
    """Full specification of a CG tensor: external edges and the internal
    left-associative fusion tree.

    Use ``from_edges`` to construct; do not call the constructor directly.

    Examples
    --------
    >>> import yuzuha
    >>> jhalf = yuzuha.Spin(1)
    >>> spec  = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.outgoing(yuzuha.Spin(2)),
    ... ])
    >>> print(spec.num_external())
    3
    >>> print(spec.om_dimension())
    1
    """

    @staticmethod
    def from_edges(edges: list[Edge]) -> CGSpec:
        """Construct a CGSpec from a list of edges.

        Automatically enumerates all valid internal spin configurations
        (alpha values) using the triangle inequality.

        Parameters
        ----------
        edges : list[Edge]
            External edges of the tensor, at least 2.

        Returns
        -------
        CGSpec
            A new spec with all valid OM configurations.

        Raises
        ------
        ValueError
            If the edge list has fewer than 2 elements or if no valid
            internal coupling exists.
        """
        ...

    @property
    def edges(self) -> list[Edge]:
        """The external edges of this CGSpec.

        Returns
        -------
        list[Edge]
            One ``Edge`` per external edge, in order.
        """
        ...

    def num_external(self) -> int:
        """Return the number of external edges.

        Returns
        -------
        int
            Number of external edges.
        """
        ...

    def om_dimension(self) -> int:
        """Return the outer-multiplicity (OM) dimension.

        The OM dimension is the number of valid internal spin paths through
        the left-associative fusion tree.

        Returns
        -------
        int
            Number of valid OM configurations.
        """
        ...

    def get_spins(self) -> list[int]:
        """Return the doubled spin values (2j) for all edges.

        Returns
        -------
        list[int]
            Doubled spin value for each external edge.
        """
        ...

    def get_directions(self) -> list[int]:
        """Return the direction signs for all edges.

        Returns
        -------
        list[int]
            ``+1`` for each incoming edge, ``-1`` for each outgoing edge.
        """
        ...

    def with_inverted_axes(self, axes: list[int]) -> CGSpec:
        """Return a new CGSpec with the edge directions at the given axes flipped.

        The internal OM configurations (alphas) are reused unchanged because
        they depend only on spin values, not directions.

        Parameters
        ----------
        axes : list[int]
            Indices of edges whose direction should be flipped
            (``Incoming`` ↔ ``Outgoing``).

        Returns
        -------
        CGSpec
            A new CGSpec with the specified edge directions inverted.

        Raises
        ------
        ValueError
            If any axis index is out of bounds.

        Examples
        --------
        >>> import yuzuha
        >>> j = yuzuha.Spin(2)
        >>> spec = yuzuha.CGSpec.from_edges([
        ...     yuzuha.Edge.incoming(j),
        ...     yuzuha.Edge.incoming(j),
        ...     yuzuha.Edge.outgoing(j),
        ... ])
        >>> flipped = spec.with_inverted_axes([0, 1])
        >>> [e.dir for e in flipped.edges]
        [Direction.outgoing, Direction.outgoing, Direction.outgoing]
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# Contraction
# ---------------------------------------------------------------------------

class Contraction:
    """Edge-pair specification for contracting two CG tensors.

    Parameters
    ----------
    axes_a : list[int]
        Edge indices from the first CGSpec to contract.
    axes_b : list[int]
        Edge indices from the second CGSpec to contract.

    Examples
    --------
    >>> import yuzuha
    >>> # Contract edge 2 of spec_a with edge 0 of spec_b
    >>> c = yuzuha.Contraction([2], [0])
    >>> # Contract two pairs simultaneously
    >>> c2 = yuzuha.Contraction([2, 3], [0, 1])
    """

    def __init__(self, axes_a: list[int], axes_b: list[int]) -> None: ...

    @property
    def axes_a(self) -> list[int]:
        """Edge indices to contract from the first CGSpec.

        Returns
        -------
        list[int]
            Axis indices into spec A.
        """
        ...

    @property
    def axes_b(self) -> list[int]:
        """Edge indices to contract from the second CGSpec.

        Returns
        -------
        list[int]
            Axis indices into spec B.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def compute_xsymbol(
    spec_a: CGSpec,
    spec_b: CGSpec,
    contraction: Contraction,
) -> tuple[NDArray[np.float64], CGSpec]:
    """Compute the X-symbol for contracting two CG tensors (Rust implementation).

    This is the low-level Rust function. In most cases you should call
    ``yuzuha.compute_xsymbol`` from the top-level package, which adds
    SQLite caching on top of this function.

    Parameters
    ----------
    spec_a : CGSpec
        First CG specification.
    spec_b : CGSpec
        Second CG specification.
    contraction : Contraction
        Specification of which edges to contract.

    Returns
    -------
    tuple[numpy.ndarray, CGSpec]
        A tuple ``(x_array, spec_c)`` where:

        - ``x_array`` has shape ``(om_a, om_b, om_c)`` and dtype ``float64``
        - ``spec_c`` describes the uncontracted output tensor

    Raises
    ------
    ValueError
        If the contraction is invalid (incompatible spins or directions).
    RuntimeError
        If computation or cache access fails.
    """
    ...


def compute_rsymbol(
    spec: CGSpec,
    permutation: list[int],
) -> tuple[NDArray[np.float64], CGSpec]:
    """Compute the R-symbol for permuting the edges of a CG tensor (Rust implementation).

    This is the low-level Rust function. In most cases you should call
    ``yuzuha.compute_rsymbol`` from the top-level package, which adds
    SQLite caching on top of this function.

    Parameters
    ----------
    spec : CGSpec
        The CGSpec whose edges are to be permuted.
    permutation : list[int]
        Permutation of external edge indices. ``permutation[i]`` is the
        index in the original spec that moves to position ``i`` in the
        permuted spec.

    Returns
    -------
    tuple[numpy.ndarray, CGSpec]
        A tuple ``(r_array, spec_permuted)`` where:

        - ``r_array`` has shape ``(om_original, om_permuted)`` and dtype ``float64``
        - ``spec_permuted`` is the permuted CGSpec

    Raises
    ------
    ValueError
        If the permutation is invalid (wrong length, out-of-range, or duplicate indices).
    RuntimeError
        If computation or cache access fails.
    """
    ...


def canonical_basis(spec: CGSpec) -> NDArray[np.float64]:
    """Compute the canonical CG basis tensor for a CGSpec.

    Builds the left-associative fusion-tree basis for the given spec. The
    result is cached in an SQLite database for reuse across calls and sessions.

    Parameters
    ----------
    spec : CGSpec
        The CGSpec for which to compute the canonical basis.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(d_0, d_1, ..., d_{n-1}, om_dim)`` and dtype
        ``float64``, where ``d_i = 2j_i + 1`` is the dimension of the i-th
        edge and ``om_dim = spec.om_dimension()``.

    Raises
    ------
    ValueError
        If the CGSpec has fewer than 2 external edges.
    RuntimeError
        If computation or cache access fails.

    Examples
    --------
    >>> import yuzuha
    >>> jhalf = yuzuha.Spin(1)
    >>> spec  = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.incoming(jhalf),
    ... ])
    >>> basis = yuzuha.canonical_basis(spec)
    >>> print(basis.shape)
    (2, 2, 2, 2)
    """
    ...


def fs_phase_for_spin(spin: Spin) -> float:
    """Return the Frobenius-Schur indicator ``(-1)^{2j}`` for a spin.

    Parameters
    ----------
    spin : Spin
        The spin quantum number.

    Returns
    -------
    float
        ``+1.0`` for integer spins, ``-1.0`` for half-integer spins.

    Examples
    --------
    >>> import yuzuha
    >>> yuzuha.fs_phase_for_spin(yuzuha.Spin(1))   # j = 1/2, half-integer
    -1.0
    >>> yuzuha.fs_phase_for_spin(yuzuha.Spin(2))   # j = 1, integer
    1.0
    """
    ...


def compute_conjugate(spec: CGSpec) -> tuple[float, CGSpec]:
    """Compute the conjugated CGSpec and the cumulated Frobenius-Schur phase.

    The conjugated spec has all edge directions reversed. The cumulated FS
    phase accounts for edges that differ from the canonical conjugate pattern
    (first ``n-1`` edges outgoing, last edge incoming): each differing edge
    contributes a factor of ``(-1)^{2j}``.

    Parameters
    ----------
    spec : CGSpec
        The CGSpec to conjugate.

    Returns
    -------
    tuple[float, CGSpec]
        A tuple ``(phase, conj_spec)`` where ``phase`` is ``+1.0`` or
        ``-1.0`` and ``conj_spec`` has the same spins as ``spec`` with all
        directions flipped.

    Raises
    ------
    ValueError
        If the conjugated edge configuration is invalid.

    Examples
    --------
    >>> import yuzuha
    >>> jhalf = yuzuha.Spin(1)
    >>> spec = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.incoming(jhalf),
    ...     yuzuha.Edge.outgoing(yuzuha.Spin(2)),
    ... ])
    >>> phase, conj = yuzuha.compute_conjugate(spec)
    >>> phase
    1.0
    """
    ...
