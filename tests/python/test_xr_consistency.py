# Copyright (C) 2026 Changkai Zhang.
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

"""
Consistency tests between X-symbols and R-symbols.

The key algebraic identity
--------------------------
When one of the two CG tensors in an X-symbol contraction is a "2-edge
identity wire" — a CGSpec with OM dimension 1 that acts as a delta/pass-
through — the contraction is equivalent to permuting the edges of the other
tensor.  The X-symbol must therefore reduce to the R-symbol for that
permutation.

Mathematical statement (wire on the right)
------------------------------------------
Let spec_A be an N-edge CG specification (N >= 3 to have non-trivial OM).
Let spec_B = [in(j), out(j)] — the canonical 2-edge identity wire, which
has OM dimension 1.
Contract A[k] = out(j) with B[0] = in(j).

The output spec_C has edges
    [A[0], ..., A[k-1], A[k+1], ..., A[N-1], B[1] = out(j)]
which is spec_A with edge k moved to position N-1.  This is a permutation

    sigma = [0, ..., k-1, k+1, ..., N-1, k]

applied to spec_A's edge list.

Since spec_B has OM dimension 1, the X-symbol has shape [dim_A, 1, dim_C].
The singleton beta-axis is squeezed out, giving the 2-D slice X[:, 0, :].

The canonical 2-edge basis tensor for [in(j), out(j)] is the delta tensor
normalized to unit Frobenius norm: delta_{m,m'} / sqrt(2j+1).  Because this
factor of 1/sqrt(2j+1) is baked into the X-symbol computation, the identity
is:

    X(spec_A, spec_B, Contraction([k], [0]))[:, 0, :] * sqrt(2j+1)
        == R(spec_A, sigma)

The same identity holds when the wire is transposed to [out(j), in(j)] and
A[k] = in(j) is contracted with B[0] = out(j).

Frobenius–Schur (FS) phase
--------------------------
Both direction setups described above yield FS phase = +1, so the X-symbol
slice equals the R-symbol up to the sqrt(2j+1) wire normalization factor,
with no additional sign corrections.

Wire on the left (spec_A is the wire)
--------------------------------------
Symmetrically, if spec_A = [in(j), out(j)] and spec_B is the large N-edge
tensor, contracting A[1] = out(j) with B[k] = in(j) inserts A[0] = in(j)
at position 0 of the output.  The X-symbol slice X[0, :, :] equals the
R-symbol for the permutation that moves B[k] to position 0 of spec_B,
again up to the same sqrt(2j+1) factor.

Test organisation
-----------------
- TestXRWireOnRight          : wire = spec_B; edge k of spec_A moved to last
- TestXRWireOnLeft           : wire = spec_A; B's edge k moved to position 0
- TestXRLargerSpins          : j = 1, 3/2, 2 with 4-edge tensors
- TestXRNontrivialOM         : 5-edge and 6-edge tensors with OM dim > 2
- TestXRIdentityPermutation  : special case k = N-1; sigma = identity => X = I
- TestXRStress               : randomised stress tests
"""

import numpy as np
import yuzuha


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def run_xr_wire_right_test(spec_a, k, j, tol=1e-10):
    """Assert X(spec_A, wire_B, Contraction([k], [0]))[:, 0, :] * sqrt(2j+1) == R(spec_A, sigma).

    The wire spec_B = [in(j), out(j)] has OM dim 1.  Its canonical basis
    tensor is the delta delta_{m,m'} / sqrt(2j+1), so the X-symbol slice
    carries a factor of 1/sqrt(2j+1) relative to the R-symbol.

    Contracting spec_A's edge k (out(j)) with B[0] = in(j) moves edge k to
    the last position of the output, giving permutation sigma.

    FS phase is +1 for all k in this setup.

    Parameters
    ----------
    spec_a : CGSpec
        N-edge spec whose edge k is out(j).
    k : int
        Index of the contracted edge in spec_A (must be an outgoing edge of
        spin j).
    j : Spin
        Spin of the contracted edge.
    tol : float
        Numerical tolerance for the comparison.
    """
    n = spec_a.num_external()
    dim_a = spec_a.om_dimension()
    assert dim_a > 0, f"spec_A OM dim = 0 (invalid spec)"

    # 2-edge identity wire: [in(j), out(j)], OM dim = 1
    spec_b = yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j),
        yuzuha.Edge.outgoing(j),
    ])
    assert spec_b.om_dimension() == 1, "Wire must have OM dim 1"

    contraction = yuzuha.Contraction([k], [0])

    # X-symbol: shape [dim_A, 1, dim_C]
    x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
    assert x_array.shape[1] == 1, f"Expected beta-dim = 1, got {x_array.shape[1]}"

    dim_c = spec_c.om_dimension()
    assert dim_c == dim_a, (
        f"Output OM dim {dim_c} != input OM dim {dim_a}; "
        "the wire must preserve the OM space"
    )

    # Squeeze out the singleton beta-axis and rescale by sqrt(2j+1)
    wire_dim = j.dimension()   # 2j+1
    x_slice = x_array[:, 0, :] * np.sqrt(wire_dim)   # shape [dim_A, dim_C]

    # Permutation: move edge k to the last position
    sigma = list(range(n))
    sigma.pop(k)
    sigma.append(k)

    r_array, _ = yuzuha.compute_rsymbol(spec_a, sigma)

    assert x_slice.shape == r_array.shape, (
        f"Shape mismatch: X-slice {x_slice.shape} vs R {r_array.shape}"
    )
    max_diff = np.max(np.abs(x_slice - r_array))
    assert max_diff < tol, (
        f"X[:, 0, :] * sqrt(2j+1) does not match R(spec_A, {sigma}): "
        f"max_diff = {max_diff:.2e}, tol = {tol:.2e}"
    )


def run_xr_wire_right_transposed_test(spec_a, k, j, tol=1e-10):
    """Assert X(spec_A, wire_B, Contraction([k], [0]))[:, 0, :] * sqrt(2j+1) == R(spec_A, sigma).

    Same identity as run_xr_wire_right_test but the wire is transposed to
    [out(j), in(j)] and the contracted edge of spec_A is in(j) paired with
    B[0] = out(j).  The permutation sigma is identical: edge k of spec_A
    moves to the last position.  FS phase = +1.

    Parameters
    ----------
    spec_a : CGSpec
        N-edge spec whose edge k is in(j).
    k : int
        Index of the contracted edge in spec_A (must be an incoming edge of
        spin j).
    j : Spin
        Spin of the contracted edge.
    tol : float
        Numerical tolerance.
    """
    n = spec_a.num_external()
    dim_a = spec_a.om_dimension()
    assert dim_a > 0, "spec_A OM dim = 0 (invalid spec)"

    # Wire: [out(j), in(j)]; we contract A[k] = in(j) with B[0] = out(j)
    spec_b = yuzuha.CGSpec.from_edges([
        yuzuha.Edge.outgoing(j),
        yuzuha.Edge.incoming(j),
    ])
    assert spec_b.om_dimension() == 1

    contraction = yuzuha.Contraction([k], [0])

    x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
    assert x_array.shape[1] == 1

    wire_dim = j.dimension()   # 2j+1
    x_slice = x_array[:, 0, :] * np.sqrt(wire_dim)   # [dim_A, dim_C]

    sigma = list(range(n))
    sigma.pop(k)
    sigma.append(k)

    r_array, _ = yuzuha.compute_rsymbol(spec_a, sigma)

    max_diff = np.max(np.abs(x_slice - r_array))
    assert max_diff < tol, (
        f"X[:, 0, :] * sqrt(2j+1) (transposed wire) does not match R(spec_A, {sigma}): "
        f"max_diff = {max_diff:.2e}"
    )


def run_xr_wire_left_test(spec_b, k, j, tol=1e-10):
    """Assert X(wire_A, spec_B, Contraction([1], [k]))[0, :, :] * sqrt(2j+1) == R(spec_B, tau).

    The wire spec_A = [in(j), out(j)] has OM dim 1.  Contracting A[1] = out(j)
    with B[k] = in(j) inserts A[0] = in(j) at position 0 of the output, which
    is the permutation tau that moves B[k] to position 0 of spec_B.

    FS phase is +1 for all valid k (B[k] must be in(j)) in this setup.
    The wire normalization factor sqrt(2j+1) applies here as well.

    Parameters
    ----------
    spec_b : CGSpec
        N-edge spec whose edge k is in(j).
    k : int
        Index of the contracted edge in spec_B (must be incoming with spin j).
    j : Spin
        Spin of the contracted edge.
    tol : float
        Numerical tolerance.
    """
    n = spec_b.num_external()
    dim_b = spec_b.om_dimension()
    assert dim_b > 0, "spec_B OM dim = 0 (invalid spec)"

    spec_a = yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j),
        yuzuha.Edge.outgoing(j),
    ])
    assert spec_a.om_dimension() == 1

    contraction = yuzuha.Contraction([1], [k])

    # X-symbol: shape [1, dim_B, dim_C]
    x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
    assert x_array.shape[0] == 1, f"Expected alpha-dim = 1, got {x_array.shape[0]}"

    dim_c = spec_c.om_dimension()
    assert dim_c == dim_b

    # Squeeze out the singleton alpha-axis and rescale by sqrt(2j+1)
    wire_dim = j.dimension()   # 2j+1
    x_slice = x_array[0, :, :] * np.sqrt(wire_dim)   # [dim_B, dim_C]

    # Permutation: B[k] moves to position 0; the other edges shift right by 1
    # tau[0] = k; tau[1..k] = 0..k-1; tau[k+1..N-1] = k+1..N-1
    tau = [k] + list(range(0, k)) + list(range(k + 1, n))

    r_array, _ = yuzuha.compute_rsymbol(spec_b, tau)

    max_diff = np.max(np.abs(x_slice - r_array))
    assert max_diff < tol, (
        f"X[0, :, :] * sqrt(2j+1) (wire on left) does not match R(spec_B, {tau}): "
        f"max_diff = {max_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# TestXRWireOnRight: wire = spec_B, contract out-edge of spec_A with B[0]
# ---------------------------------------------------------------------------

class TestXRWireOnRight:
    """X-symbol with wire on the right equals R-symbol for the induced permutation.

    spec_A has N edges with one out(j) edge at position k.  spec_B is the
    2-edge canonical wire [in(j), out(j)] (OM dim 1).  Contracting A[k] with
    B[0] moves edge k to the last position of the output, giving sigma.

    The X-symbol slice X[:, 0, :] must equal R(spec_A, sigma).
    """

    # ------------------------------------------------------------------
    # j = 1/2, 4-edge spec_A: one out edge at each position k = 0..3
    # ------------------------------------------------------------------

    def test_j_half_k0(self):
        """j=1/2, 4-edge: out at position 0; sigma = [1, 2, 3, 0] (cyclic shift)."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j),   # k = 0
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j)

    def test_j_half_k1(self):
        """j=1/2, 4-edge: out at position 1; sigma = [0, 2, 3, 1]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),   # k = 1
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=1, j=j)

    def test_j_half_k2(self):
        """j=1/2, 4-edge: out at position 2; sigma = [0, 1, 3, 2] (transpose)."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),   # k = 2
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=2, j=j)

    def test_j_half_k3_canonical(self):
        """j=1/2, 4-edge canonical spec: out at position 3; sigma = identity."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),   # k = 3 (last)
        ])
        run_xr_wire_right_test(spec_a, k=3, j=j)

    # ------------------------------------------------------------------
    # j = 1/2, 3-edge spec_A (OM dim 1; sanity-checks scalar case)
    # ------------------------------------------------------------------

    def test_j_half_3edge_k0(self):
        """j=1/2, 3-edge (j1=j2=j3=1/2 is degenerate; use j1=1, j2=j3=1/2)."""
        # [in(1), in(1/2), out(1/2)] is NOT moving the j=1/2 out edge.
        # Use a valid 3-edge spec: [in(1), in(1/2), out(1/2)] with k=0 out(1)
        # -> requires edge 0 to be out(j). Use j=1 for the contracted edge.
        j = yuzuha.Spin(2)      # j = 1
        j_half = yuzuha.Spin(1) # j = 1/2
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j),    # k = 0, spin j=1
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j)

    def test_j_half_3edge_k2_canonical(self):
        """j=1, 3-edge canonical [in(1/2), in(1/2), out(1)]: k=2, sigma = identity."""
        j = yuzuha.Spin(2)      # j = 1
        j_half = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j),    # k = 2 (last)
        ])
        run_xr_wire_right_test(spec_a, k=2, j=j)

    # ------------------------------------------------------------------
    # Transposed wire: [out(j), in(j)], contract A[k] = in(j) with B[0]
    # ------------------------------------------------------------------

    def test_j_half_transposed_k0(self):
        """j=1/2, 4-edge canonical: transposed wire, in-edge k=0 with B[0]; sigma = [1,2,3,0]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),   # k = 0, in(j) -> contract with B[0]
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_transposed_test(spec_a, k=0, j=j)

    def test_j_half_transposed_k1(self):
        """j=1/2, 4-edge canonical: transposed wire, in-edge k=1 with B[0]; sigma = [0,2,3,1]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 1
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_transposed_test(spec_a, k=1, j=j)

    def test_j_half_transposed_k2(self):
        """j=1/2, 4-edge canonical: transposed wire, in-edge k=2 with B[0]; sigma = [0,1,3,2]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 2
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_transposed_test(spec_a, k=2, j=j)


# ---------------------------------------------------------------------------
# TestXRWireOnLeft: wire = spec_A, contract A[1] with B[k]
# ---------------------------------------------------------------------------

class TestXRWireOnLeft:
    """X-symbol with wire on the left equals R-symbol for the induced permutation.

    spec_A = [in(j), out(j)] (2-edge wire, OM dim 1).  spec_B is the N-edge
    tensor.  Contracting A[1] = out(j) with B[k] = in(j) prepends A[0] = in(j)
    to the output, moving B[k] to position 0 of the result, giving tau.

    The X-symbol slice X[0, :, :] must equal R(spec_B, tau).
    """

    # ------------------------------------------------------------------
    # j = 1/2, 4-edge canonical spec_B = [in,in,in,out]
    # ------------------------------------------------------------------

    def test_j_half_k0(self):
        """Wire ⊗ B, k=0: B's edge 0 stays at 0; tau = identity."""
        j = yuzuha.Spin(1)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),   # k = 0
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_left_test(spec_b, k=0, j=j)

    def test_j_half_k1(self):
        """Wire ⊗ B, k=1: B's edge 1 moves to position 0; tau = [1,0,2,3]."""
        j = yuzuha.Spin(1)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 1
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_left_test(spec_b, k=1, j=j)

    def test_j_half_k2(self):
        """Wire ⊗ B, k=2: B's edge 2 moves to position 0; tau = [2,0,1,3]."""
        j = yuzuha.Spin(1)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 2
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_left_test(spec_b, k=2, j=j)

    # ------------------------------------------------------------------
    # j = 1/2, non-canonical spec_B: verify the same identity holds
    # ------------------------------------------------------------------

    def test_j_half_noncanonicaL_b_k1(self):
        """Wire ⊗ B (non-canonical), k=1: tau = [1,0,2,3]."""
        j = yuzuha.Spin(1)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 1
            yuzuha.Edge.outgoing(j),   # out in middle
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_left_test(spec_b, k=1, j=j)

    def test_j_half_5edge_b_k0(self):
        """Wire ⊗ 5-edge B (mixed spins), k=0: tau = identity."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        # Valid 5-edge spec: [in(1/2), in(1/2), in(1/2), in(1/2), out(1)]
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),  # k = 0
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        run_xr_wire_left_test(spec_b, k=0, j=j_half)

    def test_j_half_5edge_b_k2(self):
        """Wire ⊗ 5-edge B, k=2: tau = [2,0,1,3,4]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),  # k = 2
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        run_xr_wire_left_test(spec_b, k=2, j=j_half)


# ---------------------------------------------------------------------------
# TestXRLargerSpins: higher spin values
# ---------------------------------------------------------------------------

class TestXRLargerSpins:
    """Verify the X = R identity for j = 1, 3/2, and 2."""

    # j = 1 -----------------------------------------------------------------

    def test_j1_4edge_k0(self):
        """j=1, 4-edge: out at position 0; sigma = [1,2,3,0]."""
        j = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j)

    def test_j1_4edge_k1(self):
        """j=1, 4-edge: out at position 1; sigma = [0,2,3,1]."""
        j = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=1, j=j)

    def test_j1_4edge_k3_canonical(self):
        """j=1, 4-edge canonical: sigma = identity."""
        j = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_test(spec_a, k=3, j=j)

    # j = 3/2 ----------------------------------------------------------------

    def test_j3_half_4edge_k0(self):
        """j=3/2, 4-edge: out at position 0; sigma = [1,2,3,0]."""
        j = yuzuha.Spin(3)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j)

    def test_j3_half_4edge_k2(self):
        """j=3/2, 4-edge: out at position 2; sigma = [0,1,3,2]."""
        j = yuzuha.Spin(3)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=2, j=j)

    # j = 2 ------------------------------------------------------------------

    def test_j2_4edge_k1(self):
        """j=2, 4-edge: out at position 1; sigma = [0,2,3,1]."""
        j = yuzuha.Spin(4)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
        ])
        run_xr_wire_right_test(spec_a, k=1, j=j)

    def test_j2_4edge_k3_canonical(self):
        """j=2, 4-edge canonical: sigma = identity."""
        j = yuzuha.Spin(4)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_test(spec_a, k=3, j=j)


# ---------------------------------------------------------------------------
# TestXRIdentityPermutation: k = N-1 gives sigma = identity => X-slice = I
# ---------------------------------------------------------------------------

class TestXRIdentityPermutation:
    """When the contracted edge is already at the last position (k = N-1),
    the wire does not reorder anything: sigma = identity, so
    X[:, 0, :] * sqrt(2j+1) = I.

    This is a special case of the general identity that also serves as a
    sanity check: contracting with a wire and immediately re-appending the
    same edge type at the end is a no-op on the OM space.
    """

    def _check_identity(self, spec_a, j):
        n = spec_a.num_external()
        k = n - 1   # last position
        run_xr_wire_right_test(spec_a, k=k, j=j)
        # Additionally verify that the rescaled slice is numerically the identity matrix
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        x_array, _ = yuzuha.compute_xsymbol(
            spec_a, spec_b, yuzuha.Contraction([k], [0])
        )
        wire_dim = j.dimension()   # 2j+1
        x_slice = x_array[:, 0, :] * np.sqrt(wire_dim)
        dim = spec_a.om_dimension()
        identity = np.eye(dim)
        assert np.allclose(x_slice, identity, atol=1e-10), (
            f"X-slice * sqrt(2j+1) should be I for sigma=identity, max diff="
            f"{np.max(np.abs(x_slice - identity)):.2e}"
        )

    def test_j_half_4edge_canonical(self):
        """j=1/2, canonical 4-edge [in,in,in,out]: sigma = identity."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        self._check_identity(spec_a, j)

    def test_j1_4edge_canonical(self):
        """j=1, canonical 4-edge: sigma = identity."""
        j = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        self._check_identity(spec_a, j)

    def test_j3_half_4edge_canonical(self):
        """j=3/2, canonical 4-edge: sigma = identity."""
        j = yuzuha.Spin(3)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        self._check_identity(spec_a, j)

    def test_j_half_5edge(self):
        """j=1/2 and j=1, 5-edge canonical: sigma = identity."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        self._check_identity(spec_a, j1)


# ---------------------------------------------------------------------------
# TestXRNontrivialOM: 5-edge and 6-edge tensors (OM dim > 2)
# ---------------------------------------------------------------------------

class TestXRNontrivialOM:
    """Verify X = R for tensors with larger OM dimensions.

    - 5-edge [in(1/2)×4, out(1)]: OM dim = 3
    - 5-edge [in(1)×3, out(1/2), out(1/2)]: mixed (OM dim ≥ 1)
    - 6-edge [in(1/2)×6, out(1/2)×1] ... actually need even half-integers;
      use [in(1/2)×3, out(1/2)×3] for example
    """

    def test_5edge_j_half_x4_out_j1_k0(self):
        """5-edge [in(1/2)×4, out(1)]: move out-j1 edge from k=0 to last."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),     # k = 0
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j1)

    def test_5edge_j_half_x4_out_j1_k4_canonical(self):
        """5-edge canonical [in(1/2)×4, out(1)]: k=4 (last); sigma = identity."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),     # k = 4 (last)
        ])
        run_xr_wire_right_test(spec_a, k=4, j=j1)

    def test_5edge_j_half_x4_out_j1_k2(self):
        """5-edge [in,in,out(1),in,in]: move out-j1 from k=2 to last."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),     # k = 2
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        run_xr_wire_right_test(spec_a, k=2, j=j1)

    def test_6edge_j_half_x6_transposed_k0(self):
        """6-edge [in(1/2)×5, out(1/2)]: transposed wire, contract in-edge k=0 with B[0]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),   # k = 0, in(j) -> contract with B[0]
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_transposed_test(spec_a, k=0, j=j)

    def test_6edge_j_half_x6_transposed_k3(self):
        """6-edge [in(1/2)×5, out(1/2)]: transposed wire, contract in-edge k=3 with B[0]."""
        j = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),   # k = 3
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        run_xr_wire_right_transposed_test(spec_a, k=3, j=j)


# ---------------------------------------------------------------------------
# TestXRStress: randomised stress tests
# ---------------------------------------------------------------------------

class TestXRStress:
    """Randomised stress tests that verify X[:, 0, :] == R for many configs."""

    def _make_spec_with_out_at_k(self, j, n, k):
        """Build an N-edge spec with all in(j) except out(j) at position k."""
        edges = [yuzuha.Edge.incoming(j)] * n
        edges[k] = yuzuha.Edge.outgoing(j)
        return yuzuha.CGSpec.from_edges(edges)

    def test_stress_j_half_all_k_all_n(self):
        """j=1/2: verify for n = 4, 5, 6 edges and all valid k positions."""
        j = yuzuha.Spin(1)
        # For j=1/2, n-1 in and 1 out must produce even total 2j parity:
        # n edges each j=1/2 -> total 2j = n; need n even for coupling to 0.
        # n=4: ok (OM 2), n=6: ok (OM dim >= 1), n=5: total 2j=5 (odd) -> skip.
        for n in (4, 6):
            for k in range(n):
                spec_a = self._make_spec_with_out_at_k(j, n, k)
                if spec_a.om_dimension() == 0:
                    continue
                run_xr_wire_right_test(spec_a, k=k, j=j)

    def test_stress_j1_all_k(self):
        """j=1, 4-edge: verify for all k=0..3."""
        j = yuzuha.Spin(2)
        for k in range(4):
            spec_a = self._make_spec_with_out_at_k(j, 4, k)
            run_xr_wire_right_test(spec_a, k=k, j=j)

    def test_stress_j3_half_all_k(self):
        """j=3/2, 4-edge: verify for all k=0..3."""
        j = yuzuha.Spin(3)
        for k in range(4):
            spec_a = self._make_spec_with_out_at_k(j, 4, k)
            run_xr_wire_right_test(spec_a, k=k, j=j)

    def test_stress_j2_all_k(self):
        """j=2, 4-edge: verify for all k=0..3."""
        j = yuzuha.Spin(4)
        for k in range(4):
            spec_a = self._make_spec_with_out_at_k(j, 4, k)
            run_xr_wire_right_test(spec_a, k=k, j=j)

    def test_stress_transposed_j_half_all_k(self):
        """j=1/2, 4-edge canonical: transposed wire, contract in-edges k=0,1,2 with B[0]."""
        j = yuzuha.Spin(1)
        # Canonical spec: [in, in, in, out]. Contract in-edges (k=0,1,2) with B[0].
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        for k in range(3):   # k=0,1,2 are all in(j)
            run_xr_wire_right_transposed_test(spec_a, k=k, j=j)

    def test_stress_wire_left_j_half_all_k(self):
        """j=1/2, canonical 4-edge spec_B: wire-on-left for all in-edges k=0..2."""
        j = yuzuha.Spin(1)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        for k in range(3):   # only in-edges of spec_B
            run_xr_wire_left_test(spec_b, k=k, j=j)

    def test_stress_wire_left_j1_all_k(self):
        """j=1, 4-edge spec_B: wire-on-left for in-edges k=0..2."""
        j = yuzuha.Spin(2)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
        ])
        for k in range(3):
            run_xr_wire_left_test(spec_b, k=k, j=j)

    def test_stress_mixed_spins(self):
        """5-edge tensors with mixed spins: verify X = R for a range of k."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # Spec: [out(j1), in(1/2), in(1/2), in(1/2), in(1/2)] — out at k=0
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        run_xr_wire_right_test(spec_a, k=0, j=j1)

        # Spec: [in(1/2), in(1/2), out(j1), in(1/2), in(1/2)] — out at k=2
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        run_xr_wire_right_test(spec_a2, k=2, j=j1)
