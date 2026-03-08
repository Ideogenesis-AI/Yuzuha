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
Tests for compute_conjugate and fs_phase_for_spin.

compute_conjugate(spec) -> (phase: float, conj_spec: CGSpec)
  - conj_spec has all edge directions flipped relative to spec.
  - The cumulated FS phase is the product of (-1)^{2j} for each edge in
    conj_spec whose direction differs from the canonical conjugate pattern:
    first (n-1) edges Outgoing, last edge Incoming.

fs_phase_for_spin(spin: Spin) -> float
  - Returns (-1)^{2j}: +1.0 for integer spins, -1.0 for half-integer spins.
"""

import itertools

import numpy as np
import pytest
import yuzuha


# ===========================================================================
#  Test fs_phase_for_spin
# ===========================================================================

class TestFsPhaseForSpin:
    """Unit tests for the fs_phase_for_spin helper."""

    def test_integer_spins_give_plus_one(self):
        """j=0, 1, 2 are integer → (-1)^{2j} = +1."""
        for j_doubled in [0, 2, 4, 6]:
            phase = yuzuha.fs_phase_for_spin(yuzuha.Spin(j_doubled))
            assert phase == 1.0, f"Expected +1.0 for j={j_doubled/2}, got {phase}"

    def test_half_integer_spins_give_minus_one(self):
        """j=1/2, 3/2, 5/2 are half-integer → (-1)^{2j} = -1."""
        for j_doubled in [1, 3, 5]:
            phase = yuzuha.fs_phase_for_spin(yuzuha.Spin(j_doubled))
            assert phase == -1.0, f"Expected -1.0 for j={j_doubled/2}, got {phase}"

    def test_return_type_is_float(self):
        assert isinstance(yuzuha.fs_phase_for_spin(yuzuha.Spin(1)), float)
        assert isinstance(yuzuha.fs_phase_for_spin(yuzuha.Spin(2)), float)


# ===========================================================================
#  Test compute_conjugate directions
# ===========================================================================

class TestComputeConjugateDirections:
    """Verify that the conjugated spec has all directions flipped."""

    def test_all_directions_flipped_3edge(self):
        """3-edge spec: each direction in conj is the opposite of the original."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        _, conj = yuzuha.compute_conjugate(spec)
        for orig, conj_e in zip(spec.edges, conj.edges):
            assert orig.dir != conj_e.dir, "Expected flipped direction"

    def test_all_directions_flipped_4edge(self):
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        _, conj = yuzuha.compute_conjugate(spec)
        for orig, conj_e in zip(spec.edges, conj.edges):
            assert orig.dir != conj_e.dir

    def test_spins_preserved(self):
        """Spins must be unchanged after conjugation."""
        j_half = yuzuha.Spin(1)
        j3_2   = yuzuha.Spin(3)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j3_2),
            yuzuha.Edge.outgoing(j1),
        ])
        _, conj = yuzuha.compute_conjugate(spec)
        for orig, conj_e in zip(spec.edges, conj.edges):
            assert orig.j.twice() == conj_e.j.twice()

    def test_num_external_preserved(self):
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
        ])
        _, conj = yuzuha.compute_conjugate(spec)
        assert conj.num_external() == spec.num_external()

    def test_conj_spec_is_valid(self):
        """om_dimension() must not raise."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        _, conj = yuzuha.compute_conjugate(spec)
        _ = conj.om_dimension()


# ===========================================================================
#  Test compute_conjugate phase
# ===========================================================================

class TestComputeConjugatePhase:
    """Verify the cumulated FS phase logic.

    Canonical conjugate pattern: first (n-1) edges Outgoing, last edge Incoming.
    A mismatch on edge i with spin j_i contributes (-1)^{2j_i}.
    """

    def test_canonical_spec_phase_one_integer(self):
        """Canonical spec with integer spins → conj is canonical-conjugate → phase = 1."""
        # canonical: in, in, out  →  conj: out, out, in  (== canonical-conjugate)
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_canonical_spec_phase_one_half_integer(self):
        """Canonical spec with half-integer spins → no mismatches → phase still 1."""
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_single_first_region_mismatch_integer(self):
        """First edge Outgoing in spec → conj edge is Incoming (mismatch vs Outgoing).
        Integer j → phase = +1."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_single_first_region_mismatch_half_integer(self):
        """First edge Outgoing in spec → conj edge is Incoming (mismatch vs Outgoing).
        Half-integer j → phase = -1."""
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == -1.0

    def test_last_edge_mismatch_half_integer(self):
        """Last edge Incoming in spec → conj edge is Outgoing (mismatch vs Incoming).
        Half-integer j → phase = -1.
        Uses j=1, j=1/2, j=1/2 (two half-integers, valid SU(2) config)."""
        j1     = yuzuha.Spin(2)
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == -1.0

    def test_last_edge_mismatch_integer(self):
        """Last edge Incoming in spec → conj edge is Outgoing (mismatch).
        Integer j → phase = +1."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_two_mismatches_both_half_integer(self):
        """Two half-integer mismatches: (-1)*(-1) = +1."""
        # edges 0 and 1 both out → conj: in, in → both mismatch canonical-conj (out, out)
        # j=1/2 for both  →  (-1)^1 * (-1)^1 = +1
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_two_mismatches_one_half_one_integer(self):
        """One half-integer mismatch, one integer mismatch: (-1)*(+1) = -1.
        Uses j=1/2, j=1, j=1/2 (two half-integers, valid SU(2) config).
        All edges outgoing → conj all incoming → first two mismatch canonical-conj (outgoing)."""
        # edge 0: conj in vs canonical-conj out, j=1/2 → -1
        # edge 1: conj in vs canonical-conj out, j=1  → +1
        # edge 2: conj in vs canonical-conj in          → no contribution
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == -1.0

    def test_fully_non_canonical_3edge(self):
        """All directions inverted from canonical: out, out, in.
        Conj = in, in, out → all mismatch canonical-conj (out, out, in).
        Phase = product of (-1)^{2j_i} for all edges."""
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        expected = (
            yuzuha.fs_phase_for_spin(j_half)
            * yuzuha.fs_phase_for_spin(j_half)
            * yuzuha.fs_phase_for_spin(j1)
        )
        assert phase == expected  # (-1)*(-1)*(+1) = +1

    def test_fully_non_canonical_4edge_all_half_integer(self):
        """All four edges non-canonical, all half-integer spins.
        Spec = out, out, out, in  → conj = in, in, in, out.
        Each mismatches canonical-conj (out, out, out, in).
        Phase = (-1)^4 = +1."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        phase, _ = yuzuha.compute_conjugate(spec)
        assert phase == 1.0

    def test_phase_is_plus_or_minus_one(self):
        """phase must always be exactly +1.0 or -1.0."""
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        j2     = yuzuha.Spin(4)
        specs = [
            yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j2),
            ]),
            yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j1),
            ]),
            yuzuha.CGSpec.from_edges([
                yuzuha.Edge.outgoing(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j1),
            ]),
            yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j_half),
            ]),
        ]
        for spec in specs:
            phase, _ = yuzuha.compute_conjugate(spec)
            assert phase in (1.0, -1.0), f"Unexpected phase {phase}"

    def test_double_conjugate_phase_is_identity(self):
        """Applying compute_conjugate twice must restore directions and phases cancel.

        The net phase phase1 * phase2 is always +1: for each edge, exactly one of
        the two passes accumulates (-1)^{2j}, and valid CGSpecs have an even number
        of half-integer edges so the product is +1."""
        j_half = yuzuha.Spin(1)
        j1     = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        phase1, conj1 = yuzuha.compute_conjugate(spec)
        phase2, conj2 = yuzuha.compute_conjugate(conj1)
        for orig, final in zip(spec.edges, conj2.edges):
            assert orig.dir == final.dir, "Double conjugate should restore original directions"
        assert phase1 * phase2 == 1.0, "Double-conjugate phases must cancel to +1"


# ===========================================================================
#  Test conjugate orthogonality
# ===========================================================================

class TestConjugateOrthogonality:
    """Test that conj(A) contracted with A gives identity.

    For a CGSpec ``spec``, ``compute_conjugate`` yields ``(fs_phase, spec_conj)``
    where ``spec_conj`` has all directions flipped and ``fs_phase`` accounts for
    the FS phase correction.  Scaling the conjugate basis by ``fs_phase`` and
    computing the cross-Gram matrix must recover the identity:

        (fs_phase * B_conj)^T @ B  =  I

    This replaces the old convention where ``B_conj^T @ B = fs_phase * I`` with
    ``fs_phase`` obtained from the now-removed ``compute_fs_phase``.
    """

    def _assert_cross_gram_is_identity(self, spec):
        """Assert that (fs_phase * B_conj)^T @ B equals I.

        Derives the conjugate spec and its FS phase via ``compute_conjugate``,
        computes the canonical bases for both specs, scales the conjugate basis
        by the phase, and verifies the cross-Gram matrix is the identity.
        """
        fs_phase, spec_conj = yuzuha.compute_conjugate(spec)

        basis      = yuzuha.canonical_basis(spec)
        basis_conj = yuzuha.canonical_basis(spec_conj)
        om_dim = spec.om_dimension()

        assert spec_conj.om_dimension() == om_dim

        physical_dim = np.prod(basis.shape[:-1])
        basis_matrix      = basis.reshape(physical_dim, om_dim)
        basis_conj_matrix = basis_conj.reshape(physical_dim, om_dim)

        # Apply FS phase to the conjugate basis before computing the cross-Gram.
        cross_gram = (fs_phase * basis_conj_matrix).T @ basis_matrix

        assert np.allclose(cross_gram, np.eye(om_dim), rtol=1e-10, atol=1e-10)

    # ------------------------------------------------------------------
    # 2-edge tests
    # ------------------------------------------------------------------

    def test_cross_orthonormality_two_edge_j_half(self):
        """2-edge j=1/2: [in, out]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_two_edge_j1(self):
        """2-edge j=1: [in, out]."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_two_edge_j3_half(self):
        """2-edge j=3/2: [in, out]."""
        j3_half = yuzuha.Spin(3)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 3-edge tests — spin variety
    # ------------------------------------------------------------------

    def test_cross_orthonormality_j_half_three_edges(self):
        """3-edge: [in(j=1/2), in(j=1/2), out(j=0)] — two j=1/2 to singlet."""
        j_half = yuzuha.Spin(1)
        j0 = yuzuha.Spin(0)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j0),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j1_three_edges(self):
        """3-edge: [in(j=1), in(j=1), out(j=1)] canonical."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j3_half_three_edges(self):
        """3-edge: [in(j=3/2), in(j=3/2), out(j=1)]."""
        j3_half = yuzuha.Spin(3)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_j1_three_edges(self):
        """3-edge mixed: [in(j=1/2), in(j=1/2), out(j=1)]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_j1_three_edges_j_half_out(self):
        """3-edge mixed: [in(j=1), in(j=1/2), out(j=1/2)]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 3-edge tests — direction variety
    # ------------------------------------------------------------------

    def test_cross_orthonormality_j_half_three_edges_all_in(self):
        """3-edge [j=1/2, j=1/2, j=1]: all-incoming."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_three_edges_first_out(self):
        """3-edge [j=1/2, j=1/2, j=1]: [out, in, in]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_three_edges_middle_out(self):
        """3-edge [j=1/2, j=1/2, j=1]: [in, out, in]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 4-edge tests — spin variety
    # ------------------------------------------------------------------

    def test_cross_orthonormality_inverted_directions(self):
        """4× j=1/2: all-incoming."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j1_four_edges(self):
        """4× j=1: all-incoming."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j3_half_four_edges(self):
        """4× j=3/2: all-incoming."""
        j3_half = yuzuha.Spin(3)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j2_four_edges(self):
        """4× j=2: all-incoming."""
        j2 = yuzuha.Spin(4)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j2),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 4-edge tests — direction variety (j=1/2)
    # ------------------------------------------------------------------

    def test_cross_orthonormality_j_half_four_last_out(self):
        """4× j=1/2: [in, in, in, out] canonical."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_four_first_out(self):
        """4× j=1/2: [out, in, in, in]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_four_second_out(self):
        """4× j=1/2: [in, out, in, in]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_four_third_out(self):
        """4× j=1/2: [in, in, out, in]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_four_alternating(self):
        """4× j=1/2: [in, out, in, out]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_four_all_out(self):
        """4× j=1/2: all-outgoing."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j3_half_four_mixed_dir(self):
        """4× j=3/2: [in, out, in, out]."""
        j3_half = yuzuha.Spin(3)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_j1_four_mixed_dir(self):
        """4-edge mixed spins j=1/2, j=1: [in, out, in, out]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 5-edge tests
    # ------------------------------------------------------------------

    def test_cross_orthonormality_five_edges(self):
        """5-edge: 4× j=1/2 + 1× j=1, all-incoming."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_six_edges(self):
        """6× j=1/2: all-incoming."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_five_edges_mixed_dir(self):
        """5-edge (4× j=1/2 + 1× j=1) with mixed directions."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # 6-edge tests — direction variety
    # ------------------------------------------------------------------

    def test_cross_orthonormality_j_half_six_edges_alternating(self):
        """6× j=1/2: [in, out, in, out, in, out]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_six_edges_last_out(self):
        """6× j=1/2: first five in, last out."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    # ------------------------------------------------------------------
    # High-order tests: 7-edge and 8-edge
    # ------------------------------------------------------------------

    def test_cross_orthonormality_j_half_seven_edges(self):
        """7-edge: 6× j=1/2 + 1× j=1, all-incoming."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges(
            [yuzuha.Edge.incoming(j_half)] * 6 + [yuzuha.Edge.incoming(j1)]
        )
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_eight_edges(self):
        """8× j=1/2: all-incoming."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges(
            [yuzuha.Edge.incoming(j_half)] * 8
        )
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j_half_eight_edges_alternating(self):
        """8× j=1/2: [in, out, in, out, in, out, in, out]."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_j1_eight_edges(self):
        """8× j=1: all-incoming."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges(
            [yuzuha.Edge.incoming(j1)] * 8
        )
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_mixed_spins(self):
        """4-edge mixed j=1/2 and j=1: [in, in, in, in]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)

    def test_cross_orthonormality_mixed_directions(self):
        """4-edge mixed spins and directions: [in(j=1/2), in(j=1/2), out(j=1), in(j=1)]."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
        ])
        self._assert_cross_gram_is_identity(spec)


# ===========================================================================
#  Stress tests for conjugate orthogonality
# ===========================================================================

class TestConjugateOrthogonalityStress:
    """Stress tests for conjugate cross-Gram orthogonality.

    For each tensor size n in 3..8, systematically enumerates valid spin
    configurations, constructs the canonical spec — (n-1) incoming edges
    followed by one outgoing edge — and verifies that
    ``(fs_phase * B_conj)^T @ B = I`` using ``compute_conjugate``.

    A spin configuration (2j_1, ..., 2j_n) is valid when:
      1. Even number of half-integer spins (even count of odd 2j values).
      2. Largest spin ≤ sum of all others (generalised triangle condition).
    Configurations with om_dimension == 0 are skipped at runtime.
    """

    def _assert_cross_gram_is_identity(self, spec):
        """Assert that (fs_phase * B_conj)^T @ B equals I."""
        fs_phase, spec_conj = yuzuha.compute_conjugate(spec)

        basis      = yuzuha.canonical_basis(spec)
        basis_conj = yuzuha.canonical_basis(spec_conj)
        om_dim = spec.om_dimension()

        assert spec_conj.om_dimension() == om_dim

        physical_dim = np.prod(basis.shape[:-1])
        basis_matrix      = basis.reshape(physical_dim, om_dim)
        basis_conj_matrix = basis_conj.reshape(physical_dim, om_dim)

        cross_gram = (fs_phase * basis_conj_matrix).T @ basis_matrix

        assert np.allclose(cross_gram, np.eye(om_dim), rtol=1e-10, atol=1e-10)

    @staticmethod
    def _generate_spin_configs(n, max_2j, max_configs):
        """Yield spin tuples (each entry is 2j) of length n.

        Tuples are drawn from ``range(1, max_2j + 1)^n`` and pre-filtered by:
          - parity: even count of odd entries,
          - triangle: largest entry ≤ sum of all others.
        At most ``max_configs`` tuples are yielded.
        """
        count = 0
        for two_js in itertools.product(range(1, max_2j + 1), repeat=n):
            if count >= max_configs:
                break
            if sum(1 for x in two_js if x % 2 == 1) % 2 != 0:
                continue
            max_val = max(two_js)
            if max_val > sum(two_js) - max_val:
                continue
            count += 1
            yield two_js

    def _run_stress(self, n, max_2j, max_configs):
        """Run the cross-orthogonality check for all accepted spin configs."""
        tested = 0
        for two_js in self._generate_spin_configs(n, max_2j, max_configs):
            spins = [yuzuha.Spin(tj) for tj in two_js]
            # Canonical direction: (n-1) incoming, last outgoing.
            edges = [yuzuha.Edge.incoming(s) for s in spins[:-1]]
            edges.append(yuzuha.Edge.outgoing(spins[-1]))
            spec = yuzuha.CGSpec.from_edges(edges)

            if spec.om_dimension() == 0:
                continue

            self._assert_cross_gram_is_identity(spec)
            tested += 1

        assert tested > 0, (
            f"No non-trivial spin configs were found for n={n} edges "
            f"(max_2j={max_2j}); increase max_2j or max_configs."
        )

    def test_stress_cross_orthonormality_3_edges(self):
        """Stress: 3-edge tensors, spins up to j=3."""
        self._run_stress(n=3, max_2j=6, max_configs=50)

    def test_stress_cross_orthonormality_4_edges(self):
        """Stress: 4-edge tensors, spins up to j=2."""
        self._run_stress(n=4, max_2j=4, max_configs=50)

    def test_stress_cross_orthonormality_5_edges(self):
        """Stress: 5-edge tensors, spins up to j=2."""
        self._run_stress(n=5, max_2j=4, max_configs=50)

    def test_stress_cross_orthonormality_6_edges(self):
        """Stress: 6-edge tensors, spins up to j=3/2."""
        self._run_stress(n=6, max_2j=3, max_configs=50)

    def test_stress_cross_orthonormality_7_edges(self):
        """Stress: 7-edge tensors, spins up to j=3/2."""
        self._run_stress(n=7, max_2j=3, max_configs=50)

    def test_stress_cross_orthonormality_8_edges(self):
        """Stress: 8-edge tensors, spins up to j=3/2."""
        self._run_stress(n=8, max_2j=3, max_configs=50)
