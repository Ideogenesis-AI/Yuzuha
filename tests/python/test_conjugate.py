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

import pytest
import yuzuha


# ===========================================================================
# TestFsPhaseForSpin
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
# TestComputeConjugateDirections
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
# TestComputeConjugatePhase
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
