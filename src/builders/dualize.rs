// Copyright (C) 2026 Changkai Zhang.
//
// This file is part of Yuzuha library.
//
// Yuzuha is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Yuzuha is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Yuzuha. If not, see <https://www.gnu.org/licenses/>.

//! Dualization (conjugation) of CGSpecs and the Frobenius-Schur (FS) phase.
//!
//! The canonical direction convention for a CG basis is:
//! - first (n-1) edges: **Incoming**
//! - last edge: **Outgoing**
//!
//! The canonical direction for the **conjugate** (dual) basis is the opposite:
//! - first (n-1) edges: **Outgoing**
//! - last edge: **Incoming**
//!
//! `compute_conjugate` reverses all edge directions and accumulates a factor of
//! `(-1)^{2j}` for each edge whose direction in the conjugated spec differs from
//! the canonical conjugate pattern.

use crate::core::{CGSpec, Direction, Edge, Spin};
use crate::error::Result;

/// Returns the Frobenius-Schur phase `(-1)^{2j}` for a single spin.
///
/// - Integer spins (`2j` even): returns `+1.0`.
/// - Half-integer spins (`2j` odd): returns `-1.0`.
pub fn fs_phase_for_spin(j: Spin) -> f64 {
    if j.twice() % 2 != 0 { -1.0 } else { 1.0 }
}

/// Compute the conjugate CGSpec and the associated cumulated FS phase.
///
/// The conjugated spec has all edge directions reversed relative to `spec`.
/// The cumulated FS phase corrects for edges in the conjugated spec whose
/// direction differs from the **canonical conjugate pattern**:
/// first (n-1) edges Outgoing, last edge Incoming.
///
/// Each such differing edge contributes a factor of `(-1)^{2j}`.
///
/// # Returns
/// `(phase, conj_spec)` where `phase` is `+1.0` or `-1.0` and `conj_spec`
/// has the same spins as `spec` with all directions flipped.
pub fn compute_conjugate(spec: &CGSpec) -> Result<(f64, CGSpec)> {
    let n = spec.num_external();

    // Flip all edge directions.
    let conj_edges: Vec<Edge> = spec
        .edges
        .iter()
        .map(|e| Edge::new(e.j, e.dir.flip()))
        .collect();
    let conj_spec = CGSpec::from_edges(conj_edges)?;

    // Accumulate FS phase for each edge that differs from the canonical
    // conjugate direction (first n-1 Outgoing, last Incoming).
    let mut phase = 1.0_f64;
    for (i, edge) in conj_spec.edges.iter().enumerate() {
        let canonical_conj_dir = if i < n - 1 {
            Direction::Outgoing
        } else {
            Direction::Incoming
        };
        if edge.dir != canonical_conj_dir {
            phase *= fs_phase_for_spin(edge.j);
        }
    }

    Ok((phase, conj_spec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Edge, Spin};

    // -----------------------------------------------------------------------
    // fs_phase_for_spin
    // -----------------------------------------------------------------------

    #[test]
    fn test_fs_phase_integer_spins() {
        // j=0, 1, 2, 3  →  (-1)^{2j} = +1
        for twice_j in [0, 2, 4, 6] {
            let j = Spin::new(twice_j).unwrap();
            assert_eq!(fs_phase_for_spin(j), 1.0, "expected +1 for j={twice_j}/2");
        }
    }

    #[test]
    fn test_fs_phase_half_integer_spins() {
        // j=1/2, 3/2, 5/2  →  (-1)^{2j} = -1
        for twice_j in [1, 3, 5] {
            let j = Spin::new(twice_j).unwrap();
            assert_eq!(fs_phase_for_spin(j), -1.0, "expected -1 for j={twice_j}/2");
        }
    }

    // -----------------------------------------------------------------------
    // compute_conjugate — direction flipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_directions_flipped() {
        let j1 = Spin::new(2).unwrap();  // j=1
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
        ]).unwrap();

        let (_, conj) = compute_conjugate(&spec).unwrap();

        for (orig, conj_edge) in spec.edges.iter().zip(conj.edges.iter()) {
            assert_ne!(orig.dir, conj_edge.dir, "direction must be flipped");
        }
    }

    #[test]
    fn test_spins_preserved_after_conjugate() {
        let j_half = Spin::new(1).unwrap();
        let j1    = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (_, conj) = compute_conjugate(&spec).unwrap();

        for (orig, conj_edge) in spec.edges.iter().zip(conj.edges.iter()) {
            assert_eq!(orig.j, conj_edge.j, "spin must be unchanged");
        }
    }

    #[test]
    fn test_num_external_preserved() {
        let j1 = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
        ]).unwrap();

        let (_, conj) = compute_conjugate(&spec).unwrap();
        assert_eq!(conj.num_external(), spec.num_external());
    }

    // -----------------------------------------------------------------------
    // compute_conjugate — FS phase logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_canonical_spec_phase_is_one() {
        // Canonical spec: first (n-1) incoming, last outgoing.
        // Conj = first (n-1) outgoing, last incoming  ← exactly canonical-conj.
        // → no mismatches → phase = 1.
        let j1 = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, 1.0);
    }

    #[test]
    fn test_canonical_spec_half_integer_phase_is_one() {
        // Canonical spec with half-integer spins: still no mismatches.
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, 1.0);
    }

    #[test]
    fn test_single_first_region_mismatch_half_integer() {
        // First edge Outgoing in spec → conj first edge Incoming
        // → mismatch vs canonical-conj (Outgoing); j=1/2 → -1.
        // Spec: out(j=1/2), in(j=1/2), out(j=1)  — two half-integers, valid.
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, -1.0);
    }

    #[test]
    fn test_single_first_region_mismatch_integer() {
        // First edge Outgoing in spec; integer j=1 → mismatch contributes +1.
        // Spec: out(1), in(1), out(2)  — all integer.
        let j1 = Spin::new(2).unwrap();
        let j2 = Spin::new(4).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j1),
            Edge::incoming(j1),
            Edge::outgoing(j2),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, 1.0);
    }

    #[test]
    fn test_two_half_integer_mismatches_cancel() {
        // Two first-region edges both Outgoing, both j=1/2: (-1)*(-1) = +1.
        // Spec: out(1/2), out(1/2), out(1)
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j_half),
            Edge::outgoing(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, 1.0);
    }

    #[test]
    fn test_last_edge_mismatch_half_integer() {
        // Last edge Incoming in spec → conj last edge Outgoing
        // → mismatch vs canonical-conj (Incoming); j=1/2 → -1.
        // Spec: in(1), in(1/2), in(1/2) — two half-integers, valid.
        let j1     = Spin::new(2).unwrap();
        let j_half = Spin::new(1).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ]).unwrap();

        let (phase, _) = compute_conjugate(&spec).unwrap();
        assert_eq!(phase, -1.0);
    }

    #[test]
    fn test_double_conjugate_restores_directions_and_phases_cancel() {
        // Applying compute_conjugate twice must restore the original directions,
        // and the two phases must multiply to +1.
        //
        // Proof sketch: for each edge i, exactly one of phase1 / phase2 accumulates
        // the factor (-1)^{2j_i}.  A valid CGSpec has an even number of half-integer
        // edges, so the total product is (-1)^{even} = +1.
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase1, conj1) = compute_conjugate(&spec).unwrap();
        let (phase2, conj2) = compute_conjugate(&conj1).unwrap();

        // Directions must be fully restored.
        for (orig, final_edge) in spec.edges.iter().zip(conj2.edges.iter()) {
            assert_eq!(orig.dir, final_edge.dir, "double-conj must restore direction");
        }

        // The two phases must cancel: phase1 * phase2 == +1.
        assert_eq!(phase1 * phase2, 1.0, "double-conjugate phases must multiply to +1");
    }

    // -----------------------------------------------------------------------
    // compute_conjugate — 5-edge basis
    // -----------------------------------------------------------------------

    #[test]
    fn test_five_edge_canonical_phase_is_one() {
        // Canonical 5-edge spec: first 4 incoming, last outgoing.
        // Conj = first 4 outgoing, last incoming = canonical-conj → no mismatches.
        // Spins: j=1/2, j=1/2, j=1/2, j=1/2, j=1  (four half-integers, valid).
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, conj) = compute_conjugate(&spec).unwrap();

        assert_eq!(phase, 1.0, "canonical 5-edge spec: phase must be +1");
        assert_eq!(conj.num_external(), 5);
        // Directions must all be flipped.
        for (orig, conj_edge) in spec.edges.iter().zip(conj.edges.iter()) {
            assert_ne!(orig.dir, conj_edge.dir);
        }
    }

    #[test]
    fn test_five_edge_single_mismatch_half_integer() {
        // Non-canonical 5-edge spec: first edge Outgoing (mismatch) → conj first
        // edge Incoming vs canonical-conj Outgoing → factor (-1)^{2*(1/2)} = -1.
        // Remaining edges canonical → no further contributions.
        // Net phase = -1.
        //
        // Spec: out(1/2), in(1/2), in(1/2), in(1/2), out(1)
        //        ^-- non-canonical                     ^-- canonical last
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase, conj) = compute_conjugate(&spec).unwrap();

        assert_eq!(phase, -1.0, "single half-integer mismatch in 5-edge spec: phase must be -1");
        assert_eq!(conj.num_external(), 5);
    }

    #[test]
    fn test_five_edge_double_conjugate_phases_cancel() {
        // Verify the phase-cancellation invariant (phase1 * phase2 == +1)
        // holds for a 5-edge non-canonical spec as well.
        let j_half = Spin::new(1).unwrap();
        let j1     = Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::outgoing(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ]).unwrap();

        let (phase1, conj1) = compute_conjugate(&spec).unwrap();
        let (phase2, _)     = compute_conjugate(&conj1).unwrap();

        assert_eq!(phase1 * phase2, 1.0, "double-conjugate phases must multiply to +1 for 5-edge spec");
    }

    #[test]
    fn test_phase_is_plus_or_minus_one() {
        // Phase must always be exactly ±1 for a variety of specs.
        let configs: Vec<Vec<Edge>> = vec![
            vec![Edge::incoming(Spin::new(2).unwrap()), Edge::incoming(Spin::new(2).unwrap()), Edge::outgoing(Spin::new(4).unwrap())],
            vec![Edge::incoming(Spin::new(1).unwrap()), Edge::incoming(Spin::new(1).unwrap()), Edge::outgoing(Spin::new(2).unwrap())],
            vec![Edge::outgoing(Spin::new(1).unwrap()), Edge::incoming(Spin::new(1).unwrap()), Edge::outgoing(Spin::new(2).unwrap())],
            vec![Edge::incoming(Spin::new(1).unwrap()), Edge::incoming(Spin::new(1).unwrap()), Edge::incoming(Spin::new(1).unwrap()), Edge::outgoing(Spin::new(1).unwrap())],
        ];
        for edges in configs {
            let spec = CGSpec::from_edges(edges).unwrap();
            let (phase, _) = compute_conjugate(&spec).unwrap();
            assert!(phase == 1.0 || phase == -1.0, "phase must be ±1, got {phase}");
        }
    }
}
