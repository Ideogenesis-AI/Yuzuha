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

//! Consistency of the conjugate (dual) specification.
//!
//! `compute_conjugate` flips every edge direction and returns the Frobenius-Schur
//! phase picked up relative to the conjugate of the canonical spec. The phase is
//! the product of `(-1)^{2j}` over edges whose direction disagrees with the
//! canonical pattern, so it is always ±1, and conjugating twice must restore the
//! original directions with the two phases cancelling.

use yuzuha::builders::dualize::compute_conjugate;
use yuzuha::core::{CGSpec, Edge};

use crate::common::j;

/// Build a spec from (doubled spin, is_incoming) pairs.
fn spec(edges: &[(i32, bool)]) -> CGSpec {
    let edges: Vec<Edge> = edges
        .iter()
        .map(|&(d, incoming)| if incoming { Edge::incoming(j(d)) } else { Edge::outgoing(j(d)) })
        .collect();
    CGSpec::from_edges(edges).expect("edge spins must couple to j=0")
}

// ============================================================================
// Direction flipping
// ============================================================================

#[test]
fn test_all_directions_flipped() {
    let original = spec(&[(2, true), (2, true), (2, false)]);
    let (_, conj) = compute_conjugate(&original).unwrap();

    for (orig, conj_edge) in original.edges.iter().zip(conj.edges.iter()) {
        assert_ne!(orig.dir, conj_edge.dir, "direction must be flipped");
    }
}

#[test]
fn test_spins_preserved_after_conjugate() {
    let original = spec(&[(1, true), (1, true), (2, false)]);
    let (_, conj) = compute_conjugate(&original).unwrap();

    for (orig, conj_edge) in original.edges.iter().zip(conj.edges.iter()) {
        assert_eq!(orig.j, conj_edge.j, "spin must be unchanged");
    }
}

#[test]
fn test_num_external_preserved() {
    let original = spec(&[(2, true), (2, true), (2, true), (2, false)]);
    let (_, conj) = compute_conjugate(&original).unwrap();
    assert_eq!(conj.num_external(), original.num_external());
}

// ============================================================================
// Frobenius-Schur phase
// ============================================================================

#[test]
fn test_canonical_spec_phase_is_one() {
    // Canonical spec: first n-1 incoming, last outgoing. Its conjugate is
    // exactly the canonical conjugate, so there are no mismatches.
    let original = spec(&[(2, true), (2, true), (2, false)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, 1.0);
}

#[test]
fn test_canonical_spec_half_integer_phase_is_one() {
    let original = spec(&[(1, true), (1, true), (2, false)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, 1.0);
}

#[test]
fn test_single_first_region_mismatch_half_integer() {
    // First edge outgoing is one mismatch; j=1/2 contributes (-1)^{2j} = -1.
    let original = spec(&[(1, false), (1, true), (2, false)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, -1.0);
}

#[test]
fn test_single_first_region_mismatch_integer() {
    // Same mismatch with an integer spin contributes +1.
    let original = spec(&[(2, false), (2, true), (4, false)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, 1.0);
}

#[test]
fn test_two_half_integer_mismatches_cancel() {
    let original = spec(&[(1, false), (1, false), (2, false)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, 1.0);
}

#[test]
fn test_last_edge_mismatch_half_integer() {
    // Last edge incoming is the mismatch here; j=1/2 gives -1.
    let original = spec(&[(2, true), (1, true), (1, true)]);
    let (phase, _) = compute_conjugate(&original).unwrap();
    assert_eq!(phase, -1.0);
}

#[test]
fn test_phase_is_plus_or_minus_one() {
    let configs: Vec<Vec<(i32, bool)>> = vec![
        vec![(2, true), (2, true), (4, false)],
        vec![(1, true), (1, true), (2, false)],
        vec![(1, false), (1, true), (2, false)],
        vec![(1, true), (1, true), (1, true), (1, false)],
    ];
    for config in configs {
        let (phase, _) = compute_conjugate(&spec(&config)).unwrap();
        assert!(phase == 1.0 || phase == -1.0, "phase must be ±1, got {phase}");
    }
}

// ============================================================================
// Double conjugation
// ============================================================================

#[test]
fn test_double_conjugate_restores_directions_and_phases_cancel() {
    // For each edge exactly one of the two phases picks up (-1)^{2j}, and a
    // valid CGSpec has an even number of half-integer edges, so the product
    // over both conjugations is +1.
    let original = spec(&[(1, false), (1, true), (2, false)]);

    let (phase1, conj1) = compute_conjugate(&original).unwrap();
    let (phase2, conj2) = compute_conjugate(&conj1).unwrap();

    for (orig, final_edge) in original.edges.iter().zip(conj2.edges.iter()) {
        assert_eq!(orig.dir, final_edge.dir, "double-conj must restore direction");
    }
    assert_eq!(phase1 * phase2, 1.0, "double-conjugate phases must multiply to +1");
}

// ============================================================================
// Five-edge specs
// ============================================================================

#[test]
fn test_five_edge_canonical_phase_is_one() {
    let original = spec(&[(1, true), (1, true), (1, true), (1, true), (2, false)]);
    let (phase, conj) = compute_conjugate(&original).unwrap();

    assert_eq!(phase, 1.0, "canonical 5-edge spec: phase must be +1");
    assert_eq!(conj.num_external(), 5);
    for (orig, conj_edge) in original.edges.iter().zip(conj.edges.iter()) {
        assert_ne!(orig.dir, conj_edge.dir);
    }
}

#[test]
fn test_five_edge_single_mismatch_half_integer() {
    let original = spec(&[(1, false), (1, true), (1, true), (1, true), (2, false)]);
    let (phase, conj) = compute_conjugate(&original).unwrap();

    assert_eq!(phase, -1.0, "single half-integer mismatch in 5-edge spec: phase must be -1");
    assert_eq!(conj.num_external(), 5);
}

#[test]
fn test_five_edge_double_conjugate_phases_cancel() {
    let original = spec(&[(1, false), (1, true), (1, true), (1, true), (2, false)]);

    let (phase1, conj1) = compute_conjugate(&original).unwrap();
    let (phase2, _) = compute_conjugate(&conj1).unwrap();

    assert_eq!(phase1 * phase2, 1.0, "double-conjugate phases must multiply to +1 for 5-edge spec");
}
