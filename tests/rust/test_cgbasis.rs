// Copyright (C) 2025-2026 Changkai Zhang.
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

//! Orthonormality of the canonical basis.
//!
//! For every supported edge count and direction pattern, `build_canonical_basis_data`
//! must return OM slices of unit Frobenius norm that are mutually orthogonal when
//! contracted over all but the terminal edge, leaving `delta_{alpha,beta}/(2j+1) * I`.
//! `assert_canonical_basis_orthonormal` in `common` checks both properties; the tests
//! here enumerate the configurations it is checked on.
//!
//! The enumeration varies three things independently, because the basis builder
//! treats them through different code paths:
//!
//! - **Edge count** (2 through 6): two edges is a special-cased identity, three is
//!   the recursion base case, and four and above exercise the fusion-tree recursion
//!   with a growing number of internal spins.
//! - **Spin content**: uniform spins keep the outer multiplicity small, while mixed
//!   spins (especially with a large middle spin) push the OM dimension above one, so
//!   the off-diagonal `alpha != beta` orthogonality is actually non-trivial.
//! - **Edge directions**: non-canonical patterns route through the metric tensor to
//!   raise or lower indices, which must not disturb orthonormality.
//!
//! Every configuration must contain an even number of half-integer spins; otherwise
//! the edges cannot couple to j=0 and `CGSpec::from_edges` rejects the spec before a
//! basis is ever built. Comments below flag the cases where that parity constraint
//! dictates the chosen spins.

use approx::assert_relative_eq;
use yuzuha::builders::builders::build_canonical_basis_data;
use yuzuha::builders::TestCacheGuard;
use yuzuha::core::{CGSpec, Direction, Edge, Spin};

use crate::common::{assert_canonical_basis_orthonormal, j};

// ----------------------------------------------------------------------------
// Edge-list constructors
//
// Spins are given as doubled integers throughout, so 1 is j=1/2 and 2 is j=1.
// ----------------------------------------------------------------------------

/// Edges with the canonical direction pattern: all incoming but the last.
///
/// This is the fusion-tree convention — several edges fuse into one outgoing
/// result — and is the pattern the builder is optimized for.
fn canonical(doubled: &[i32]) -> Vec<Edge> {
    let last = doubled.len() - 1;
    doubled
        .iter()
        .enumerate()
        .map(|(i, &d)| if i == last { Edge::outgoing(j(d)) } else { Edge::incoming(j(d)) })
        .collect()
}

/// Edges that all share one direction.
///
/// All-incoming and all-outgoing are the extreme non-canonical cases: every
/// index relative to the canonical form has to be raised, or every one lowered.
fn uniform(doubled: &[i32], dir: Direction) -> Vec<Edge> {
    doubled.iter().map(|&d| Edge::new(j(d), dir)).collect()
}

/// Edges from explicit (doubled spin, direction) pairs.
fn mixed(spec: &[(i32, Direction)]) -> Vec<Edge> {
    spec.iter().map(|&(d, dir)| Edge::new(j(d), dir)).collect()
}

// ============================================================================
// Two-edge identity tests
//
// A two-edge spec is the degenerate case: the only way two spins couple to j=0
// is if they are equal, the outer multiplicity is always 1, and the basis is
// the identity matrix scaled to unit Frobenius norm. The generic helper does
// not apply (there is nothing to contract over before the terminal edge), so
// the structure is checked directly.
// ============================================================================

/// Verify that the two-edge basis for `spin` is a normalized identity.
///
/// Checks the shape is `[dim, dim, 1]` (the trailing 1 being the single OM
/// index), that the one slice has unit Frobenius norm, and that it squares to
/// `(1/dim) * I` — the residue of the `sqrt(2j+1)` normalization, matching how
/// bases with three or more edges are normalized.
fn test_two_edges_helper(spin: Spin, dir0: Direction, dir1: Direction) {
    // Redirect the basis cache to a private database so concurrent tests
    // neither see nor overwrite each other's entries.
    let _guard = TestCacheGuard::new();

    let edges = vec![Edge::new(spin, dir0), Edge::new(spin, dir1)];
    let spec = CGSpec::from_edges(edges).unwrap();

    let data = build_canonical_basis_data(&spec).unwrap();
    let dim = spin.dimension();

    // Shape: [dim, dim, 1]
    assert_eq!(data.shape(), &[dim, dim, 1]);

    let slice = data.slice(ndarray::s![.., .., 0]);

    // Frobenius norm of the slice must be 1.
    let frob_sq: f64 = slice.iter().map(|x| x * x).sum();
    assert_relative_eq!(frob_sq, 1.0, epsilon = 1e-10);

    // slice * slice^T == (1/dim) I for a normalized identity-like basis.
    let expected_diag = 1.0 / (dim as f64);
    for i in 0..dim {
        for k in 0..dim {
            let mut dot = 0.0_f64;
            for m in 0..dim {
                dot += slice[[i, m]] * slice[[k, m]];
            }
            let expected = if i == k { expected_diag } else { 0.0 };
            assert_relative_eq!(dot, expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_two_edges_canonical_j_half() {
    // Canonical (in, out) — plain identity, no metric involved.
    test_two_edges_helper(j(1), Direction::Incoming, Direction::Outgoing);
}

#[test]
fn test_two_edges_canonical_j1() {
    // Same as above at integer spin, where the Frobenius-Schur phase is +1.
    test_two_edges_helper(j(2), Direction::Incoming, Direction::Outgoing);
}

#[test]
fn test_two_edges_canonical_j2() {
    // A larger spin, so the identity being checked is 5x5.
    test_two_edges_helper(j(4), Direction::Incoming, Direction::Outgoing);
}

#[test]
fn test_two_edges_both_incoming_j_half() {
    // (in, in) at half-integer spin: the second index is lowered through the
    // metric, which is antisymmetric here, yet normalization must survive.
    test_two_edges_helper(j(1), Direction::Incoming, Direction::Incoming);
}

#[test]
fn test_two_edges_both_incoming_j1() {
    // (in, in) at integer spin, where the metric is symmetric instead.
    test_two_edges_helper(j(2), Direction::Incoming, Direction::Incoming);
}

#[test]
fn test_two_edges_both_outgoing_j1() {
    // (out, out) — the first index is raised instead, the mirror image of the
    // all-incoming case.
    test_two_edges_helper(j(2), Direction::Outgoing, Direction::Outgoing);
}

#[test]
fn test_two_edges_reversed_j1() {
    // (out, in) — reversed canonical.
    test_two_edges_helper(j(2), Direction::Outgoing, Direction::Incoming);
}

// ============================================================================
// Three-edge orthonormality tests
//
// Three edges is the base case of the fusion-tree recursion: the basis is a
// single Clebsch-Gordan coefficient array, and the outer multiplicity is 1 for
// every admissible spin triple. These tests therefore mainly pin down the
// normalization convention across a range of spins.
// ============================================================================

#[test]
fn test_orthonormality_three_j1() {
    // Three j=1 spins — the smallest all-integer triple.
    assert_canonical_basis_orthonormal(canonical(&[2, 2, 2]));
}

#[test]
fn test_orthonormality_three_j_half() {
    // j=1/2, j=1/2, j=1: two half-integers, so the parity constraint is met.
    assert_canonical_basis_orthonormal(canonical(&[1, 1, 2]));
}

#[test]
fn test_orthonormality_three_j2() {
    // Three j=2 spins — larger dimensions, same structure.
    assert_canonical_basis_orthonormal(canonical(&[4, 4, 4]));
}

#[test]
fn test_orthonormality_three_mixed_1() {
    // j=1/2, j=1, j=1/2 — unequal spins, with the half-integers split apart.
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 1]));
}

#[test]
fn test_orthonormality_three_mixed_2() {
    // j=1, j=2, j=1 — the triangle inequality is saturated here (1 + 1 = 2).
    assert_canonical_basis_orthonormal(canonical(&[2, 4, 2]));
}

#[test]
fn test_orthonormality_three_mixed_3() {
    // j=1/2, j=2, j=3/2 — three distinct spins mixing both parities.
    assert_canonical_basis_orthonormal(canonical(&[1, 4, 3]));
}

// ============================================================================
// Four-edge orthonormality tests
//
// Four edges introduces one internal spin, so the outer multiplicity can
// exceed 1 and the `alpha != beta` half of the orthogonality check starts
// carrying weight.
// ============================================================================

#[test]
fn test_orthonormality_four_j_half() {
    // Four j=1/2 spins — the canonical two-dimensional OM example: the four
    // spin-halves couple to j=0 through either internal spin 0 or 1.
    assert_canonical_basis_orthonormal(canonical(&[1, 1, 1, 1]));
}

#[test]
fn test_orthonormality_four_j1() {
    // Four j=1 spins — a larger OM space (internal spin 0, 1 or 2).
    assert_canonical_basis_orthonormal(canonical(&[2, 2, 2, 2]));
}

#[test]
fn test_orthonormality_four_j2() {
    // Four j=2 spins — the largest uniform four-edge case still cheap to build.
    assert_canonical_basis_orthonormal(canonical(&[4, 4, 4, 4]));
}

#[test]
fn test_orthonormality_four_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 1, 2]));
}

#[test]
fn test_orthonormality_four_mixed_2() {
    // j=1, j=2, j=1, j=2
    assert_canonical_basis_orthonormal(canonical(&[2, 4, 2, 4]));
}

#[test]
fn test_orthonormality_four_mixed_3() {
    // j=1/2, j=1, j=2, j=3/2 — four distinct spins, no symmetry to fall back on.
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 4, 3]));
}

// ============================================================================
// Five-edge orthonormality tests
//
// Two internal spins, so the recursion nests twice and the OM index enumerates
// pairs of internal spins.
// ============================================================================

#[test]
fn test_orthonormality_five_j_half() {
    // Five spins cannot all be half-integer and still couple to j=0, so the
    // terminal edge is raised to j=1: j=1/2 x4, j=1.
    assert_canonical_basis_orthonormal(canonical(&[1, 1, 1, 1, 2]));
}

#[test]
fn test_orthonormality_five_j1() {
    // Five j=1 spins — all integer, so parity is satisfied automatically.
    assert_canonical_basis_orthonormal(canonical(&[2, 2, 2, 2, 2]));
}

#[test]
fn test_orthonormality_five_j2() {
    // Five j=2 spins — the widest OM space exercised at this edge count.
    assert_canonical_basis_orthonormal(canonical(&[4, 4, 4, 4, 4]));
}

#[test]
fn test_orthonormality_five_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1, j=1 — two half-integers, as parity requires.
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 1, 2, 2]));
}

#[test]
fn test_orthonormality_five_mixed_2() {
    // j=1, j=2, j=1, j=2, j=1
    assert_canonical_basis_orthonormal(canonical(&[2, 4, 2, 4, 2]));
}

#[test]
fn test_orthonormality_five_mixed_3() {
    // j=1/2, j=1, j=3/2, j=2, j=1 — half-integers at non-adjacent positions,
    // so they are fused at different depths of the tree.
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 3, 4, 2]));
}

// ============================================================================
// Six-edge orthonormality tests
//
// The deepest recursion covered here, with three internal spins. These are the
// most expensive cases, so the spin values are kept small.
// ============================================================================

#[test]
fn test_orthonormality_six_j_half() {
    // Six j=1/2 spins — an even count, so they couple to j=0 directly.
    assert_canonical_basis_orthonormal(canonical(&[1, 1, 1, 1, 1, 1]));
}

#[test]
fn test_orthonormality_six_j1() {
    // Six j=1 spins.
    assert_canonical_basis_orthonormal(canonical(&[2, 2, 2, 2, 2, 2]));
}

#[test]
fn test_orthonormality_six_j2() {
    // Six j=2 spins.
    assert_canonical_basis_orthonormal(canonical(&[4, 4, 4, 4, 4, 4]));
}

#[test]
fn test_orthonormality_six_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1, j=1/2, j=1/2 — four half-integers, so even.
    assert_canonical_basis_orthonormal(canonical(&[1, 2, 1, 2, 1, 1]));
}

#[test]
fn test_orthonormality_six_mixed_2() {
    // j=1, j=2, j=1, j=2, j=1, j=2
    assert_canonical_basis_orthonormal(canonical(&[2, 4, 2, 4, 2, 4]));
}

#[test]
fn test_orthonormality_six_mixed_3() {
    // j=1, j=1, j=3/2, j=2, j=3/2, j=1 — exactly two half-integers, placed in
    // the middle of the chain.
    assert_canonical_basis_orthonormal(canonical(&[2, 2, 3, 4, 3, 2]));
}

// ============================================================================
// Non-canonical direction tests - Three edges
//
// From here on the spins repeat configurations already covered above, and the
// variable under test is the direction pattern: each edge whose direction
// departs from the canonical "all incoming but the last" form has its index
// raised or lowered through the metric tensor. Orthonormality must be
// invariant under that, because the metric is itself an isometry.
// ============================================================================

#[test]
fn test_orthonormality_three_edges_all_outgoing() {
    // Three j=1 spins, all outgoing: every index is raised.
    assert_canonical_basis_orthonormal(uniform(&[2, 2, 2], Direction::Outgoing));
}

#[test]
fn test_orthonormality_three_edges_all_incoming() {
    // Three edges all incoming: j=1/2, j=1/2, j=1. Only the terminal edge
    // differs from canonical, so this isolates a single index lowering.
    assert_canonical_basis_orthonormal(uniform(&[1, 1, 2], Direction::Incoming));
}

#[test]
fn test_orthonormality_three_edges_out_in_out() {
    // Three j=2 spins: out, in, out — directions alternate at a large spin,
    // where the metric is a 5x5 anti-diagonal.
    assert_canonical_basis_orthonormal(mixed(&[
        (4, Direction::Outgoing),
        (4, Direction::Incoming),
        (4, Direction::Outgoing),
    ]));
}

#[test]
fn test_orthonormality_three_edges_mixed_spins_out_in_out() {
    // Mixed spins and mixed directions together: j=1/2, j=1/2, j=1 as
    // out, in, out. The half-integer metrics are antisymmetric, so this is
    // the case where a lost sign would be easiest to miss.
    assert_canonical_basis_orthonormal(mixed(&[
        (1, Direction::Outgoing),
        (1, Direction::Incoming),
        (2, Direction::Outgoing),
    ]));
}

// ============================================================================
// Non-canonical direction tests - Four edges
// ============================================================================

#[test]
fn test_orthonormality_four_edges_all_outgoing() {
    // Four j=1/2 spins, all outgoing — the OM space is two-dimensional, so the
    // metric must preserve orthogonality between OM slices, not just norms.
    assert_canonical_basis_orthonormal(uniform(&[1, 1, 1, 1], Direction::Outgoing));
}

#[test]
fn test_orthonormality_four_edges_all_incoming() {
    // Four j=1/2 spins, all incoming — the opposite extreme of the same spec.
    assert_canonical_basis_orthonormal(uniform(&[1, 1, 1, 1], Direction::Incoming));
}

#[test]
fn test_orthonormality_four_edges_alternating() {
    // Four j=1 spins: out, in, out, in.
    assert_canonical_basis_orthonormal(mixed(&[
        (2, Direction::Outgoing),
        (2, Direction::Incoming),
        (2, Direction::Outgoing),
        (2, Direction::Incoming),
    ]));
}

// ============================================================================
// Non-canonical direction tests - Five edges
// ============================================================================

#[test]
fn test_orthonormality_five_edges_all_incoming() {
    // j=1/2 x4, j=1 all incoming — an even number of half-integer spins, as
    // parity requires, now with every index lowered.
    assert_canonical_basis_orthonormal(uniform(&[1, 1, 1, 1, 2], Direction::Incoming));
}

#[test]
fn test_orthonormality_five_edges_mixed_complex() {
    // Five j=1 spins: in, out, in, out, in.
    assert_canonical_basis_orthonormal(mixed(&[
        (2, Direction::Incoming),
        (2, Direction::Outgoing),
        (2, Direction::Incoming),
        (2, Direction::Outgoing),
        (2, Direction::Incoming),
    ]));
}

// ============================================================================
// Non-canonical direction tests - Six edges
// ============================================================================

#[test]
fn test_orthonormality_six_edges_all_outgoing() {
    // Six j=1/2 spins, all outgoing.
    assert_canonical_basis_orthonormal(uniform(&[1, 1, 1, 1, 1, 1], Direction::Outgoing));
}

#[test]
fn test_orthonormality_six_edges_alternating_pattern() {
    // Six j=1 spins: out, out, in, in, out, out — directions change in blocks,
    // so the reversals do not line up with the pairs that the fusion tree
    // groups together.
    assert_canonical_basis_orthonormal(mixed(&[
        (2, Direction::Outgoing),
        (2, Direction::Outgoing),
        (2, Direction::Incoming),
        (2, Direction::Incoming),
        (2, Direction::Outgoing),
        (2, Direction::Outgoing),
    ]));
}
