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
