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

//! Frobenius-Schur (FS) phase factor for X-symbol contractions.
//!
//! When contracting two CGTs, a pair of contracted edges may carry a FS phase
//! of `(-1)^{2j}`.  The triggering condition depends on which canonical region
//! the contracted pair occupies:
//!
//! - **First region** (both axes `< n-1`): phase applies when directions are
//!   `(Incoming, Outgoing)`.
//! - **Last edge** (both axes `== n-1`): phase applies when directions are
//!   `(Outgoing, Incoming)`.

use crate::core::{CGSpec, Contraction, Direction};

/// Compute the overall Frobenius-Schur (FS) phase factor for a contraction.
///
/// For each contracted pair `(axis_a, axis_b)`, a factor of `(-1)^{2j}` is
/// accumulated when the pair is half-integer and satisfies:
///
/// - **First region** (`axis_a < n_a - 1` and `axis_b < n_b - 1`):
///   directions are `(Incoming, Outgoing)`.
/// - **Last edge** (`axis_a == n_a - 1` and `axis_b == n_b - 1`):
///   directions are `(Outgoing, Incoming)`.
///
/// Returns `+1.0` or `-1.0`.
pub fn compute_fs_phase(
    spec_a: &CGSpec,
    spec_b: &CGSpec,
    contraction: &Contraction,
) -> f64 {
    let n_a = spec_a.num_external();
    let n_b = spec_b.num_external();
    let mut phase = 1.0_f64;

    for (&axis_a, &axis_b) in contraction.axes_a.iter().zip(contraction.axes_b.iter()) {
        let edge_a = &spec_a.edges[axis_a];
        let edge_b = &spec_b.edges[axis_b];

        let both_first_region = axis_a < n_a - 1 && axis_b < n_b - 1;
        let both_last = axis_a == n_a - 1 && axis_b == n_b - 1;

        let applies = (both_first_region
            && edge_a.dir == Direction::Incoming
            && edge_b.dir == Direction::Outgoing)
            || (both_last
                && edge_a.dir == Direction::Outgoing
                && edge_b.dir == Direction::Incoming);

        if applies {
            // (-1)^{2j}: flip sign only for half-integer spins
            if edge_a.j.twice() % 2 != 0 {
                phase *= -1.0;
            }
        }
    }

    phase
}
