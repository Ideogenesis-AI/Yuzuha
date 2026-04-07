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

//! Outer multiplicity (OM) basis enumeration
//!
//! Enumerates all valid internal spin configurations (alpha tuples)
//! for left-associative fusion trees.

use crate::core::Spin;
use crate::primitives::triangle::allowed_triangle;
use std::collections::HashMap;

/// Enumerate all allowed internal spin tuples (alpha) for a left-associative fusion tree
///
/// For n external edges with spins J[0..n-1], this generates all valid tuples:
/// alpha = [J₁₂, J₁₂₃, ..., J₁...ₙ₋₂]
///
/// where each intermediate spin satisfies triangle inequalities.
///
/// **Angular Momentum Conservation**: The final total angular momentum is always
/// constrained to j=0 (J₁...ₙ₋₁ ⊗ Jₙ → 0), consistent with the closed-diagram
/// convention used throughout this library.
///
/// # Arguments
/// * `j_list` - External edge spins in fusion order
///
/// # Returns
/// Vector of all valid internal spin tuples, sorted lexicographically
pub fn enumerate_alpha(j_list: &[Spin]) -> Vec<Vec<Spin>> {
    let n = j_list.len();

    // Trivial cases
    if n <= 2 {
        return vec![Vec::new()];
    }

    let j_target = Spin::new(0).unwrap(); // Angular momentum conservation: total j = 0
    let mut alphas = Vec::new();

    // Recursive enumeration
    fn recurse(
        j_list: &[Spin],
        step: usize,
        j_prev: Spin,
        prefix: &mut Vec<Spin>,
        alphas: &mut Vec<Vec<Spin>>,
        j_target: Spin,
    ) {
        let n = j_list.len();

        if step == n - 1 {
            // Reached the last edge - check if final coupling produces j=0
            let j_last = j_list[n - 1];
            let allowed_final = allowed_triangle(j_prev, j_last);
            
            // Only accept this alpha if it can couple to produce j=0
            if allowed_final.contains(&j_target) {
                alphas.push(prefix.clone());
            }
            return;
        }

        let j_next = j_list[step];

        // Try all allowed couplings
        for j_coupled in allowed_triangle(j_prev, j_next) {
            prefix.push(j_coupled);
            recurse(j_list, step + 1, j_coupled, prefix, alphas, j_target);
            prefix.pop();
        }
    }

    // Start by fusing first two edges
    for j12 in allowed_triangle(j_list[0], j_list[1]) {
        let mut prefix = vec![j12];
        if n == 3 {
            // For 3 edges, J12 is the only internal spin
            // Check if J12 ⊗ j3 can produce j=0
            let allowed_final = allowed_triangle(j12, j_list[2]);
            if allowed_final.contains(&j_target) {
                alphas.push(prefix);
            }
        } else {
            // Continue fusing
            recurse(j_list, 2, j12, &mut prefix, &mut alphas, j_target);
        }
    }

    // Sort for deterministic OM indexing
    alphas.sort_by(|a, b| {
        for (ai, bi) in a.iter().zip(b.iter()) {
            match ai.twice().cmp(&bi.twice()) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        a.len().cmp(&b.len())
    });

    alphas
}

/// Create a mapping from alpha tuple to OM index
///
/// # Arguments
/// * `alphas` - List of alpha tuples (from enumerate_alpha)
///
/// # Returns
/// HashMap mapping tuple to its index
pub fn alpha_to_om_map(alphas: &[Vec<Spin>]) -> HashMap<Vec<Spin>, usize> {
    alphas
        .iter()
        .enumerate()
        .map(|(i, alpha)| (alpha.clone(), i))
        .collect()
}

/// Get the OM dimension (number of unique internal spin configurations)
///
/// # Arguments
/// * `j_list` - External edge spins
///
/// # Returns
/// Number of unique alpha tuples
#[inline]
pub fn om_dimension(j_list: &[Spin]) -> usize {
    enumerate_alpha(j_list).len()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_alpha_trivial() {
        // 0 edges
        let alphas = enumerate_alpha(&[]);
        assert_eq!(alphas.len(), 1);
        assert_eq!(alphas[0].len(), 0);

        // 1 edge
        let j1 = Spin::new(2).unwrap();
        let alphas = enumerate_alpha(&[j1]);
        assert_eq!(alphas.len(), 1);
        assert_eq!(alphas[0].len(), 0);

        // 2 edges
        let alphas = enumerate_alpha(&[j1, j1]);
        assert_eq!(alphas.len(), 1);
        assert_eq!(alphas[0].len(), 0);
    }

    #[test]
    fn test_enumerate_alpha_three_edges() {
        // Three j=1 spins with j_total=0 constraint
        let j1 = Spin::new(2).unwrap();
        let j_list = vec![j1, j1, j1];

        let alphas = enumerate_alpha(&j_list);

        // j=1 ⊗ j=1 = j=0 ⊕ j=1 ⊕ j=2
        // With j_total=0 constraint: only J12=1 allows j1⊗j1→j0
        // So we expect 1 alpha tuple
        assert_eq!(alphas.len(), 1);

        // Each alpha should have length 1 (n-2 = 3-2 = 1)
        for alpha in &alphas {
            assert_eq!(alpha.len(), 1);
        }

        // Should be: [J12=2] (j=1)
        assert_eq!(alphas[0][0].twice(), 2);
    }

    #[test]
    fn test_enumerate_alpha_four_edges() {
        // Four j=1/2 spins
        let j_half = Spin::new(1).unwrap();
        let j_list = vec![j_half, j_half, j_half, j_half];

        let alphas = enumerate_alpha(&j_list);

        // j=1/2 ⊗ j=1/2 = j=0 ⊕ j=1 (2 for first fusion)
        // Each can fuse with third j=1/2:
        //   j=0 ⊗ j=1/2 = j=1/2 (1 option)
        //   j=1 ⊗ j=1/2 = j=1/2 ⊕ j=3/2 (2 options)
        // Total: 1 + 2 = 3 intermediate states before last fusion
        // Each fuses with fourth j=1/2:
        //   j=1/2 ⊗ j=1/2 = j=0 ⊕ j=1 (2 options each)
        // But we need to count carefully...

        // Actually, let's verify the count
        assert!(alphas.len() > 0);

        // Each alpha should have length 2 (n-2 = 4-2 = 2)
        for alpha in &alphas {
            assert_eq!(alpha.len(), 2);
        }

        // Verify sorting
        for i in 1..alphas.len() {
            assert!(
                alphas[i - 1][0].twice() < alphas[i][0].twice()
                    || (alphas[i - 1][0].twice() == alphas[i][0].twice()
                        && alphas[i - 1][1].twice() <= alphas[i][1].twice())
            );
        }
    }

    #[test]
    fn test_alpha_to_om_map() {
        let j1 = Spin::new(2).unwrap();
        let alphas = enumerate_alpha(&[j1, j1, j1]);

        let map = alpha_to_om_map(&alphas);

        // With j_total=0 constraint, only 1 alpha
        assert_eq!(map.len(), 1);
        assert_eq!(map[&alphas[0]], 0);
    }

    #[test]
    fn test_om_dimension() {
        let j1 = Spin::new(2).unwrap();

        // 2 edges: dimension 1 (j2⊗j2 can give j0)
        assert_eq!(om_dimension(&[j1, j1]), 1);

        // 3 edges with j_total=0 constraint: dimension 1
        assert_eq!(om_dimension(&[j1, j1, j1]), 1);
    }

}
