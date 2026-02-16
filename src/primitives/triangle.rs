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

//! Triangle rules for SU(2) angular momentum coupling
//!
//! The triangle inequality constrains which angular momenta can couple:
//! |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂, with j₁ + j₂ + j₃ even

use crate::core::Spin;

/// Check if three angular momenta satisfy the triangle inequality
///
/// For SU(2), three spins can couple if:
/// - |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
/// - j₁ + j₂ + j₃ is even (integer)
///
/// # Arguments
/// * `j1`, `j2`, `j3` - The three spin quantum numbers
///
/// # Returns
/// `true` if the triangle inequality is satisfied
pub fn triangle_check(j1: Spin, j2: Spin, j3: Spin) -> bool {
    let j1_val = j1.twice();
    let j2_val = j2.twice();
    let j3_val = j3.twice();

    // Check parity: sum must be even
    if (j1_val + j2_val + j3_val) % 2 != 0 {
        return false;
    }

    // Check triangle inequality
    let min_j3 = (j1_val - j2_val).abs();
    let max_j3 = j1_val + j2_val;

    j3_val >= min_j3 && j3_val <= max_j3
}

/// Generate all allowed spins from coupling j₁ and j₂
///
/// Returns all valid j₃ values such that |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
/// with j₃ stepping by 2 (in doubled units).
///
/// # Arguments
/// * `j1`, `j2` - The two spin quantum numbers to couple
///
/// # Returns
/// Vector of allowed resultant spins, in ascending order
pub fn allowed_triangle(j1: Spin, j2: Spin) -> Vec<Spin> {
    let j1_val = j1.twice();
    let j2_val = j2.twice();

    let min_j = (j1_val - j2_val).abs();
    let max_j = j1_val + j2_val;

    let mut result = Vec::new();
    let mut j = min_j;
    while j <= max_j {
        // Safety: j is guaranteed to be non-negative
        result.push(Spin::new(j).unwrap());
        j += 2;
    }

    result
}

/// Get the minimum allowed spin from coupling j₁ and j₂
#[inline]
pub fn min_coupled_spin(j1: Spin, j2: Spin) -> Spin {
    let min = (j1.twice() - j2.twice()).abs();
    Spin::new(min).unwrap()
}

/// Get the maximum allowed spin from coupling j₁ and j₂
#[inline]
pub fn max_coupled_spin(j1: Spin, j2: Spin) -> Spin {
    let max = j1.twice() + j2.twice();
    Spin::new(max).unwrap()
}

/// Count the number of allowed coupled spins
///
/// For j₁ and j₂, this returns the dimension of the outer multiplicity space.
#[inline]
pub fn num_allowed_spins(j1: Spin, j2: Spin) -> usize {
    let min = (j1.twice() - j2.twice()).abs();
    let max = j1.twice() + j2.twice();
    ((max - min) / 2 + 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_check() {
        let j0 = Spin::new(0).unwrap();
        let j1 = Spin::new(2).unwrap(); // j=1
        let j2 = Spin::new(4).unwrap(); // j=2

        // Valid triangles
        assert!(triangle_check(j1, j1, j0));
        assert!(triangle_check(j1, j1, j1));
        assert!(triangle_check(j1, j1, j2));
        assert!(triangle_check(j1, j2, j1));
        assert!(triangle_check(j1, j2, Spin::new(6).unwrap())); // j=3

        // Invalid: violates inequality
        assert!(!triangle_check(j0, j1, j2));

        // Invalid: wrong parity (half-integer sum)
        let j_half = Spin::new(1).unwrap(); // j=1/2
        assert!(!triangle_check(j_half, j1, j1)); // 1/2 + 1 + 1 = 5/2 is half-integer
    }

    #[test]
    fn test_allowed_triangle() {
        let j0 = Spin::new(0).unwrap();
        let j1 = Spin::new(2).unwrap(); // j=1
        let j2 = Spin::new(4).unwrap(); // j=2

        // j=1 ⊗ j=1 = j=0 ⊕ j=1 ⊕ j=2
        let allowed = allowed_triangle(j1, j1);
        assert_eq!(allowed.len(), 3);
        assert_eq!(allowed[0].twice(), 0);
        assert_eq!(allowed[1].twice(), 2);
        assert_eq!(allowed[2].twice(), 4);

        // j=1 ⊗ j=2 = j=1 ⊕ j=2 ⊕ j=3
        let allowed = allowed_triangle(j1, j2);
        assert_eq!(allowed.len(), 3);
        assert_eq!(allowed[0].twice(), 2);
        assert_eq!(allowed[1].twice(), 4);
        assert_eq!(allowed[2].twice(), 6);

        // j=0 ⊗ j=1 = j=1 (singlet coupling)
        let allowed = allowed_triangle(j0, j1);
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0].twice(), 2);
    }

    #[test]
    fn test_min_max_coupled_spin() {
        let j1 = Spin::new(2).unwrap(); // j=1
        let j2 = Spin::new(4).unwrap(); // j=2

        assert_eq!(min_coupled_spin(j1, j2).twice(), 2); // |1-2| = 1
        assert_eq!(max_coupled_spin(j1, j2).twice(), 6); // 1+2 = 3
    }

    #[test]
    fn test_num_allowed_spins() {
        let j1 = Spin::new(2).unwrap(); // j=1
        let j2 = Spin::new(4).unwrap(); // j=2

        assert_eq!(num_allowed_spins(j1, j1), 3); // 0, 1, 2
        assert_eq!(num_allowed_spins(j1, j2), 3); // 1, 2, 3

        let j_half = Spin::new(1).unwrap(); // j=1/2
        assert_eq!(num_allowed_spins(j_half, j_half), 2); // 0, 1
    }
}
