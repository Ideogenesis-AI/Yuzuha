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

//! Clebsch-Gordan coefficients for SU(2)
//!
//! Implements real-valued Condon-Shortley convention CG coefficients:
//! ⟨j₁ m₁, j₂ m₂ | j₃ m₃⟩
//!
//! Using the Racah formula for efficient computation with caching.

use crate::core::{MagneticNumber, Spin};
use crate::primitives::triangle::triangle_check;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

/// Cache for computed CG coefficients
static CG_CACHE: Lazy<Mutex<HashMap<(i32, i32, i32, i32, i32, i32), f64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Compute Clebsch-Gordan coefficient ⟨j₁ m₁, j₂ m₂ | j₃ m₃⟩
///
/// Uses the Racah formula with Condon-Shortley phase convention.
/// Results are cached for performance.
///
/// # Arguments
/// * `j1`, `m1` - First angular momentum and its projection
/// * `j2`, `m2` - Second angular momentum and its projection
/// * `j3`, `m3` - Coupled angular momentum and its projection
///
/// # Returns
/// The CG coefficient value (real-valued)
pub fn clebsch_gordan(
    j1: Spin,
    m1: MagneticNumber,
    j2: Spin,
    m2: MagneticNumber,
    j3: Spin,
    m3: MagneticNumber,
) -> f64 {
    // Check selection rules first (fast rejection)
    if !selection_rules(j1, m1, j2, m2, j3, m3) {
        return 0.0;
    }

    let key = (
        j1.twice(),
        m1.twice(),
        j2.twice(),
        m2.twice(),
        j3.twice(),
        m3.twice(),
    );

    // Check cache
    {
        let cache = CG_CACHE.lock().unwrap();
        if let Some(&value) = cache.get(&key) {
            return value;
        }
    }

    // Compute using Racah formula
    let value = compute_cg_racah(j1, m1, j2, m2, j3, m3);

    // Store in cache
    {
        let mut cache = CG_CACHE.lock().unwrap();
        cache.insert(key, value);
    }

    value
}

/// Check selection rules for CG coefficients
#[inline]
fn selection_rules(
    j1: Spin,
    m1: MagneticNumber,
    j2: Spin,
    m2: MagneticNumber,
    j3: Spin,
    m3: MagneticNumber,
) -> bool {
    // Magnetic number conservation
    if m1.twice() + m2.twice() != m3.twice() {
        return false;
    }

    // Triangle inequality
    if !triangle_check(j1, j2, j3) {
        return false;
    }

    true
}

/// Compute CG coefficient using Racah formula
fn compute_cg_racah(
    j1: Spin,
    m1: MagneticNumber,
    j2: Spin,
    m2: MagneticNumber,
    j3: Spin,
    m3: MagneticNumber,
) -> f64 {
    let j1_val = j1.twice() as f64 / 2.0;
    let m1_val = m1.twice() as f64 / 2.0;
    let j2_val = j2.twice() as f64 / 2.0;
    let m2_val = m2.twice() as f64 / 2.0;
    let j3_val = j3.twice() as f64 / 2.0;
    let m3_val = m3.twice() as f64 / 2.0;

    // Special case: j1 = 0
    if j1.twice() == 0 {
        if j2.twice() == j3.twice() && m2.twice() == m3.twice() {
            return 1.0;
        }
        return 0.0;
    }

    // Special case: j2 = 0
    if j2.twice() == 0 {
        if j1.twice() == j3.twice() && m1.twice() == m3.twice() {
            return 1.0;
        }
        return 0.0;
    }

    // Compute normalization factor
    let delta_factor = delta(j1_val, j2_val, j3_val);
    if delta_factor == 0.0 {
        return 0.0;
    }

    let norm = delta_factor
        * ((2.0 * j3_val + 1.0)
            * factorial_f64((j3_val + m3_val) as i32)
            * factorial_f64((j3_val - m3_val) as i32)
            * factorial_f64((j1_val - m1_val) as i32)
            * factorial_f64((j1_val + m1_val) as i32)
            * factorial_f64((j2_val - m2_val) as i32)
            * factorial_f64((j2_val + m2_val) as i32))
        .sqrt();

    // Sum over k in Racah formula
    let k_min = [
        0,
        (j2_val - j3_val - m1_val) as i32,
        (j1_val - j3_val + m2_val) as i32,
    ]
    .iter()
    .copied()
    .max()
    .unwrap();

    let k_max = [
        (j1_val + j2_val - j3_val) as i32,
        (j1_val - m1_val) as i32,
        (j2_val + m2_val) as i32,
    ]
    .iter()
    .copied()
    .min()
    .unwrap();

    let mut sum = 0.0;
    for k in k_min..=k_max {
        let term = if k % 2 == 0 { 1.0 } else { -1.0 };
        let denom = factorial_f64(k)
            * factorial_f64((j1_val + j2_val - j3_val) as i32 - k)
            * factorial_f64((j1_val - m1_val) as i32 - k)
            * factorial_f64((j2_val + m2_val) as i32 - k)
            * factorial_f64((j3_val - j2_val + m1_val) as i32 + k)
            * factorial_f64((j3_val - j1_val - m2_val) as i32 + k);

        sum += term / denom;
    }

    norm * sum
}

/// Triangle coefficient (delta function) for Racah formula
fn delta(j1: f64, j2: f64, j3: f64) -> f64 {
    let num = factorial_f64((j1 + j2 - j3) as i32)
        * factorial_f64((j1 - j2 + j3) as i32)
        * factorial_f64((-j1 + j2 + j3) as i32);
    let denom = factorial_f64((j1 + j2 + j3 + 1.0) as i32);

    (num / denom).sqrt()
}

/// Factorial function (works with half-integers via doubling)
fn factorial_f64(n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    if n == 0 {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 1..=n {
        result *= i as f64;
    }
    result
}

/// Clear the CG coefficient cache (useful for testing)
#[allow(dead_code)]
pub fn clear_cache() {
    let mut cache = CG_CACHE.lock().unwrap();
    cache.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_selection_rules() {
        let j1 = Spin::new(2).unwrap(); // j=1
        let m1 = MagneticNumber::new_unchecked(2);

        let j2 = Spin::new(2).unwrap();
        let m2 = MagneticNumber::new_unchecked(0);

        let j3 = Spin::new(2).unwrap();
        let m3 = MagneticNumber::new_unchecked(2);

        // Valid: m1 + m2 = m3 and triangle satisfied
        assert!(selection_rules(j1, m1, j2, m2, j3, m3));

        // Invalid: m conservation violated
        let m3_bad = MagneticNumber::new_unchecked(0);
        assert!(!selection_rules(j1, m1, j2, m2, j3, m3_bad));
    }

    #[test]
    fn test_cg_special_cases() {
        // j=0 coupling: should give delta functions
        let j0 = Spin::new(0).unwrap();
        let m0 = MagneticNumber::new_unchecked(0);

        let j1 = Spin::new(2).unwrap();
        let m1 = MagneticNumber::new_unchecked(2);

        // ⟨0,0; j,m | j,m⟩ = 1
        let cg = clebsch_gordan(j0, m0, j1, m1, j1, m1);
        assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cg_spin_half() {
        // j=1/2 ⊗ j=1/2 system
        let j_half = Spin::new(1).unwrap();
        let m_up = MagneticNumber::new_unchecked(1);
        let m_down = MagneticNumber::new_unchecked(-1);

        let j0 = Spin::new(0).unwrap();
        let m0 = MagneticNumber::new_unchecked(0);

        let j1 = Spin::new(2).unwrap();

        // Singlet: |↑↓⟩ - |↓↑⟩ / √2
        // ⟨1/2,1/2; 1/2,-1/2 | 0,0⟩ = 1/√2
        let cg_singlet = clebsch_gordan(j_half, m_up, j_half, m_down, j0, m0);
        assert_relative_eq!(cg_singlet, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);

        // Triplet: |↑↓⟩ + |↓↑⟩ / √2
        // ⟨1/2,1/2; 1/2,-1/2 | 1,0⟩ = 1/√2
        let m1_0 = MagneticNumber::new_unchecked(0);
        let cg_triplet = clebsch_gordan(j_half, m_up, j_half, m_down, j1, m1_0);
        assert_relative_eq!(cg_triplet, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_cg_orthonormality() {
        // Test orthonormality: Σ_m1,m2 CG(j1,m1,j2,m2,j3,m3) * CG(j1,m1,j2,m2,j3',m3') = δ_j3,j3' δ_m3,m3'
        let j_half = Spin::new(1).unwrap();
        let j0 = Spin::new(0).unwrap();
        let j1 = Spin::new(2).unwrap();

        // Sum over all m1, m2 for j3=j3'=0, m3=m3'=0
        let m0 = MagneticNumber::new_unchecked(0);
        let mut sum = 0.0;

        for m1_val in [-1, 1] {
            for m2_val in [-1, 1] {
                if m1_val + m2_val == 0 {
                    let m1 = MagneticNumber::new_unchecked(m1_val);
                    let m2 = MagneticNumber::new_unchecked(m2_val);
                    let cg = clebsch_gordan(j_half, m1, j_half, m2, j0, m0);
                    sum += cg * cg;
                }
            }
        }

        // Should sum to 1 (normalized)
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check orthogonality: j3=0 vs j3=1
        sum = 0.0;
        for m1_val in [-1, 1] {
            for m2_val in [-1, 1] {
                if m1_val + m2_val == 0 {
                    let m1 = MagneticNumber::new_unchecked(m1_val);
                    let m2 = MagneticNumber::new_unchecked(m2_val);
                    let cg1 = clebsch_gordan(j_half, m1, j_half, m2, j0, m0);
                    let cg2 = clebsch_gordan(j_half, m1, j_half, m2, j1, m0);
                    sum += cg1 * cg2;
                }
            }
        }

        // Should be 0 (orthogonal)
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cg_caching() {
        clear_cache();

        let j1 = Spin::new(2).unwrap();
        let m1 = MagneticNumber::new_unchecked(2);
        let j2 = Spin::new(2).unwrap();
        let m2 = MagneticNumber::new_unchecked(0);
        let j3 = Spin::new(2).unwrap();
        let m3 = MagneticNumber::new_unchecked(2);

        // Compute twice - second should be from cache
        let cg1 = clebsch_gordan(j1, m1, j2, m2, j3, m3);
        let cg2 = clebsch_gordan(j1, m1, j2, m2, j3, m3);

        assert_eq!(cg1, cg2);
    }
}
