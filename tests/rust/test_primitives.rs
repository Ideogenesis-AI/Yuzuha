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

//! Comprehensive tests for primitive operations

use approx::assert_relative_eq;
use yuzuha::primitives::*;
use yuzuha::{MagneticNumber, Spin};

#[test]
fn test_cg_symmetries() {
    // Test CG coefficient symmetry relations
    let j_half = Spin::new(1).unwrap();
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();

    let m_up = MagneticNumber::new_unchecked(1);
    let m_down = MagneticNumber::new_unchecked(-1);
    let m0 = MagneticNumber::new_unchecked(0);

    // Test ⟨1/2,1/2; 1/2,-1/2 | 0,0⟩ = 1/√2
    let cg1 = clebsch_gordan(j_half, m_up, j_half, m_down, j0, m0);
    assert_relative_eq!(cg1, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);

    // Test ⟨1/2,1/2; 1/2,-1/2 | 1,0⟩ = 1/√2
    let cg2 = clebsch_gordan(j_half, m_up, j_half, m_down, j1, m0);
    assert_relative_eq!(cg2, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);

    // Test ⟨1/2,-1/2; 1/2,1/2 | 1,0⟩ should have opposite sign for triplet
    let cg3 = clebsch_gordan(j_half, m_down, j_half, m_up, j1, m0);
    assert_relative_eq!(cg3, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
}

#[test]
fn test_cg_completeness() {
    // Sum rule: Σ_j3,m3 |⟨j1,m1; j2,m2 | j3,m3⟩|² = 1
    // For FIXED m1 and m2, sum over all j3, m3
    let j_half = Spin::new(1).unwrap();
    let m1 = MagneticNumber::new_unchecked(1);  // 1/2
    let m2 = MagneticNumber::new_unchecked(-1); // -1/2

    let mut sum = 0.0;

    // j=1/2 ⊗ j=1/2 couples to j=0 or j=1
    for j3 in [Spin::new(0).unwrap(), Spin::new(2).unwrap()] {
        for m3_val in -j3.twice()..=j3.twice() {
            if m3_val % 2 != j3.twice() % 2 {
                continue;
            }
            let m3 = MagneticNumber::new_unchecked(m3_val);

            let cg = clebsch_gordan(j_half, m1, j_half, m2, j3, m3);
            sum += cg * cg;
        }
    }

    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

#[test]
fn test_metric_involution() {
    // g² = I: applying metric twice gives identity
    let j1 = Spin::new(2).unwrap();

    for m_val in [-2, 0, 2] {
        let m = MagneticNumber::new_unchecked(m_val);
        let m_neg = m.negate();

        let g1 = g(j1, m, m_neg);
        let g2 = g(j1, m_neg, m);

        assert_relative_eq!(g1 * g2, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_triangle_inequality_edge_cases() {
    // Test edge cases of triangle inequality
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    let j3 = Spin::new(6).unwrap();

    // Valid: j1 + j2 = j3
    assert!(triangle_check(j1, j2, j3));

    // Valid: |j1 - j2| = j0
    assert!(!triangle_check(j0, j1, j2)); // But parity wrong!

    // Invalid: j3 > j1 + j2
    let j4 = Spin::new(8).unwrap();
    assert!(!triangle_check(j1, j2, j4));
}

#[test]
fn test_allowed_triangle_count() {
    // Verify number of allowed coupled spins
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();

    let allowed = allowed_triangle(j1, j2);

    // |1-2| = 1 to 1+2 = 3, so j ∈ {1,2,3} → 3 values
    assert_eq!(allowed.len(), 3);
    assert_eq!(allowed[0].twice(), 2);
    assert_eq!(allowed[1].twice(), 4);
    assert_eq!(allowed[2].twice(), 6);
}

#[test]
fn test_cg_special_values() {
    // Test some known special values

    // j=0 coupling is trivial
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();
    let m0 = MagneticNumber::new_unchecked(0);
    let m1 = MagneticNumber::new_unchecked(2);

    let cg = clebsch_gordan(j0, m0, j1, m1, j1, m1);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);

    // Highest weight coupling
    let j_half = Spin::new(1).unwrap();
    let m_up = MagneticNumber::new_unchecked(1);
    let j1_coupled = Spin::new(2).unwrap();
    let m1_coupled = MagneticNumber::new_unchecked(2);

    let cg = clebsch_gordan(j_half, m_up, j_half, m_up, j1_coupled, m1_coupled);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
}

// ========================================================================
// COMPREHENSIVE CG COEFFICIENT TESTS WITH ANALYTIC RESULTS
// ========================================================================

#[test]
fn test_spin_half_tensor_spin_half_complete() {
    // Complete test of j=1/2 ⊗ j=1/2 → j=0,1
    // This system has all analytically known coefficients
    
    let j_half = Spin::new(1).unwrap(); // j=1/2 (twice=1)
    let j0 = Spin::new(0).unwrap();     // j=0 (twice=0)
    let j1 = Spin::new(2).unwrap();     // j=1 (twice=2)
    
    let m_up = MagneticNumber::new_unchecked(1);    // m=+1/2
    let m_down = MagneticNumber::new_unchecked(-1); // m=-1/2
    let m0 = MagneticNumber::new_unchecked(0);      // m=0
    
    // ====== Singlet state (j=0, m=0) ======
    // |0,0⟩ = (|↑↓⟩ - |↓↑⟩)/√2
    
    // ⟨1/2,1/2; 1/2,-1/2 | 0,0⟩ = 1/√2
    let cg = clebsch_gordan(j_half, m_up, j_half, m_down, j0, m0);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 1/2,1/2 | 0,0⟩ = -1/√2
    let cg = clebsch_gordan(j_half, m_down, j_half, m_up, j0, m0);
    assert_relative_eq!(cg, -1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ====== Triplet states (j=1) ======
    
    // |1,1⟩ = |↑↑⟩
    // ⟨1/2,1/2; 1/2,1/2 | 1,1⟩ = 1
    let m_up_2 = MagneticNumber::new_unchecked(2); // m=1
    let cg = clebsch_gordan(j_half, m_up, j_half, m_up, j1, m_up_2);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
    
    // |1,0⟩ = (|↑↓⟩ + |↓↑⟩)/√2
    // ⟨1/2,1/2; 1/2,-1/2 | 1,0⟩ = 1/√2
    let cg = clebsch_gordan(j_half, m_up, j_half, m_down, j1, m0);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 1/2,1/2 | 1,0⟩ = 1/√2
    let cg = clebsch_gordan(j_half, m_down, j_half, m_up, j1, m0);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // |1,-1⟩ = |↓↓⟩
    // ⟨1/2,-1/2; 1/2,-1/2 | 1,-1⟩ = 1
    let m_down_2 = MagneticNumber::new_unchecked(-2); // m=-1
    let cg = clebsch_gordan(j_half, m_down, j_half, m_down, j1, m_down_2);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
}

#[test]
fn test_spin_half_tensor_spin_one() {
    // Test j=1/2 ⊗ j=1 → j=1/2, 3/2
    // Many analytic results available
    
    let j_half = Spin::new(1).unwrap();   // j=1/2
    let j1 = Spin::new(2).unwrap();       // j=1
    let j_half_out = Spin::new(1).unwrap(); // j=1/2
    let j_3half = Spin::new(3).unwrap();  // j=3/2
    
    // Test highest weight: ⟨1/2,1/2; 1,1 | 3/2,3/2⟩ = 1
    let m_half = MagneticNumber::new_unchecked(1);   // 1/2
    let m_1 = MagneticNumber::new_unchecked(2);      // 1
    let m_3half = MagneticNumber::new_unchecked(3);  // 3/2
    
    let cg = clebsch_gordan(j_half, m_half, j1, m_1, j_3half, m_3half);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
    
    // Test: ⟨1/2,1/2; 1,0 | 3/2,1/2⟩ = √(2/3)
    let m0 = MagneticNumber::new_unchecked(0);
    let m_half_out = MagneticNumber::new_unchecked(1);
    
    let cg = clebsch_gordan(j_half, m_half, j1, m0, j_3half, m_half_out);
    assert_relative_eq!(cg, (2.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // Test: ⟨1/2,-1/2; 1,1 | 3/2,1/2⟩ = √(1/3)
    let m_minus_half = MagneticNumber::new_unchecked(-1);
    
    let cg = clebsch_gordan(j_half, m_minus_half, j1, m_1, j_3half, m_half_out);
    assert_relative_eq!(cg, (1.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // Test: ⟨1/2,1/2; 1,0 | 1/2,1/2⟩ = √(1/3)
    let cg = clebsch_gordan(j_half, m_half, j1, m0, j_half_out, m_half_out);
    assert_relative_eq!(cg, (1.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // Test: ⟨1/2,-1/2; 1,1 | 1/2,1/2⟩ = -√(2/3)
    let cg = clebsch_gordan(j_half, m_minus_half, j1, m_1, j_half_out, m_half_out);
    assert_relative_eq!(cg, -(2.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_spin_one_tensor_spin_one() {
    // Test j=1 ⊗ j=1 → j=0,1,2
    // Important system with known analytic results
    
    let j1 = Spin::new(2).unwrap();  // j=1
    let j0 = Spin::new(0).unwrap();  // j=0
    let j2 = Spin::new(4).unwrap();  // j=2
    
    let m1 = MagneticNumber::new_unchecked(2);    // m=1
    let m0 = MagneticNumber::new_unchecked(0);    // m=0
    let m_neg1 = MagneticNumber::new_unchecked(-2); // m=-1
    
    // ====== j=2 (quintet) ======
    
    // Highest weight: ⟨1,1; 1,1 | 2,2⟩ = 1
    let m2 = MagneticNumber::new_unchecked(4); // m=2
    let cg = clebsch_gordan(j1, m1, j1, m1, j2, m2);
    assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
    
    // ⟨1,1; 1,0 | 2,1⟩ = 1/√2
    let cg = clebsch_gordan(j1, m1, j1, m0, j2, m1);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 1,1 | 2,1⟩ = 1/√2
    let cg = clebsch_gordan(j1, m0, j1, m1, j2, m1);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 1,-1 | 2,0⟩ = 1/√6
    let cg = clebsch_gordan(j1, m1, j1, m_neg1, j2, m0);
    assert_relative_eq!(cg, 1.0 / 6.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 1,0 | 2,0⟩ = √(2/3)
    let cg = clebsch_gordan(j1, m0, j1, m0, j2, m0);
    assert_relative_eq!(cg, (2.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // ====== j=1 (triplet) ======
    
    // ⟨1,1; 1,0 | 1,1⟩ = 1/√2
    let cg = clebsch_gordan(j1, m1, j1, m0, j1, m1);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 1,1 | 1,1⟩ = -1/√2
    let cg = clebsch_gordan(j1, m0, j1, m1, j1, m1);
    assert_relative_eq!(cg, -1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 1,-1 | 1,0⟩ = 1/√2
    let cg = clebsch_gordan(j1, m1, j1, m_neg1, j1, m0);
    assert_relative_eq!(cg, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,-1; 1,1 | 1,0⟩ = -1/√2
    let cg = clebsch_gordan(j1, m_neg1, j1, m1, j1, m0);
    assert_relative_eq!(cg, -1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 1,0 | 1,0⟩ = 0 (vanishes!)
    let cg = clebsch_gordan(j1, m0, j1, m0, j1, m0);
    assert_relative_eq!(cg, 0.0, epsilon = 1e-10);
    
    // ====== j=0 (singlet) ======
    // Note: The exact signs depend on phase conventions (Condon-Shortley)
    // What matters is that they're consistent and orthonormal
    
    // Test that all three contributions to the singlet have equal magnitude
    let cg_1 = clebsch_gordan(j1, m1, j1, m_neg1, j0, m0);
    let cg_2 = clebsch_gordan(j1, m0, j1, m0, j0, m0);
    let cg_3 = clebsch_gordan(j1, m_neg1, j1, m1, j0, m0);
    
    // All should have magnitude 1/√3
    assert_relative_eq!(cg_1.abs(), 1.0 / 3.0_f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(cg_2.abs(), 1.0 / 3.0_f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(cg_3.abs(), 1.0 / 3.0_f64.sqrt(), epsilon = 1e-10);
    
    // Verify they form a properly normalized singlet
    // |0,0⟩ = Σ CG * |m1,m2⟩ should be normalized
    let norm_sq = cg_1 * cg_1 + cg_2 * cg_2 + cg_3 * cg_3;
    assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);
}

#[test]
fn test_cg_highest_lowest_weight_states() {
    // For any j1 ⊗ j2, the highest weight state should have CG = 1
    // ⟨j1,j1; j2,j2 | j1+j2,j1+j2⟩ = 1
    
    // Test with various spins
    let test_cases = vec![
        (1, 1), // 1/2 ⊗ 1/2
        (1, 2), // 1/2 ⊗ 1
        (2, 2), // 1 ⊗ 1
        (2, 4), // 1 ⊗ 2
        (3, 3), // 3/2 ⊗ 3/2
    ];
    
    for (twice_j1, twice_j2) in test_cases {
        let j1 = Spin::new(twice_j1).unwrap();
        let j2 = Spin::new(twice_j2).unwrap();
        let j_max = Spin::new(twice_j1 + twice_j2).unwrap();
        
        let m1 = MagneticNumber::new_unchecked(twice_j1);
        let m2 = MagneticNumber::new_unchecked(twice_j2);
        let m_max = MagneticNumber::new_unchecked(twice_j1 + twice_j2);
        
        let cg = clebsch_gordan(j1, m1, j2, m2, j_max, m_max);
        assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
        
        // Similarly for lowest weight states
        let m1_low = MagneticNumber::new_unchecked(-twice_j1);
        let m2_low = MagneticNumber::new_unchecked(-twice_j2);
        let m_min = MagneticNumber::new_unchecked(-(twice_j1 + twice_j2));
        
        let cg_low = clebsch_gordan(j1, m1_low, j2, m2_low, j_max, m_min);
        assert_relative_eq!(cg_low, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_cg_symmetry_exchange() {
    // Symmetry under exchange: ⟨j1,m1; j2,m2 | j3,m3⟩ = (-1)^(j1+j2-j3) ⟨j2,m2; j1,m1 | j3,m3⟩
    
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    
    let m_up = MagneticNumber::new_unchecked(1);
    let m_down = MagneticNumber::new_unchecked(-1);
    let m0 = MagneticNumber::new_unchecked(0);
    
    // For j=1/2 ⊗ j=1/2 → j=0: phase = (-1)^(1/2+1/2-0) = (-1)^1 = -1
    let cg1 = clebsch_gordan(j_half, m_up, j_half, m_down, Spin::new(0).unwrap(), m0);
    let cg2 = clebsch_gordan(j_half, m_down, j_half, m_up, Spin::new(0).unwrap(), m0);
    assert_relative_eq!(cg1, -cg2, epsilon = 1e-10);
    
    // For j=1/2 ⊗ j=1/2 → j=1: phase = (-1)^(1/2+1/2-1) = (-1)^0 = +1
    let cg1 = clebsch_gordan(j_half, m_up, j_half, m_down, j1, m0);
    let cg2 = clebsch_gordan(j_half, m_down, j_half, m_up, j1, m0);
    assert_relative_eq!(cg1, cg2, epsilon = 1e-10);
}

#[test]
fn test_cg_orthogonality_different_j() {
    // Test that CG coefficients for different final j are orthogonal
    // Σ_{m1,m2} ⟨j1,m1; j2,m2 | j3,m3⟩ ⟨j1,m1; j2,m2 | j3',m3⟩ = δ_{j3,j3'}
    
    let j_half = Spin::new(1).unwrap();
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();
    
    let m0 = MagneticNumber::new_unchecked(0);
    
    // Sum over all valid (m1, m2) pairs that give m3 = 0
    let mut sum = 0.0;
    for m1_val in [-1, 1] {
        let m2_val = -m1_val; // To get m3 = 0
        let m1 = MagneticNumber::new_unchecked(m1_val);
        let m2 = MagneticNumber::new_unchecked(m2_val);
        
        let cg_j0 = clebsch_gordan(j_half, m1, j_half, m2, j0, m0);
        let cg_j1 = clebsch_gordan(j_half, m1, j_half, m2, j1, m0);
        
        sum += cg_j0 * cg_j1;
    }
    
    // Should be zero (orthogonal)
    assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
}

#[test]
fn test_cg_completeness_relation() {
    // Completeness: for fixed m1, m2, sum over all allowed j3, m3 gives 1
    // Σ_{j3,m3} |⟨j1,m1; j2,m2 | j3,m3⟩|² = 1
    
    let j_half = Spin::new(1).unwrap();
    
    // Test for all combinations of (m1, m2)
    let m_values = [-1, 1];
    
    for &m1_val in &m_values {
        for &m2_val in &m_values {
            let m1 = MagneticNumber::new_unchecked(m1_val);
            let m2 = MagneticNumber::new_unchecked(m2_val);
            let m3_val = m1_val + m2_val; // Conservation
            
            let mut sum = 0.0;
            
            // j=1/2 ⊗ j=1/2 → j=0 or j=1
            for twice_j3 in [0, 2] {
                // Check if m3 is valid for this j3
                if m3_val.abs() > twice_j3 {
                    continue;
                }
                if m3_val % 2 != twice_j3 % 2 {
                    continue;
                }
                
                let j3 = Spin::new(twice_j3).unwrap();
                let m3 = MagneticNumber::new_unchecked(m3_val);
                
                let cg = clebsch_gordan(j_half, m1, j_half, m2, j3, m3);
                sum += cg * cg;
            }
            
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_metric_values() {
    // Test specific metric values for j=1
    let j1 = Spin::new(2).unwrap();

    // g^(1)_{1,-1} = (-1)^0 = 1
    let m1 = MagneticNumber::new_unchecked(2);
    let m_neg1 = MagneticNumber::new_unchecked(-2);
    assert_eq!(g(j1, m1, m_neg1), 1.0);

    // g^(1)_{0,0} = (-1)^1 = -1
    let m0 = MagneticNumber::new_unchecked(0);
    assert_eq!(g(j1, m0, m0.negate()), -1.0);

    // g^(1)_{-1,1} = (-1)^2 = 1
    assert_eq!(g(j1, m_neg1, m1), 1.0);
}
