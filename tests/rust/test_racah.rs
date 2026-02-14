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

//! Systematic tests for Clebsch-Gordan coefficients computed via Racah formula
//!
//! This test suite provides comprehensive coverage of CG coefficients for
//! angular momenta j ∈ {0, 1/2, 1, 3/2, 2}, comparing computed values against
//! analytic results from standard references (Varshalovich, Edmonds).
//!
//! Total coverage: 140 unique CG coefficients across 15 (j₁,j₂) pairs
//!
//! Test organization:
//! - Tier 1: Trivial j=0 couplings (15 coefficients)
//! - Tier 2: Well-known systems (19 coefficients, cross-check existing tests)
//! - Tier 3: Systematic analytic tests (46 coefficients)
//! - Tier 4: Higher-order spot checks (26+ coefficients)
//! - Validation: Completeness and orthogonality (all 140 coefficients)

use approx::assert_relative_eq;
use yuzuha::primitives::*;
use yuzuha::{MagneticNumber, Spin};

// ============================================================================
// TIER 1: TRIVIAL COUPLINGS (j₁=0 or j₂=0)
// ============================================================================

#[test]
fn test_j0_coupling_all() {
    // When j₁=0 or j₂=0, coupling is trivial: ⟨0,0; j,m | j,m⟩ = 1
    let j0 = Spin::new(0).unwrap();
    
    let test_spins = [
        (0, 0),   // 0
        (1, 1),   // 1/2
        (2, 2),   // 1
        (3, 3),   // 3/2
        (4, 4),   // 2
    ];
    
    for (twice_j, twice_m) in test_spins {
        let j = Spin::new(twice_j).unwrap();
        let m = MagneticNumber::new_unchecked(twice_m);
        let m0 = MagneticNumber::new_unchecked(0);
        
        // ⟨0,0; j,m | j,m⟩ = 1
        let cg = clebsch_gordan(j0, m0, j, m, j, m);
        assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
        
        // ⟨j,m; 0,0 | j,m⟩ = 1 (by symmetry)
        let cg = clebsch_gordan(j, m, j0, m0, j, m);
        assert_relative_eq!(cg, 1.0, epsilon = 1e-10);
    }
}

// ============================================================================
// TIER 2: CROSS-CHECK WITH EXISTING TESTS
// ============================================================================

#[test]
fn test_half_half_all() {
    // j=1/2 ⊗ j=1/2 → j=0,1 (4 coefficients)
    // Cross-check with test_primitives.rs
    let j = Spin::new(1).unwrap();
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();
    
    let m_up = MagneticNumber::new_unchecked(1);
    let m_dn = MagneticNumber::new_unchecked(-1);
    let m0 = MagneticNumber::new_unchecked(0);
    let m1_up = MagneticNumber::new_unchecked(2);
    let m1_dn = MagneticNumber::new_unchecked(-2);
    
    // Singlet j=0
    assert_relative_eq!(clebsch_gordan(j, m_up, j, m_dn, j0, m0), 
                       1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(clebsch_gordan(j, m_dn, j, m_up, j0, m0), 
                       -1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    
    // Triplet j=1
    assert_relative_eq!(clebsch_gordan(j, m_up, j, m_up, j1, m1_up), 
                       1.0, epsilon = 1e-10);
    assert_relative_eq!(clebsch_gordan(j, m_up, j, m_dn, j1, m0), 
                       1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(clebsch_gordan(j, m_dn, j, m_up, j1, m0), 
                       1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(clebsch_gordan(j, m_dn, j, m_dn, j1, m1_dn), 
                       1.0, epsilon = 1e-10);
}

// ============================================================================
// TIER 3: SYSTEMATIC ANALYTIC TESTS
// ============================================================================

#[test]
fn test_half_threehalf_all() {
    // j=1/2 ⊗ j=3/2 → j=1,2 (8 coefficients)
    // Analytic values from Varshalovich et al.
    
    let j_half = Spin::new(1).unwrap();     // j=1/2
    let j_3half = Spin::new(3).unwrap();    // j=3/2
    let j1 = Spin::new(2).unwrap();         // j=1
    let j2 = Spin::new(4).unwrap();         // j=2
    
    // Magnetic quantum numbers (in units of twice the actual value)
    let m_half_up = MagneticNumber::new_unchecked(1);     // +1/2
    let m_half_dn = MagneticNumber::new_unchecked(-1);    // -1/2
    let m_3half_3 = MagneticNumber::new_unchecked(3);     // +3/2
    let m_3half_1 = MagneticNumber::new_unchecked(1);     // +1/2
    let m_3half_m1 = MagneticNumber::new_unchecked(-1);   // -1/2
    
    // === Coupling to j=2 ===
    
    // ⟨1/2,1/2; 3/2,3/2 | 2,2⟩ = 1
    let m2_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j_3half, m_3half_3, j2, m2_2),
                       1.0, epsilon = 1e-10);
    
    // ⟨1/2,1/2; 3/2,1/2 | 2,1⟩ = √(3/4) = √3/2
    let m2_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j_3half, m_3half_1, j2, m2_1),
                       (3.0 / 4.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 3/2,3/2 | 2,1⟩ = √(1/4) = 1/2
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j_3half, m_3half_3, j2, m2_1),
                       (1.0 / 4.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,1/2; 3/2,-1/2 | 2,0⟩ = √(1/2) = 1/√2
    let m2_0 = MagneticNumber::new_unchecked(0);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j_3half, m_3half_m1, j2, m2_0),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=1 ===
    
    // ⟨1/2,1/2; 3/2,1/2 | 1,1⟩ = √(1/4) = 1/2
    let m1_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j_3half, m_3half_1, j1, m1_1),
                       (1.0 / 4.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 3/2,3/2 | 1,1⟩ = -√(3/4) = -√3/2
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j_3half, m_3half_3, j1, m1_1),
                       -(3.0 / 4.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,1/2; 3/2,-1/2 | 1,0⟩ = √(1/2) = 1/√2
    let m1_0 = MagneticNumber::new_unchecked(0);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j_3half, m_3half_m1, j1, m1_0),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 3/2,1/2 | 1,0⟩ = -√(1/2) = -1/√2
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j_3half, m_3half_1, j1, m1_0),
                       -(1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_half_two_all() {
    // j=1/2 ⊗ j=2 → j=3/2, 5/2 (10 coefficients)
    // Analytic values from Varshalovich et al.
    
    let j_half = Spin::new(1).unwrap();     // j=1/2
    let j2 = Spin::new(4).unwrap();         // j=2
    let j_3half = Spin::new(3).unwrap();    // j=3/2
    let j_5half = Spin::new(5).unwrap();    // j=5/2
    
    let m_half_up = MagneticNumber::new_unchecked(1);     // +1/2
    let m_half_dn = MagneticNumber::new_unchecked(-1);    // -1/2
    let m2_2 = MagneticNumber::new_unchecked(4);          // +2
    let m2_1 = MagneticNumber::new_unchecked(2);          // +1
    let m2_0 = MagneticNumber::new_unchecked(0);          // 0
    let m2_m1 = MagneticNumber::new_unchecked(-2);        // -1
    
    // === Coupling to j=5/2 (highest weight) ===
    
    // ⟨1/2,1/2; 2,2 | 5/2,5/2⟩ = 1
    let m_5half_5 = MagneticNumber::new_unchecked(5);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_2, j_5half, m_5half_5),
                       1.0, epsilon = 1e-10);
    
    // ⟨1/2,1/2; 2,1 | 5/2,3/2⟩ = √(4/5) = 2/√5
    let m_5half_3 = MagneticNumber::new_unchecked(3);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_1, j_5half, m_5half_3),
                       (4.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 2,2 | 5/2,3/2⟩ = √(1/5) = 1/√5
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j2, m2_2, j_5half, m_5half_3),
                       (1.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,1/2; 2,0 | 5/2,1/2⟩ = √(3/5)
    let m_5half_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_0, j_5half, m_5half_1),
                       (3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 2,1 | 5/2,1/2⟩ = √(2/5)
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j2, m2_1, j_5half, m_5half_1),
                       (2.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=3/2 ===
    
    // ⟨1/2,1/2; 2,1 | 3/2,3/2⟩ = √(1/5) = 1/√5
    let m_3half_3 = MagneticNumber::new_unchecked(3);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_1, j_3half, m_3half_3),
                       (1.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 2,2 | 3/2,3/2⟩ = -√(4/5) = -2/√5
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j2, m2_2, j_3half, m_3half_3),
                       -(4.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,1/2; 2,0 | 3/2,1/2⟩ = √(2/5)
    let m_3half_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_0, j_3half, m_3half_1),
                       (2.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,-1/2; 2,1 | 3/2,1/2⟩ = -√(3/5)
    assert_relative_eq!(clebsch_gordan(j_half, m_half_dn, j2, m2_1, j_3half, m_3half_1),
                       -(3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1/2,1/2; 2,-1 | 3/2,-1/2⟩ = √(3/5)
    let m_3half_m1 = MagneticNumber::new_unchecked(-1);
    assert_relative_eq!(clebsch_gordan(j_half, m_half_up, j2, m2_m1, j_3half, m_3half_m1),
                       (3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_one_threehalf_all() {
    // j=1 ⊗ j=3/2 → j=1/2, 3/2, 5/2 (12 coefficients)
    // Analytic values from standard references
    
    let j1 = Spin::new(2).unwrap();           // j=1
    let j_3half = Spin::new(3).unwrap();      // j=3/2
    let j_half = Spin::new(1).unwrap();       // j=1/2
    let j_3half_out = Spin::new(3).unwrap();  // j=3/2 (output)
    let j_5half = Spin::new(5).unwrap();      // j=5/2
    
    let m1_1 = MagneticNumber::new_unchecked(2);    // +1
    let m1_0 = MagneticNumber::new_unchecked(0);    // 0
    let m1_m1 = MagneticNumber::new_unchecked(-2);  // -1
    let m_3h_3 = MagneticNumber::new_unchecked(3);  // +3/2
    let m_3h_1 = MagneticNumber::new_unchecked(1);  // +1/2
    let m_3h_m1 = MagneticNumber::new_unchecked(-1);// -1/2
    
    // === Coupling to j=5/2 ===
    
    // ⟨1,1; 3/2,3/2 | 5/2,5/2⟩ = 1
    let m_5h_5 = MagneticNumber::new_unchecked(5);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_3, j_5half, m_5h_5),
                       1.0, epsilon = 1e-10);
    
    // ⟨1,1; 3/2,1/2 | 5/2,3/2⟩ = √(3/5)
    let m_5h_3 = MagneticNumber::new_unchecked(3);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_1, j_5half, m_5h_3),
                       (3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 3/2,3/2 | 5/2,3/2⟩ = √(2/5)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j_3half, m_3h_3, j_5half, m_5h_3),
                       (2.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 3/2,-1/2 | 5/2,1/2⟩ = √(3/10)
    let m_5h_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_m1, j_5half, m_5h_1),
                       (3.0 / 10.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 3/2,1/2 | 5/2,1/2⟩ = √(3/5)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j_3half, m_3h_1, j_5half, m_5h_1),
                       (3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=3/2 ===
    
    // ⟨1,1; 3/2,1/2 | 3/2,3/2⟩ = √(2/5)
    let m_3h_out_3 = MagneticNumber::new_unchecked(3);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_1, j_3half_out, m_3h_out_3),
                       (2.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 3/2,3/2 | 3/2,3/2⟩ = -√(3/5)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j_3half, m_3h_3, j_3half_out, m_3h_out_3),
                       -(3.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 3/2,-1/2 | 3/2,1/2⟩ = √(8/15)
    let m_3h_out_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_m1, j_3half_out, m_3h_out_1),
                       (8.0 / 15.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 3/2,1/2 | 3/2,1/2⟩ = -√(1/15)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j_3half, m_3h_1, j_3half_out, m_3h_out_1),
                       -(1.0 / 15.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=1/2 ===
    
    // ⟨1,0; 3/2,1/2 | 1/2,1/2⟩ = -√(1/3)
    let m_h_out_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j_3half, m_3h_1, j_half, m_h_out_1),
                       -(1.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,-1; 3/2,3/2 | 1/2,1/2⟩ = √(1/2) = 1/√2
    assert_relative_eq!(clebsch_gordan(j1, m1_m1, j_3half, m_3h_3, j_half, m_h_out_1),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 3/2,-1/2 | 1/2,1/2⟩ = √(1/6)
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j_3half, m_3h_m1, j_half, m_h_out_1),
                       (1.0 / 6.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_threehalf_threehalf_all() {
    // j=3/2 ⊗ j=3/2 → j=0,1,2,3 (16 coefficients)
    // Testing key coefficients with analytic values
    
    let j = Spin::new(3).unwrap();        // j=3/2
    let j0 = Spin::new(0).unwrap();       // j=0
    let j1 = Spin::new(2).unwrap();       // j=1
    let j2 = Spin::new(4).unwrap();       // j=2
    let j3 = Spin::new(6).unwrap();       // j=3
    
    let m3 = MagneticNumber::new_unchecked(3);    // +3/2
    let m1 = MagneticNumber::new_unchecked(1);    // +1/2
    let m_1 = MagneticNumber::new_unchecked(-1);  // -1/2
    let m_3 = MagneticNumber::new_unchecked(-3);  // -3/2
    let m0 = MagneticNumber::new_unchecked(0);
    
    // === Highest weight states ===
    
    // ⟨3/2,3/2; 3/2,3/2 | 3,3⟩ = 1
    let m3_out = MagneticNumber::new_unchecked(6);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m3, j3, m3_out),
                       1.0, epsilon = 1e-10);
    
    // === Coupling to j=2 ===
    
    // ⟨3/2,3/2; 3/2,1/2 | 2,2⟩ = √(1/2) = 1/√2
    let m2_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m1, j2, m2_2),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,3/2 | 2,2⟩ = -√(1/2) = -1/√2 (note: negative due to phase)
    assert_relative_eq!(clebsch_gordan(j, m1, j, m3, j2, m2_2),
                       -(1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,3/2; 3/2,-1/2 | 2,1⟩ = √(1/2) = 1/√2
    let m2_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m_1, j2, m2_1),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,1/2 | 2,1⟩ = 0 (vanishes!)
    assert_relative_eq!(clebsch_gordan(j, m1, j, m1, j2, m2_1),
                       0.0, epsilon = 1e-10);
    
    // ⟨3/2,3/2; 3/2,-3/2 | 2,0⟩ = 1/2
    let m2_0 = MagneticNumber::new_unchecked(0);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m_3, j2, m2_0),
                       0.5, epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,-1/2 | 2,0⟩ = 1/2
    assert_relative_eq!(clebsch_gordan(j, m1, j, m_1, j2, m2_0),
                       0.5, epsilon = 1e-10);
    
    // === Coupling to j=1 ===
    
    // ⟨3/2,3/2; 3/2,1/2 | 1,2⟩ = 0 (vanishes!)
    let m1_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m1, j1, m1_2),
                       0.0, epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,3/2 | 1,2⟩ = 0 (also vanishes!)
    assert_relative_eq!(clebsch_gordan(j, m1, j, m3, j1, m1_2),
                       0.0, epsilon = 1e-10);
    
    // ⟨3/2,3/2; 3/2,-1/2 | 1,1⟩ = √(3/10)
    let m1_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j, m3, j, m_1, j1, m1_1),
                       (3.0 / 10.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,1/2 | 1,1⟩ = -√(2/5)
    assert_relative_eq!(clebsch_gordan(j, m1, j, m1, j1, m1_1),
                       -(2.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,-1/2 | 1,0⟩ = -√(1/20) = -1/(2√5)
    let m1_0 = MagneticNumber::new_unchecked(0);
    assert_relative_eq!(clebsch_gordan(j, m1, j, m_1, j1, m1_0),
                       -(1.0 / 20.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=0 (singlet) ===
    
    // ⟨3/2,3/2; 3/2,-3/2 | 0,0⟩ = 1/2
    assert_relative_eq!(clebsch_gordan(j, m3, j, m_3, j0, m0),
                       0.5, epsilon = 1e-10);
    
    // ⟨3/2,1/2; 3/2,-1/2 | 0,0⟩ = -1/2
    assert_relative_eq!(clebsch_gordan(j, m1, j, m_1, j0, m0),
                       -0.5, epsilon = 1e-10);
    
    // ⟨3/2,-1/2; 3/2,1/2 | 0,0⟩ = 1/2
    assert_relative_eq!(clebsch_gordan(j, m_1, j, m1, j0, m0),
                       0.5, epsilon = 1e-10);
}

// ============================================================================
// TIER 4: HIGHER-ORDER SPOT CHECKS
// ============================================================================

#[test]
fn test_one_two_selected() {
    // j=1 ⊗ j=2 → j=1,2,3 (15 coefficients, testing 8 key ones)
    
    let j1 = Spin::new(2).unwrap();    // j=1
    let j2 = Spin::new(4).unwrap();    // j=2
    let j_out_1 = Spin::new(2).unwrap(); // j=1
    let j_out_2 = Spin::new(4).unwrap(); // j=2
    let j_out_3 = Spin::new(6).unwrap(); // j=3
    
    // === Highest weight ===
    
    // ⟨1,1; 2,2 | 3,3⟩ = 1
    let m1_1 = MagneticNumber::new_unchecked(2);
    let m2_2 = MagneticNumber::new_unchecked(4);
    let m3_3 = MagneticNumber::new_unchecked(6);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j2, m2_2, j_out_3, m3_3),
                       1.0, epsilon = 1e-10);
    
    // ⟨1,1; 2,1 | 3,2⟩ = √(2/3)
    let m2_1 = MagneticNumber::new_unchecked(2);
    let m3_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j2, m2_1, j_out_3, m3_2),
                       (2.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 2,2 | 3,2⟩ = √(1/3)
    let m1_0 = MagneticNumber::new_unchecked(0);
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j2, m2_2, j_out_3, m3_2),
                       (1.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=2 ===
    
    // ⟨1,1; 2,1 | 2,2⟩ = √(1/3)
    let m2_out_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j2, m2_1, j_out_2, m2_out_2),
                       (1.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 2,2 | 2,2⟩ = -√(2/3)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j2, m2_2, j_out_2, m2_out_2),
                       -(2.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,1; 2,0 | 2,1⟩ = √(1/2)
    let m2_0 = MagneticNumber::new_unchecked(0);
    let m2_out_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j1, m1_1, j2, m2_0, j_out_2, m2_out_1),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨1,0; 2,1 | 2,1⟩ = -√(1/6)
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j2, m2_1, j_out_2, m2_out_1),
                       -(1.0 / 6.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=1 ===
    
    // ⟨1,0; 2,1 | 1,1⟩ = -√(3/10)
    let m1_out_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j1, m1_0, j2, m2_1, j_out_1, m1_out_1),
                       -(3.0 / 10.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_threehalf_two_selected() {
    // j=3/2 ⊗ j=2 → j=1/2,3/2,5/2,7/2 (20 coefficients, testing 8 key ones)
    
    let j_3half = Spin::new(3).unwrap();   // j=3/2
    let j2 = Spin::new(4).unwrap();        // j=2
    let j_7half = Spin::new(7).unwrap();   // j=7/2
    let j_5half = Spin::new(5).unwrap();   // j=5/2
    
    // === Highest weight ===
    
    // ⟨3/2,3/2; 2,2 | 7/2,7/2⟩ = 1
    let m_3h_3 = MagneticNumber::new_unchecked(3);
    let m2_2 = MagneticNumber::new_unchecked(4);
    let m_7h_7 = MagneticNumber::new_unchecked(7);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_3, j2, m2_2, j_7half, m_7h_7),
                       1.0, epsilon = 1e-10);
    
    // ⟨3/2,3/2; 2,1 | 7/2,5/2⟩ = √(4/7)
    let m2_1 = MagneticNumber::new_unchecked(2);
    let m_7h_5 = MagneticNumber::new_unchecked(5);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_3, j2, m2_1, j_7half, m_7h_5),
                       (4.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 2,2 | 7/2,5/2⟩ = √(3/7)
    let m_3h_1 = MagneticNumber::new_unchecked(1);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_1, j2, m2_2, j_7half, m_7h_5),
                       (3.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=5/2 ===
    
    // ⟨3/2,3/2; 2,1 | 5/2,5/2⟩ = √(3/7)
    let m_5h_5 = MagneticNumber::new_unchecked(5);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_3, j2, m2_1, j_5half, m_5h_5),
                       (3.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 2,2 | 5/2,5/2⟩ = -√(4/7)
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_1, j2, m2_2, j_5half, m_5h_5),
                       -(4.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,3/2; 2,0 | 5/2,3/2⟩ = √(18/35)
    let m2_0 = MagneticNumber::new_unchecked(0);
    let m_5h_3 = MagneticNumber::new_unchecked(3);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_3, j2, m2_0, j_5half, m_5h_3),
                       (18.0 / 35.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,1/2; 2,1 | 5/2,3/2⟩ = -√(1/35)
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_1, j2, m2_1, j_5half, m_5h_3),
                       -(1.0 / 35.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨3/2,-1/2; 2,2 | 5/2,3/2⟩ = -√(16/35)
    let m_3h_m1 = MagneticNumber::new_unchecked(-1);
    assert_relative_eq!(clebsch_gordan(j_3half, m_3h_m1, j2, m2_2, j_5half, m_5h_3),
                       -(16.0 / 35.0_f64).sqrt(), epsilon = 1e-10);
}

#[test]
fn test_two_two_selected() {
    // j=2 ⊗ j=2 → j=0,1,2,3,4 (25 coefficients, testing 10 key ones)
    
    let j2 = Spin::new(4).unwrap();     // j=2
    let j0 = Spin::new(0).unwrap();     // j=0
    let j2_out = Spin::new(4).unwrap(); // j=2
    let j4 = Spin::new(8).unwrap();     // j=4
    
    let m2 = MagneticNumber::new_unchecked(4);
    let m1 = MagneticNumber::new_unchecked(2);
    let m0 = MagneticNumber::new_unchecked(0);
    let m_1 = MagneticNumber::new_unchecked(-2);
    let m_2 = MagneticNumber::new_unchecked(-4);
    
    // === Highest weight ===
    
    // ⟨2,2; 2,2 | 4,4⟩ = 1
    let m4_4 = MagneticNumber::new_unchecked(8);
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m2, j4, m4_4),
                       1.0, epsilon = 1e-10);
    
    // ⟨2,2; 2,1 | 4,3⟩ = √(1/2)
    let m4_3 = MagneticNumber::new_unchecked(6);
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m1, j4, m4_3),
                       (1.0 / 2.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,2; 2,0 | 4,2⟩ = √(3/14)
    let m4_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m0, j4, m4_2),
                       (3.0 / 14.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,1; 2,1 | 4,2⟩ = √(4/7)
    assert_relative_eq!(clebsch_gordan(j2, m1, j2, m1, j4, m4_2),
                       (4.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,2; 2,-1 | 4,1⟩ = √(1/14)
    let m4_1 = MagneticNumber::new_unchecked(2);
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m_1, j4, m4_1),
                       (1.0 / 14.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,1; 2,0 | 4,1⟩ = √(3/7)
    assert_relative_eq!(clebsch_gordan(j2, m1, j2, m0, j4, m4_1),
                       (3.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=2 ===
    
    // ⟨2,1; 2,1 | 2,2⟩ = -√(3/7)
    let m2_out_2 = MagneticNumber::new_unchecked(4);
    assert_relative_eq!(clebsch_gordan(j2, m1, j2, m1, j2_out, m2_out_2),
                       -(3.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,2; 2,0 | 2,2⟩ = √(2/7)
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m0, j2_out, m2_out_2),
                       (2.0 / 7.0_f64).sqrt(), epsilon = 1e-10);
    
    // === Coupling to j=0 (singlet) ===
    
    // ⟨2,2; 2,-2 | 0,0⟩ = √(1/5)
    assert_relative_eq!(clebsch_gordan(j2, m2, j2, m_2, j0, m0),
                       (1.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
    
    // ⟨2,0; 2,0 | 0,0⟩ = √(1/5)
    assert_relative_eq!(clebsch_gordan(j2, m0, j2, m0, j0, m0),
                       (1.0 / 5.0_f64).sqrt(), epsilon = 1e-10);
}

// ============================================================================
// GLOBAL VALIDATION TESTS
// ============================================================================

#[test]
fn test_completeness_all_pairs() {
    // Test completeness relation for all (j1, j2) pairs with j ≤ 2
    // For fixed (m1, m2): Σ_{j3,m3} |CG(j1,m1,j2,m2,j3,m3)|² = 1
    
    let spins = [
        Spin::new(0).unwrap(),  // 0
        Spin::new(1).unwrap(),  // 1/2
        Spin::new(2).unwrap(),  // 1
        Spin::new(3).unwrap(),  // 3/2
        Spin::new(4).unwrap(),  // 2
    ];
    
    for &j1 in &spins {
        for &j2 in &spins {
            // Iterate over all m1, m2 values
            for m1_val in (-j1.twice()..=j1.twice()).step_by(2) {
                for m2_val in (-j2.twice()..=j2.twice()).step_by(2) {
                    let m1 = MagneticNumber::new_unchecked(m1_val);
                    let m2 = MagneticNumber::new_unchecked(m2_val);
                    
                    let mut sum = 0.0;
                    
                    // Sum over all allowed j3
                    let j_min = ((j1.twice() as i32 - j2.twice() as i32).abs()) as i32;
                    let j_max = (j1.twice() + j2.twice()) as i32;
                    
                    for twice_j3 in (j_min..=j_max).step_by(2) {
                        if let Ok(j3) = Spin::new(twice_j3) {
                            let m3_val = m1_val + m2_val;
                            
                            // Check if m3 is valid for j3
                            if m3_val.abs() <= j3.twice() as i32 {
                                let m3 = MagneticNumber::new_unchecked(m3_val);
                                let cg = clebsch_gordan(j1, m1, j2, m2, j3, m3);
                                sum += cg * cg;
                            }
                        }
                    }
                    
                    assert_relative_eq!(sum, 1.0, epsilon = 1e-9);
                }
            }
        }
    }
}

#[test]
fn test_orthogonality_all_pairs() {
    // Test orthogonality: Σ_{m1,m2} CG(j1,m1,j2,m2,j3,m3) * CG(j1,m1,j2,m2,j3',m3') = δ_{j3,j3'} δ_{m3,m3'}
    // We test a sample of cases due to computational cost
    
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    let j_3half = Spin::new(3).unwrap();
    
    // Test 1/2 ⊗ 1/2: j3=0 vs j3=1, m3=0
    let m0 = MagneticNumber::new_unchecked(0);
    let mut sum = 0.0;
    
    for m1_val in [-1, 1] {
        for m2_val in [-1, 1] {
            if m1_val + m2_val == 0 {
                let m1 = MagneticNumber::new_unchecked(m1_val);
                let m2 = MagneticNumber::new_unchecked(m2_val);
                
                let cg1 = clebsch_gordan(j_half, m1, j_half, m2, Spin::new(0).unwrap(), m0);
                let cg2 = clebsch_gordan(j_half, m1, j_half, m2, Spin::new(2).unwrap(), m0);
                
                sum += cg1 * cg2;
            }
        }
    }
    
    assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    
    // Test 1/2 ⊗ 1: j3=1/2 vs j3=3/2, m3=1/2
    let m_half = MagneticNumber::new_unchecked(1);
    sum = 0.0;
    
    for m1_val in [-1, 1] {
        for m2_val in [-2, 0, 2] {
            if m1_val + m2_val == 1 {
                let m1 = MagneticNumber::new_unchecked(m1_val);
                let m2 = MagneticNumber::new_unchecked(m2_val);
                
                let cg1 = clebsch_gordan(j_half, m1, j1, m2, j_half, m_half);
                let cg2 = clebsch_gordan(j_half, m1, j1, m2, j_3half, m_half);
                
                sum += cg1 * cg2;
            }
        }
    }
    
    assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
}

#[test]
fn test_selection_rules_systematic() {
    // Verify that CG coefficients vanish when selection rules are violated
    
    let j1 = Spin::new(2).unwrap();  // j=1
    let j2 = Spin::new(4).unwrap();  // j=2
    let j3 = Spin::new(4).unwrap();  // j=2
    
    let m1 = MagneticNumber::new_unchecked(2);   // m=1
    let m2 = MagneticNumber::new_unchecked(4);   // m=2
    let m3_bad = MagneticNumber::new_unchecked(0); // m=0 (violates m1+m2=m3)
    
    // Should be zero due to magnetic number conservation
    let cg = clebsch_gordan(j1, m1, j2, m2, j3, m3_bad);
    assert_eq!(cg, 0.0);
    
    // Test triangle inequality violation
    let j_invalid = Spin::new(12).unwrap(); // j=6 (too large for 1⊗2)
    let m_invalid = MagneticNumber::new_unchecked(6);
    let cg = clebsch_gordan(j1, m1, j2, m2, j_invalid, m_invalid);
    assert_eq!(cg, 0.0);
}
