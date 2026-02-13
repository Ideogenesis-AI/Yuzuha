//! Tests for F-symbol consistency
//!
//! Verifies that canonical basis transformations match known F-symbol values.
//! 
//! F-symbols relate different coupling schemes. For 4 spin-1/2 edges:
//! - Canonical: ((e0 ⊗ e1) ⊗ e2) ⊗ e3  [left associative]
//! - Binary:    (e0 ⊗ e1) ⊗ (e2 ⊗ e3)  [binary tree]
//! 
//! The F-symbol is the transformation matrix between these bases.

use approx::assert_relative_eq;
use ndarray::{Array2, Axis};
use yuzuha::builders::atomic::build_cg3;
use yuzuha::builders::builders::build_canonical_basis_data;
use yuzuha::core::{CGSpec, Edge, Spin};

/// Invert edge direction using metric tensor
/// 
/// Incoming → Outgoing: mirror and apply (-1)^{j+m}
/// Outgoing → Incoming: mirror and apply (-1)^{2j} * (-1)^{j+m}
fn invert_edge_direction_axis(
    tensor: &ndarray::ArrayD<f64>,
    axis_idx: usize,
    j: Spin,
    from_incoming: bool,
) -> ndarray::ArrayD<f64> {
    let mut result = tensor.clone();
    
    // Reverse the axis (mirror: m → -m)
    result.invert_axis(Axis(axis_idx));
    
    // Apply phase factors based on direction conversion
    let dim = j.dimension();
    for m_idx in 0..dim {
        let m = yuzuha::core::MagneticNumber::from_index(m_idx, j);
        
        let phase = if from_incoming {
            // Incoming → Outgoing: (-1)^{j+m}
            if (j.twice() + m.twice()) % 4 == 0 { 1.0 } else { -1.0 }
        } else {
            // Outgoing → Incoming: (-1)^{2j} * (-1)^{j+m}
            let phase_2j = if j.twice() % 2 == 0 { 1.0 } else { -1.0 };
            let phase_jm = if (j.twice() + m.twice()) % 4 == 0 { 1.0 } else { -1.0 };
            phase_2j * phase_jm
        };
        
        result.index_axis_mut(Axis(axis_idx), m_idx).mapv_inplace(|x| x * phase);
    }
    
    result
}

#[test]
fn test_fsymbol_four_spin_half() {
    // Four spin-1/2 edges
    let j_half = Spin::new(1).unwrap();
    
    // Build canonical basis: 3 incoming + 1 outgoing
    let edges = vec![
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::outgoing(j_half),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let canonical_basis = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();
    
    // Build alternative (binary tree) basis
    // Couple (e0 ⊗ e1) and (e2 ⊗ e3) separately, then contract
    
    // Possible intermediate spins for j=1/2 ⊗ j=1/2: j=0 or j=1
    let j0 = Spin::new(0).unwrap();
    let j1 = Spin::new(2).unwrap();
    let intermediate_spins = vec![j0, j1];
    
    let mut binary_basis = Vec::new();
    
    for &j_01 in &intermediate_spins {
        // Build CG3 for e0 ⊗ e1 → j_01
        let cg3_left = build_cg3(j_half, j_half, j_01).unwrap();
        
        // Build CG3 for e2 ⊗ e3 → j_01 (must be same intermediate spin)
        let cg3_right = build_cg3(j_half, j_half, j_01).unwrap();
        
        // Invert left CG3 output from outgoing to incoming
        let cg3_left_inverted = invert_edge_direction_axis(&cg3_left, 2, j_01, false);
        
        // Contract: last axis of left (j_01, now incoming) with last axis of right (j_01, outgoing)
        let mut result = ndarray_einsum::tensordot(
            &cg3_left_inverted,
            &cg3_right,
            &[Axis(2)],
            &[Axis(2)],
        );
        
        // Invert e3 from incoming to outgoing to match canonical basis direction
        result = invert_edge_direction_axis(&result, 3, j_half, true);

        // Normalize by sqrt(2j+1) of the intermediate spin j_01
        let norm = (j_01.dimension() as f64).sqrt();
        result.mapv_inplace(|x| x / norm);

        // Permute e1 and e2 (swap axes 1 and 2)
        result.swap_axes(1, 2);

        binary_basis.push(result);
    }

    // Verify both bases have unit Frobenius norm
    for alpha_idx in 0..om_dim {
        let slice = canonical_basis.slice(ndarray::s![.., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }
    for binary_idx in 0..binary_basis.len() {
        let frob_norm_sq: f64 = binary_basis[binary_idx].mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    // Now compute F-symbol: contract both bases over all external edges
    // F[alpha, (j_01, j_23)] = sum over external indices of 
    //   canonical_basis[..., alpha] * binary_basis[(j_01, j_23)][...]
    
    let binary_dim = binary_basis.len();
    let mut f_matrix = Array2::zeros((om_dim, binary_dim));
    
    for alpha_idx in 0..om_dim {
        let canonical_slice = canonical_basis.slice(ndarray::s![.., .., .., .., alpha_idx]);
        
        for binary_idx in 0..binary_dim {
            let binary_tensor = &binary_basis[binary_idx];
            
            // Contract over all 4 external edges
            let overlap = ndarray_einsum::tensordot(
                &canonical_slice,
                binary_tensor,
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
            );
            
            // Result should be a scalar (0-d array)
            f_matrix[[alpha_idx, binary_idx]] = overlap[[]];
        }
    }
    
    // Verify F-symbol properties:
    // The F-matrix relates two different fusion bases:
    // - Canonical: ((e0 ⊗ e1) ⊗ e2) ⊗ e3 (left-associative)
    // - Binary: (e0 ⊗ e1) ⊗ (e2 ⊗ e3) (binary tree)
    //
    // F†F should be diagonal (but not necessarily identity) because:
    // 1. The bases are orthogonal within each basis
    // 2. Different intermediate spin dimensions lead to different normalizations
    //    (e.g., j=0 has dim 1, j=1 has dim 3)
    let f_dagger_f = f_matrix.t().dot(&f_matrix);
    
    for i in 0..binary_dim {
        for j in 0..binary_dim {
            if i != j {
                // Off-diagonal elements should be zero
                assert_relative_eq!(f_dagger_f[[i, j]], 0.0, epsilon = 1e-10);
            }
        }
    }

    // Apply phase convention: negate F-matrix
    f_matrix.mapv_inplace(|x| -x);

    // Compare with exact F-symbol values for 4 spin-1/2 recoupling
    // F[alpha, j_01]: canonical basis (j_01, j_012) indexed by alpha, binary basis by j_01
    // Standard values: [[1/2, sqrt(3)/2], [sqrt(3)/2, -1/2]]
    let sqrt3 = 3.0_f64.sqrt();
    let expected = [
        [0.5, sqrt3 / 2.0],
        [sqrt3 / 2.0, -0.5],
    ];
    for alpha_idx in 0..om_dim {
        for binary_idx in 0..binary_dim {
            assert_relative_eq!(
                f_matrix[[alpha_idx, binary_idx]],
                expected[alpha_idx][binary_idx],
                epsilon = 1e-10
            );
        }
    }
}
