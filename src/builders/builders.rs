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

//! Canonical basis builder functions
//!
//! Functions to construct canonical basis tensors from CGSpec.

use crate::builders::atomic::build_cg3;
use crate::builders::cache;
use crate::core::{CGSpec, Direction, Edge, MagneticNumber, Spin};
use crate::error::{Result, YuzuhaError};
use ndarray::{Array, ArrayD, Axis, IxDyn};

/// Build canonical basis data for ALL OM configurations
///
/// This is the primary builder for canonical basis tensors. It computes the
/// canonical basis element for each OM configuration (alpha) and stacks them
/// along the last axis.
///
/// Handles arbitrary arrow directions by converting to canonical form (n-1 incoming, 1 outgoing).
/// Uses SQLite caching to avoid recomputation of expensive basis tensors.
///
/// # Arguments
/// * `spec` - CG specification with edges and alphas
///
/// # Returns
/// Array with shape [external_dims..., om_dim]
pub fn build_canonical_basis_data(spec: &CGSpec) -> Result<ArrayD<f64>> {
    let n = spec.num_external();
    
    // Canonical basis is only defined for n >= 3
    if n < 3 {
        return Err(YuzuhaError::InvalidCGTSpec(
            format!("Cannot build canonical basis with {} external edges (minimum is 3)", n)
        ));
    }
    
    // Extract spins only (ignore directions)
    let spins: Vec<Spin> = spec.edges.iter().map(|e| e.j).collect();
    
    // Try to get from cache first
    let canonical_data = if let Some(cached) = cache::query_canonical_basis(&spins)? {
        cached
    } else {
        // Cache miss - compute it
        let canonical_spec = create_canonical_spec(&spins)?;
        let data = compute_canonical_basis_data(&canonical_spec)?;
        
        // Store in cache
        cache::store_canonical_basis(&spins, &data)?;
        
        data
    };
    
    // Apply direction inversions to match the original spec
    let result = invert_directions_for_spec(&canonical_data, spec)?;
    
    Ok(result)
}

/// Create canonical spec from spins
///
/// Creates a CGSpec with canonical arrow directions (n-1 incoming, 1 outgoing).
///
/// # Arguments
/// * `spins` - Array of spin quantum numbers
///
/// # Returns
/// CGSpec with canonical directions
fn create_canonical_spec(spins: &[Spin]) -> Result<CGSpec> {
    let n = spins.len();
    let canonical_edges: Vec<_> = spins.iter().enumerate().map(|(i, &j)| {
        let canonical_dir = if i < n - 1 {
            Direction::Incoming
        } else {
            Direction::Outgoing
        };
        Edge::new(j, canonical_dir)
    }).collect();
    
    CGSpec::from_edges(canonical_edges)
}

/// Compute canonical basis data (internal, cacheable part)
///
/// This is the extracted computation logic that was previously inline in
/// build_canonical_basis_data. It computes the basis for canonical directions only.
///
/// # Arguments
/// * `canonical_spec` - CGSpec with canonical directions
///
/// # Returns
/// Canonical basis array [external_dims..., om_dim]
fn compute_canonical_basis_data(canonical_spec: &CGSpec) -> Result<ArrayD<f64>> {
    let n = canonical_spec.num_external();
    let mut data = ArrayD::zeros(IxDyn(&canonical_spec.shape()));
    
    let j_last = canonical_spec.edges[n - 1].j;
    let norm = (j_last.dimension() as f64).sqrt();

    for (om_idx, alpha) in canonical_spec.alphas.iter().enumerate() {
        let mut alpha_data = build_single_om_tensor(&canonical_spec, alpha)?;
        alpha_data.mapv_inplace(|x| x / norm);
        
        // No direction inversion here - store in canonical form only
        set_om_slice(&mut data, om_idx, &alpha_data, n)?;
    }
    
    Ok(data)
}

/// Invert direction for a single edge across ALL OM slices
///
/// Similar to `invert_edge_direction` but operates on the full basis array
/// including the OM axis at the end. The OM axis is preserved.
///
/// # Arguments
/// * `tensor` - Input tensor with shape [external_dims..., om_dim]
/// * `axis_idx` - Index of the external axis to invert
/// * `j` - Spin of the edge being inverted
/// * `from_dir` - Current direction of the edge
/// * `_n_external` - Number of external axes (excluding OM axis)
///
/// # Returns
/// Tensor with inverted edge direction
fn invert_edge_direction_full_basis(
    tensor: &ArrayD<f64>,
    axis_idx: usize,
    j: Spin,
    from_dir: Direction,
    _n_external: usize,
) -> Result<ArrayD<f64>> {
    let mut result = tensor.clone();
    
    // Reverse the axis (mirror: m → -m)
    result.invert_axis(Axis(axis_idx));
    
    // Apply phase factors based on direction conversion
    let dim = j.dimension();
    for m_idx in 0..dim {
        let m = MagneticNumber::from_index(m_idx, j);
        
        // Calculate phase
        let phase = if from_dir == Direction::Incoming {
            // Incoming → Outgoing: (-1)^{j+m}
            if (j.twice() + m.twice()) % 4 == 0 { 1.0 } else { -1.0 }
        } else {
            // Outgoing → Incoming: (-1)^{2j} * (-1)^{j+m}
            let phase_2j = if j.twice() % 2 == 0 { 1.0 } else { -1.0 };
            let phase_jm = if (j.twice() + m.twice()) % 4 == 0 { 1.0 } else { -1.0 };
            phase_2j * phase_jm
        };
        
        // Apply phase to the entire slice at this m value (including all OM values)
        result.index_axis_mut(Axis(axis_idx), m_idx).mapv_inplace(|x| x * phase);
    }
    
    Ok(result)
}

/// Invert edge directions for entire canonical basis array
///
/// Takes the canonical basis array (shape [external_dims..., om_dim])
/// and inverts directions to match the target spec.
///
/// # Arguments
/// * `canonical_data` - Canonical basis array [external_dims..., om_dim]
/// * `spec` - Target CGSpec with potentially non-canonical directions
///
/// # Returns
/// Basis array with directions matching spec
fn invert_directions_for_spec(
    canonical_data: &ArrayD<f64>,
    spec: &CGSpec,
) -> Result<ArrayD<f64>> {
    let n = spec.num_external();
    let mut result = canonical_data.clone();
    
    // For each axis, check if direction needs inverting
    for (axis_idx, edge) in spec.edges.iter().enumerate() {
        let canonical_dir = if axis_idx < n - 1 {
            Direction::Incoming
        } else {
            Direction::Outgoing
        };
        
        if edge.dir != canonical_dir {
            result = invert_edge_direction_full_basis(&result, axis_idx, edge.j, canonical_dir, n)?;
        }
    }
    
    Ok(result)
}

/// Build canonical tensor for a single OM configuration
///
/// Creates the fusion tree for a specific alpha and contracts it.
/// Uses a recursive approach, expanding from the right.
///
/// # Arguments
/// * `spec` - CG specification
/// * `alpha` - Internal spin configuration [j_12, j_123, ..., j_1...n-1]
///
/// # Returns
/// Array with shape [external_dims...]
fn build_single_om_tensor(spec: &CGSpec, alpha: &[Spin]) -> Result<ArrayD<f64>> {
    let n = spec.num_external();
    
    // Error for invalid cases
    if n < 3 {
        return Err(YuzuhaError::InvalidCGTSpec(
            format!("Cannot build OM tensor with {} external edges (minimum is 3)", n)
        ));
    }
    
    // Base case: n = 3 (CG3 tensor)
    if n == 3 {
        // CG3: edges[0] ⊗ edges[1] → edges[2]
        return build_cg3(
            spec.edges[0].j,
            spec.edges[1].j,
            spec.edges[2].j,
        );
    }
    
    // For n >= 4: Build fusion tree recursively
    build_fusion_tree_recursive(&spec.edges, alpha)
}

/// Recursively build the fusion tree tensor for n >= 4
///
/// Builds left-associative fusion tree: ((...((j_0 ⊗ j_1) ⊗ j_2) ⊗ ...) ⊗ j_{n-1})
/// where the first n-1 spins fuse together with the final spin.
///
/// # Arguments
/// * `edges` - External edge spins [j_0, j_1, ..., j_{n-1}] (n >= 4)
/// * `alpha` - Internal spins [j_01, j_012, ..., j_0...n-2] where n = edges.len()
///
/// # Returns
/// Tensor with shape [dim(j_0), dim(j_1), ..., dim(j_{n-1})]
fn build_fusion_tree_recursive(
    edges: &[crate::core::Edge],
    alpha: &[Spin],
) -> Result<ArrayD<f64>> {
    let n = edges.len();
    
    // This function should only be called for n >= 4
    if n < 4 {
        return Err(YuzuhaError::InvalidCGTSpec(
            format!("build_fusion_tree_recursive requires n >= 4, got n = {}", n)
        ));
    }
    
    // Recursive case: n >= 4
    // Build from the right: first n-1 edges fuse, then couple with the last edge
    // 
    // For n edges with alpha = [j_01, j_012, ..., j_0...n-2]:
    //   The left tensor (first n-1 edges) has last dimension corresponding to alpha[n-4]
    //   Then build CG3: alpha[n-4] ⊗ edges[n-2] → alpha[n-3]
    // For n=4: alpha[0] = j_01 (coupling of first 2 edges)
    // For n=5: alpha[1] = j_012 (coupling of first 3 edges)
    
    // Base case for recursion: n = 4
    // Build tensor for first 3 edges (which is handled by build_cg3)
    let left_tensor = if n == 4 {
        // For n=4, alpha has 2 elements: [j_01, j_012]
        // We need to build CG3 for first 2 edges: e0 ⊗ e1 → alpha[0]
        build_cg3(
            edges[0].j,
            edges[1].j,
            alpha[0],
        )?
    } else {
        // n > 4: recursively build tensor for first n-1 edges
        build_fusion_tree_recursive(&edges[..n-1], &alpha[..n-3])?
    };
    
    // Build CG3: alpha[n-4] ⊗ edges[n-2] → alpha[n-3]
    let cg3_tensor = build_cg3(
        alpha[n - 4],
        edges[n - 2].j,
        alpha[n - 3],
    )?;
    
    // Contract: last axis of left_tensor with first axis of cg3_tensor
    contract_last_with_first(&left_tensor, &cg3_tensor)
}

/// Contract last axis of tensor A with first axis of tensor B
///
/// Uses a simple for-loop implementation for small index dimensions.
///
/// # Arguments
/// * `a` - Left tensor with shape [..., k]
/// * `b` - Right tensor with shape [k, ...]
///
/// # Returns
/// Contracted tensor with shape [..., ...]
fn contract_last_with_first(a: &ArrayD<f64>, b: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();
    
    if a_ndim == 0 || b_ndim == 0 {
        return Err(YuzuhaError::InvalidContraction(
            "Cannot contract scalar tensors".to_string(),
        ));
    }
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    let contract_dim = a_shape[a_ndim - 1];
    if contract_dim != b_shape[0] {
        return Err(YuzuhaError::InvalidContraction(
            format!("Incompatible contraction dimensions: {} vs {}", contract_dim, b_shape[0]),
        ));
    }
    
    // Build output shape: a.shape[..a_ndim-1] + b.shape[1..]
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&a_shape[..a_ndim - 1]);
    out_shape.extend_from_slice(&b_shape[1..]);
    
    let mut result = Array::zeros(IxDyn(&out_shape));
    
    // Compute strides for efficient indexing
    let a_outer_size: usize = a_shape[..a_ndim - 1].iter().product();
    let b_inner_size: usize = b_shape[1..].iter().product();
    
    // Perform contraction using explicit loops
    for a_outer_idx in 0..a_outer_size {
        for b_inner_idx in 0..b_inner_size {
            let mut sum = 0.0;
            
            for k in 0..contract_dim {
                // Get value from a
                let a_flat_idx = a_outer_idx * contract_dim + k;
                let a_val = a.as_slice().unwrap()[a_flat_idx];
                
                // Get value from b
                let b_flat_idx = k * b_inner_size + b_inner_idx;
                let b_val = b.as_slice().unwrap()[b_flat_idx];
                
                sum += a_val * b_val;
            }
            
            // Store in result
            let out_flat_idx = a_outer_idx * b_inner_size + b_inner_idx;
            result.as_slice_mut().unwrap()[out_flat_idx] = sum;
        }
    }
    
    Ok(result)
}

/// Helper: Set a slice of an array along the OM axis (last axis)
fn set_om_slice(
    data: &mut ArrayD<f64>,
    om_idx: usize,
    slice_data: &ArrayD<f64>,
    n_external: usize,
) -> Result<()> {
    // Build a slice specification for the OM axis
    let mut slice_vec = vec![ndarray::SliceInfoElem::Slice {
        start: 0,
        end: None,
        step: 1,
    }; n_external];
    slice_vec.push(ndarray::SliceInfoElem::Index(om_idx as isize));
    
    let slice_info = ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::try_from(slice_vec)
        .map_err(|_| YuzuhaError::InvalidCGTSpec("Invalid slice".to_string()))?;
    
    let mut view = data.slice_mut(slice_info);
    view.assign(slice_data);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::TestCacheGuard;
    use crate::core::Edge;
    use approx::assert_relative_eq;
    use ndarray::Axis;

    #[test]
    fn test_build_canonical_basis_data_single_edge_errors() {
        let _guard = TestCacheGuard::new();
        
        let j1 = crate::core::Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![Edge::incoming(j1)]).unwrap();
        
        // Should error for n=1 (less than minimum of 3)
        assert!(build_canonical_basis_data(&spec).is_err());
    }
    
    #[test]
    fn test_build_canonical_basis_data_two_edges_errors() {
        let _guard = TestCacheGuard::new();
        
        let j1 = crate::core::Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
        ]).unwrap();
        
        // Should error for n=2 (less than minimum of 3)
        assert!(build_canonical_basis_data(&spec).is_err());
    }

    #[test]
    fn test_build_canonical_basis_three_edges() {
        let _guard = TestCacheGuard::new();
        
        // Three j=1 spins: two incoming, one outgoing
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Shape should be [3, 3, 3, om_dim]
        assert_eq!(data.ndim(), 4);
        assert_eq!(data.shape()[0], 3); // dim(j1)
        assert_eq!(data.shape()[1], 3); // dim(j1)
        assert_eq!(data.shape()[2], 3); // dim(j1)
        assert_eq!(data.shape()[3], om_dim);

        // Verify each basis element has unit Frobenius norm
        for alpha_idx in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: contract over first 2 axes should give delta * identity
        // With sqrt(2j+1) normalization, diagonal is 1/(2j+1) for last leg
        let j_last = spec.edges[2].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha_idx in 0..om_dim {
            for beta_idx in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., alpha_idx]);
                let slice_beta = data.slice(ndarray::s![.., .., .., beta_idx]);
                
                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1)],
                    &[Axis(0), Axis(1)],
                );
                
                for i in 0..3 {
                    for j in 0..3 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_canonical_basis_four_edges() {
        let _guard = TestCacheGuard::new();
        
        // Four j=1/2 spins: three incoming, one outgoing
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Shape should be [2, 2, 2, 2, om_dim]
        assert_eq!(data.ndim(), 5);
        for i in 0..4 {
            assert_eq!(data.shape()[i], 2); // dim(j=1/2)
        }
        assert_eq!(data.shape()[4], om_dim);

        for alpha_idx in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: contract over first 3 axes should give delta * identity
        // With sqrt(2j+1) normalization, diagonal is 1/(2j+1) for last leg
        let j_last = spec.edges[3].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha_idx in 0..om_dim {
            for beta_idx in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
                let slice_beta = data.slice(ndarray::s![.., .., .., .., beta_idx]);
                
                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1), Axis(2)],
                    &[Axis(0), Axis(1), Axis(2)],
                );
                
                for i in 0..2 {
                    for j in 0..2 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_canonical_basis_three_edges_all_outgoing() {
        let _guard = TestCacheGuard::new();
        
        // Three j=1 spins: all outgoing (non-canonical)
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::outgoing(j1),
            Edge::outgoing(j1),
            Edge::outgoing(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();

        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();

        for alpha_idx in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: contract over first 2 axes should give delta * identity
        // With sqrt(2j+1) normalization, diagonal is 1/(2j+1) for last leg
        let j_last = spec.edges[2].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha_idx in 0..om_dim {
            for beta_idx in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., alpha_idx]);
                let slice_beta = data.slice(ndarray::s![.., .., .., beta_idx]);
                
                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1)],
                    &[Axis(0), Axis(1)],
                );
                
                for i in 0..3 {
                    for j in 0..3 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_canonical_basis_three_edges_mixed_directions() {
        let _guard = TestCacheGuard::new();
        
        // Three edges with mixed directions: out, in, out (non-canonical)
        // Use j=1/2, j=1/2, j=1 which can couple to j=0
        let j_half = Spin::new(1).unwrap();
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::outgoing(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();

        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();

        for alpha in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., alpha]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: with sqrt(2j+1) normalization, diagonal is 1/(2j+1)
        let j_last = spec.edges[2].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha in 0..om_dim {
            for beta in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., alpha]);
                let slice_beta = data.slice(ndarray::s![.., .., .., beta]);

                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1)],
                    &[Axis(0), Axis(1)],
                );

                for i in 0..3 {
                    for j in 0..3 {
                        let expected = if alpha == beta && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_canonical_basis_four_edges_all_incoming() {
        let _guard = TestCacheGuard::new();
        
        // Four j=1/2 spins: all incoming (non-canonical)
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();

        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();

        for alpha_idx in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: with sqrt(2j+1) normalization, diagonal is 1/(2j+1)
        let j_last = spec.edges[3].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha_idx in 0..om_dim {
            for beta_idx in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
                let slice_beta = data.slice(ndarray::s![.., .., .., .., beta_idx]);

                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1), Axis(2)],
                    &[Axis(0), Axis(1), Axis(2)],
                );

                for i in 0..2 {
                    for j in 0..2 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_canonical_basis_four_edges_alternating_directions() {
        let _guard = TestCacheGuard::new();
        
        // Four j=1 spins: out, in, out, in (non-canonical)
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::outgoing(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
            Edge::incoming(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();

        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();

        for alpha_idx in 0..om_dim {
            let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
            let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
            assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
        }

        // Test orthonormality: with sqrt(2j+1) normalization, diagonal is 1/(2j+1)
        let j_last = spec.edges[3].j;
        let expected_diag = 1.0 / (j_last.dimension() as f64);
        for alpha_idx in 0..om_dim {
            for beta_idx in 0..om_dim {
                let slice_alpha = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
                let slice_beta = data.slice(ndarray::s![.., .., .., .., beta_idx]);
                
                let result = ndarray_einsum::tensordot(
                    &slice_alpha,
                    &slice_beta,
                    &[Axis(0), Axis(1), Axis(2)],
                    &[Axis(0), Axis(1), Axis(2)],
                );
                
                for i in 0..3 {
                    for j in 0..3 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            expected_diag
                        } else {
                            0.0
                        };
                        assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_build_single_om_tensor_three_edges() {
        let _guard = TestCacheGuard::new();
        
        // Three j=1 spins: two incoming, one outgoing
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::outgoing(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        // Test each alpha configuration
        for alpha in spec.alphas.iter() {
            let tensor = build_single_om_tensor(&spec, alpha).unwrap();
            
            // Shape should be [3, 3, 3] for three j=1 spins
            assert_eq!(tensor.shape(), &[3, 3, 3]);
            
            // Test normalization: contract over first 2 axes should give identity
            let result = ndarray_einsum::tensordot(
                &tensor,
                &tensor,
                &[Axis(0), Axis(1)],
                &[Axis(0), Axis(1)],
            );
            
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_build_single_om_tensor_four_edges() {
        let _guard = TestCacheGuard::new();
        
        // Four j=1/2 spins: three incoming, one outgoing
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::outgoing(j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        // Test each alpha configuration
        for alpha in spec.alphas.iter() {
            let tensor = build_single_om_tensor(&spec, alpha).unwrap();
            
            // Shape should be [2, 2, 2, 2] for four j=1/2 spins
            assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
            
            // Test normalization: contract over first 3 axes should give identity
            let result = ndarray_einsum::tensordot(
                &tensor,
                &tensor,
                &[Axis(0), Axis(1), Axis(2)],
                &[Axis(0), Axis(1), Axis(2)],
            );
            
            for i in 0..2 {
                for j in 0..2 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(result[[i, j]], expected, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_build_single_om_tensor_invalid_cases() {
        let _guard = TestCacheGuard::new();
        
        // Test that n < 3 returns error
        let j1 = Spin::new(2).unwrap();
        
        // n = 2 case
        let edges_2 = vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
        ];
        let spec_2 = CGSpec::from_edges(edges_2).unwrap();
        let alpha_2 = &spec_2.alphas[0];
        
        let result = build_single_om_tensor(&spec_2, alpha_2);
        assert!(result.is_err(), "n=2 should return error");
        
        // n = 1 case
        let edges_1 = vec![Edge::incoming(j1)];
        let spec_1 = CGSpec::from_edges(edges_1).unwrap();
        let alpha_1 = &spec_1.alphas[0];
        
        let result = build_single_om_tensor(&spec_1, alpha_1);
        assert!(result.is_err(), "n=1 should return error");
    }

    #[test]
    fn test_canonical_basis_normalization() {
        let _guard = TestCacheGuard::new();
        
        // Test that each OM slice is properly normalized
        // Use j=1/2, j=1/2, j=1 which can couple to j=0
        let j_half = Spin::new(1).unwrap();
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        
        // For each OM configuration, check that the tensor has reasonable values
        for om_idx in 0..spec.om_dimension() {
            let mut sum_squares = 0.0;
            
            // Sum over all external indices for this OM configuration
            let dim0 = spec.edges[0].dimension();
            let dim1 = spec.edges[1].dimension();
            let dim2 = spec.edges[2].dimension();
            for i0 in 0..dim0 {
                for i1 in 0..dim1 {
                    for i2 in 0..dim2 {
                        let val = data[[i0, i1, i2, om_idx]];
                        sum_squares += val * val;
                    }
                }
            }
            
            // Each OM configuration should have some non-zero contribution
            assert!(sum_squares > 0.0, "OM configuration {} should be non-zero", om_idx);
        }
    }
}
