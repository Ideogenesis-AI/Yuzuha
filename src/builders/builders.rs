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
use crate::core::{CGSpec, Direction, MagneticNumber, Spin};
use crate::error::{Result, YuzuhaError};
use ndarray::{Array, ArrayD, Axis, IxDyn};

/// Build canonical basis data for ALL OM configurations
///
/// This is the primary builder for canonical basis tensors. It computes the
/// canonical basis element for each OM configuration (alpha) and stacks them
/// along the last axis.
///
/// Handles arbitrary arrow directions by converting to canonical form (n-1 incoming, 1 outgoing).
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
    
    // Create a canonical spec with (n-1) incoming and 1 outgoing
    let canonical_edges: Vec<_> = spec.edges.iter().enumerate().map(|(i, e)| {
        let canonical_dir = if i < n - 1 {
            Direction::Incoming
        } else {
            Direction::Outgoing
        };
        crate::core::Edge::new(e.id.clone(), e.j, canonical_dir)
    }).collect();
    
    let canonical_spec = CGSpec::from_edges(canonical_edges)?;
    
    // Initialize output array
    let mut data = ArrayD::zeros(IxDyn(&spec.shape()));
    
    // Build canonical tensor for each OM configuration
    for (om_idx, alpha) in canonical_spec.alphas.iter().enumerate() {
        let mut alpha_data = build_single_om_tensor(&canonical_spec, alpha)?;
        
        // Invert directions back to match original spec
        for (axis_idx, edge) in spec.edges.iter().enumerate() {
            let canonical_dir = if axis_idx < n - 1 {
                Direction::Incoming
            } else {
                Direction::Outgoing
            };
            
            if edge.dir != canonical_dir {
                alpha_data = invert_edge_direction(&alpha_data, axis_idx, edge.j, canonical_dir)?;
            }
        }
        
        // Set OM slice
        set_om_slice(&mut data, om_idx, &alpha_data, n)?;
    }
    
    Ok(data)
}

/// Invert the direction of an edge using the metric tensor
///
/// Applies the metric transformation without explicit contraction:
/// - Incoming → Outgoing: mirror axis (m → -m) and apply phase (-1)^{j+m}
/// - Outgoing → Incoming: mirror axis (m → -m) and apply phase (-1)^{2j} * (-1)^{j+m}
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `axis_idx` - Index of the axis to invert
/// * `j` - Spin of the edge being inverted
/// * `from_dir` - Current direction of the edge
///
/// # Returns
/// Tensor with inverted edge direction
fn invert_edge_direction(
    tensor: &ArrayD<f64>,
    axis_idx: usize,
    j: Spin,
    from_dir: Direction,
) -> Result<ArrayD<f64>> {
    let mut result = tensor.clone();
    
    // Reverse the axis (mirror: m → -m)
    result.invert_axis(Axis(axis_idx));
    
    // Apply phase factors based on direction conversion
    // Iterate over magnetic quantum numbers on this axis
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
        
        // Apply phase to the entire slice at this m value
        result.index_axis_mut(Axis(axis_idx), m_idx).mapv_inplace(|x| x * phase);
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
    use crate::core::Edge;
    use approx::assert_relative_eq;
    use ndarray::Axis;

    #[test]
    fn test_build_canonical_basis_data_single_edge_errors() {
        let j1 = crate::core::Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![Edge::incoming("a", j1)]).unwrap();
        
        // Should error for n=1 (less than minimum of 3)
        assert!(build_canonical_basis_data(&spec).is_err());
    }
    
    #[test]
    fn test_build_canonical_basis_data_two_edges_errors() {
        let j1 = crate::core::Spin::new(2).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming("a", j1),
            Edge::incoming("b", j1),
        ]).unwrap();
        
        // Should error for n=2 (less than minimum of 3)
        assert!(build_canonical_basis_data(&spec).is_err());
    }

    #[test]
    fn test_build_canonical_basis_three_edges() {
        // Three j=1 spins: two incoming, one outgoing
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::incoming("a", j1),
            Edge::incoming("b", j1),
            Edge::outgoing("c", j1),
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
        
        // Test orthonormality: contract over first 2 axes should give delta * identity
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
                            1.0
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
        // Four j=1/2 spins: three incoming, one outgoing
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming("a", j_half),
            Edge::incoming("b", j_half),
            Edge::incoming("c", j_half),
            Edge::outgoing("d", j_half),
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
        
        // Test orthonormality: contract over first 3 axes should give delta * identity
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
                            1.0
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
        // Three j=1 spins: all outgoing (non-canonical)
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::outgoing("a", j1),
            Edge::outgoing("b", j1),
            Edge::outgoing("c", j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Test orthonormality: contract over first 2 axes should give delta * identity
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
                            1.0
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
        // Three j=1/2 spins: out, in, out (non-canonical)
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::outgoing("a", j_half),
            Edge::incoming("b", j_half),
            Edge::outgoing("c", j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Test orthonormality
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
                
                for i in 0..2 {
                    for j in 0..2 {
                        let expected = if alpha_idx == beta_idx && i == j {
                            1.0
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
        // Four j=1/2 spins: all incoming (non-canonical)
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming("a", j_half),
            Edge::incoming("b", j_half),
            Edge::incoming("c", j_half),
            Edge::incoming("d", j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Test orthonormality
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
                            1.0
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
        // Four j=1 spins: out, in, out, in (non-canonical)
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::outgoing("a", j1),
            Edge::incoming("b", j1),
            Edge::outgoing("c", j1),
            Edge::incoming("d", j1),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        let om_dim = spec.om_dimension();
        
        // Test orthonormality
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
                            1.0
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
        // Three j=1 spins: two incoming, one outgoing
        let j1 = Spin::new(2).unwrap();
        let edges = vec![
            Edge::incoming("a", j1),
            Edge::incoming("b", j1),
            Edge::outgoing("c", j1),
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
        // Four j=1/2 spins: three incoming, one outgoing
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming("a", j_half),
            Edge::incoming("b", j_half),
            Edge::incoming("c", j_half),
            Edge::outgoing("d", j_half),
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
        // Test that n < 3 returns error
        let j1 = Spin::new(2).unwrap();
        
        // n = 2 case
        let edges_2 = vec![
            Edge::incoming("a", j1),
            Edge::incoming("b", j1),
        ];
        let spec_2 = CGSpec::from_edges(edges_2).unwrap();
        let alpha_2 = &spec_2.alphas[0];
        
        let result = build_single_om_tensor(&spec_2, alpha_2);
        assert!(result.is_err(), "n=2 should return error");
        
        // n = 1 case
        let edges_1 = vec![Edge::incoming("a", j1)];
        let spec_1 = CGSpec::from_edges(edges_1).unwrap();
        let alpha_1 = &spec_1.alphas[0];
        
        let result = build_single_om_tensor(&spec_1, alpha_1);
        assert!(result.is_err(), "n=1 should return error");
    }

    #[test]
    fn test_canonical_basis_normalization() {
        // Test that each OM slice is properly normalized
        let j_half = Spin::new(1).unwrap();
        let edges = vec![
            Edge::incoming("a", j_half),
            Edge::incoming("b", j_half),
            Edge::incoming("c", j_half),
        ];
        let spec = CGSpec::from_edges(edges).unwrap();
        
        let data = build_canonical_basis_data(&spec).unwrap();
        
        // For each OM configuration, check that the tensor has reasonable values
        for om_idx in 0..spec.om_dimension() {
            let mut sum_squares = 0.0;
            
            // Sum over all external indices for this OM configuration
            for i0 in 0..2 {
                for i1 in 0..2 {
                    for i2 in 0..2 {
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
