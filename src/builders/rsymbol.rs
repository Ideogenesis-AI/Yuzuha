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

//! R-symbol computation for SU(2) tensor permutations
//!
//! This module provides the main API for computing R-symbols, which represent
//! the transformation of outer multiplicity (OM) indices under permutation of
//! tensor legs.

use crate::builders::builders::build_canonical_basis_data;
use crate::core::CGSpec;
use crate::error::Result;
use ndarray::{Array2, Axis};

/// R-symbol matrix representing permutation in OM space
///
/// Represents R^β_α where α is the OM index of the original CGSpec
/// and β is the OM index of the permuted CGSpec.
#[derive(Debug, Clone)]
pub struct RSymbol {
    /// 2D array: R[α, β] where α is original OM index, β is permuted OM index
    pub data: Array2<f64>,
    /// Original CG specification (before permutation)
    pub spec_original: CGSpec,
    /// Permuted CG specification (after permutation)
    pub spec_permuted: CGSpec,
    /// Permutation applied to external edges
    pub permutation: Vec<usize>,
}

impl RSymbol {
    /// Get the OM dimensions (dim_original, dim_permuted)
    pub fn dimensions(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    /// Get a specific R-symbol element
    pub fn get(&self, alpha_idx: usize, beta_idx: usize) -> Option<f64> {
        self.data.get([alpha_idx, beta_idx]).copied()
    }

    /// Check if R-symbol is real-valued (for SU(2) it should be)
    pub fn is_real(&self, _epsilon: f64) -> bool {
        // For SU(2), R-symbols should be real
        // In Rust all our f64 are real
        true
    }

    /// Check if R-symbol is unitary (R†R = I and RR† = I)
    pub fn is_unitary(&self, epsilon: f64) -> bool {
        let (m, n) = self.dimensions();
        
        // For a unitary matrix, we need m == n
        if m != n {
            return false;
        }
        
        // Check R†R = I
        let r_dagger_r = self.data.t().dot(&self.data);
        for i in 0..m {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (r_dagger_r[[i, j]] - expected).abs() > epsilon {
                    return false;
                }
            }
        }
        
        true
    }
}

/// Compute R-symbol for permuting tensor legs
///
/// Given a CGSpec and a permutation, computes the R-symbol matrix R^β_α
/// representing how OM indices transform under the permutation.
///
/// The R-symbol is computed by:
/// 1. Building canonical basis tensor for the original CGSpec
/// 2. Permuting the external axes of the tensor
/// 3. Building canonical basis tensor for the permuted CGSpec
/// 4. Projecting the permuted tensor onto the new canonical basis
///
/// # Arguments
/// * `spec` - Original CG specification
/// * `permutation` - Permutation of external edge indices (must be a valid permutation)
///
/// # Returns
/// R-symbol matrix with shape [om_original, om_permuted]
///
/// # Errors
/// Returns an error if:
/// - The permutation is invalid (wrong length, invalid indices, etc.)
/// - The canonical basis cannot be built
///
/// # Examples
///
/// ```
/// # use yuzuha::{CGSpec, Edge, Spin};
/// # use yuzuha::builders::rsymbol::compute_rsymbol;
/// let j_half = Spin::new(1).unwrap();
/// let spec = CGSpec::from_edges(vec![
///     Edge::incoming(j_half),
///     Edge::incoming(j_half),
///     Edge::incoming(j_half),
///     Edge::incoming(j_half),
/// ]).unwrap();
///
/// // Swap first two legs: [1, 0, 2, 3]
/// let permutation = vec![1, 0, 2, 3];
/// let r_symbol = compute_rsymbol(&spec, &permutation).unwrap();
/// ```
pub fn compute_rsymbol(spec: &CGSpec, permutation: &[usize]) -> Result<RSymbol> {
    // Validate permutation
    validate_permutation(spec, permutation)?;

    // Build the permuted CGSpec
    let spec_permuted = build_permuted_spec(spec, permutation)?;

    // Build canonical basis for original spec
    // basis_original: [dims_original..., om_original]
    let basis_original = build_canonical_basis_data(spec)?;

    // Build canonical basis for permuted spec
    // basis_permuted: [dims_permuted..., om_permuted]
    let basis_permuted = build_canonical_basis_data(&spec_permuted)?;

    // Permute the external axes of basis_original
    // Last axis is OM, which we don't permute
    let n_external = spec.num_external();
    let basis_original_permuted = permute_external_axes(&basis_original, permutation, n_external);

    // Now contract: basis_original_permuted with basis_permuted over all external dimensions
    // basis_original_permuted: [dims_permuted..., om_original]
    // basis_permuted: [dims_permuted..., om_permuted]
    // Result: [om_original, om_permuted]

    // Build list of external axes to contract (all except the last OM axis)
    let external_axes: Vec<_> = (0..n_external).map(Axis).collect();

    // Contract all external dimensions
    let r_tensor = ndarray_einsum::tensordot(
        &basis_original_permuted,
        &basis_permuted,
        &external_axes,
        &external_axes,
    );

    // r_tensor now has shape [om_original, om_permuted]
    // Convert to Array2 (it should already be 2D)
    let r_data = r_tensor
        .into_dimensionality::<ndarray::Ix2>()
        .expect("R-symbol should be 2-dimensional");

    Ok(RSymbol {
        data: r_data,
        spec_original: spec.clone(),
        spec_permuted,
        permutation: permutation.to_vec(),
    })
}

/// Validate that the permutation is valid for the given CGSpec
fn validate_permutation(spec: &CGSpec, permutation: &[usize]) -> Result<()> {
    let n = spec.num_external();

    // Check length
    if permutation.len() != n {
        return Err(crate::error::YuzuhaError::InvalidCGTSpec(format!(
            "Permutation length {} does not match number of external edges {}",
            permutation.len(),
            n
        )));
    }

    // Check that all indices are valid and unique
    let mut seen = vec![false; n];
    for &idx in permutation {
        if idx >= n {
            return Err(crate::error::YuzuhaError::InvalidCGTSpec(format!(
                "Permutation index {} out of range (max {})",
                idx,
                n - 1
            )));
        }
        if seen[idx] {
            return Err(crate::error::YuzuhaError::InvalidCGTSpec(format!(
                "Permutation index {} appears more than once",
                idx
            )));
        }
        seen[idx] = true;
    }

    Ok(())
}

/// Build the permuted CGSpec by reordering edges according to permutation
fn build_permuted_spec(spec: &CGSpec, permutation: &[usize]) -> Result<CGSpec> {
    let mut permuted_edges = Vec::with_capacity(spec.edges.len());
    
    for &idx in permutation {
        permuted_edges.push(spec.edges[idx].clone());
    }

    CGSpec::from_edges(permuted_edges)
}

/// Permute external axes of a tensor
///
/// Given a tensor with shape [dims..., om] where the last axis is the OM index,
/// permute only the external axes according to the given permutation.
fn permute_external_axes(
    tensor: &ndarray::ArrayD<f64>,
    permutation: &[usize],
    n_external: usize,
) -> ndarray::ArrayD<f64> {
    // Build the full permutation including the OM axis
    // External axes are permuted according to `permutation`,
    // OM axis stays at the end
    let mut full_permutation = Vec::with_capacity(n_external + 1);
    for &idx in permutation {
        full_permutation.push(idx);
    }
    full_permutation.push(n_external); // OM axis stays last

    tensor.clone().permuted_axes(full_permutation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::TestCacheGuard;
    use crate::core::{Edge, Spin};
    use approx::assert_relative_eq;

    #[test]
    fn test_rsymbol_identity_permutation() {
        let _guard = TestCacheGuard::new();
        
        // Identity permutation should give identity matrix
        let j_half = Spin::new(1).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ])
        .unwrap();

        let permutation = vec![0, 1, 2, 3]; // Identity
        let r = compute_rsymbol(&spec, &permutation).unwrap();

        let (dim_original, dim_permuted) = r.dimensions();
        assert_eq!(dim_original, dim_permuted);

        // Should be identity matrix
        for i in 0..dim_original {
            for j in 0..dim_permuted {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(r.get(i, j).unwrap(), expected, epsilon = 1e-10);
            }
        }

        assert!(r.is_unitary(1e-10));
    }

    #[test]
    fn test_rsymbol_swap_first_two() {
        let _guard = TestCacheGuard::new();
        
        // Swap first two legs
        let j_half = Spin::new(1).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ])
        .unwrap();

        let permutation = vec![1, 0, 2, 3]; // Swap first two
        let r = compute_rsymbol(&spec, &permutation).unwrap();

        let (dim_original, dim_permuted) = r.dimensions();
        assert!(dim_original > 0);
        assert_eq!(dim_original, dim_permuted);

        // R should be unitary
        assert!(r.is_unitary(1e-10));

        // Applying the permutation twice should give identity
        let r_twice = r.data.dot(&r.data);
        for i in 0..dim_original {
            for j in 0..dim_original {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(r_twice[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rsymbol_three_legs() {
        let _guard = TestCacheGuard::new();
        
        // Test with 3 legs (simpler case)
        let j1 = Spin::new(2).unwrap(); // j=1
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j1),
            Edge::incoming(j1),
            Edge::incoming(j1),
        ])
        .unwrap();

        // Cyclic permutation: [1, 2, 0]
        let permutation = vec![1, 2, 0];
        let r = compute_rsymbol(&spec, &permutation).unwrap();

        let (dim_original, dim_permuted) = r.dimensions();
        assert!(dim_original > 0);
        assert_eq!(dim_original, dim_permuted);

        // R should be unitary
        assert!(r.is_unitary(1e-10));
    }

    #[test]
    fn test_validate_permutation() {
        let j_half = Spin::new(1).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ])
        .unwrap();

        // Valid permutation
        assert!(validate_permutation(&spec, &[3, 2, 1, 0]).is_ok());
        assert!(validate_permutation(&spec, &[0, 1, 2, 3]).is_ok());

        // Invalid: wrong length
        assert!(validate_permutation(&spec, &[0, 1]).is_err());
        assert!(validate_permutation(&spec, &[0, 1, 2, 3, 4]).is_err());

        // Invalid: out of range index
        assert!(validate_permutation(&spec, &[0, 1, 2, 4]).is_err());

        // Invalid: duplicate index
        assert!(validate_permutation(&spec, &[0, 0, 2, 3]).is_err());
    }

    #[test]
    fn test_build_permuted_spec() {
        let j_half = Spin::new(1).unwrap();
        let j1 = Spin::new(2).unwrap();
        
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j1),
            Edge::incoming(j_half),
        ])
        .unwrap();

        let permutation = vec![2, 0, 1]; // Rearrange to [j_half, j_half, j1]
        let spec_permuted = build_permuted_spec(&spec, &permutation).unwrap();

        assert_eq!(spec_permuted.num_external(), 3);
        assert_eq!(spec_permuted.edges[0].j, j_half);
        assert_eq!(spec_permuted.edges[1].j, j_half);
        assert_eq!(spec_permuted.edges[2].j, j1);
    }

    #[test]
    fn test_rsymbol_dimensions() {
        let _guard = TestCacheGuard::new();
        
        let j_half = Spin::new(1).unwrap();
        let spec = CGSpec::from_edges(vec![
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
            Edge::incoming(j_half),
        ])
        .unwrap();

        let permutation = vec![1, 0, 2, 3];
        let r = compute_rsymbol(&spec, &permutation).unwrap();

        let (dim_a, dim_b) = r.dimensions();
        assert_eq!(dim_a, spec.om_dimension());
        assert_eq!(dim_b, spec.om_dimension()); // Same spec structure, so same OM dimension
    }
}
