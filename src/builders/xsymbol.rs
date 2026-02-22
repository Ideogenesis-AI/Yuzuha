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

//! X-symbol computation for SU(2) tensor network contractions
//!
//! This module provides the main API for computing X-symbols, which represent
//! the coupling coefficients for contracting two coupled gauge trees (CGTs).

use crate::builders::builders::build_canonical_basis_data;
use crate::builders::fs_phase::compute_fs_phase;
use crate::core::{CGSpec, Contraction};
use crate::error::Result;
use ndarray::{Array3, Axis};
use std::collections::HashSet;

/// X-symbol tensor with outer multiplicity (OM) index metadata
///
/// Represents X^γ_{αβ} where α, β, γ are OM indices (internal spin configurations).
#[derive(Debug, Clone)]
pub struct XSymbol {
    /// 3D array: X[α, β, γ]
    pub data: Array3<f64>,
    /// CG specification for input CGT A (contains edges and OM configurations)
    pub spec_a: CGSpec,
    /// CG specification for input CGT B (contains edges and OM configurations)
    pub spec_b: CGSpec,
    /// CG specification for output CGT C (contains edges and OM configurations)
    pub spec_c: CGSpec,
}

impl XSymbol {
    /// Get the OM dimensions (dim_α, dim_β, dim_γ)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1], shape[2])
    }

    /// Get a specific X-symbol element
    pub fn get(&self, alpha_idx: usize, beta_idx: usize, gamma_idx: usize) -> Option<f64> {
        self.data.get([alpha_idx, beta_idx, gamma_idx]).copied()
    }

    /// Check if X-symbol is real-valued (for SU(2) it should be)
    pub fn is_real(&self, _epsilon: f64) -> bool {
        // For SU(2), X-symbols should be real
        // In Rust all our f64 are real
        true
    }
}

/// Compute X-symbol for contracting two CGTs
///
/// Given two CGTs A and B with a contraction specification, computes the
/// X-symbol tensor X^γ_{αβ} representing the coupling to output CGT C.
///
/// The X-symbol is computed by:
/// 1. Building canonical basis tensors for A and B
/// 2. Contracting them over the specified edges (keeping OM indices separate)
/// 3. Projecting the result onto the canonical basis of C
///
/// # Arguments
/// * `spec_a` - First CG specification
/// * `spec_b` - Second CG specification
/// * `contraction` - Specification of which edges to contract
///
/// # Returns
/// X-symbol with full OM index metadata: X[α, β, γ]
pub fn compute_xsymbol(
    spec_a: &CGSpec,
    spec_b: &CGSpec,
    contraction: &Contraction,
) -> Result<XSymbol> {
    // Validate contraction
    contraction.validate(spec_a, spec_b)?;

    // Determine output spec
    let spec_c = build_output_spec(spec_a, spec_b, contraction)?;

    // Build canonical basis tensors
    // basis_a: [dims_a..., om_a]
    // basis_b: [dims_b..., om_b]
    // basis_c: [dims_c..., om_c]
    let basis_a = build_canonical_basis_data(spec_a)?;
    let basis_b = build_canonical_basis_data(spec_b)?;
    let basis_c = build_canonical_basis_data(&spec_c)?;

    // Compute the Frobenius-Schur (FS) phase factor for this contraction.
    let fs_phase = compute_fs_phase(spec_a, spec_b, contraction);

    // Contract basis_a and basis_b over specified external edges
    // The last axis of each basis is the OM axis, which we keep separate
    let axes_a_to_contract: Vec<_> = contraction.axes_a.iter().map(|&i| Axis(i)).collect();
    let axes_b_to_contract: Vec<_> = contraction.axes_b.iter().map(|&i| Axis(i)).collect();
    
    let contracted = ndarray_einsum::tensordot(
        &basis_a,
        &basis_b,
        &axes_a_to_contract,
        &axes_b_to_contract,
    );
    
    // After tensordot, contracted has shape: [remaining_a_external..., om_a, remaining_b_external..., om_b]
    // Calculate positions of OM axes:
    let n_a_external = spec_a.num_external();
    let n_remaining_a_external = n_a_external - contraction.axes_a.len();
    // om_a is at position: n_remaining_a_external
    // om_b is at position: contracted.ndim() - 1 (always last)
    let om_a_axis = n_remaining_a_external;
    let om_b_axis = contracted.ndim() - 1;
    
    // Contract over all external dimensions at once, keeping OM dimensions uncontracted
    // contracted: [remaining_a_external..., om_a, remaining_b_external..., om_b]
    // basis_c: [c_external..., om_c]
    // Result: [om_a, om_b, om_c]
    
    // Build list of external axes in contracted tensor (all except om_a and om_b)
    let mut contracted_external_axes = Vec::new();
    for i in 0..contracted.ndim() {
        if i != om_a_axis && i != om_b_axis {
            contracted_external_axes.push(Axis(i));
        }
    }
    
    // Build list of external axes in basis_c (all except last, which is om_c)
    let basis_c_external_axes: Vec<_> = (0..basis_c.ndim() - 1).map(Axis).collect();
    
    // Contract all external dimensions in one go
    let x_tensor = ndarray_einsum::tensordot(
        &contracted,
        &basis_c,
        &contracted_external_axes,
        &basis_c_external_axes,
    );
    
    // x_tensor now has shape [om_a, om_b, om_c]
    // Convert to Array3 (it should already be 3D) and apply the FS phase
    let x_data = x_tensor
        .into_dimensionality::<ndarray::Ix3>()
        .expect("X-symbol should be 3-dimensional")
        .mapv(|v| v * fs_phase);

    Ok(XSymbol {
        data: x_data,
        spec_a: spec_a.clone(),
        spec_b: spec_b.clone(),
        spec_c,
    })
}

/// Build the output spec from two input specs and contraction specification
pub fn build_output_spec(
    spec_a: &CGSpec,
    spec_b: &CGSpec,
    contraction: &Contraction,
) -> Result<CGSpec> {
    let contracted_a: HashSet<_> = contraction.axes_a.iter().copied().collect();
    let contracted_b: HashSet<_> = contraction.axes_b.iter().copied().collect();

    // Collect uncontracted edges from A and B
    let mut output_edges = Vec::new();

    for (idx, edge) in spec_a.edges.iter().enumerate() {
        if !contracted_a.contains(&idx) {
            output_edges.push(edge.clone());
        }
    }

    for (idx, edge) in spec_b.edges.iter().enumerate() {
        if !contracted_b.contains(&idx) {
            output_edges.push(edge.clone());
        }
    }

    // Use from_edges to automatically enumerate alphas
    CGSpec::from_edges(output_edges)
}


#[allow(dead_code)]
fn _future_api_placeholder() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Edge, Spin};

    #[test]
    fn test_xsymbol_basic() {
        let j_half = Spin::new(1).unwrap(); // j=1/2
        let j1 = Spin::new(2).unwrap();     // j=1

        // Need at least 3 edges for canonical basis
        // Use j=1/2 ⊗ j=1/2 ⊗ j=1 which can couple to j=0
        let spec_a = CGSpec::from_edges(
            vec![
                Edge::incoming(j_half),
                Edge::incoming(j_half),
                Edge::outgoing(j1),
            ],
        )
        .unwrap();

        let spec_b = CGSpec::from_edges(
            vec![
                Edge::incoming(j1),
                Edge::incoming(j_half),
                Edge::outgoing(j_half),
            ],
        )
        .unwrap();

        // Contract edge 2 from spec_a (j=1) with edge 0 from spec_b (j=1)
        let contraction = Contraction::new(&[2], &[0]);

        let x = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();

        // Check dimensions
        let (dim_a, dim_b, dim_c) = x.dimensions();
        assert!(dim_a > 0);
        assert!(dim_b > 0);
        assert!(dim_c > 0);
        
        // Check X†X by summing over alpha and beta for each gamma
        // For this specific configuration, we expect:
        // gamma=0: sum = 0 (inaccessible)
        // gamma=1: sum = 1/3 (1/dim(j=1))
        
        let mut sum_gamma0 = 0.0;
        let mut sum_gamma1 = 0.0;
        
        for alpha in 0..dim_a {
            for beta in 0..dim_b {
                let val0 = x.get(alpha, beta, 0).unwrap();
                sum_gamma0 += val0 * val0;
                
                let val1 = x.get(alpha, beta, 1).unwrap();
                sum_gamma1 += val1 * val1;
            }
        }
        
        assert!(sum_gamma0.abs() < 1e-10, "Expected gamma=0 to be inaccessible, got {}", sum_gamma0);
        assert!((sum_gamma1 - 1.0/3.0).abs() < 1e-6, "Expected gamma=1 sum = 1/3, got {}", sum_gamma1);
    }

    #[test]
    fn test_build_output_spec() {
        let j1 = Spin::new(2).unwrap();

        let spec_a = CGSpec::from_edges(
            vec![
                Edge::incoming(j1),
                Edge::incoming(j1),
                Edge::incoming(j1),
            ],
        )
        .unwrap();

        let spec_b = CGSpec::from_edges(
            vec![Edge::incoming(j1), Edge::incoming(j1)],
        )
        .unwrap();

        // Contract edge 1 from spec_a with edge 0 from spec_b
        let contraction = Contraction::new(&[1], &[0]);

        let spec_c = build_output_spec(&spec_a, &spec_b, &contraction).unwrap();

        // Should have 3 uncontracted edges: edges 0, 2 (from A), edge 1 (from B)
        // (edge 1 from A and edge 0 from B are contracted)
        assert_eq!(spec_c.num_external(), 3);
    }
}
