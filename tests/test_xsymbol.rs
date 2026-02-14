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

//! Tests for X-symbol consistency
//!
//! Verifies that X-symbols correctly transform outer multiplicity indices
//! during tensor contractions by comparing two equivalent computation paths.

use yuzuha::core::{CGSpec, Contraction, Edge, Spin};
use yuzuha::builders::builders::build_canonical_basis_data;
use yuzuha::builders::xsymbol::{compute_xsymbol, build_output_spec};
use yuzuha::builders::TestCacheGuard;
use ndarray::{ArrayD, Axis};
use rand::Rng;

/// Test X-symbol by comparing two ways of computing the same contracted tensor:
/// 1. Direct contraction: Contract CGTensor_A with CGTensor_B
/// 2. Via X-symbol: Use X-symbol to transform OM weights and build from basis_C
#[test]
fn test_xsymbol_consistency() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=1/2, j=1/2, j=1] (incoming, incoming, outgoing)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(1).unwrap()), // j=1/2
        Edge::incoming(Spin::new(1).unwrap()), // j=1/2
        Edge::outgoing(Spin::new(2).unwrap()), // j=1
    ]).unwrap();
    
    // CGSpec B has [j=1, j=1/2, j=1/2] (incoming, outgoing, outgoing)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edge 2 of A (j=1 outgoing) with edge 0 of B (j=1 incoming)
    let contraction = Contraction::new(&[2], &[0]);
    
    // Get OM dimensions
    let dim_a = spec_a.om_dimension();
    let dim_b = spec_b.om_dimension();
    
    assert!(dim_a > 0, "spec_a should have non-zero OM dimension");
    assert!(dim_b > 0, "spec_b should have non-zero OM dimension");
    
    // Generate random weights for OM indices
    let mut rng = rand::thread_rng();
    let w_a: Vec<f64> = (0..dim_a).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let w_b: Vec<f64> = (0..dim_b).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Build basis tensors
    let basis_a = build_canonical_basis_data(&spec_a).unwrap();
    let basis_b = build_canonical_basis_data(&spec_b).unwrap();
    
    // Build CGTensor A = sum_alpha w_a[alpha] * basis_a[..., alpha]
    let cg_tensor_a = build_weighted_tensor(&basis_a, &w_a);
    
    // Build CGTensor B = sum_beta w_b[beta] * basis_b[..., beta]
    let cg_tensor_b = build_weighted_tensor(&basis_b, &w_b);
    
    // Direct contraction: Contract CGTensor A with CGTensor B
    let cg_tensor_c = ndarray_einsum::tensordot(
        &cg_tensor_a,
        &cg_tensor_b,
        &[Axis(2)], // Contract axis 2 of A (j=1 edge)
        &[Axis(0)], // Contract axis 0 of B (j=1 edge)
    );
    
    // Compute X-symbol
    let xsymbol = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();
    let (dim_alpha, dim_beta, dim_gamma) = xsymbol.dimensions();
    
    assert_eq!(dim_alpha, dim_a);
    assert_eq!(dim_beta, dim_b);
    
    // Compute w_c via X-symbol: w_c[gamma] = sum_{alpha,beta} w_a[alpha] * w_b[beta] * X[alpha,beta,gamma]
    let mut w_c = vec![0.0; dim_gamma];
    for gamma in 0..dim_gamma {
        for alpha in 0..dim_alpha {
            for beta in 0..dim_beta {
                let x_val = xsymbol.get(alpha, beta, gamma).unwrap();
                w_c[gamma] += w_a[alpha] * w_b[beta] * x_val;
            }
        }
    }
    
    // Build output CGSpec
    let spec_c = build_output_spec(&spec_a, &spec_b, &contraction).unwrap();
    
    // Build basis_c
    let basis_c = build_canonical_basis_data(&spec_c).unwrap();
    
    // Build CGTensor C' = sum_gamma w_c[gamma] * basis_c[..., gamma]
    let cg_tensor_c_prime = build_weighted_tensor(&basis_c, &w_c);
    
    // Check that C == C'
    assert_eq!(cg_tensor_c.shape(), cg_tensor_c_prime.shape(),
        "Tensor shapes don't match");
    
    for idx in cg_tensor_c.iter().zip(cg_tensor_c_prime.iter()) {
        let (val_c, val_c_prime) = idx;
        assert!((val_c - val_c_prime).abs() < 1e-10,
            "Tensor mismatch: direct={}, via X-symbol={}", val_c, val_c_prime);
    }
}

/// Test with a different configuration: larger spins
#[test]
fn test_xsymbol_larger_spins() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=1, j=1, j=1] (incoming, incoming, outgoing)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has [j=1, j=1/2, j=1/2] (incoming, outgoing, outgoing)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edge 2 of A (j=1 outgoing) with edge 0 of B (j=1 incoming)
    let contraction = Contraction::new(&[2], &[0]);
    
    // Get OM dimensions
    let dim_a = spec_a.om_dimension();
    let dim_b = spec_b.om_dimension();
    
    assert!(dim_a > 0);
    assert!(dim_b > 0);
    
    // Generate random weights
    let mut rng = rand::thread_rng();
    let w_a: Vec<f64> = (0..dim_a).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let w_b: Vec<f64> = (0..dim_b).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Build basis tensors
    let basis_a = build_canonical_basis_data(&spec_a).unwrap();
    let basis_b = build_canonical_basis_data(&spec_b).unwrap();
    
    // Build CGTensors
    let cg_tensor_a = build_weighted_tensor(&basis_a, &w_a);
    let cg_tensor_b = build_weighted_tensor(&basis_b, &w_b);
    
    // Direct contraction
    let cg_tensor_c = ndarray_einsum::tensordot(
        &cg_tensor_a,
        &cg_tensor_b,
        &[Axis(2)],
        &[Axis(0)],
    );
    
    // Compute X-symbol
    let xsymbol = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();
    let (dim_alpha, dim_beta, dim_gamma) = xsymbol.dimensions();
    
    // Compute w_c
    let mut w_c = vec![0.0; dim_gamma];
    for gamma in 0..dim_gamma {
        for alpha in 0..dim_alpha {
            for beta in 0..dim_beta {
                let x_val = xsymbol.get(alpha, beta, gamma).unwrap();
                w_c[gamma] += w_a[alpha] * w_b[beta] * x_val;
            }
        }
    }
    
    // Build basis_c and CGTensor C'
    let spec_c = build_output_spec(&spec_a, &spec_b, &contraction).unwrap();
    let basis_c = build_canonical_basis_data(&spec_c).unwrap();
    let cg_tensor_c_prime = build_weighted_tensor(&basis_c, &w_c);
    
    // Check equality
    assert_eq!(cg_tensor_c.shape(), cg_tensor_c_prime.shape());
    
    for (val_c, val_c_prime) in cg_tensor_c.iter().zip(cg_tensor_c_prime.iter()) {
        assert!((val_c - val_c_prime).abs() < 1e-10,
            "Tensor mismatch: direct={}, via X-symbol={}", val_c, val_c_prime);
    }
}

/// Test with multiple contractions
#[test]
fn test_xsymbol_multiple_contractions() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=1/2, j=1/2, j=1, j=1] (in, in, out, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has [j=1, j=1, j=1/2, j=1/2] (in, in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edges 2,3 of A with edges 0,1 of B
    let contraction = Contraction::new(&[2, 3], &[0, 1]);
    
    // Get OM dimensions
    let dim_a = spec_a.om_dimension();
    let dim_b = spec_b.om_dimension();
    
    assert!(dim_a > 0);
    assert!(dim_b > 0);
    
    // Generate random weights
    let mut rng = rand::thread_rng();
    let w_a: Vec<f64> = (0..dim_a).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let w_b: Vec<f64> = (0..dim_b).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Build basis tensors
    let basis_a = build_canonical_basis_data(&spec_a).unwrap();
    let basis_b = build_canonical_basis_data(&spec_b).unwrap();
    
    // Build CGTensors
    let cg_tensor_a = build_weighted_tensor(&basis_a, &w_a);
    let cg_tensor_b = build_weighted_tensor(&basis_b, &w_b);
    
    // Direct contraction
    let cg_tensor_c = ndarray_einsum::tensordot(
        &cg_tensor_a,
        &cg_tensor_b,
        &[Axis(2), Axis(3)],
        &[Axis(0), Axis(1)],
    );
    
    // Compute X-symbol
    let xsymbol = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();
    let (dim_alpha, dim_beta, dim_gamma) = xsymbol.dimensions();
    
    // Compute w_c
    let mut w_c = vec![0.0; dim_gamma];
    for gamma in 0..dim_gamma {
        for alpha in 0..dim_alpha {
            for beta in 0..dim_beta {
                let x_val = xsymbol.get(alpha, beta, gamma).unwrap();
                w_c[gamma] += w_a[alpha] * w_b[beta] * x_val;
            }
        }
    }
    
    // Build basis_c and CGTensor C'
    let spec_c = build_output_spec(&spec_a, &spec_b, &contraction).unwrap();
    let basis_c = build_canonical_basis_data(&spec_c).unwrap();
    let cg_tensor_c_prime = build_weighted_tensor(&basis_c, &w_c);
    
    // Check equality
    assert_eq!(cg_tensor_c.shape(), cg_tensor_c_prime.shape());
    
    for (val_c, val_c_prime) in cg_tensor_c.iter().zip(cg_tensor_c_prime.iter()) {
        assert!((val_c - val_c_prime).abs() < 1e-10,
            "Tensor mismatch: direct={}, via X-symbol={}", val_c, val_c_prime);
    }
}

/// Test with spin-3/2 particles
#[test]
fn test_xsymbol_spin_3_2() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=3/2, j=1/2, j=1] (in, in, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(3).unwrap()),  // j=3/2
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has [j=1, j=3/2, j=1/2] (in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(3).unwrap()),  // j=3/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edge 2 of A with edge 0 of B
    let contraction = Contraction::new(&[2], &[0]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(2)], &[Axis(0)]);
}

/// Test with spin-2 particles
#[test]
fn test_xsymbol_spin_2() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=2, j=1, j=1] (in, in, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(4).unwrap()),  // j=2
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has [j=1, j=2, j=1] (in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(4).unwrap()),  // j=2
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // Contract edge 2 of A with edge 0 of B
    let contraction = Contraction::new(&[2], &[0]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(2)], &[Axis(0)]);
}

/// Test with 5-edge tensors
#[test]
fn test_xsymbol_5_edges() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has 5 edges [j=1/2, j=1/2, j=1, j=1, j=1] (in, in, in, out, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has 4 edges [j=1, j=1, j=1/2, j=1/2] (in, in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edges 3,4 of A with edges 0,1 of B
    let contraction = Contraction::new(&[3, 4], &[0, 1]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(3), Axis(4)], &[Axis(0), Axis(1)]);
}

/// Test with 6-edge tensors
#[test]
fn test_xsymbol_6_edges() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has 6 edges [j=1/2, j=1/2, j=1, j=1, j=1, j=1] (in, in, in, in, out, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has 4 edges [j=1, j=1, j=1/2, j=1/2] (in, in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edges 4,5 of A with edges 0,1 of B
    let contraction = Contraction::new(&[4, 5], &[0, 1]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(4), Axis(5)], &[Axis(0), Axis(1)]);
}

/// Test with mixed large spins and many edges
#[test]
fn test_xsymbol_mixed_large() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has [j=3/2, j=1, j=1/2, j=2, j=1] (in, in, in, out, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(3).unwrap()),  // j=3/2
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(1).unwrap()),  // j=1/2
        Edge::outgoing(Spin::new(4).unwrap()),  // j=2
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // CGSpec B has [j=2, j=1, j=3/2, j=1/2] (in, in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(4).unwrap()),  // j=2
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(3).unwrap()),  // j=3/2
        Edge::outgoing(Spin::new(1).unwrap()),  // j=1/2
    ]).unwrap();
    
    // Contract edges 3,4 of A with edges 0,1 of B
    let contraction = Contraction::new(&[3, 4], &[0, 1]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(3), Axis(4)], &[Axis(0), Axis(1)]);
}

/// Test with single edge contraction on large tensors
#[test]
fn test_xsymbol_single_contraction_large() {
    let _guard = TestCacheGuard::new();
    
    // Setup: CGSpec A has 6 edges [j=1, j=1, j=1, j=1, j=1, j=2] (in, in, in, in, in, out)
    let spec_a = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::incoming(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(4).unwrap()),  // j=2
    ]).unwrap();
    
    // CGSpec B has 3 edges [j=2, j=1, j=1] (in, out, out)
    let spec_b = CGSpec::from_edges(vec![
        Edge::incoming(Spin::new(4).unwrap()),  // j=2
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
        Edge::outgoing(Spin::new(2).unwrap()),  // j=1
    ]).unwrap();
    
    // Contract edge 5 of A with edge 0 of B
    let contraction = Contraction::new(&[5], &[0]);
    
    run_consistency_test(&spec_a, &spec_b, &contraction, &[Axis(5)], &[Axis(0)]);
}

/// Helper function to run consistency test
fn run_consistency_test(
    spec_a: &CGSpec,
    spec_b: &CGSpec,
    contraction: &Contraction,
    axes_a: &[Axis],
    axes_b: &[Axis],
) {
    let dim_a = spec_a.om_dimension();
    let dim_b = spec_b.om_dimension();
    
    assert!(dim_a > 0, "spec_a should have non-zero OM dimension");
    assert!(dim_b > 0, "spec_b should have non-zero OM dimension");
    
    // Generate random weights for OM indices
    let mut rng = rand::thread_rng();
    let w_a: Vec<f64> = (0..dim_a).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let w_b: Vec<f64> = (0..dim_b).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Build basis tensors
    let basis_a = build_canonical_basis_data(&spec_a).unwrap();
    let basis_b = build_canonical_basis_data(&spec_b).unwrap();
    
    // Build CGTensors
    let cg_tensor_a = build_weighted_tensor(&basis_a, &w_a);
    let cg_tensor_b = build_weighted_tensor(&basis_b, &w_b);
    
    // Direct contraction
    let cg_tensor_c = ndarray_einsum::tensordot(
        &cg_tensor_a,
        &cg_tensor_b,
        axes_a,
        axes_b,
    );
    
    // Compute X-symbol
    let xsymbol = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();
    let (dim_alpha, dim_beta, dim_gamma) = xsymbol.dimensions();
    
    assert_eq!(dim_alpha, dim_a);
    assert_eq!(dim_beta, dim_b);
    
    // Compute w_c via X-symbol
    let mut w_c = vec![0.0; dim_gamma];
    for gamma in 0..dim_gamma {
        for alpha in 0..dim_alpha {
            for beta in 0..dim_beta {
                let x_val = xsymbol.get(alpha, beta, gamma).unwrap();
                w_c[gamma] += w_a[alpha] * w_b[beta] * x_val;
            }
        }
    }
    
    // Build basis_c and CGTensor C'
    let spec_c = build_output_spec(&spec_a, &spec_b, &contraction).unwrap();
    let basis_c = build_canonical_basis_data(&spec_c).unwrap();
    let cg_tensor_c_prime = build_weighted_tensor(&basis_c, &w_c);
    
    // Check equality
    assert_eq!(cg_tensor_c.shape(), cg_tensor_c_prime.shape(),
        "Tensor shapes don't match: direct={:?}, via X-symbol={:?}",
        cg_tensor_c.shape(), cg_tensor_c_prime.shape());
    
    let mut max_diff = 0.0;
    for (val_c, val_c_prime) in cg_tensor_c.iter().zip(cg_tensor_c_prime.iter()) {
        let diff = (val_c - val_c_prime).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(diff < 1e-10,
            "Tensor mismatch: direct={}, via X-symbol={}, diff={}",
            val_c, val_c_prime, diff);
    }
}

/// Helper function to build a weighted tensor from basis and weights
/// tensor[..., i] = sum_alpha w[alpha] * basis[..., i, alpha]
fn build_weighted_tensor(basis: &ArrayD<f64>, weights: &[f64]) -> ArrayD<f64> {
    let ndim = basis.ndim();
    let om_axis = ndim - 1;
    
    // Get shape without OM axis
    let mut result_shape = basis.shape().to_vec();
    result_shape.pop(); // Remove last dimension (OM axis)
    
    let mut result = ArrayD::zeros(result_shape);
    
    // Sum over OM configurations: result[...] = sum_alpha w[alpha] * basis[..., alpha]
    for (alpha, &weight) in weights.iter().enumerate() {
        let basis_slice = basis.index_axis(Axis(om_axis), alpha);
        result = result + &(basis_slice.to_owned() * weight);
    }
    
    result
}
