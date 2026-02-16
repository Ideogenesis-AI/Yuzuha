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

//! Tests for canonical basis construction and orthonormality
//!
//! Tests the build_single_om_tensor function for proper orthonormality properties.

use approx::assert_relative_eq;
use ndarray::Axis;
use yuzuha::builders::builders::build_canonical_basis_data;
use yuzuha::builders::TestCacheGuard;
use yuzuha::core::{CGSpec, Direction, Edge, Spin};

// ============================================================================
// Three-edge orthonormality tests
// ============================================================================

#[test]
fn test_orthonormality_three_j1() {
    // Three j=1 spins
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_three_edges_helper(j1, j1, j1);
}

#[test]
fn test_orthonormality_three_j_half() {
    // Three spins: j=1/2, j=1/2, j=1 (valid configuration)
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_three_edges_helper(j_half, j_half, j1);
}

#[test]
fn test_orthonormality_three_j2() {
    // Three j=2 spins
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_three_edges_helper(j2, j2, j2);
}

#[test]
fn test_orthonormality_three_mixed_1() {
    // j=1/2, j=1, j=1/2
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_three_edges_helper(j_half, j1, j_half);
}

#[test]
fn test_orthonormality_three_mixed_2() {
    // j=1, j=2, j=1
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_three_edges_helper(j1, j2, j1);
}

#[test]
fn test_orthonormality_three_mixed_3() {
    // j=1/2, j=2, j=3/2
    let j_half = Spin::new(1).unwrap();
    let j2 = Spin::new(4).unwrap();
    let j_3half = Spin::new(3).unwrap();
    test_orthonormality_three_edges_helper(j_half, j2, j_3half);
}

/// Helper function for three-edge orthonormality tests
fn test_orthonormality_three_edges_helper(j0: Spin, j1: Spin, j2: Spin) {
    let _guard = TestCacheGuard::new();
    
    let edges = vec![
        Edge::new(j0, Direction::Incoming),
        Edge::new(j1, Direction::Incoming),
        Edge::new(j2, Direction::Outgoing),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    
    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();
    let dim_out = j2.dimension();

    // Verify each basis element has unit Frobenius norm
    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    // Contract over first 2 axes using tensordot
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
            
            // With sqrt(2j+1) normalization, diagonal is 1/(2j+1)
            let expected_diag = 1.0 / (dim_out as f64);
            for i in 0..dim_out {
                for j in 0..dim_out {
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

// ============================================================================
// Four-edge orthonormality tests
// ============================================================================

#[test]
fn test_orthonormality_four_j_half() {
    // Four j=1/2 spins
    let j_half = Spin::new(1).unwrap();
    test_orthonormality_four_edges_helper(j_half, j_half, j_half, j_half);
}

#[test]
fn test_orthonormality_four_j1() {
    // Four j=1 spins
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_four_edges_helper(j1, j1, j1, j1);
}

#[test]
fn test_orthonormality_four_j2() {
    // Four j=2 spins
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_four_edges_helper(j2, j2, j2, j2);
}

#[test]
fn test_orthonormality_four_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_four_edges_helper(j_half, j1, j_half, j1);
}

#[test]
fn test_orthonormality_four_mixed_2() {
    // j=1, j=2, j=1, j=2
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_four_edges_helper(j1, j2, j1, j2);
}

#[test]
fn test_orthonormality_four_mixed_3() {
    // j=1/2, j=1, j=2, j=3/2
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    let j_3half = Spin::new(3).unwrap();
    test_orthonormality_four_edges_helper(j_half, j1, j2, j_3half);
}

/// Helper function for four-edge orthonormality tests
fn test_orthonormality_four_edges_helper(j0: Spin, j1: Spin, j2: Spin, j3: Spin) {
    let _guard = TestCacheGuard::new();
    
    let edges = vec![
        Edge::new(j0, Direction::Incoming),
        Edge::new(j1, Direction::Incoming),
        Edge::new(j2, Direction::Incoming),
        Edge::new(j3, Direction::Outgoing),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    
    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();
    let dim_out = j3.dimension();

    // Verify each basis element has unit Frobenius norm
    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    // Contract over first 3 axes using tensordot
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
            
            // With sqrt(2j+1) normalization, diagonal is 1/(2j+1)
            let expected_diag = 1.0 / (dim_out as f64);
            for i in 0..dim_out {
                for j in 0..dim_out {
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

// ============================================================================
// Five-edge orthonormality tests
// ============================================================================

#[test]
fn test_orthonormality_five_j_half() {
    // Five spins with even number of fermions: j=1/2, j=1/2, j=1/2, j=1/2, j=1
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_five_edges_helper(j_half, j_half, j_half, j_half, j1);
}

#[test]
fn test_orthonormality_five_j1() {
    // Five j=1 spins
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_five_edges_helper(j1, j1, j1, j1, j1);
}

#[test]
fn test_orthonormality_five_j2() {
    // Five j=2 spins
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_five_edges_helper(j2, j2, j2, j2, j2);
}

#[test]
fn test_orthonormality_five_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1, j=1 (even number of fermions)
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_five_edges_helper(j_half, j1, j_half, j1, j1);
}

#[test]
fn test_orthonormality_five_mixed_2() {
    // j=1, j=2, j=1, j=2, j=1
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_five_edges_helper(j1, j2, j1, j2, j1);
}

#[test]
fn test_orthonormality_five_mixed_3() {
    // j=1/2, j=1, j=3/2, j=2, j=1
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    let j_3half = Spin::new(3).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_five_edges_helper(j_half, j1, j_3half, j2, j1);
}

/// Helper function for five-edge orthonormality tests
fn test_orthonormality_five_edges_helper(j0: Spin, j1: Spin, j2: Spin, j3: Spin, j4: Spin) {
    let _guard = TestCacheGuard::new();
    
    let edges = vec![
        Edge::new(j0, Direction::Incoming),
        Edge::new(j1, Direction::Incoming),
        Edge::new(j2, Direction::Incoming),
        Edge::new(j3, Direction::Incoming),
        Edge::new(j4, Direction::Outgoing),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    
    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();
    let dim_out = j4.dimension();

    // Verify each basis element has unit Frobenius norm
    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    // Contract over first 4 axes using tensordot
    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., beta]);
            
            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
            );
            
            // With sqrt(2j+1) normalization, diagonal is 1/(2j+1)
            let expected_diag = 1.0 / (dim_out as f64);
            for i in 0..dim_out {
                for j in 0..dim_out {
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

// ============================================================================
// Six-edge orthonormality tests
// ============================================================================

#[test]
fn test_orthonormality_six_j_half() {
    // Six j=1/2 spins
    let j_half = Spin::new(1).unwrap();
    test_orthonormality_six_edges_helper(j_half, j_half, j_half, j_half, j_half, j_half);
}

#[test]
fn test_orthonormality_six_j1() {
    // Six j=1 spins
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_six_edges_helper(j1, j1, j1, j1, j1, j1);
}

#[test]
fn test_orthonormality_six_j2() {
    // Six j=2 spins
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_six_edges_helper(j2, j2, j2, j2, j2, j2);
}

#[test]
fn test_orthonormality_six_mixed_1() {
    // j=1/2, j=1, j=1/2, j=1, j=1/2, j=1/2 (even number of fermions)
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    test_orthonormality_six_edges_helper(j_half, j1, j_half, j1, j_half, j_half);
}

#[test]
fn test_orthonormality_six_mixed_2() {
    // j=1, j=2, j=1, j=2, j=1, j=2
    let j1 = Spin::new(2).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_six_edges_helper(j1, j2, j1, j2, j1, j2);
}

#[test]
fn test_orthonormality_six_mixed_3() {
    // j=1, j=1, j=3/2, j=2, j=3/2, j=1 (even number of half-integers: 2)
    let j1 = Spin::new(2).unwrap();
    let j_3half = Spin::new(3).unwrap();
    let j2 = Spin::new(4).unwrap();
    test_orthonormality_six_edges_helper(j1, j1, j_3half, j2, j_3half, j1);
}

/// Helper function for six-edge orthonormality tests
fn test_orthonormality_six_edges_helper(j0: Spin, j1: Spin, j2: Spin, j3: Spin, j4: Spin, j5: Spin) {
    let _guard = TestCacheGuard::new();
    
    let edges = vec![
        Edge::new(j0, Direction::Incoming),
        Edge::new(j1, Direction::Incoming),
        Edge::new(j2, Direction::Incoming),
        Edge::new(j3, Direction::Incoming),
        Edge::new(j4, Direction::Incoming),
        Edge::new(j5, Direction::Outgoing),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    
    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();
    let dim_out = j5.dimension();

    // Verify each basis element has unit Frobenius norm
    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    // Contract over first 5 axes using tensordot
    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., .., beta]);
            
            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
            );
            
            // With sqrt(2j+1) normalization, diagonal is 1/(2j+1)
            let expected_diag = 1.0 / (dim_out as f64);
            for i in 0..dim_out {
                for j in 0..dim_out {
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

// ============================================================================
// Non-canonical direction tests - Three edges
// ============================================================================

#[test]
fn test_orthonormality_three_edges_all_outgoing() {
    let _guard = TestCacheGuard::new();
    
    // Three j=1 spins, all outgoing
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::outgoing(j1),
        Edge::outgoing(j1),
        Edge::outgoing(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

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
fn test_orthonormality_three_edges_all_incoming() {
    let _guard = TestCacheGuard::new();
    
    // Three edges all incoming: j=1/2, j=1/2, j=1 (valid configuration)
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::incoming(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

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
fn test_orthonormality_three_edges_out_in_out() {
    let _guard = TestCacheGuard::new();
    
    // Three j=2 spins: out, in, out
    let j2 = Spin::new(4).unwrap();
    let edges = vec![
        Edge::outgoing(j2),
        Edge::incoming(j2),
        Edge::outgoing(j2),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j2.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

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

            for i in 0..5 {
                for j in 0..5 {
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

// ============================================================================
// Non-canonical direction tests - Four edges
// ============================================================================

#[test]
fn test_orthonormality_four_edges_all_outgoing() {
    let _guard = TestCacheGuard::new();
    
    // Four j=1/2 spins, all outgoing
    let j_half = Spin::new(1).unwrap();
    let edges = vec![
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j_half.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

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
fn test_orthonormality_four_edges_alternating() {
    let _guard = TestCacheGuard::new();
    
    // Four j=1 spins: out, in, out, in
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::outgoing(j1),
        Edge::incoming(j1),
        Edge::outgoing(j1),
        Edge::incoming(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

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

// ============================================================================
// Non-canonical direction tests - Five edges
// ============================================================================

#[test]
fn test_orthonormality_five_edges_all_incoming() {
    let _guard = TestCacheGuard::new();
    
    // Five edges all incoming: j=1/2, j=1/2, j=1/2, j=1/2, j=1 (even fermions)
    let j_half = Spin::new(1).unwrap();
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::incoming(j_half),
        Edge::incoming(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., alpha]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., beta]);

            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
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
fn test_orthonormality_five_edges_mixed_complex() {
    let _guard = TestCacheGuard::new();
    
    // Five j=1 spins: in, out, in, out, in
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::incoming(j1),
        Edge::outgoing(j1),
        Edge::incoming(j1),
        Edge::outgoing(j1),
        Edge::incoming(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., beta]);

            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
                &[Axis(0), Axis(1), Axis(2), Axis(3)],
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

// ============================================================================
// Non-canonical direction tests - Six edges
// ============================================================================

#[test]
fn test_orthonormality_six_edges_all_outgoing() {
    let _guard = TestCacheGuard::new();
    
    // Six j=1/2 spins, all outgoing
    let j_half = Spin::new(1).unwrap();
    let edges = vec![
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
        Edge::outgoing(j_half),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j_half.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., .., beta]);

            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
            );

            for i in 0..2 {
                for j in 0..2 {
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
fn test_orthonormality_six_edges_alternating_pattern() {
    let _guard = TestCacheGuard::new();
    
    // Six j=1 spins: out, out, in, in, out, out
    let j1 = Spin::new(2).unwrap();
    let edges = vec![
        Edge::outgoing(j1),
        Edge::outgoing(j1),
        Edge::incoming(j1),
        Edge::incoming(j1),
        Edge::outgoing(j1),
        Edge::outgoing(j1),
    ];
    let spec = CGSpec::from_edges(edges).unwrap();
    let expected_diag = 1.0 / (j1.dimension() as f64);

    let data = build_canonical_basis_data(&spec).unwrap();
    let om_dim = spec.om_dimension();

    for alpha_idx in 0..om_dim {
        let slice = data.slice(ndarray::s![.., .., .., .., .., .., alpha_idx]);
        let frob_norm_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frob_norm_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.slice(ndarray::s![.., .., .., .., .., .., alpha]);
            let slice_beta = data.slice(ndarray::s![.., .., .., .., .., .., beta]);

            let result = ndarray_einsum::tensordot(
                &slice_alpha,
                &slice_beta,
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
                &[Axis(0), Axis(1), Axis(2), Axis(3), Axis(4)],
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

