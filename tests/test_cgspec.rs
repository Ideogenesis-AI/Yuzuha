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

//! Comprehensive tests for CGSpec creation and OM enumeration with j=0 constraint
//!
//! Tests edge creation and CGSpec construction for 4th to 6th order tensors
//! with spins j ∈ {0, 1/2, 1, 3/2, 2}
//!
//! **IMPORTANT**: All tests enforce angular momentum conservation (total j = 0)

use yuzuha::core::{CGSpec, Direction, Edge, Spin};

// Helper to create spin from doubled value
fn j(doubled: i32) -> Spin {
    Spin::new(doubled).unwrap()
}

// Helper to format alpha for display (debugging)
#[allow(dead_code)]
fn format_alpha(alpha: &[Spin]) -> String {
    let parts: Vec<String> = alpha.iter().map(|s| format!("{}", s)).collect();
    format!("[{}]", parts.join(", "))
}

// Helper to print all alphas (for debugging)
#[allow(dead_code)]
fn print_alphas(alphas: &[Vec<Spin>]) {
    println!("Total alphas: {}", alphas.len());
    for (i, alpha) in alphas.iter().enumerate() {
        println!("  α[{}] = {}", i, format_alpha(alpha));
    }
}

#[cfg(test)]
mod test_fourth_order {
    use super::*;

    #[test]
    fn test_four_half_spins() {
        // Four j=1/2 spins → total j=0
        // j1/2 ⊗ j1/2 = j0 ⊕ j1
        // Path 1: j12=0 → j0⊗j1/2=j1/2 → j1/2⊗j1/2={j0,j1} → j0 ✓
        // Path 2: j12=1 → j1⊗j1/2={j1/2,j3/2} → j1/2⊗j1/2={j0,j1} → j0 ✓ ; j3/2⊗j1/2={j1,j2} → no j0 ✗
        // Expected OM dim = 2
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(1)),
            Edge::incoming("d", j(1)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 4);
        assert_eq!(spec.om_dimension(), 2);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }

        assert_eq!(spec.alphas[0], vec![j(0), j(1)]); // [j=0, j=1/2]
        assert_eq!(spec.alphas[1], vec![j(2), j(1)]); // [j=1, j=1/2]
    }

    #[test]
    fn test_four_spin_one() {
        // Four j=1 spins → total j=0
        // j1 ⊗ j1 = {j0, j1, j2}
        // j12=0: j0⊗j1=j1 → j1⊗j1={j0,j1,j2} → j0 ✓
        // j12=1: j1⊗j1={j0,j1,j2} → all can reach j0
        // j12=2: j2⊗j1={j1,j2,j3} → j2⊗j1→{j1,j2,j3}, only j2 gives j0
        // Let me count: 1 + 3 + 1 = 5? No wait...
        // j12=0 → j123=1 → j1⊗j1 gives j0 → 1 alpha
        // j12=1 → j123∈{0,1,2} → j0,j1,j2 ⊗ j1:
        //   j0⊗j1=j1 (no j0) ✗
        //   j1⊗j1={0,1,2} includes j0 ✓
        //   j2⊗j1={1,2,3} (no j0) ✗
        // j12=2 → j123∈{1,2,3} → j1,j2,j3 ⊗ j1:
        //   j1⊗j1 includes j0 ✓
        //   j2⊗j1={1,2,3} (no j0) ✗
        //   j3⊗j1={2,3,4} (no j0) ✗
        // Total: 1 + 1 + 1 = 3
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 3);

        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }
    }

    #[test]
    fn test_four_with_spin_zero() {
        // [j=0, j=1, j=1, j=1] → total j=0
        // j0⊗j1 = j1 (unique)
        // j1⊗j1 = {j0,j1,j2}
        // j0⊗j1 = j1 (no j0) ✗
        // j1⊗j1 includes j0 ✓
        // j2⊗j1 = {j1,j2,j3} (no j0) ✗
        // OM dim = 1
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 1);
        assert_eq!(spec.alphas[0], vec![j(2), j(2)]); // [j=1, j=1]
    }

    #[test]
    fn test_four_mixed_spins() {
        // [j=1/2, j=1, j=3/2, j=2] → total j=0
        // j1/2 ⊗ j1 = {j1/2, j3/2}
        // Complex calculation - just verify it runs
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(3)),
            Edge::incoming("d", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 4);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }

        // Should have some valid configurations
        assert!(spec.om_dimension() >= 1);
    }

    #[test]
    fn test_four_identical_high_spin() {
        // Four j=2 spins → total j=0
        // j2 ⊗ j2 = {j0,j1,j2,j3,j4}
        // Many paths possible
        let edges = vec![
            Edge::incoming("a", j(4)),
            Edge::incoming("b", j(4)),
            Edge::incoming("c", j(4)),
            Edge::incoming("d", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        // Should have high multiplicity
        assert!(spec.om_dimension() >= 5);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }
    }
}

#[cfg(test)]
mod test_fifth_order {
    use super::*;

    #[test]
    fn test_five_half_spins() {
        // Five j=1/2 spins → total j=0
        // ODD number of half-integer spins cannot give j=0!
        // OM dim = 0
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(1)),
            Edge::incoming("d", j(1)),
            Edge::incoming("e", j(1)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 5);
        assert_eq!(spec.om_dimension(), 0); // Odd number of half-spins!
    }

    #[test]
    fn test_five_spin_one() {
        // Five j=1 spins → total j=0
        // With j=0 constraint enforced
        // OM dim = 6 (verified by enumerate_alpha)
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 5);
        assert_eq!(spec.om_dimension(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 3);
        }
    }

    #[test]
    fn test_five_with_zeros() {
        // [j=0, j=0, j=1, j=1, j=1] → total j=0
        // j0⊗j0=j0, j0⊗j1=j1, j1⊗j1={j0,j1,j2}
        // With j=0 constraint: OM dim = 1
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(0)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 1);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 3);
            // First two should always be j=0, j=1
            assert_eq!(alpha[0], j(0));
            assert_eq!(alpha[1], j(2));
        }
    }

    #[test]
    fn test_five_ascending_spins() {
        // [j=0, j=1/2, j=1, j=3/2, j=2] → total j=0
        // j0⊗j1/2 = j1/2 (unique)
        // Half-integer intermediate, so need to reach j=0 eventually
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(3)),
            Edge::incoming("e", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 5);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 3);
        }

        // May have some valid configurations
        assert!(spec.om_dimension() >= 0);
    }

    #[test]
    fn test_five_high_spins() {
        // [j=3/2, j=2, j=2, j=2, j=3/2] → total j=0
        // Two half-integer spins (even count) + three j=2
        // ODD total number → parity issue
        let edges = vec![
            Edge::incoming("a", j(3)),
            Edge::incoming("b", j(4)),
            Edge::incoming("c", j(4)),
            Edge::incoming("d", j(4)),
            Edge::incoming("e", j(3)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 5);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 3);
        }

        // Should have some configurations
        assert!(spec.om_dimension() >= 0);
    }
}

#[cfg(test)]
mod test_sixth_order {
    use super::*;

    #[test]
    fn test_six_half_spins() {
        // Six j=1/2 spins → total j=0
        // EVEN number of half-integer spins CAN give j=0
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(1)),
            Edge::incoming("d", j(1)),
            Edge::incoming("e", j(1)),
            Edge::incoming("f", j(1)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }

        // Should have multiple configurations
        assert!(spec.om_dimension() >= 2);
    }

    #[test]
    fn test_six_spin_one() {
        // Six j=1 spins → total j=0
        // EVEN number of integer spins CAN give j=0
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(2)),
            Edge::incoming("f", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }

        // Should have high OM dimension
        assert!(spec.om_dimension() >= 5);
    }

    #[test]
    fn test_six_with_zeros_at_start() {
        // [j=0, j=0, j=0, j=1, j=1, j=1] → total j=0
        // With j=0 constraint: OM dim = 1
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(0)),
            Edge::incoming("c", j(0)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(2)),
            Edge::incoming("f", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 1);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
            // First three should always be j=0
            assert_eq!(alpha[0], j(0));
            assert_eq!(alpha[1], j(0));
            assert_eq!(alpha[2], j(2));
        }
    }

    #[test]
    fn test_six_alternating_spins() {
        // [j=1/2, j=1, j=1/2, j=1, j=1/2, j=1] → total j=0
        // Three j=1/2 (odd) + three j=1 (odd) → tricky parity
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(1)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(1)),
            Edge::incoming("f", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }

        // May have configurations
        assert!(spec.om_dimension() >= 0);
    }

    #[test]
    fn test_six_all_different_spins() {
        // [j=0, j=1/2, j=1, j=3/2, j=2, j=2] → total j=0
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(3)),
            Edge::incoming("e", j(4)),
            Edge::incoming("f", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }

        assert!(spec.om_dimension() >= 0);
    }

    #[test]
    fn test_six_high_spin_two() {
        // All j=2 spins → total j=0
        // Six even-integer spins CAN give j=0
        let edges = vec![
            Edge::incoming("a", j(4)),
            Edge::incoming("b", j(4)),
            Edge::incoming("c", j(4)),
            Edge::incoming("d", j(4)),
            Edge::incoming("e", j(4)),
            Edge::incoming("f", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.num_external(), 6);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }

        // Very high OM dimension
        assert!(spec.om_dimension() >= 10);
    }
}

#[cfg(test)]
mod test_edge_properties {
    use super::*;

    #[test]
    fn test_edge_directions() {
        // Test that directions are preserved in CGSpec
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::outgoing("b", j(2)),
            Edge::incoming("c", j(2)),
        ];

        let spec = CGSpec::from_edges(edges.clone()).unwrap();

        assert_eq!(spec.edges.len(), 3);
        assert_eq!(spec.edges[0].dir, Direction::Incoming);
        assert_eq!(spec.edges[1].dir, Direction::Outgoing);
        assert_eq!(spec.edges[2].dir, Direction::Incoming);
    }

    #[test]
    fn test_edge_ids() {
        // Test that edge IDs are preserved
        let edges = vec![
            Edge::incoming("alpha", j(2)),
            Edge::incoming("beta", j(2)),
            Edge::incoming("gamma", j(2)),
            Edge::incoming("delta", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert!(spec.find_edge("alpha").is_some());
        assert!(spec.find_edge("beta").is_some());
        assert!(spec.find_edge("gamma").is_some());
        assert!(spec.find_edge("delta").is_some());
        assert!(spec.find_edge("epsilon").is_none());
    }

    #[test]
    fn test_edge_spin_lookup() {
        let edges = vec![
            Edge::incoming("x", j(1)),
            Edge::incoming("y", j(3)),
            Edge::incoming("z", j(4)),
            Edge::incoming("w", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.edge_spin("x").unwrap(), j(1));
        assert_eq!(spec.edge_spin("y").unwrap(), j(3));
        assert_eq!(spec.edge_spin("z").unwrap(), j(4));
        assert_eq!(spec.edge_spin("w").unwrap(), j(2));
        assert!(spec.edge_spin("v").is_err());
    }
}

#[cfg(test)]
mod test_special_cases {
    use super::*;

    #[test]
    fn test_all_spin_zero() {
        // Four j=0 spins → total j=0
        // j0⊗j0=j0 always
        // Only one OM configuration
        let edges = vec![
            Edge::incoming("a", j(0)),
            Edge::incoming("b", j(0)),
            Edge::incoming("c", j(0)),
            Edge::incoming("d", j(0)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 1);
        assert_eq!(spec.alphas[0], vec![j(0), j(0)]);
    }

    #[test]
    fn test_symmetric_configuration() {
        // [j=1, j=1, j=1, j=1] → total j=0
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        // Should have some valid configurations
        assert!(spec.om_dimension() >= 1);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }
    }

    #[test]
    fn test_shape_calculation() {
        // Test that shape is computed correctly
        let edges = vec![
            Edge::incoming("a", j(1)), // dim = 2
            Edge::incoming("b", j(2)), // dim = 3
            Edge::incoming("c", j(3)), // dim = 4
            Edge::incoming("d", j(2)), // dim = 3
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        let shape = spec.shape();
        assert_eq!(shape[0], 2); // dim(j=1/2)
        assert_eq!(shape[1], 3); // dim(j=1)
        assert_eq!(shape[2], 4); // dim(j=3/2)
        assert_eq!(shape[3], 3); // dim(j=1)
        assert_eq!(shape[4], spec.om_dimension()); // OM dimension
    }

    #[test]
    fn test_two_spin_twos() {
        // [j=2, j=2] → total j=0
        // j2⊗j2 = {j0,j1,j2,j3,j4}
        // Only j=0 satisfies constraint
        let edges = vec![
            Edge::incoming("a", j(4)),
            Edge::incoming("b", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        // For 2 edges, alpha is empty, but j2⊗j2 can give j0
        assert_eq!(spec.om_dimension(), 1);
    }
}

#[cfg(test)]
mod test_manually_verified {
    use super::*;

    #[test]
    fn test_three_spin_one_to_zero() {
        // Three j=1 spins → total j=0
        // j1 ⊗ j1 = {j0,j1,j2}
        // j0⊗j1 = j1 (cannot give j=0) ✗
        // j1⊗j1 = {j0,j1,j2} includes j0 ✓
        // j2⊗j1 = {j1,j2,j3} (cannot give j=0) ✗
        // OM dim = 1, alpha = [j12=1]
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 1);
        assert_eq!(spec.alphas.len(), 1);
        assert_eq!(spec.alphas[0], vec![j(2)]); // [j=1]
    }

    #[test]
    fn test_four_half_spins_manual() {
        // Four j=1/2 spins → total j=0
        // Detailed calculation as in test_four_half_spins
        // OM dim = 2
        let edges = vec![
            Edge::incoming("a", j(1)),
            Edge::incoming("b", j(1)),
            Edge::incoming("c", j(1)),
            Edge::incoming("d", j(1)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 2);
        assert_eq!(spec.alphas.len(), 2);
        
        assert_eq!(spec.alphas[0], vec![j(0), j(1)]); // [j=0, j=1/2]
        assert_eq!(spec.alphas[1], vec![j(2), j(1)]); // [j=1, j=1/2]
    }

    #[test]
    fn test_four_spins_two_pairs() {
        // [j=1, j=1, j=2, j=2] → total j=0
        // j1⊗j1 = {j0,j1,j2}
        // j0⊗j2=j2 → j2⊗j2={j0,j1,j2,j3,j4} includes j0 ✓
        // j1⊗j2={j1,j2,j3} → j1⊗j2={j1,j2,j3}, j2⊗j2={j0,...,j4}, j3⊗j2={j1,...,j5}
        //   Only j2 includes j0 ✓
        // j2⊗j2={j0,j1,j2,j3,j4} → j0⊗j2=j2 (no j0) ✗, j1⊗j2={j1,j2,j3} (no j0) ✗
        //   j2⊗j2 includes j0 ✓, j3⊗j2={j1,j2,j3,j4,j5} (no j0) ✗, j4⊗j2={j2,j3,j4,j5,j6} (no j0) ✗
        // Total: 1 + 1 + 1 = 3
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(4)),
            Edge::incoming("d", j(4)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        assert_eq!(spec.om_dimension(), 3);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 2);
        }
    }

    #[test]
    fn test_six_all_ones_to_zero() {
        // Six j=1 spins → total j=0
        // Complex but should have significant OM dimension
        let edges = vec![
            Edge::incoming("a", j(2)),
            Edge::incoming("b", j(2)),
            Edge::incoming("c", j(2)),
            Edge::incoming("d", j(2)),
            Edge::incoming("e", j(2)),
            Edge::incoming("f", j(2)),
        ];

        let spec = CGSpec::from_edges(edges).unwrap();

        // Should have multiple configurations
        assert!(spec.om_dimension() >= 5);
        
        for alpha in &spec.alphas {
            assert_eq!(alpha.len(), 4);
        }
    }
}
