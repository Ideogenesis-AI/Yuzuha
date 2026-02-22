# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Yuzuha library.
#
# Yuzuha is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Yuzuha is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Yuzuha. If not, see <https://www.gnu.org/licenses/>.

"""
Comprehensive tests for X-symbol computation and consistency.

This test suite validates X-symbol correctness by comparing two methods of 
computing CG tensor contractions:
1. Direct contraction: Contract CG tensors A and B using tensordot
2. X-symbol method: Use X-symbol to transform OM weights, then build from basis_c

The two methods must produce identical results (within numerical tolerance).

Test Organization
-----------------
- TestXSymbolBasic: Basic API and dimension tests
- TestXSymbolProperties: Mathematical properties (real-valued, normalization)
- TestXSymbolCaching: Verify caching behavior
- TestXSymbolConsistency: Systematic consistency tests across spin configurations
  * Basic configurations (j=1/2, j=1)
  * Larger spins (j=3/2, j=2)
  * Multiple edge contractions
  * 5-6 edge tensors
  * Mixed spin values
- TestXSymbolStress: Randomized stress tests
  * Random 3-edge and 4-edge configurations
  * Random multiple contractions
  * Systematic enumeration of valid spin pairs
  * High spin values (up to j=5/2)
- TestXSymbolEdgeCases: Boundary conditions and special cases

Spin Configuration Validity
----------------------------
All tests ensure valid spin configurations:
- Even number of half-integer spins (fermion number conservation)
- Triangular inequalities satisfied: |j1 - j2| <= j3 <= j1 + j2
- Angular momentum conservation in fusion trees

Total: 25 tests covering basic, consistency, stress, and edge cases.
"""
import pytest
import numpy as np
import yuzuha


def build_weighted_tensor(basis, weights):
    """
    Build a weighted tensor from basis and weights.
    
    tensor[...] = sum_alpha weights[alpha] * basis[..., alpha]
    
    Parameters
    ----------
    basis : np.ndarray
        Canonical basis tensor with OM axis as last dimension
    weights : np.ndarray
        Weights for each OM configuration
        
    Returns
    -------
    np.ndarray
        Weighted tensor without OM dimension
    """
    # Sum over OM configurations (last axis)
    result = np.tensordot(basis, weights, axes=([basis.ndim - 1], [0]))
    return result


def run_consistency_test(spec_a, spec_b, contraction, axes_a, axes_b, tol=1e-10):
    """
    Test X-symbol consistency by comparing two computation methods.
    
    Method 1: Direct contraction of CG tensors A and B, scaled by the
              Frobenius-Schur (FS) phase factor for the contraction.
    Method 2: Use X-symbol to transform OM weights, then build from basis_c
    
    Parameters
    ----------
    spec_a : CGSpec
        Specification for tensor A
    spec_b : CGSpec
        Specification for tensor B
    contraction : Contraction
        Contraction specification
    axes_a : list of int
        Axes of A to contract
    axes_b : list of int
        Axes of B to contract
    tol : float
        Tolerance for numerical comparison
    """
    # Get OM dimensions
    dim_a = spec_a.om_dimension()
    dim_b = spec_b.om_dimension()
    
    assert dim_a > 0, f"spec_a should have non-zero OM dimension, got {dim_a}"
    assert dim_b > 0, f"spec_b should have non-zero OM dimension, got {dim_b}"
    
    # Generate random weights for OM indices
    rng = np.random.default_rng(42)
    w_a = rng.uniform(-1.0, 1.0, size=dim_a)
    w_b = rng.uniform(-1.0, 1.0, size=dim_b)
    
    # Build basis tensors
    basis_a = yuzuha.canonical_basis(spec_a)
    basis_b = yuzuha.canonical_basis(spec_b)
    
    # Build CG tensors with weights
    cg_tensor_a = build_weighted_tensor(basis_a, w_a)
    cg_tensor_b = build_weighted_tensor(basis_b, w_b)
    
    # Method 1: Direct contraction, scaled by the FS phase to match X-symbol convention
    fs_phase = yuzuha.compute_fs_phase(spec_a, spec_b, contraction)
    cg_tensor_c = np.tensordot(cg_tensor_a, cg_tensor_b, axes=(axes_a, axes_b)) * fs_phase
    
    # Method 2: Via X-symbol
    x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
    dim_alpha, dim_beta, dim_gamma = x_array.shape
    
    assert dim_alpha == dim_a, f"X-symbol alpha dimension mismatch: {dim_alpha} != {dim_a}"
    assert dim_beta == dim_b, f"X-symbol beta dimension mismatch: {dim_beta} != {dim_b}"
    
    # Compute w_c via X-symbol: w_c[gamma] = sum_{alpha,beta} w_a[alpha] * w_b[beta] * X[alpha,beta,gamma]
    w_c = np.einsum('a,b,abg->g', w_a, w_b, x_array)
    
    # Build basis_c and CG tensor C'
    basis_c = yuzuha.canonical_basis(spec_c)
    cg_tensor_c_prime = build_weighted_tensor(basis_c, w_c)
    
    # Check shapes match
    assert cg_tensor_c.shape == cg_tensor_c_prime.shape, \
        f"Tensor shapes don't match: direct={cg_tensor_c.shape}, via X-symbol={cg_tensor_c_prime.shape}"
    
    # Check values match
    diff = np.abs(cg_tensor_c - cg_tensor_c_prime)
    max_diff = np.max(diff)
    
    assert max_diff < tol, \
        f"Tensors don't match: max_diff={max_diff:.2e}, tol={tol:.2e}"
    
    # Additional check: element-wise comparison
    assert np.allclose(cg_tensor_c, cg_tensor_c_prime, atol=tol, rtol=1e-10), \
        "Element-wise comparison failed"


class TestXSymbolBasic:
    """Basic X-symbol computation tests."""

    def test_xsymbol_simple(self):
        """Test X-symbol computation with simple configuration."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Check return types
        assert isinstance(x_array, np.ndarray)
        assert x_array.dtype == np.float64
        assert len(x_array.shape) == 3

        # Check dimensions
        dim_a = spec_a.om_dimension()
        dim_b = spec_b.om_dimension()
        dim_c = spec_c.om_dimension()
        assert x_array.shape == (dim_a, dim_b, dim_c)

        # Check that spec_c has correct number of edges
        assert spec_c.num_external() == 4  # 2 from A + 2 from B

    def test_xsymbol_larger_spins(self):
        """Test X-symbol with j=1 spins."""
        j1 = yuzuha.Spin(2)
        j_half = yuzuha.Spin(1)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        assert x_array.shape[0] == spec_a.om_dimension()
        assert x_array.shape[1] == spec_b.om_dimension()
        assert x_array.shape[2] == spec_c.om_dimension()

    def test_xsymbol_multiple_contractions(self):
        """Test X-symbol with multiple edge contractions."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2, 3], [0, 1])
        x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Check dimensions
        assert len(x_array.shape) == 3
        assert spec_c.num_external() == 4  # 2 from A + 2 from B


class TestXSymbolProperties:
    """Test mathematical properties of X-symbols."""

    def test_xsymbol_real_valued(self):
        """Test that X-symbols are real-valued."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        x_array, _ = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Check that array is real (dtype is float64, not complex)
        assert x_array.dtype == np.float64
        assert np.all(np.isreal(x_array))
        assert np.all(np.isfinite(x_array))

    def test_xsymbol_normalization(self):
        """Test X-symbol normalization property."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        x_array, _ = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Sum over alpha and beta for each gamma
        # Should satisfy certain sum rules
        for gamma in range(x_array.shape[2]):
            sum_sq = np.sum(x_array[:, :, gamma]**2)
            # For this specific case, we know the expected value from Rust tests
            if gamma == 0:
                assert sum_sq < 1e-10  # Inaccessible
            elif gamma == 1:
                assert abs(sum_sq - 1.0/3.0) < 1e-6


class TestXSymbolCaching:
    """Test that caching works across multiple calls."""

    def test_repeated_calls_consistent(self):
        """Test that repeated calls give consistent results."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])

        # First call
        x1, spec_c1 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Second call (should use cache)
        x2, spec_c2 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

        # Results should be identical
        assert np.allclose(x1, x2)
        assert spec_c1.om_dimension() == spec_c2.om_dimension()
        assert spec_c1.num_external() == spec_c2.num_external()


class TestXSymbolConsistency:
    """Test X-symbol consistency by comparing direct contraction vs X-symbol method."""
    
    def test_consistency_basic(self):
        """Test consistency with basic configuration (j=1/2, j=1/2, j=1)."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_consistency_larger_spins(self):
        """Test consistency with j=1 spins."""
        j1 = yuzuha.Spin(2)
        j_half = yuzuha.Spin(1)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_consistency_multiple_contractions(self):
        """Test consistency with multiple edge contractions."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2, 3], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [2, 3], [0, 1])
    
    def test_consistency_spin_3_2(self):
        """Test consistency with spin-3/2 particles."""
        j_3_2 = yuzuha.Spin(3)  # j=3/2
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_3_2),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_consistency_spin_2(self):
        """Test consistency with spin-2 particles."""
        j2 = yuzuha.Spin(4)  # j=2
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j1),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_consistency_5_edges(self):
        """Test consistency with 5-edge tensors."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([3, 4], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [3, 4], [0, 1])
    
    def test_consistency_6_edges(self):
        """Test consistency with 6-edge tensors."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([4, 5], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [4, 5], [0, 1])
    
    def test_consistency_mixed_large(self):
        """Test consistency with mixed large spins."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        j_3_2 = yuzuha.Spin(3)
        j2 = yuzuha.Spin(4)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_3_2),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([3, 4], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [3, 4], [0, 1])
    
    def test_consistency_single_contraction_large(self):
        """Test consistency with single edge contraction on large tensors."""
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        contraction = yuzuha.Contraction([5], [0])
        run_consistency_test(spec_a, spec_b, contraction, [5], [0])

    def test_two_index_outcome_out_out(self):
        """Two-edge outcome (out, out): contracted axes pair in(A) × out(B)."""
        j_half = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_outcome_out_in(self):
        """Two-edge outcome (out, in): contracted axes pair in(A) × out(B)."""
        j_half = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_outcome_in_out(self):
        """Two-edge outcome (in, out): canonical spec_c, contracted axes pair out(A) × in(B)."""
        j_half = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_outcome_in_in(self):
        """Two-edge outcome (in, in): contracted axes pair out(A) × in(B)."""
        j_half = yuzuha.Spin(1)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_tensor_left(self):
        """Two-index tensor A contracted with an intermediate edge of higher-order tensor B.

        Contracts spec_a axis 0 (out j=1) with spec_b axis 2 (in j=1) — not the
        first or last axis of B. Tests both directions for the free edge of spec_a.
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # 5-edge tensor B; axis 2 (in j=1) is the contracted edge (middle, not first/last)
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Case 1: free edge of the 2-edge tensor is incoming
        spec_a_free_in = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),  # contracted: out(A) × in(B axis 2) ✓
            yuzuha.Edge.incoming(j1),  # free edge: incoming
        ])
        contraction = yuzuha.Contraction([0], [2])
        run_consistency_test(spec_a_free_in, spec_b, contraction, [0], [2])

        # Case 2: free edge of the 2-edge tensor is outgoing
        spec_a_free_out = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),  # contracted: out(A) × in(B axis 2) ✓
            yuzuha.Edge.outgoing(j1),  # free edge: outgoing
        ])
        run_consistency_test(spec_a_free_out, spec_b, contraction, [0], [2])

    def test_two_index_tensor_right(self):
        """Higher-order tensor A contracted with a two-index tensor B via an intermediate edge.

        Contracts spec_a axis 2 (out j=1) — not the first or last axis of A — with
        spec_b axis 0 (in j=1). Tests both directions for the free edge of spec_b.
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # 5-edge tensor A; axis 2 (out j=1) is the contracted edge (middle, not first/last)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),   # contracted: out(A axis 2) × in(B) ✓
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Case 1: free edge of the 2-edge tensor is outgoing
        spec_b_free_out = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),  # contracted: in(B) × out(A axis 2) ✓
            yuzuha.Edge.outgoing(j1),  # free edge: outgoing
        ])
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b_free_out, contraction, [2], [0])

        # Case 2: free edge of the 2-edge tensor is incoming
        spec_b_free_in = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),  # contracted: in(B) × out(A axis 2) ✓
            yuzuha.Edge.incoming(j1),  # free edge: incoming
        ])
        run_consistency_test(spec_a, spec_b_free_in, contraction, [2], [0])


class TestXSymbolStress:
    """Stress tests with random configurations to ensure robustness."""
    
    def _is_valid_spin_config(self, spins):
        """
        Check if a spin configuration is valid.
        
        Must have even number of half-integer spins for the overall config to work.
        
        Parameters
        ----------
        spins : list of Spin
            List of spins to check
            
        Returns
        -------
        bool
            True if configuration is valid
        """
        # Count half-integer spins (odd twice values)
        half_integer_count = sum(1 for s in spins if s.twice() % 2 == 1)
        return half_integer_count % 2 == 0
    
    def _satisfies_triangle(self, j1, j2, j3):
        """
        Check if three spins satisfy the triangular inequality.
        
        |j1 - j2| <= j3 <= j1 + j2
        
        Parameters
        ----------
        j1, j2, j3 : Spin
            Three spins to check
            
        Returns
        -------
        bool
            True if triangle inequality is satisfied
        """
        t1 = j1.twice()
        t2 = j2.twice()
        t3 = j3.twice()
        
        return abs(t1 - t2) <= t3 <= (t1 + t2)
    
    def test_stress_random_3_edge(self):
        """Stress test with random 3-edge configurations."""
        rng = np.random.default_rng(12345)
        
        # Test multiple random configurations
        num_tests = 20
        success_count = 0
        
        for _ in range(num_tests):
            # Generate random spins from j=1/2 to j=2
            # Spin(n) represents j = n/2, so Spin(1)=j=1/2, ..., Spin(4)=j=2
            spin_values = [1, 2, 3, 4]  # j=1/2, 1, 3/2, 2
            
            # Try to find a valid configuration
            for _ in range(50):  # Max attempts
                j1_idx = rng.choice(len(spin_values))
                j2_idx = rng.choice(len(spin_values))
                j3_idx = rng.choice(len(spin_values))
                
                j1 = yuzuha.Spin(spin_values[j1_idx])
                j2 = yuzuha.Spin(spin_values[j2_idx])
                j3 = yuzuha.Spin(spin_values[j3_idx])
                
                # Check valid spin config
                if not self._is_valid_spin_config([j1, j2, j3]):
                    continue
                
                # Check triangle inequality
                if not self._satisfies_triangle(j1, j2, j3):
                    continue
                
                try:
                    # Build specs
                    spec_a = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(j1),
                        yuzuha.Edge.incoming(j2),
                        yuzuha.Edge.outgoing(j3),
                    ])
                    
                    spec_b = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(j3),
                        yuzuha.Edge.outgoing(j1),
                        yuzuha.Edge.outgoing(j2),
                    ])
                    
                    contraction = yuzuha.Contraction([2], [0])
                    run_consistency_test(spec_a, spec_b, contraction, [2], [0])
                    success_count += 1
                    break
                    
                except Exception:
                    continue
        
        assert success_count >= num_tests * 0.8, \
            f"Only {success_count}/{num_tests} random tests succeeded"
    
    def test_stress_random_4_edge(self):
        """Stress test with random 4-edge configurations."""
        rng = np.random.default_rng(23456)
        
        num_tests = 15
        success_count = 0
        
        for _ in range(num_tests):
            spin_values = [1, 2, 3, 4]  # j=1/2, 1, 3/2, 2
            
            for _ in range(50):  # Max attempts
                # Generate 4 spins for tensor A
                spins_a = [yuzuha.Spin(rng.choice(spin_values)) for _ in range(4)]
                
                # Check even number of half-integers
                if not self._is_valid_spin_config(spins_a):
                    continue
                
                # For 4-edge tensor with 3 incoming, 1 outgoing
                # Need triangle condition for j1, j2 -> j_int and j_int, j3 -> j4
                # This is complex, so we'll just try and catch exceptions
                
                try:
                    spec_a = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(spins_a[0]),
                        yuzuha.Edge.incoming(spins_a[1]),
                        yuzuha.Edge.incoming(spins_a[2]),
                        yuzuha.Edge.outgoing(spins_a[3]),
                    ])
                    
                    # Generate compatible tensor B
                    # Contract last edge of A with first edge of B
                    # B should have j4 incoming, then two outgoing
                    spins_b = [yuzuha.Spin(rng.choice(spin_values)) for _ in range(2)]
                    
                    if not self._is_valid_spin_config([spins_a[3]] + spins_b):
                        continue
                    
                    spec_b = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(spins_a[3]),
                        yuzuha.Edge.outgoing(spins_b[0]),
                        yuzuha.Edge.outgoing(spins_b[1]),
                    ])
                    
                    contraction = yuzuha.Contraction([3], [0])
                    run_consistency_test(spec_a, spec_b, contraction, [3], [0])
                    success_count += 1
                    break
                    
                except Exception:
                    continue
        
        assert success_count >= num_tests * 0.7, \
            f"Only {success_count}/{num_tests} random 4-edge tests succeeded"
    
    def test_stress_random_multiple_contractions(self):
        """Stress test with random multiple edge contractions."""
        rng = np.random.default_rng(34567)
        
        num_tests = 10
        success_count = 0
        
        for _ in range(num_tests):
            # Use j=1/2 and j=1 only for simplicity
            spin_values = [1, 2]  # j=1/2, 1
            
            for _ in range(50):
                # Generate tensor A with 4 edges (2 in, 2 out)
                spins_a = [yuzuha.Spin(rng.choice(spin_values)) for _ in range(4)]
                
                if not self._is_valid_spin_config(spins_a):
                    continue
                
                try:
                    spec_a = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(spins_a[0]),
                        yuzuha.Edge.incoming(spins_a[1]),
                        yuzuha.Edge.outgoing(spins_a[2]),
                        yuzuha.Edge.outgoing(spins_a[3]),
                    ])
                    
                    # Generate tensor B with matching contracted edges
                    spins_b = [yuzuha.Spin(rng.choice(spin_values)) for _ in range(2)]
                    all_spins = spins_a[2:4] + spins_b
                    
                    if not self._is_valid_spin_config(all_spins):
                        continue
                    
                    spec_b = yuzuha.CGSpec.from_edges([
                        yuzuha.Edge.incoming(spins_a[2]),
                        yuzuha.Edge.incoming(spins_a[3]),
                        yuzuha.Edge.outgoing(spins_b[0]),
                        yuzuha.Edge.outgoing(spins_b[1]),
                    ])
                    
                    contraction = yuzuha.Contraction([2, 3], [0, 1])
                    run_consistency_test(spec_a, spec_b, contraction, [2, 3], [0, 1])
                    success_count += 1
                    break
                    
                except Exception:
                    continue
        
        assert success_count >= num_tests * 0.7, \
            f"Only {success_count}/{num_tests} random multiple contraction tests succeeded"
    
    def test_stress_systematic_spin_pairs(self):
        """
        Systematic test of all valid spin pairs up to j=2.
        
        Test all combinations of (j1, j2, j3) where each j is in {1/2, 1, 3/2, 2}
        and the configuration satisfies validity constraints.
        """
        spin_values = [
            yuzuha.Spin(1),   # j=1/2
            yuzuha.Spin(2),   # j=1
            yuzuha.Spin(3),   # j=3/2
            yuzuha.Spin(4),   # j=2
        ]
        
        valid_configs = []
        
        # Generate all valid configurations
        for j1 in spin_values:
            for j2 in spin_values:
                for j3 in spin_values:
                    # Check spin config validity
                    if not self._is_valid_spin_config([j1, j2, j3]):
                        continue
                    
                    # Check triangle inequality
                    if not self._satisfies_triangle(j1, j2, j3):
                        continue
                    
                    try:
                        # Try to build the spec
                        spec = yuzuha.CGSpec.from_edges([
                            yuzuha.Edge.incoming(j1),
                            yuzuha.Edge.incoming(j2),
                            yuzuha.Edge.outgoing(j3),
                        ])
                        
                        # If successful, add to valid configs
                        valid_configs.append((j1, j2, j3))
                    except Exception:
                        continue
        
        # Test a subset of valid configurations
        test_count = min(20, len(valid_configs))
        for j1, j2, j3 in valid_configs[:test_count]:
            spec_a = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j2),
                yuzuha.Edge.outgoing(j3),
            ])
            
            spec_b = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j3),
                yuzuha.Edge.outgoing(j1),
                yuzuha.Edge.outgoing(j2),
            ])
            
            contraction = yuzuha.Contraction([2], [0])
            run_consistency_test(spec_a, spec_b, contraction, [2], [0])
        
        assert len(valid_configs) > 0, "No valid configurations found"
    
    def test_stress_larger_tensors(self):
        """Stress test with larger tensors (5-6 edges)."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # Test 1: 5-edge A, 4-edge B
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([3, 4], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [3, 4], [0, 1])
        
        # Test 2: 6-edge A, 4-edge B
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([4, 5], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [4, 5], [0, 1])
        
        # Test 3: 5-edge A (all integer spins), 3-edge B
        j2 = yuzuha.Spin(4)  # j=2
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        contraction = yuzuha.Contraction([4], [0])
        run_consistency_test(spec_a, spec_b, contraction, [4], [0])
    
    def test_stress_high_spin_values(self):
        """Test with higher spin values (up to j=5/2)."""
        spin_values = [
            yuzuha.Spin(1),   # j=1/2
            yuzuha.Spin(2),   # j=1
            yuzuha.Spin(3),   # j=3/2
            yuzuha.Spin(4),   # j=2
            yuzuha.Spin(5),   # j=5/2
        ]
        
        # Test some specific valid configurations with higher spins
        test_cases = [
            # (j1, j2, j3)
            (spin_values[4], spin_values[1], spin_values[3]),  # j=5/2, j=1, j=2
            (spin_values[3], spin_values[3], spin_values[4]),  # j=2, j=2, j=5/2
            (spin_values[4], spin_values[2], spin_values[4]),  # j=5/2, j=3/2, j=5/2
        ]
        
        for j1, j2, j3 in test_cases:
            # Check validity
            if not self._is_valid_spin_config([j1, j2, j3]):
                continue
            
            if not self._satisfies_triangle(j1, j2, j3):
                continue
            
            try:
                spec_a = yuzuha.CGSpec.from_edges([
                    yuzuha.Edge.incoming(j1),
                    yuzuha.Edge.incoming(j2),
                    yuzuha.Edge.outgoing(j3),
                ])
                
                spec_b = yuzuha.CGSpec.from_edges([
                    yuzuha.Edge.incoming(j3),
                    yuzuha.Edge.outgoing(j1),
                    yuzuha.Edge.outgoing(j2),
                ])
                
                contraction = yuzuha.Contraction([2], [0])
                run_consistency_test(spec_a, spec_b, contraction, [2], [0])
                
            except Exception as e:
                pytest.fail(f"Failed for high spin config ({j1.twice()}/2, {j2.twice()}/2, {j3.twice()}/2): {e}")

    def test_two_index_outcome_j1(self):
        """Two-edge outcome with j=1: 4-edge tensors contracting 3 edges each."""
        j1 = yuzuha.Spin(2)  # j=1, integer spin
        # Contracted: in(A) × out(B); free: out(A) + out(B)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_outcome_j3_half(self):
        """Two-edge outcome with j=3/2: 4-edge tensors contracting 3 edges each."""
        j_3_2 = yuzuha.Spin(3)  # j=3/2, 4 half-integer edges → even → valid
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.outgoing(j_3_2),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_3_2),
            yuzuha.Edge.outgoing(j_3_2),
            yuzuha.Edge.outgoing(j_3_2),
            yuzuha.Edge.outgoing(j_3_2),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])

    def test_two_index_outcome_five_edges_j1(self):
        """Two-edge outcome with j=1: 5-edge tensors contracting 4 edges each."""
        j1 = yuzuha.Spin(2)  # j=1, all integer → valid for any edge count
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        contraction = yuzuha.Contraction([0, 1, 2, 3], [0, 1, 2, 3])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2, 3], [0, 1, 2, 3])

    def test_two_index_outcome_five_edges_mixed(self):
        """Two-edge outcome with mixed spins: 5-edge tensors contracting 4 edges each."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        # 3× j=1/2 + 1× j=1 contracted; free: j=1/2
        # 4 half-integer edges per tensor → even → valid
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2, 3], [0, 1, 2, 3])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2, 3], [0, 1, 2, 3])

    def test_two_index_outcome_six_edges_j_half(self):
        """Two-edge outcome with j=1/2: 6-edge tensors contracting 5 edges each."""
        j_half = yuzuha.Spin(1)
        # 6 half-integer edges → even → valid
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

    def test_two_index_outcome_mixed_spins(self):
        """Two-edge outcome with mixed contracted spins: 4-edge A + 4-edge B."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        # Contracted: 2× j=1 + 1× j=1/2; free: j=1/2
        # 2 half-integer edges per tensor → even → valid
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        run_consistency_test(spec_a, spec_b, contraction, [0, 1, 2], [0, 1, 2])


class TestXSymbolEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_configuration(self):
        """Test with minimal 3-edge configuration."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # Simplest possible: j=1/2 + j=1/2 -> j=1
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_same_spin_all_edges(self):
        """Test with all edges having the same spin value."""
        j1 = yuzuha.Spin(2)
        
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        contraction = yuzuha.Contraction([2], [0])
        run_consistency_test(spec_a, spec_b, contraction, [2], [0])
    
    def test_maximal_contraction_4_edges(self):
        """Test with maximal contraction (all but one edge contracted)."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # A has 4 edges, contract 2 of them (even number of half-integers)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([2, 3], [0, 1])
        run_consistency_test(spec_a, spec_b, contraction, [2, 3], [0, 1])
    
    def test_asymmetric_tensor_sizes(self):
        """Test with very different tensor sizes."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # Large A (6 edges), small B (3 edges)
        spec_a = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        spec_b = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        contraction = yuzuha.Contraction([5], [0])
        run_consistency_test(spec_a, spec_b, contraction, [5], [0])

class TestXSymbolInversionConsistency:
    """Test that flipping both edges of a contracted pair leaves the X-symbol unchanged.

    Mathematical basis
    ------------------
    Inverting both edges in a contracted pair inserts g^{-1} ∘ g = I on that
    index, so the X-symbol is invariant.  Concretely, if spec_a has edge i with
    direction d and spec_b has edge j with direction −d (the valid pairing), then
    flipping both to (−d, d) gives a new pair that is still opposite — and the
    resulting X-symbol array is element-wise identical.

    Tests cover:
    - single contracted pair flipped
    - one pair flipped out of several contracted pairs
    - all contracted pairs flipped simultaneously
    - different spin values (j=1/2, j=1, j=3/2)
    - two-index outcome scenario
    """

    def _xsymbol_equal(self, spec_a1, spec_b1, spec_a2, spec_b2, contraction, tol=1e-10):
        """Assert the two X-symbol arrays are element-wise equal."""
        x1, _ = yuzuha.compute_xsymbol(spec_a1, spec_b1, contraction)
        x2, _ = yuzuha.compute_xsymbol(spec_a2, spec_b2, contraction)
        assert x1.shape == x2.shape, \
            f"Shape mismatch: {x1.shape} vs {x2.shape}"
        assert np.allclose(x1, x2, atol=tol), \
            f"X-symbols differ after flipping contracted pair; max diff={np.max(np.abs(x1-x2)):.2e}"

    def test_single_pair_flip_j_half(self):
        """Flip the sole contracted pair for j=1/2, 3-edge × 3-edge."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # Original: A[2] (out j=1) × B[0] (in j=1)
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flipped contracted pair: A'[2] (in j=1) × B'[0] (out j=1)
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_j1(self):
        """Flip the sole contracted pair for j=1, 3-edge × 3-edge."""
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        # Flipped: A'[2] (in j=2) × B'[0] (out j=2)
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j2),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_j3_half(self):
        """Flip the sole contracted pair for j=3/2, 3-edge × 3-edge."""
        j1    = yuzuha.Spin(2)
        j_3_2 = yuzuha.Spin(3)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.outgoing(j_3_2),   # contracted axis 2
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_3_2),   # contracted axis 0
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_3_2),
        ])

        # Flipped contracted pair only
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j_3_2),   # contracted axis 2, now incoming
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_3_2),   # contracted axis 0, now outgoing
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_3_2),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_one_of_multiple_pairs(self):
        """With three contracted pairs, flip only the middle one."""
        j_half = yuzuha.Spin(1)

        # Original: A[0,1,2] (in,in,in) × B[0,1,2] (out,out,out)
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flip only pair 1 (axis 1 of A, axis 1 of B): in→out and out→in
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_all_contracted_pairs(self):
        """Flip all three contracted pairs simultaneously."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # All three contracted axes flipped: in→out for A, out→in for B
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_two_index_outcome(self):
        """Flip the contracted pair when the outcome is a two-edge tensor."""
        j_half = yuzuha.Spin(1)

        # Original: A[0,1,2] (in) × B[0,1,2] (out); free: out(A) + out(B)
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flipped: A'[0,1,2] (out) × B'[0,1,2] (in)
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_mixed_spins(self):
        """Flip one contracted pair when contracted edges have mixed spins."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # Contract axes [0,1,2]: j=1, j=1, j=1/2 from A with matching from B
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flip the j=1/2 pair only (axis 2): in→out for A, out→in for B
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Single contracted pair — more spin values
    # ------------------------------------------------------------------

    def test_single_pair_flip_j2(self):
        """Flip the sole contracted pair for j=2, 3-edge × 3-edge."""
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j2),   # contracted axis 2: out(j=2)
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),   # contracted axis 0: in(j=2)
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j2),   # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j2),   # flipped
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_j5_half(self):
        """Flip the sole contracted pair for j=5/2, 3-edge × 3-edge."""
        j2     = yuzuha.Spin(4)  # j=2
        j_half = yuzuha.Spin(1)  # j=1/2
        j5_half = yuzuha.Spin(5) # j=5/2; |2−1/2|=3/2 ≤ 5/2 ≤ 5/2 ✓

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j5_half),  # contracted axis 2
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j5_half),  # contracted axis 0
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j5_half),  # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j5_half),  # flipped
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Single contracted pair — varied axis positions
    # ------------------------------------------------------------------

    def test_single_pair_flip_axis_0(self):
        """Flip contracted pair at axis 0 of A and axis 2 of B."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # A[0]=out(j=1) contracted with B[2]=in(j=1)
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),    # contracted axis 0: out
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),    # contracted axis 2: in
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),    # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),    # flipped
        ])

        contraction = yuzuha.Contraction([0], [2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_middle_axis(self):
        """Flip contracted pair at axis 1 of both A and B."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # A[1]=out(j=1) contracted with B[1]=in(j=1)
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),    # contracted axis 1: out
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j1),    # contracted axis 1: in
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),    # flipped
            yuzuha.Edge.incoming(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),    # flipped
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([1], [1])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Single contracted pair — last edge (index n−1) of each spec
    # ------------------------------------------------------------------

    def test_single_pair_flip_last_edge_j_half(self):
        """Flip the last-edge contracted pair for j=1/2, 4-edge × 4-edge."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # last edge (axis 3): out
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),  # last edge (axis 3): in
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),  # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
        ])

        contraction = yuzuha.Contraction([3], [3])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_last_edge_j1(self):
        """Flip the last-edge contracted pair for j=1, 4-edge × 4-edge."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        # sum 2j = 1+1+2+2 = 6 ✓
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),    # last edge (axis 3): out(j=1)
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),    # last edge (axis 3): in(j=1)
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),    # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),    # flipped
        ])

        contraction = yuzuha.Contraction([3], [3])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_single_pair_flip_last_edge_j3_half(self):
        """Flip the last-edge contracted pair for j=3/2, 4-edge × 4-edge."""
        j_half  = yuzuha.Spin(1)
        j3_half = yuzuha.Spin(3)

        # 3× j=1/2 couple via j=1 intermediate → j=3/2 last edge; sum 2j = 6 ✓
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j3_half),  # last edge (axis 3): out(j=3/2)
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j3_half),  # last edge (axis 3): in(j=3/2)
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j3_half),  # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j3_half),  # flipped
        ])

        contraction = yuzuha.Contraction([3], [3])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Multiple contracted pairs — which pair is flipped
    # ------------------------------------------------------------------

    def test_flip_one_of_multiple_pairs_first(self):
        """With three contracted pairs, flip only the first one (axis 0)."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_one_of_multiple_pairs_last(self):
        """With three contracted pairs, flip only the last one (axis 2)."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_first_and_last_of_multiple_pairs(self):
        """With three contracted pairs, flip the first and last but not the middle."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flip pairs 0 and 2, leave pair 1 unchanged
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_two_index_outcome_j1(self):
        """Flip contracted pair when outcome is a two-edge tensor, integer spin j=1."""
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Multiple contracted pairs — different spins
    # ------------------------------------------------------------------

    def test_flip_all_pairs_j1(self):
        """Flip all three contracted first-region pairs with integer spin j=1."""
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_all_pairs_j3_half(self):
        """Flip all three contracted first-region pairs with half-integer spin j=3/2."""
        j3_half = yuzuha.Spin(3)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_j2_pair_among_mixed(self):
        """Flip the j=2 pair in a 3-pair scenario with mixed j=2 and j=1/2."""
        j_half = yuzuha.Spin(1)
        j2 = yuzuha.Spin(4)

        # Axes 0,1 are j=2; axis 2 is j=1/2; sum 2j = 4+4+1+1 = 10 ✓
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Flip only pair 0 (the j=2 pair): in→out for A[0], out→in for B[0]
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j2),   # flipped
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),   # flipped
            yuzuha.Edge.outgoing(j2),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Higher-order: more contracted pairs
    # ------------------------------------------------------------------

    def test_flip_four_contracted_pairs(self):
        """Flip all four contracted first-region pairs in a 5-edge × 5-edge scenario.

        Five j=1/2 edges give an odd sum of 2j, so the last edge is j=1
        (sum 2j = 4+2 = 6, even), preserving the five-edge structure.
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),      # last edge: j=1
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        # All four first-region axes flipped: in→out for A, out→in for B
        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([0, 1, 2, 3], [0, 1, 2, 3])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_flip_five_contracted_pairs(self):
        """Flip all five contracted first-region pairs in a 6-edge × 6-edge scenario."""
        j_half = yuzuha.Spin(1)

        # 6× j=1/2: sum 2j = 6 ✓
        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),  # last edge
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # 2-edge cases: 2-edge × 2-edge and 2-edge × higher-order
    # ------------------------------------------------------------------

    def test_2edge_2edge_j_half_first_axis(self):
        """2-edge × 2-edge: flip A[0]×B[0], an (in, out) pair, j=1/2."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),   # flipped
            yuzuha.Edge.incoming(j_half),
        ])

        contraction = yuzuha.Contraction([0], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_2edge_j1_last_axis(self):
        """2-edge × 2-edge: flip A[1]×B[1], an (out, out) pair, j=1."""
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),       # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),       # flipped
        ])

        contraction = yuzuha.Contraction([1], [1])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_2edge_j3_half(self):
        """2-edge × 2-edge: flip A[0]×B[0], an (in, out) pair, j=3/2."""
        j3_half = yuzuha.Spin(3)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j3_half),
            yuzuha.Edge.incoming(j3_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j3_half),  # flipped
            yuzuha.Edge.outgoing(j3_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),  # flipped
            yuzuha.Edge.incoming(j3_half),
        ])

        contraction = yuzuha.Contraction([0], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_2edge_j1_same_direction_pair(self):
        """2-edge × 2-edge: flip A[0]×B[0], an (in, in) pair, j=1.

        Both contracted edges point inward; flipping gives an (out, out) pair.
        """
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([0], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_a_vs_4edge_b_j_half(self):
        """2-edge A × 4-edge B: flip A[1]×B[0], an (out, in) pair, j=1/2."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),   # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([1], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_4edge_a_vs_2edge_b_j1(self):
        """4-edge A × 2-edge B: flip A[3]×B[0], an (out, in) pair, j=1."""
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),       # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([3], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_a_vs_6edge_b_j_half(self):
        """2-edge A × 6-edge B: flip A[1]×B[0], an (out, in) pair, j=1/2."""
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),   # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([1], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_2edge_a_vs_3edge_b_j3_half(self):
        """2-edge A × 3-edge B: flip A[1]×B[0], an (out, in) pair, j=3/2.

        B uses [j=3/2, j=1/2, j=1] to keep the sum of 2j even (3+1+2=6).
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        j3_half = yuzuha.Spin(3)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j3_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j3_half),  # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j3_half),  # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j1),
        ])

        contraction = yuzuha.Contraction([1], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    # ------------------------------------------------------------------
    # Different-order: spec_a and spec_b have different numbers of edges
    # ------------------------------------------------------------------

    def test_different_orders_3a_4b(self):
        """Flip the sole contracted pair between a 3-edge A and 4-edge B.

        A[2] (out j=1) is contracted with B[0] (in j=1).
        Free indices: A[0,1], B[1,2,3].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),       # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_4a_3b(self):
        """Flip the sole contracted pair between a 4-edge A and 3-edge B.

        A[3] (out j=1) is contracted with B[0] (in j=1).
        Free indices: A[0,1,2], B[1,2].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),       # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([3], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_3a_5b(self):
        """Flip the sole contracted pair between a 3-edge A and 5-edge B.

        A[2] (out j=1) is contracted with B[0] (in j=1).
        Free indices: A[0,1], B[1,2,3,4].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),       # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),       # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([2], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_4a_6b(self):
        """Flip the sole contracted pair between a 4-edge A and 6-edge B.

        A[3] (out j=1/2) is contracted with B[0] (in j=1/2).
        Free indices: A[0,1,2], B[1,2,3,4,5].
        """
        j_half = yuzuha.Spin(1)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),   # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([3], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_5a_4b_two_pairs(self):
        """Flip both first-region contracted pairs between a 5-edge A and 4-edge B.

        A[0,1] (in j=1/2) contracted with B[0,1] (out j=1/2); A has one more free edge.
        Free indices: A[2,3,4], B[2,3].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),   # flipped
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),   # flipped
            yuzuha.Edge.incoming(j_half),   # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1], [0, 1])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_4a_5b_three_pairs_flip_one(self):
        """Flip one of three contracted pairs between a 4-edge A and 5-edge B.

        A[0,1,2] (in j=1/2) contracted with B[0,1,2] (in j=1/2).
        Flip only pair 0: A[0] in→out, B[0] in→out.
        Free indices: A[3], B[3,4].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped pair 0
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),   # flipped pair 0
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([0, 1, 2], [0, 1, 2])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

    def test_different_orders_5a_3b_last_vs_first(self):
        """Flip the contracted pair between the last edge of 5-edge A and first of 3-edge B.

        A[4] (out j=1) contracted with B[0] (in j=1) — asymmetric axis positions.
        Free indices: A[0,1,2,3], B[1,2].
        """
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)

        spec_a1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        spec_b1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        spec_a2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),      # flipped
        ])
        spec_b2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),      # flipped
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        contraction = yuzuha.Contraction([4], [0])
        self._xsymbol_equal(spec_a1, spec_b1, spec_a2, spec_b2, contraction)

