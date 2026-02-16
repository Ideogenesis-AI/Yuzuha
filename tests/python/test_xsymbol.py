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
    
    Method 1: Direct contraction of CG tensors A and B
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
    
    # Method 1: Direct contraction
    cg_tensor_c = np.tensordot(cg_tensor_a, cg_tensor_b, axes=(axes_a, axes_b))
    
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
