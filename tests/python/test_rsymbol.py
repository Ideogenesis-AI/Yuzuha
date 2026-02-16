# Copyright (C) 2026 Changkai Zhang.
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
Comprehensive tests for R-symbol computation and consistency.

R-symbols represent basis transformations for different leg orderings in
tensor networks. They possess strong mathematical properties that can be
tested for self-consistency.

Test Organization
-----------------
- TestRSymbolBasic: Basic API and dimension tests
- TestRSymbolUnitarity: Verify unitarity property (R^T R = I)
- TestRSymbolInversion: Test permutation inversion (R[perm] * R[perm^{-1}] = I)
- TestRSymbolComposition: Test composition (R[perm1 ∘ perm2] = R[perm2] * R[perm1])
- TestRSymbolFSymbolCoincidence: Compare with known F-symbol values
- TestRSymbolStress: Randomized stress tests
- TestRSymbolEdgeCases: Boundary conditions and special cases
- TestRSymbolCaching: Verify caching behavior

Key Properties
--------------
1. **Unitarity**: R-symbols are unitary matrices (basis change)
2. **Inversion**: R[perm] * R[perm^{-1}] = I
3. **Composition**: R[perm1 ∘ perm2] = R[perm2] * R[perm1]
4. **F-symbol relation**: For specific permutations, R-symbols coincide with F-symbols

Total: 30+ tests covering basic, unitarity, inversion, composition, F-symbols, and stress tests.
"""
import pytest
import numpy as np
import yuzuha


class TestRSymbolBasic:
    """Basic R-symbol computation tests."""

    def test_rsymbol_identity_permutation(self):
        """Test R-symbol with identity permutation gives identity matrix."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [0, 1, 2, 3]  # Identity
        r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)

        # Check return types
        assert isinstance(r_array, np.ndarray)
        assert r_array.dtype == np.float64
        assert len(r_array.shape) == 2

        # Check dimensions
        dim = spec.om_dimension()
        assert r_array.shape == (dim, dim)

        # Should be identity matrix
        expected = np.eye(dim)
        assert np.allclose(r_array, expected, atol=1e-10)

        # Spec should be unchanged
        assert spec_permuted.om_dimension() == spec.om_dimension()

    def test_rsymbol_swap_first_two(self):
        """Test R-symbol with swap of first two legs."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]  # Swap first two
        r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)

        dim = spec.om_dimension()
        assert r_array.shape == (dim, dim)
        assert spec_permuted.om_dimension() == dim

    def test_rsymbol_three_legs(self):
        """Test R-symbol with three legs."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])

        permutation = [1, 2, 0]  # Cyclic permutation
        r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)

        dim = spec.om_dimension()
        assert r_array.shape == (dim, dim)

    def test_rsymbol_invalid_permutation(self):
        """Test that invalid permutations raise errors."""
        j1 = yuzuha.Spin(2)  # j=1
        j_half = yuzuha.Spin(1)  # j=1/2
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])

        # Wrong length
        with pytest.raises(ValueError):
            yuzuha.compute_rsymbol(spec, [0, 1])

        # Out of range
        with pytest.raises(ValueError):
            yuzuha.compute_rsymbol(spec, [0, 1, 5])

        # Duplicate index
        with pytest.raises(ValueError):
            yuzuha.compute_rsymbol(spec, [0, 0, 2])


class TestRSymbolProperties:
    """Test mathematical properties of R-symbols."""

    def test_rsymbol_unitary(self):
        """Test that R-symbols are unitary matrices."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)

        # Check R^T R = I
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)

    def test_rsymbol_involution(self):
        """Test that swapping twice gives identity."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)

        # R * R should be identity (swapping twice)
        r_squared = r_array @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_squared, identity, atol=1e-10)

    def test_rsymbol_real_valued(self):
        """Test that R-symbols are real-valued."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)

        assert r_array.dtype == np.float64
        assert np.all(np.isreal(r_array))
        assert np.all(np.isfinite(r_array))


class TestRSymbolCaching:
    """Test that caching works across multiple calls."""

    def test_repeated_calls_consistent(self):
        """Test that repeated calls give consistent results."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]

        # First call
        r1, spec1 = yuzuha.compute_rsymbol(spec, permutation)

        # Second call (should use cache)
        r2, spec2 = yuzuha.compute_rsymbol(spec, permutation)

        # Results should be identical
        assert np.allclose(r1, r2)
        assert spec1.om_dimension() == spec2.om_dimension()


class TestRSymbolUnitarity:
    """Test unitarity property: R^T R = I."""
    
    def test_unitarity_simple_swap(self):
        """Test unitarity for simple two-element swap."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Check R^T R = I
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10), \
            f"R^T R is not identity: max diff = {np.max(np.abs(r_t_r - identity)):.2e}"
        
        # Also check R R^T = I
        r_r_t = r_array @ r_array.T
        assert np.allclose(r_r_t, identity, atol=1e-10), \
            f"R R^T is not identity: max diff = {np.max(np.abs(r_r_t - identity)):.2e}"
    
    def test_unitarity_cyclic_permutation(self):
        """Test unitarity for cyclic permutation."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [1, 2, 0]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)
    
    def test_unitarity_complex_permutation(self):
        """Test unitarity for complex permutation."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [3, 1, 0, 2]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)
    
    def test_unitarity_larger_spins(self):
        """Test unitarity with larger spin values."""
        j_3_2 = yuzuha.Spin(3)  # j=3/2
        j2 = yuzuha.Spin(4)  # j=2
        
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_3_2),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j_3_2),
        ])
        
        permutation = [2, 0, 1]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)
    
    def test_unitarity_five_edges(self):
        """Test unitarity with 5-edge tensor."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # Need even number of half-integers (4 in this case)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [4, 2, 1, 3, 0]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)


class TestRSymbolInversion:
    """Test inversion property: R[perm] * R[perm^{-1}] = I."""
    
    def _inverse_permutation(self, perm):
        """Compute inverse permutation."""
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return inv
    
    def test_inversion_swap(self):
        """Test that swapping twice returns to identity."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # For a swap, inverse is itself
        r_squared = r_array @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_squared, identity, atol=1e-10), \
            "Swap applied twice should give identity"
    
    def test_inversion_cyclic(self):
        """Test inversion of cyclic permutation."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        # Forward: [0, 1, 2] -> [1, 2, 0]
        permutation = [1, 2, 0]
        r_forward, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Inverse: [0, 1, 2] -> [2, 0, 1]
        inv_permutation = self._inverse_permutation(permutation)
        r_inverse, _ = yuzuha.compute_rsymbol(spec, inv_permutation)
        
        # R[perm] * R[perm^{-1}] should be identity
        product = r_forward @ r_inverse
        identity = np.eye(r_forward.shape[0])
        assert np.allclose(product, identity, atol=1e-10), \
            "R[perm] * R[perm^{-1}] should be identity"
    
    def test_inversion_complex(self):
        """Test inversion for complex permutation."""
        j_half = yuzuha.Spin(1)
        
        # Use all j=1/2 for simpler case
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [3, 1, 0, 2]
        r_forward, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        inv_permutation = self._inverse_permutation(permutation)
        r_inverse, _ = yuzuha.compute_rsymbol(spec, inv_permutation)
        
        product = r_forward @ r_inverse
        identity = np.eye(r_forward.shape[0])
        assert np.allclose(product, identity, atol=1e-10)
        
        # Also check the reverse order
        product_rev = r_inverse @ r_forward
        assert np.allclose(product_rev, identity, atol=1e-10)
    
    def test_inversion_all_adjacent_swaps(self):
        """Test inversion for all adjacent swaps in 4-edge system."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        # Test all adjacent swaps
        adjacent_swaps = [
            [1, 0, 2, 3],  # Swap 0-1
            [0, 2, 1, 3],  # Swap 1-2
            [0, 1, 3, 2],  # Swap 2-3
        ]
        
        for perm in adjacent_swaps:
            r_array, _ = yuzuha.compute_rsymbol(spec, perm)
            # Adjacent swap is self-inverse
            r_squared = r_array @ r_array
            identity = np.eye(r_array.shape[0])
            assert np.allclose(r_squared, identity, atol=1e-10), \
                f"Adjacent swap {perm} should be self-inverse"


class TestRSymbolComposition:
    """Test composition property: R[perm1 ∘ perm2] = R[perm2] * R[perm1]."""
    
    def _compose_permutations(self, perm1, perm2):
        """Compose two permutations: apply perm1 first, then perm2."""
        return [perm2[i] for i in perm1]
    
    def test_composition_two_swaps(self):
        """Test composition of two adjacent swaps."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        perm1 = [1, 0, 2, 3]  # Swap 0-1
        perm2 = [0, 2, 1, 3]  # Swap 1-2
        
        r1, _ = yuzuha.compute_rsymbol(spec, perm1)
        r2, _ = yuzuha.compute_rsymbol(spec, perm2)
        
        # Composed permutation: apply perm1 first, then perm2
        composed_perm = self._compose_permutations(perm1, perm2)
        r_composed, _ = yuzuha.compute_rsymbol(spec, composed_perm)
        
        # R[perm2 ∘ perm1] = R[perm2] * R[perm1]
        r_product = r2 @ r1
        assert np.allclose(r_composed, r_product, atol=1e-10), \
            "Composition property failed: R[perm2 ∘ perm1] != R[perm2] * R[perm1]"
    
    def test_composition_three_permutations(self):
        """Test composition of three permutations."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        perm1 = [1, 0, 2, 3]  # Swap 0-1
        perm2 = [0, 2, 1, 3]  # Swap 1-2
        perm3 = [0, 1, 3, 2]  # Swap 2-3
        
        r1, _ = yuzuha.compute_rsymbol(spec, perm1)
        r2, _ = yuzuha.compute_rsymbol(spec, perm2)
        r3, _ = yuzuha.compute_rsymbol(spec, perm3)
        
        # Compose all three: apply perm1, then perm2, then perm3
        perm_12 = self._compose_permutations(perm1, perm2)
        perm_123 = self._compose_permutations(perm_12, perm3)
        
        r_composed, _ = yuzuha.compute_rsymbol(spec, perm_123)
        r_product = r3 @ r2 @ r1
        
        assert np.allclose(r_composed, r_product, atol=1e-10)
    
    def test_composition_cyclic(self):
        """Test composition of cyclic permutations."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        # Cyclic permutation: [0, 1, 2] -> [1, 2, 0]
        perm = [1, 2, 0]
        r_perm, _ = yuzuha.compute_rsymbol(spec, perm)
        
        # Apply twice: should get [2, 0, 1]
        perm2 = self._compose_permutations(perm, perm)
        r_perm2, _ = yuzuha.compute_rsymbol(spec, perm2)
        
        r_squared = r_perm @ r_perm
        assert np.allclose(r_perm2, r_squared, atol=1e-10)
        
        # Apply three times: should get identity [0, 1, 2]
        r_cubed = r_perm @ r_perm @ r_perm
        identity = np.eye(r_perm.shape[0])
        assert np.allclose(r_cubed, identity, atol=1e-10), \
            "Cyclic permutation applied 3 times should give identity"


class TestRSymbolFSymbolCoincidence:
    """Test R-symbol coincidence with known F-symbol values."""
    
    def test_fsymbol_4_spin_half(self):
        """
        Test R-symbol matches F-symbol for 4 spin-1/2 recoupling.
        
        For 4 spin-1/2 edges, the permutation [0, 2, 1, 3] (swapping e1 and e2)
        corresponds to the F-symbol transformation between:
        - Canonical: ((e0 ⊗ e1) ⊗ e2) ⊗ e3
        - Binary: (e0 ⊗ e1) ⊗ (e2 ⊗ e3) with axes rearranged
        
        Known F-symbol values:
        F = [[1/2, sqrt(3)/2],
             [sqrt(3)/2, -1/2]]
        """
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        # Permutation that swaps e1 and e2
        permutation = [0, 2, 1, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected F-symbol values
        sqrt3 = np.sqrt(3)
        expected = np.array([
            [0.5, sqrt3 / 2.0],
            [sqrt3 / 2.0, -0.5]
        ])
        
        assert r_array.shape == (2, 2), f"Expected 2x2 matrix, got {r_array.shape}"
        
        # Check each element
        for i in range(2):
            for j in range(2):
                assert abs(r_array[i, j] - expected[i, j]) < 1e-10, \
                    f"R[{i},{j}] = {r_array[i, j]:.6f}, expected {expected[i, j]:.6f}"
    
    def test_fsymbol_identity_permutation(self):
        """Test that identity permutation gives identity matrix (trivial F-symbol)."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [0, 1, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_array, identity, atol=1e-10), \
            "Identity permutation should give identity matrix"
    
    def test_fsymbol_determinant(self):
        """Test that F-symbol (as R-symbol) has determinant ±1."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [0, 2, 1, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        det = np.linalg.det(r_array)
        assert abs(abs(det) - 1.0) < 1e-10, \
            f"Determinant should be ±1, got {det:.6f}"
    
    def test_fsymbol_4_spin_half_swap_01(self):
        """
        Test F-symbol for 4 spin-1/2 with swap of edges 0-1.
        
        Permutation [1, 0, 2, 3]: swap first two edges
        Exact F-symbol: diag(-1, 1)
        """
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected: diagonal matrix with -1, 1
        expected = np.array([
            [-1.0, 0.0],
            [0.0, 1.0]
        ])
        
        assert r_array.shape == (2, 2)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"Expected diag(-1, 1), got:\n{r_array}"
    
    def test_fsymbol_4_spin_half_swap_23(self):
        """
        Test F-symbol for 4 spin-1/2 with swap of edges 2-3.
        
        Permutation [0, 1, 3, 2]: swap last two edges
        Exact F-symbol: diag(-1, 1)
        """
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [0, 1, 3, 2]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected: diagonal matrix with -1, 1
        expected = np.array([
            [-1.0, 0.0],
            [0.0, 1.0]
        ])
        
        assert r_array.shape == (2, 2)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"Expected diag(-1, 1), got:\n{r_array}"
    
    def test_fsymbol_4_spin_half_cyclic(self):
        """
        Test F-symbol for 4 spin-1/2 with cyclic permutation.
        
        Permutation [1, 2, 3, 0]: rotate all edges
        Exact F-symbol:
        [[1/2,   -√3/2],
         [-√3/2, -1/2 ]]
        """
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        permutation = [1, 2, 3, 0]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected exact values
        sqrt3 = np.sqrt(3)
        expected = np.array([
            [0.5, -sqrt3/2],
            [-sqrt3/2, -0.5]
        ])
        
        assert r_array.shape == (2, 2)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"Expected [[1/2, -√3/2], [-√3/2, -1/2]], got:\n{r_array}"
        
        # Verify cyclic property: R^4 = I
        r_power = r_array
        for _ in range(3):
            r_power = r_power @ r_array
        identity = np.eye(2)
        assert np.allclose(r_power, identity, atol=1e-10)
    
    def test_fsymbol_4_spin_1_swap_12(self):
        """
        Test F-symbol for 4 spin-1 with swap of edges 1-2.
        
        Permutation [0, 2, 1, 3]: swap middle two edges
        Exact F-symbol (3×3):
        [[1/3,    √3/3,    √5/3  ],
         [√3/3,   1/2,    -√15/6 ],
         [√5/3,  -√15/6,   1/6   ]]
        """
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [0, 2, 1, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected exact values
        sqrt3 = np.sqrt(3)
        sqrt5 = np.sqrt(5)
        sqrt15 = np.sqrt(15)
        
        expected = np.array([
            [1/3,      sqrt3/3,   sqrt5/3   ],
            [sqrt3/3,  1/2,      -sqrt15/6  ],
            [sqrt5/3, -sqrt15/6,  1/6       ]
        ])
        
        assert r_array.shape == (3, 3)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"F-symbol mismatch. Expected:\n{expected}\nGot:\n{r_array}"
        
        # Verify properties
        r_t_r = r_array.T @ r_array
        identity = np.eye(3)
        assert np.allclose(r_t_r, identity, atol=1e-10), \
            "R-symbol should be unitary"
        
        r_squared = r_array @ r_array
        assert np.allclose(r_squared, identity, atol=1e-10), \
            "Swap should be an involution"
    
    def test_fsymbol_4_spin_1_swap_01(self):
        """
        Test F-symbol for 4 spin-1 with swap of edges 0-1.
        
        Permutation [1, 0, 2, 3]: swap first two edges
        Exact F-symbol: diag(1, -1, 1)
        """
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [1, 0, 2, 3]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected: diagonal matrix with 1, -1, 1
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        assert r_array.shape == (3, 3)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"Expected diag(1, -1, 1), got:\n{r_array}"
    
    def test_fsymbol_4_spin_1_cyclic(self):
        """
        Test F-symbol for 4 spin-1 with cyclic permutation.
        
        Permutation [1, 2, 3, 0]: rotate all edges
        Exact F-symbol (3×3):
        [[1/3,   -√3/3,    √5/3  ],
         [-√3/3,  1/2,     √15/6 ],
         [√5/3,   √15/6,   1/6   ]]
        """
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [1, 2, 3, 0]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Expected exact values
        sqrt3 = np.sqrt(3)
        sqrt5 = np.sqrt(5)
        sqrt15 = np.sqrt(15)
        
        expected = np.array([
            [1/3,     -sqrt3/3,   sqrt5/3  ],
            [-sqrt3/3, 1/2,       sqrt15/6 ],
            [sqrt5/3,  sqrt15/6,  1/6      ]
        ])
        
        assert r_array.shape == (3, 3)
        assert np.allclose(r_array, expected, atol=1e-10), \
            f"F-symbol mismatch. Expected:\n{expected}\nGot:\n{r_array}"
        
        # Verify unitarity
        r_t_r = r_array.T @ r_array
        identity = np.eye(3)
        assert np.allclose(r_t_r, identity, atol=1e-10)
        
        # Verify cyclic property: R^4 = I
        r_power = r_array
        for _ in range(3):
            r_power = r_power @ r_array
        assert np.allclose(r_power, identity, atol=1e-10)


class TestRSymbolStress:
    """Stress tests with random permutations."""
    
    def _random_permutation(self, n, rng):
        """Generate a random permutation of [0, 1, ..., n-1]."""
        perm = list(range(n))
        rng.shuffle(perm)
        return perm
    
    def _inverse_permutation(self, perm):
        """Compute inverse permutation."""
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return inv
    
    def test_stress_random_unitarity(self):
        """Test unitarity for random permutations."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # Different configurations
        configs = [
            [j_half] * 4,
            [j1] * 3,
            [j_half, j1, j_half, j1],
            [j1, j1, j_half, j_half],
        ]
        
        rng = np.random.default_rng(42)
        
        for spins in configs:
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(s) for s in spins
            ])
            
            # Test multiple random permutations
            for _ in range(5):
                perm = self._random_permutation(len(spins), rng)
                r_array, _ = yuzuha.compute_rsymbol(spec, perm)
                
                # Check unitarity
                r_t_r = r_array.T @ r_array
                identity = np.eye(r_array.shape[0])
                assert np.allclose(r_t_r, identity, atol=1e-10), \
                    f"Unitarity failed for permutation {perm}"
    
    def test_stress_random_inversion(self):
        """Test inversion property for random permutations."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        rng = np.random.default_rng(12345)
        
        for _ in range(10):
            perm = self._random_permutation(4, rng)
            inv_perm = self._inverse_permutation(perm)
            
            r_forward, _ = yuzuha.compute_rsymbol(spec, perm)
            r_inverse, _ = yuzuha.compute_rsymbol(spec, inv_perm)
            
            product = r_forward @ r_inverse
            identity = np.eye(r_forward.shape[0])
            assert np.allclose(product, identity, atol=1e-10), \
                f"Inversion failed for permutation {perm}"
    
    def test_stress_larger_systems(self):
        """Test with larger tensor systems (5-6 edges)."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # 5-edge system (need even number of half-integers)
        spec5 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        perm5 = [4, 2, 0, 3, 1]
        r5, _ = yuzuha.compute_rsymbol(spec5, perm5)
        
        # Check unitarity
        r5_t_r5 = r5.T @ r5
        identity5 = np.eye(r5.shape[0])
        assert np.allclose(r5_t_r5, identity5, atol=1e-10)
        
        # 6-edge system
        spec6 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        perm6 = [5, 3, 1, 4, 2, 0]
        r6, _ = yuzuha.compute_rsymbol(spec6, perm6)
        
        r6_t_r6 = r6.T @ r6
        identity6 = np.eye(r6.shape[0])
        assert np.allclose(r6_t_r6, identity6, atol=1e-10)


class TestRSymbolEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_three_edge_identity(self):
        """Test three-edge identity permutation (minimum required)."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        permutation = [0, 1, 2]  # Identity
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Should be identity matrix
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_array, identity, atol=1e-10)
    
    def test_three_edge_swap(self):
        """Test three-edge swap."""
        j1 = yuzuha.Spin(2)  # Use integer spins
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        # Swap first two edges
        permutation = [1, 0, 2]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Should be unitary
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)
        
        # Should be self-inverse
        r_squared = r_array @ r_array
        assert np.allclose(r_squared, identity, atol=1e-10)
    
    def test_reverse_permutation(self):
        """Test complete reversal of edge order."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        # Reverse: [0, 1, 2, 3] -> [3, 2, 1, 0]
        permutation = [3, 2, 1, 0]
        r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
        
        # Should be unitary
        r_t_r = r_array.T @ r_array
        identity = np.eye(r_array.shape[0])
        assert np.allclose(r_t_r, identity, atol=1e-10)
        
        # Should be self-inverse (reversing twice gives original)
        r_squared = r_array @ r_array
        assert np.allclose(r_squared, identity, atol=1e-10)
