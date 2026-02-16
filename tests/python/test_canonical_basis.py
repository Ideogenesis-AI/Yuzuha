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
Tests for canonical basis computation.

Tests the canonical_basis function for various spin configurations,
edge directions, and validates normalization properties.
"""
import pytest
import numpy as np
import yuzuha


class TestCanonicalBasisErrors:
    """Test error handling for invalid inputs."""

    def test_single_edge_error(self):
        """Test that single edge raises ValueError."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
        ])
        
        with pytest.raises(ValueError, match="Cannot build canonical basis with 1 external edges"):
            yuzuha.canonical_basis(spec)

    def test_two_edges_error(self):
        """Test that two edges raises ValueError."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        with pytest.raises(ValueError, match="Cannot build canonical basis with 2 external edges"):
            yuzuha.canonical_basis(spec)


class TestCanonicalBasisShape:
    """Test shape of canonical basis arrays."""

    def test_three_edges_j1(self):
        """Test canonical basis shape for three j=1 edges."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Shape should be [3, 3, 3, om_dim] for j=1 (dimension 3)
        assert basis.ndim == 4
        assert basis.shape == (3, 3, 3, om_dim)
        assert basis.shape[3] >= 1  # At least one OM configuration

    def test_three_edges_half_integer(self):
        """Test canonical basis shape for three j=1/2, 1/2, 1 edges."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Shape should be [2, 2, 3, 1]
        assert basis.ndim == 4
        assert basis.shape == (2, 2, 3, 1)

    def test_four_edges_j_half(self):
        """Test canonical basis shape for four j=1/2 edges."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Shape should be [2, 2, 2, 2, om_dim] for j=1/2 (dimension 2)
        assert basis.ndim == 5
        assert basis.shape[0] == 2
        assert basis.shape[1] == 2
        assert basis.shape[2] == 2
        assert basis.shape[3] == 2
        assert basis.shape[4] == om_dim
        assert om_dim == 2  # Known from theory

    def test_five_edges(self):
        """Test canonical basis shape for five edges."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Shape should be [3, 3, 3, 3, 3, om_dim]
        assert basis.ndim == 6
        for i in range(5):
            assert basis.shape[i] == 3
        assert basis.shape[5] == om_dim

    def test_mixed_spins(self):
        """Test canonical basis shape with mixed spin values."""
        j_half = yuzuha.Spin(1)  # j=1/2
        j1 = yuzuha.Spin(2)      # j=1
        j3_half = yuzuha.Spin(3)  # j=3/2
        
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j3_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Shape should be [2, 3, 4, om_dim]
        assert basis.ndim == 4
        assert basis.shape[0] == 2  # dim(j=1/2)
        assert basis.shape[1] == 3  # dim(j=1)
        assert basis.shape[2] == 4  # dim(j=3/2)


class TestCanonicalBasisDirections:
    """Test canonical basis with various arrow directions."""

    def test_all_incoming(self):
        """Test canonical basis with all incoming edges."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Should succeed and return proper shape
        assert basis.shape == (2, 2, 2, 2, om_dim)

    def test_all_outgoing(self):
        """Test canonical basis with all outgoing edges."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Should succeed and return proper shape
        assert basis.shape == (3, 3, 3, om_dim)

    def test_mixed_directions(self):
        """Test canonical basis with mixed arrow directions."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Should succeed
        assert basis.shape == (2, 2, 3, spec.om_dimension())

    def test_alternating_directions(self):
        """Test canonical basis with alternating arrow directions."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Should succeed and return proper shape
        assert basis.shape == (3, 3, 3, 3, om_dim)


class TestCanonicalBasisProperties:
    """Test mathematical properties of canonical basis."""

    def test_output_type(self):
        """Test that output is a numpy array."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        assert isinstance(basis, np.ndarray)
        assert basis.dtype == np.float64

    def test_real_valued(self):
        """Test that basis is real-valued."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Should not be complex
        assert not np.iscomplexobj(basis)
        assert np.all(np.isreal(basis))

    def test_normalization_three_edges(self):
        """Test that each OM basis vector is normalized for three edges."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Each OM basis vector (last index) should have Frobenius norm = 1
        for alpha in range(om_dim):
            vec = basis[..., alpha]
            frob_norm = np.linalg.norm(vec)
            assert np.isclose(frob_norm, 1.0, rtol=1e-10, atol=1e-10)

    def test_normalization_four_edges(self):
        """Test that each OM basis vector is normalized for four edges."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Each OM basis vector should have Frobenius norm = 1
        for alpha in range(om_dim):
            vec = basis[..., alpha]
            frob_norm = np.linalg.norm(vec)
            assert np.isclose(frob_norm, 1.0, rtol=1e-10, atol=1e-10)

    def test_normalization_mixed_directions(self):
        """Test normalization with mixed arrow directions."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.outgoing(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        for alpha in range(om_dim):
            vec = basis[..., alpha]
            frob_norm = np.linalg.norm(vec)
            assert np.isclose(frob_norm, 1.0, rtol=1e-10, atol=1e-10)

    def test_no_nans_or_infs(self):
        """Test that basis contains no NaNs or infinities."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        assert not np.any(np.isnan(basis))
        assert not np.any(np.isinf(basis))


class TestCanonicalBasisOrthogonality:
    """Test orthogonality properties of canonical basis."""

    def test_om_vectors_orthogonal(self):
        """Test that different OM basis vectors are orthogonal."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # For multiple OM configurations, check orthogonality
        if om_dim > 1:
            for alpha1 in range(om_dim):
                for alpha2 in range(alpha1 + 1, om_dim):
                    vec1 = basis[..., alpha1].flatten()
                    vec2 = basis[..., alpha2].flatten()
                    inner_product = np.dot(vec1, vec2)
                    assert np.isclose(inner_product, 0.0, rtol=1e-10, atol=1e-10)

    def test_orthonormality_matrix(self):
        """Test that OM basis forms an orthonormal set."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        om_dim = spec.om_dimension()
        
        # Reshape to matrix form [product of physical dims, om_dim]
        physical_dim = np.prod(basis.shape[:-1])
        basis_matrix = basis.reshape(physical_dim, om_dim)
        
        # Compute Gram matrix: B^T @ B
        gram = basis_matrix.T @ basis_matrix
        
        # Should be identity matrix
        expected = np.eye(om_dim)
        assert np.allclose(gram, expected, rtol=1e-10, atol=1e-10)


class TestCanonicalBasisConsistency:
    """Test consistency and determinism of canonical basis."""

    def test_deterministic(self):
        """Test that repeated calls give same result."""
        j1 = yuzuha.Spin(2)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis1 = yuzuha.canonical_basis(spec)
        basis2 = yuzuha.canonical_basis(spec)
        
        assert np.allclose(basis1, basis2, rtol=1e-15, atol=1e-15)

    def test_cache_consistency(self):
        """Test that caching doesn't affect results."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        
        # First call (may populate cache)
        spec1 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        basis1 = yuzuha.canonical_basis(spec1)
        
        # Second call with same spins (should hit cache)
        spec2 = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j1),  # Different direction
        ])
        basis2 = yuzuha.canonical_basis(spec2)
        
        # Shapes should match except for direction transformations
        assert basis1.shape[:-1] == basis2.shape[:-1]

    def test_different_specs_same_spins(self):
        """Test that different directions with same spins give related results."""
        j1 = yuzuha.Spin(2)
        
        # Canonical direction: n-1 incoming, 1 outgoing
        spec_canonical = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.outgoing(j1),
        ])
        
        # All incoming
        spec_all_in = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j1),
        ])
        
        basis_canonical = yuzuha.canonical_basis(spec_canonical)
        basis_all_in = yuzuha.canonical_basis(spec_all_in)
        
        # Should have same shape and both be normalized
        assert basis_canonical.shape == basis_all_in.shape
        
        # Both should satisfy normalization
        om_dim = spec_canonical.om_dimension()
        for alpha in range(om_dim):
            assert np.isclose(np.linalg.norm(basis_canonical[..., alpha]), 1.0)
            assert np.isclose(np.linalg.norm(basis_all_in[..., alpha]), 1.0)


class TestCanonicalBasisHigherSpins:
    """Test canonical basis with higher spin values."""

    def test_j2_spins(self):
        """Test canonical basis with j=2 spins."""
        j2 = yuzuha.Spin(4)  # j=2
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.outgoing(j2),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Shape should be [5, 5, 5, om_dim] for j=2 (dimension 5)
        assert basis.shape[0] == 5
        assert basis.shape[1] == 5
        assert basis.shape[2] == 5
        
        # Check normalization
        om_dim = spec.om_dimension()
        for alpha in range(om_dim):
            frob_norm = np.linalg.norm(basis[..., alpha])
            assert np.isclose(frob_norm, 1.0, rtol=1e-10)

    def test_j3_half_spins(self):
        """Test canonical basis with j=3/2 spins."""
        j3_half = yuzuha.Spin(3)  # j=3/2
        j_half = yuzuha.Spin(1)   # j=1/2
        
        # Use j=3/2, 1/2, 1 which can couple to j=0
        j1 = yuzuha.Spin(2)  # j=1
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j3_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Shape should be [4, 2, 3, om_dim] for j=3/2, 1/2, 1
        assert basis.shape[0] == 4  # dim(j=3/2)
        assert basis.shape[1] == 2  # dim(j=1/2)
        assert basis.shape[2] == 3  # dim(j=1)

    def test_large_spin_system(self):
        """Test canonical basis with larger spin configuration."""
        j1 = yuzuha.Spin(2)
        j2 = yuzuha.Spin(4)
        
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j2),
            yuzuha.Edge.outgoing(j2),
        ])
        
        basis = yuzuha.canonical_basis(spec)
        
        # Shape should be [3, 5, 5, om_dim]
        assert basis.shape[0] == 3  # dim(j=1)
        assert basis.shape[1] == 5  # dim(j=2)
        assert basis.shape[2] == 5  # dim(j=2)
        
        # Should still be normalized
        om_dim = spec.om_dimension()
        for alpha in range(om_dim):
            frob_norm = np.linalg.norm(basis[..., alpha])
            assert np.isclose(frob_norm, 1.0, rtol=1e-10)
