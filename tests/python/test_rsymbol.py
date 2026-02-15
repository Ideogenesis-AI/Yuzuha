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
Tests for R-symbol computation.

Tests R-symbol computation for tensor leg permutations.
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
