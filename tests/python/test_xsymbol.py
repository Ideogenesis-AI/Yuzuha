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
Tests for X-symbol computation.

Tests X-symbol computation for various tensor configurations.
"""
import pytest
import numpy as np
import yuzuha


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
