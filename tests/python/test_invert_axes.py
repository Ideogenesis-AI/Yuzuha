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
Tests for CGSpec.with_inverted_axes.

CGSpec.with_inverted_axes(axes: list[int]) -> CGSpec
  - Returns a new CGSpec with edge directions at the given axes flipped.
  - Spins and OM configurations (alphas) are unchanged.
  - Raises ValueError for any out-of-bounds axis index.

Fixtures:
  - 3-edge spec (j=1, in/in/out): used for metadata-only assertions.
  - 4-edge spec (j=1/2 x4, in/in/in/out): used for direction and OM tests;
    has om_dim=2, providing a non-trivial case.
"""

import pytest
import yuzuha


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec3():
    """3-edge spec: in, in, out — j=1 throughout. om_dim=1."""
    j1 = yuzuha.Spin(2)
    return yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j1),
        yuzuha.Edge.incoming(j1),
        yuzuha.Edge.outgoing(j1),
    ])


@pytest.fixture
def spec4():
    """4-edge spec: in, in, in, out — j=1/2 throughout. om_dim=2."""
    j_half = yuzuha.Spin(1)
    return yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.outgoing(j_half),
    ])


# ===========================================================================
#  Metadata preserved (3-edge)
# ===========================================================================

class TestMetadataPreserved:
    """with_inverted_axes must not change num_external or om_dimension."""

    def test_num_external_preserved_3edge(self, spec3):
        result = spec3.with_inverted_axes([0])
        assert result.num_external() == spec3.num_external()

    def test_om_dimension_preserved_3edge(self, spec3):
        result = spec3.with_inverted_axes([0, 1])
        assert result.om_dimension() == spec3.om_dimension()

    def test_spins_preserved_3edge(self, spec3):
        result = spec3.with_inverted_axes([0, 1, 2])
        for orig, new in zip(spec3.edges, result.edges):
            assert orig.j.twice() == new.j.twice()

    def test_num_external_preserved_4edge(self, spec4):
        result = spec4.with_inverted_axes([1, 3])
        assert result.num_external() == spec4.num_external()

    def test_om_dimension_preserved_4edge(self, spec4):
        result = spec4.with_inverted_axes([0, 2])
        assert result.om_dimension() == spec4.om_dimension()
        assert result.om_dimension() == 2  # non-trivial OM

    def test_spins_preserved_4edge(self, spec4):
        result = spec4.with_inverted_axes([0, 1, 2, 3])
        for orig, new in zip(spec4.edges, result.edges):
            assert orig.j.twice() == new.j.twice()


# ===========================================================================
#  Direction flipping (4-edge)
# ===========================================================================

class TestDirectionFlipping:
    """Verify that exactly the specified axes are flipped, others unchanged."""

    def test_flip_single_axis(self, spec4):
        result = spec4.with_inverted_axes([0])
        assert result.edges[0].dir.is_outgoing()  # flipped
        assert result.edges[1].dir.is_incoming()   # unchanged
        assert result.edges[2].dir.is_incoming()   # unchanged
        assert result.edges[3].dir.is_outgoing()   # unchanged

    def test_flip_two_axes(self, spec4):
        result = spec4.with_inverted_axes([1, 3])
        assert result.edges[0].dir.is_incoming()   # unchanged
        assert result.edges[1].dir.is_outgoing()   # flipped
        assert result.edges[2].dir.is_incoming()   # unchanged
        assert result.edges[3].dir.is_incoming()   # flipped

    def test_flip_all_axes(self, spec4):
        result = spec4.with_inverted_axes([0, 1, 2, 3])
        assert result.edges[0].dir.is_outgoing()
        assert result.edges[1].dir.is_outgoing()
        assert result.edges[2].dir.is_outgoing()
        assert result.edges[3].dir.is_incoming()

    def test_empty_axes_leaves_directions_unchanged(self, spec4):
        result = spec4.with_inverted_axes([])
        for orig, new in zip(spec4.edges, result.edges):
            assert orig.dir == new.dir


# ===========================================================================
#  Immutability of original
# ===========================================================================

class TestOriginalUnchanged:
    """with_inverted_axes must return a new spec without mutating the original."""

    def test_original_3edge_unchanged(self, spec3):
        _ = spec3.with_inverted_axes([0, 1, 2])
        assert spec3.edges[0].dir.is_incoming()
        assert spec3.edges[1].dir.is_incoming()
        assert spec3.edges[2].dir.is_outgoing()

    def test_original_4edge_unchanged(self, spec4):
        _ = spec4.with_inverted_axes([0, 1, 2, 3])
        assert spec4.edges[0].dir.is_incoming()
        assert spec4.edges[1].dir.is_incoming()
        assert spec4.edges[2].dir.is_incoming()
        assert spec4.edges[3].dir.is_outgoing()


# ===========================================================================
#  Double inversion (4-edge)
# ===========================================================================

class TestDoubleInvert:
    """Inverting the same axes twice must restore the original directions."""

    def test_double_invert_subset(self, spec4):
        once = spec4.with_inverted_axes([0, 2])
        twice = once.with_inverted_axes([0, 2])
        for orig, restored in zip(spec4.edges, twice.edges):
            assert orig.dir == restored.dir

    def test_double_invert_all(self, spec4):
        once = spec4.with_inverted_axes([0, 1, 2, 3])
        twice = once.with_inverted_axes([0, 1, 2, 3])
        for orig, restored in zip(spec4.edges, twice.edges):
            assert orig.dir == restored.dir


# ===========================================================================
#  Error handling
# ===========================================================================

class TestErrorHandling:

    def test_out_of_bounds_exact(self, spec3):
        with pytest.raises(Exception):
            spec3.with_inverted_axes([3])

    def test_out_of_bounds_large(self, spec4):
        with pytest.raises(Exception):
            spec4.with_inverted_axes([99])

    def test_out_of_bounds_mixed_valid_invalid(self, spec4):
        with pytest.raises(Exception):
            spec4.with_inverted_axes([0, 4])
