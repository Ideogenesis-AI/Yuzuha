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
Basic tests for Yuzuha Python bindings.

Tests basic type construction and properties.
"""

import pytest
import yuzuha


class TestDirection:
    """Test Direction class."""

    def test_incoming_sign(self):
        """Incoming direction has sign +1."""
        d = yuzuha.Direction.incoming()
        assert d.sign() == 1

    def test_outgoing_sign(self):
        """Outgoing direction has sign -1."""
        d = yuzuha.Direction.outgoing()
        assert d.sign() == -1

    def test_from_sign_incoming(self):
        """from_sign(+1) yields incoming."""
        d = yuzuha.Direction.from_sign(1)
        assert d.is_incoming()
        assert d.sign() == 1

    def test_from_sign_outgoing(self):
        """from_sign(-1) yields outgoing."""
        d = yuzuha.Direction.from_sign(-1)
        assert d.is_outgoing()
        assert d.sign() == -1

    def test_from_sign_invalid(self):
        """from_sign with a value other than ±1 raises ValueError."""
        with pytest.raises(ValueError):
            yuzuha.Direction.from_sign(0)
        with pytest.raises(ValueError):
            yuzuha.Direction.from_sign(2)

    def test_is_incoming_outgoing(self):
        """is_incoming / is_outgoing are mutually exclusive."""
        inc = yuzuha.Direction.incoming()
        out = yuzuha.Direction.outgoing()
        assert inc.is_incoming()
        assert not inc.is_outgoing()
        assert out.is_outgoing()
        assert not out.is_incoming()

    def test_flip(self):
        """flip() returns the opposite direction."""
        inc = yuzuha.Direction.incoming()
        out = yuzuha.Direction.outgoing()
        assert inc.flip().is_outgoing()
        assert out.flip().is_incoming()
        assert inc.flip().sign() == -1
        assert out.flip().sign() == 1

    def test_flip_twice_is_identity(self):
        """Flipping twice returns the original direction."""
        inc = yuzuha.Direction.incoming()
        out = yuzuha.Direction.outgoing()
        assert inc.flip().flip().is_incoming()
        assert out.flip().flip().is_outgoing()

    def test_equality_same(self):
        """Two directions of the same kind compare equal."""
        assert yuzuha.Direction.incoming() == yuzuha.Direction.incoming()
        assert yuzuha.Direction.outgoing() == yuzuha.Direction.outgoing()

    def test_equality_different(self):
        """Incoming and outgoing compare not equal."""
        assert yuzuha.Direction.incoming() != yuzuha.Direction.outgoing()

    def test_compare_two_edge_dirs(self):
        """edge.dir from two edges of the same orientation compare equal."""
        j = yuzuha.Spin(1)
        edge_a = yuzuha.Edge.incoming(j)
        edge_b = yuzuha.Edge.incoming(j)
        assert edge_a.dir == edge_b.dir

    def test_compare_two_edge_dirs_different(self):
        """edge.dir from edges of opposite orientations compare not equal."""
        j = yuzuha.Spin(1)
        assert yuzuha.Edge.incoming(j).dir != yuzuha.Edge.outgoing(j).dir

    def test_hashable_in_set(self):
        """Direction objects can be stored in sets and used as dict keys."""
        directions = {yuzuha.Direction.incoming(), yuzuha.Direction.outgoing()}
        assert len(directions) == 2
        assert yuzuha.Direction.incoming() in directions

    def test_repr(self):
        """repr contains the direction name."""
        assert "incoming" in repr(yuzuha.Direction.incoming())
        assert "outgoing" in repr(yuzuha.Direction.outgoing())


class TestSpin:
    """Test Spin class."""

    def test_spin_creation(self):
        """Test creating spin quantum numbers."""
        j_half = yuzuha.Spin(1)
        assert j_half.twice() == 1
        assert j_half.value() == 0.5
        assert j_half.dimension() == 2
        assert j_half.is_half_integer()
        assert not j_half.is_integer()

    def test_spin_j1(self):
        """Test j=1 spin."""
        j1 = yuzuha.Spin(2)
        assert j1.twice() == 2
        assert j1.value() == 1.0
        assert j1.dimension() == 3
        assert not j1.is_half_integer()
        assert j1.is_integer()

    def test_spin_j0(self):
        """Test j=0 spin."""
        j0 = yuzuha.Spin(0)
        assert j0.twice() == 0
        assert j0.value() == 0.0
        assert j0.dimension() == 1
        assert j0.is_integer()

    def test_spin_invalid(self):
        """Test that negative spins raise ValueError."""
        with pytest.raises(ValueError):
            yuzuha.Spin(-1)

    def test_spin_repr(self):
        """Test string representation."""
        j_half = yuzuha.Spin(1)
        assert "Spin" in repr(j_half)
        j1 = yuzuha.Spin(2)
        assert "1" in str(j1)


class TestEdge:
    """Test Edge class."""

    def test_incoming_edge(self):
        """Test creating incoming edges."""
        j = yuzuha.Spin(1)
        edge = yuzuha.Edge.incoming(j)
        assert edge.is_incoming()
        assert not edge.is_outgoing()
        assert edge.j.twice() == 1
        assert edge.dir == yuzuha.Direction.incoming()
        assert edge.dir.sign() == 1

    def test_outgoing_edge(self):
        """Test creating outgoing edges."""
        j = yuzuha.Spin(2)
        edge = yuzuha.Edge.outgoing(j)
        assert edge.is_outgoing()
        assert not edge.is_incoming()
        assert edge.j.twice() == 2
        assert edge.dir == yuzuha.Direction.outgoing()
        assert edge.dir.sign() == -1

    def test_edge_repr(self):
        """Test string representation."""
        j = yuzuha.Spin(1)
        edge_in = yuzuha.Edge.incoming(j)
        assert "incoming" in repr(edge_in)
        edge_out = yuzuha.Edge.outgoing(j)
        assert "outgoing" in repr(edge_out)


class TestCGSpec:
    """Test CGSpec class."""

    def test_cgspec_three_edges(self):
        """Test CGSpec with three edges - using valid spin configuration."""
        j1 = yuzuha.Spin(2)  # j=1
        j_half = yuzuha.Spin(1)  # j=1/2
        edges = [
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ]
        spec = yuzuha.CGSpec.from_edges(edges)
        assert spec.num_external() == 3
        assert spec.om_dimension() >= 1

    def test_cgspec_four_edges(self):
        """Test CGSpec with four edges."""
        j_half = yuzuha.Spin(1)
        edges = [
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ]
        spec = yuzuha.CGSpec.from_edges(edges)
        assert spec.num_external() == 4
        assert spec.om_dimension() == 2  # Known from Rust tests

    def test_cgspec_repr(self):
        """Test string representation."""
        j1 = yuzuha.Spin(2)  # j=1
        j_half = yuzuha.Spin(1)  # j=1/2
        edges = [
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ]
        spec = yuzuha.CGSpec.from_edges(edges)
        assert "CGSpec" in repr(spec)
        assert "3" in repr(spec)  # num_external


class TestContraction:
    """Test Contraction class."""

    def test_contraction_creation(self):
        """Test creating a contraction specification."""
        contraction = yuzuha.Contraction([2], [0])
        assert "Contraction" in repr(contraction)
        assert "[2]" in repr(contraction)
        assert "[0]" in repr(contraction)

    def test_contraction_multiple_edges(self):
        """Test contracting multiple edges."""
        contraction = yuzuha.Contraction([2, 3], [0, 1])
        assert "Contraction" in repr(contraction)
