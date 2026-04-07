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

    def test_frozen(self):
        """Direction instances are immutable: attribute assignment raises AttributeError."""
        d = yuzuha.Direction.incoming()
        with pytest.raises(AttributeError):
            d.new_attr = 42


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

    def test_spin_equality(self):
        """Spins with the same doubled value compare equal."""
        assert yuzuha.Spin(1) == yuzuha.Spin(1)
        assert yuzuha.Spin(2) == yuzuha.Spin(2)
        assert yuzuha.Spin(0) == yuzuha.Spin(0)
        assert yuzuha.Spin(1) != yuzuha.Spin(2)
        assert yuzuha.Spin(0) != yuzuha.Spin(1)

    def test_spin_hash_consistency(self):
        """Equal Spins produce the same hash."""
        assert hash(yuzuha.Spin(1)) == hash(yuzuha.Spin(1))
        assert hash(yuzuha.Spin(2)) == hash(yuzuha.Spin(2))

    def test_spin_hashable_in_set(self):
        """Spin objects can be stored in sets and used as dict keys."""
        spins = {yuzuha.Spin(0), yuzuha.Spin(1), yuzuha.Spin(2), yuzuha.Spin(1)}
        assert len(spins) == 3
        assert yuzuha.Spin(1) in spins

        lookup = {yuzuha.Spin(1): "half", yuzuha.Spin(2): "one"}
        assert lookup[yuzuha.Spin(1)] == "half"
        assert lookup[yuzuha.Spin(2)] == "one"

    def test_spin_ordering(self):
        """Spins support < ordering by their doubled value."""
        assert yuzuha.Spin(0) < yuzuha.Spin(1)
        assert yuzuha.Spin(1) < yuzuha.Spin(2)
        assert not (yuzuha.Spin(2) < yuzuha.Spin(1))
        assert not (yuzuha.Spin(1) < yuzuha.Spin(1))

        spins = [yuzuha.Spin(3), yuzuha.Spin(1), yuzuha.Spin(0), yuzuha.Spin(2)]
        assert sorted(spins) == [yuzuha.Spin(0), yuzuha.Spin(1), yuzuha.Spin(2), yuzuha.Spin(3)]

    def test_spin_frozen(self):
        """Spin instances are immutable: attribute assignment raises AttributeError."""
        j = yuzuha.Spin(1)
        with pytest.raises(AttributeError):
            j.new_attr = 42


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

    def test_edge_equality(self):
        """Edges with the same spin and direction compare equal."""
        j = yuzuha.Spin(1)
        assert yuzuha.Edge.incoming(j) == yuzuha.Edge.incoming(j)
        assert yuzuha.Edge.outgoing(j) == yuzuha.Edge.outgoing(j)
        assert yuzuha.Edge.incoming(j) != yuzuha.Edge.outgoing(j)
        assert yuzuha.Edge.incoming(yuzuha.Spin(1)) != yuzuha.Edge.incoming(yuzuha.Spin(2))

    def test_edge_hash_consistency(self):
        """Equal Edges produce the same hash."""
        j = yuzuha.Spin(2)
        assert hash(yuzuha.Edge.incoming(j)) == hash(yuzuha.Edge.incoming(j))
        assert hash(yuzuha.Edge.outgoing(j)) == hash(yuzuha.Edge.outgoing(j))

    def test_edge_hashable_in_set(self):
        """Edge objects can be stored in sets and used as dict keys."""
        j = yuzuha.Spin(1)
        edges = {
            yuzuha.Edge.incoming(j),
            yuzuha.Edge.outgoing(j),
            yuzuha.Edge.incoming(j),  # duplicate
        }
        assert len(edges) == 2
        assert yuzuha.Edge.incoming(j) in edges

        lookup = {
            yuzuha.Edge.incoming(j): "in",
            yuzuha.Edge.outgoing(j): "out",
        }
        assert lookup[yuzuha.Edge.incoming(j)] == "in"
        assert lookup[yuzuha.Edge.outgoing(j)] == "out"

    def test_edge_frozen(self):
        """Edge instances are immutable: attribute assignment raises AttributeError."""
        edge = yuzuha.Edge.incoming(yuzuha.Spin(1))
        with pytest.raises(AttributeError):
            edge.new_attr = 42


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

    def test_cgspec_equality(self):
        """CGSpecs built from the same edge list compare equal."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        edges = [
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ]
        spec_a = yuzuha.CGSpec.from_edges(edges)
        spec_b = yuzuha.CGSpec.from_edges(edges)
        assert spec_a == spec_b

        edges_different = [
            yuzuha.Edge.incoming(j1),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j_half),
        ]
        spec_c = yuzuha.CGSpec.from_edges(edges_different)
        assert spec_a != spec_c

    def test_cgspec_hash_consistency(self):
        """Equal CGSpecs produce the same hash."""
        j_half = yuzuha.Spin(1)
        j1 = yuzuha.Spin(2)
        edges = [
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ]
        spec_a = yuzuha.CGSpec.from_edges(edges)
        spec_b = yuzuha.CGSpec.from_edges(edges)
        assert hash(spec_a) == hash(spec_b)

    def test_cgspec_hashable_in_set(self):
        """CGSpec objects can be stored in sets and used as dict keys."""
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
        specs = {spec_a, spec_b, spec_a}  # duplicate spec_a
        assert len(specs) == 2

        cache = {spec_a: "result_a", spec_b: "result_b"}
        assert cache[yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.outgoing(j1),
        ])] == "result_a"

    def test_cgspec_frozen(self):
        """CGSpec instances are immutable: attribute assignment raises AttributeError."""
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(yuzuha.Spin(1)),
            yuzuha.Edge.incoming(yuzuha.Spin(1)),
            yuzuha.Edge.outgoing(yuzuha.Spin(2)),
        ])
        with pytest.raises(AttributeError):
            spec.new_attr = 42


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

    def test_contraction_equality(self):
        """Contractions with the same axes compare equal."""
        assert yuzuha.Contraction([2], [0]) == yuzuha.Contraction([2], [0])
        assert yuzuha.Contraction([2, 3], [0, 1]) == yuzuha.Contraction([2, 3], [0, 1])
        assert yuzuha.Contraction([2], [0]) != yuzuha.Contraction([1], [0])
        assert yuzuha.Contraction([2], [0]) != yuzuha.Contraction([2], [1])
        assert yuzuha.Contraction([2, 3], [0, 1]) != yuzuha.Contraction([2], [0])

    def test_contraction_hash_consistency(self):
        """Equal Contractions produce the same hash."""
        assert hash(yuzuha.Contraction([2], [0])) == hash(yuzuha.Contraction([2], [0]))
        assert hash(yuzuha.Contraction([2, 3], [0, 1])) == hash(yuzuha.Contraction([2, 3], [0, 1]))

    def test_contraction_hashable_in_set(self):
        """Contraction objects can be stored in sets and used as dict keys."""
        c1 = yuzuha.Contraction([2], [0])
        c2 = yuzuha.Contraction([1], [2])
        contractions = {c1, c2, yuzuha.Contraction([2], [0])}  # duplicate c1
        assert len(contractions) == 2

        lookup = {c1: "single", c2: "other"}
        assert lookup[yuzuha.Contraction([2], [0])] == "single"

    def test_contraction_frozen(self):
        """Contraction instances are immutable: attribute assignment raises AttributeError."""
        c = yuzuha.Contraction([2], [0])
        with pytest.raises(AttributeError):
            c.new_attr = 42
