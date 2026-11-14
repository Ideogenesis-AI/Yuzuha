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
Translation of Rust errors into Python exceptions.

Every `YuzuhaError` variant the Rust core can produce is mapped to a Python
exception by `impl From<YuzuhaError> for PyErr` in `src/bindings.rs`. These
tests pin that mapping: invalid *input* becomes `ValueError`, while failures of
the surrounding machinery become `RuntimeError`.

Two variants are deliberately absent. `InvalidMagneticNumber` can only come from
`MagneticNumber::new`, and `TriangleViolation` is never constructed at all;
neither type is part of the Python surface, so no call can reach those arms.
"""

import sqlite3

import pytest

import yuzuha
from yuzuha import CGSpec, Contraction, Direction, Edge, Spin, canonical_basis


def spec(*doubled, last_outgoing=True):
    """Build a CGSpec from doubled spin values, last edge outgoing."""
    edges = [Edge.incoming(Spin(d)) for d in doubled[:-1]]
    tail = Spin(doubled[-1])
    edges.append(Edge.outgoing(tail) if last_outgoing else Edge.incoming(tail))
    return CGSpec.from_edges(edges)


class TestInvalidInput:
    """Bad arguments surface as ValueError."""

    def test_negative_spin(self):
        with pytest.raises(ValueError, match="Invalid spin value"):
            Spin(-1)

    def test_direction_from_bad_sign(self):
        # InvalidCGTSpec carries the explanatory message through unchanged.
        for sign in (0, 2, -5):
            with pytest.raises(ValueError, match="Invalid direction sign"):
                Direction.from_sign(sign)

    def test_direction_from_valid_sign(self):
        assert Direction.from_sign(1) == Direction.incoming()
        assert Direction.from_sign(-1) == Direction.outgoing()

    def test_spins_that_cannot_couple_to_zero(self):
        # An odd number of half-integer spins has no valid fusion tree.
        with pytest.raises(ValueError, match="Invalid CGSpec"):
            spec(1, 2, 2)

    def test_canonical_basis_needs_two_edges(self):
        single = CGSpec.from_edges([Edge.incoming(Spin(0))])
        with pytest.raises(ValueError, match="Cannot build canonical basis"):
            canonical_basis(single)


class TestInvalidIndices:
    """Out-of-range indices reach the catch-all arm as RuntimeError."""

    def test_inverted_axis_out_of_bounds(self):
        s = spec(1, 1, 2)
        with pytest.raises(RuntimeError, match="out of bounds"):
            s.with_inverted_axes([3])

    def test_permutation_is_validated(self):
        s = spec(1, 1, 1, 1)
        # Wrong length, repeated index, and out-of-range index are all rejected.
        for bad in ([0, 1, 2], [0, 0, 1, 2], [0, 1, 2, 9]):
            with pytest.raises(ValueError):
                yuzuha.compute_rsymbol(s, bad)


class TestCacheErrors:
    """Cache failures become RuntimeError instead of ValueError."""

    def test_unwritable_cache_directory(self, monkeypatch, tmp_path):
        # A regular file cannot hold the cache directory, so the Rust side
        # fails in create_dir_all and raises CacheError.
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(blocker / "cache"))

        with pytest.raises(RuntimeError, match="Cache error"):
            canonical_basis(spec(1, 1, 2))

    def test_cache_recovers_after_failure(self, monkeypatch, tmp_path):
        # The failed connection above must not be memoized: pointing the cache
        # somewhere writable has to work again in the same process.
        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(tmp_path / "good"))
        basis = canonical_basis(spec(1, 1, 2))
        assert basis.shape[-1] >= 1


class TestStringRepresentations:
    """__repr__ and __str__ on the binding types."""

    def test_spin(self):
        assert str(Spin(1)) == "1/2"
        assert str(Spin(2)) == "1"
        assert repr(Spin(3)) == "Spin(3/2)"

    def test_direction(self):
        assert str(Direction.incoming()) == "Direction.incoming"
        assert str(Direction.outgoing()) == "Direction.outgoing"
        assert repr(Direction.incoming()) == "Direction.incoming"
        assert repr(Direction.outgoing()) == "Direction.outgoing"

    def test_edge(self):
        text = repr(Edge.incoming(Spin(2)))
        assert "Edge" in text and "1" in text

    def test_cgspec(self):
        text = repr(spec(1, 1, 2))
        assert "num_external=3" in text

    def test_contraction(self):
        assert "Contraction" in repr(Contraction([2], [0]))


class TestPythonCacheErrors:
    """The Python-side caches swallow sqlite3 failures instead of propagating."""

    def test_query_survives_a_broken_database(self, monkeypatch):
        # A cache miss and a cache failure are indistinguishable to callers:
        # both mean "recompute". Verify the failure path returns None rather
        # than raising.
        cache = yuzuha.cache.get_xsymbol_cache()

        def boom(*args, **kwargs):
            raise sqlite3.DatabaseError("database is locked")

        monkeypatch.setattr(cache, "_get_connection", boom)

        a, b = spec(1, 1, 2), spec(2, 1, 1)
        assert cache.query(a, b, Contraction([2], [0])) is None

    def test_store_survives_a_broken_database(self, monkeypatch):
        import numpy as np

        cache = yuzuha.cache.get_rsymbol_cache()

        def boom(*args, **kwargs):
            raise sqlite3.DatabaseError("disk I/O error")

        monkeypatch.setattr(cache, "_get_connection", boom)

        s = spec(1, 1, 1, 1)
        # Must not raise: caching is best-effort.
        cache.store(s, [0, 1, 2, 3], np.eye(s.om_dimension()), s)

    def test_xsymbol_store_survives_a_broken_database(self, monkeypatch):
        import numpy as np

        cache = yuzuha.cache.get_xsymbol_cache()

        def boom(*args, **kwargs):
            raise sqlite3.DatabaseError("attempt to write a readonly database")

        monkeypatch.setattr(cache, "_get_connection", boom)

        a, b = spec(1, 1, 2), spec(2, 1, 1)
        x_data, spec_c = yuzuha.compute_xsymbol(a, b, Contraction([2], [0]))
        cache.store(a, b, Contraction([2], [0]), np.asarray(x_data), spec_c)

    def test_rsymbol_query_survives_a_broken_database(self, monkeypatch):
        cache = yuzuha.cache.get_rsymbol_cache()

        def boom(*args, **kwargs):
            raise sqlite3.DatabaseError("database disk image is malformed")

        monkeypatch.setattr(cache, "_get_connection", boom)

        assert cache.query(spec(1, 1, 1, 1), [1, 0, 2, 3]) is None
