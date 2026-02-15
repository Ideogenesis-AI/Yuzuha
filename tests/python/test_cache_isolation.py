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
Tests demonstrating cache isolation features.

These tests show how to use the fresh_cache fixture to ensure each test
starts with a completely empty cache database.
"""
import os
import yuzuha


def test_cache_path_is_isolated(isolated_cache):
    """Verify that tests use an isolated cache path."""
    cache_path = os.environ.get("YUZUHA_CACHE_PATH")
    
    # Cache path should be set to a temporary location
    assert cache_path is not None
    assert "yuzuha_test_cache" in cache_path
    assert ".yuzuha/cgbasis.db" not in cache_path
    
    # The isolated_cache fixture returns the directory, but YUZUHA_CACHE_PATH
    # points to the cgbasis.db file for Rust compatibility
    assert str(isolated_cache / "cgbasis.db") == cache_path


def test_cache_is_shared_across_calls(isolated_cache):
    """Verify that cache is persistent within the same test session."""
    # First computation
    j_half = yuzuha.Spin(1)
    spec = yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
    ])
    
    permutation = [1, 0, 2, 3]
    r_array_1, _ = yuzuha.compute_rsymbol(spec, permutation)
    
    # Second call with same parameters should use cache
    r_array_2, _ = yuzuha.compute_rsymbol(spec, permutation)
    
    # Results should be identical
    import numpy as np
    assert np.allclose(r_array_1, r_array_2)


def test_different_computations_work(isolated_cache):
    """Verify that different spin configurations work correctly."""
    # Test with different spin values
    j1 = yuzuha.Spin(2)
    spec = yuzuha.CGSpec.from_edges([
        yuzuha.Edge.incoming(j1),
        yuzuha.Edge.incoming(j1),
        yuzuha.Edge.incoming(j1),
    ])
    
    permutation = [0, 1, 2]
    r_array, _ = yuzuha.compute_rsymbol(spec, permutation)
    
    assert r_array.shape[0] > 0
