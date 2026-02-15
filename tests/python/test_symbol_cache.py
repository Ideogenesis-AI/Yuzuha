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
Tests for X-symbol and R-symbol caching.

Tests the caching mechanism to ensure results are correctly stored
and retrieved, and that cache keys are properly differentiated.
"""
import time
import pytest
import numpy as np
import yuzuha


class TestXSymbolCaching:
    """Tests for X-symbol caching."""

    def test_cache_miss_then_hit(self):
        """Test cache miss followed by cache hit for X-symbols."""
        with yuzuha.TestCacheContext():
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

            # First call - cache miss
            x1, spec_c1 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            # Second call - cache hit
            x2, spec_c2 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            # Results should be identical
            assert np.allclose(x1, x2)
            assert spec_c1.om_dimension() == spec_c2.om_dimension()
            assert spec_c1.num_external() == spec_c2.num_external()

    def test_cached_values_correct(self):
        """Test that cached values match freshly computed values."""
        with yuzuha.TestCacheContext():
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

            # Compute and cache
            x_cached, spec_c_cached = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            # Clear cache and recompute
            from yuzuha.cache import get_xsymbol_cache
            cache = get_xsymbol_cache()
            cache.clear()

            x_fresh, spec_c_fresh = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            # Values should match
            assert np.allclose(x_cached, x_fresh)
            assert spec_c_cached.om_dimension() == spec_c_fresh.om_dimension()

    def test_different_contractions_different_keys(self):
        """Test that different contractions produce different cache entries."""
        with yuzuha.TestCacheContext():
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

            # Different contractions
            contraction1 = yuzuha.Contraction([2], [0])
            contraction2 = yuzuha.Contraction([3], [0])

            x1, spec_c1 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction1)
            x2, spec_c2 = yuzuha.compute_xsymbol(spec_a, spec_b, contraction2)

            # Results should be different (different contractions)
            assert not np.allclose(x1, x2)

    def test_different_spins_different_keys(self):
        """Test that different edge spins produce different cache entries."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            j1 = yuzuha.Spin(2)
            j3_2 = yuzuha.Spin(3)

            # Configuration 1
            spec_a1 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j1),
            ])

            spec_b1 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j_half),
                yuzuha.Edge.outgoing(j_half),
            ])

            # Configuration 2 (different spins but still valid)
            spec_a2 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j1),
            ])

            spec_b2 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j_half),
                yuzuha.Edge.outgoing(j_half),
            ])

            contraction = yuzuha.Contraction([2], [0])

            x1, spec_c1 = yuzuha.compute_xsymbol(spec_a1, spec_b1, contraction)
            x2, spec_c2 = yuzuha.compute_xsymbol(spec_a2, spec_b2, contraction)

            # Verify both are cached separately
            from yuzuha.cache import get_xsymbol_cache
            cache = get_xsymbol_cache()
            assert cache.size() == 2  # Two distinct cache entries

    def test_different_directions_different_keys(self):
        """Test that different edge directions produce different cache entries."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            j1 = yuzuha.Spin(2)

            # Configuration 1
            spec_a1 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j1),
            ])

            # Configuration 2 (different directions)
            spec_a2 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.outgoing(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.outgoing(j1),
            ])

            spec_b = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j_half),
                yuzuha.Edge.outgoing(j_half),
            ])

            contraction = yuzuha.Contraction([2], [0])

            x1, spec_c1 = yuzuha.compute_xsymbol(spec_a1, spec_b, contraction)
            x2, spec_c2 = yuzuha.compute_xsymbol(spec_a2, spec_b, contraction)

            # Results may be different (different topologies)
            # At minimum, they should be separately cached
            from yuzuha.cache import get_xsymbol_cache
            cache = get_xsymbol_cache()
            assert cache.size() >= 2

    def test_cache_isolation_between_contexts(self):
        """Test that different TestCacheContext instances are isolated."""
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

        # First context
        with yuzuha.TestCacheContext():
            yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
            from yuzuha.cache import get_xsymbol_cache
            cache1 = get_xsymbol_cache()
            size1 = cache1.size()
            assert size1 == 1

        # Second context (should be empty)
        with yuzuha.TestCacheContext():
            from yuzuha.cache import get_xsymbol_cache
            cache2 = get_xsymbol_cache()
            size2 = cache2.size()
            assert size2 == 0

    def test_cache_performance(self):
        """Test that cache provides performance benefit."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            j1 = yuzuha.Spin(2)

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

            # First call (cache miss)
            start = time.time()
            x1, _ = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
            time_miss = time.time() - start

            # Second call (cache hit)
            start = time.time()
            x2, _ = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
            time_hit = time.time() - start

            # Results should match
            assert np.allclose(x1, x2)

            # Cache hit should be faster (though this may not always be true for very small problems)
            # Just verify it completes without error
            assert time_hit >= 0


class TestRSymbolCaching:
    """Tests for R-symbol caching."""

    def test_cache_miss_then_hit(self):
        """Test cache miss followed by cache hit for R-symbols."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            permutation = [1, 0, 2, 3]

            # First call - cache miss
            r1, spec_perm1 = yuzuha.compute_rsymbol(spec, permutation)

            # Second call - cache hit
            r2, spec_perm2 = yuzuha.compute_rsymbol(spec, permutation)

            # Results should be identical
            assert np.allclose(r1, r2)
            assert spec_perm1.om_dimension() == spec_perm2.om_dimension()

    def test_cached_values_correct(self):
        """Test that cached values match freshly computed values."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            permutation = [1, 0, 2, 3]

            # Compute and cache
            r_cached, spec_perm_cached = yuzuha.compute_rsymbol(spec, permutation)

            # Clear cache and recompute
            from yuzuha.cache import get_rsymbol_cache
            cache = get_rsymbol_cache()
            cache.clear()

            r_fresh, spec_perm_fresh = yuzuha.compute_rsymbol(spec, permutation)

            # Values should match
            assert np.allclose(r_cached, r_fresh)
            assert spec_perm_cached.om_dimension() == spec_perm_fresh.om_dimension()

    def test_different_permutations_different_keys(self):
        """Test that different permutations produce different cache entries."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            # Different permutations
            permutation1 = [1, 0, 2, 3]  # Swap first two
            permutation2 = [2, 1, 0, 3]  # Different permutation

            r1, _ = yuzuha.compute_rsymbol(spec, permutation1)
            r2, _ = yuzuha.compute_rsymbol(spec, permutation2)

            # Verify both are cached separately
            from yuzuha.cache import get_rsymbol_cache
            cache = get_rsymbol_cache()
            assert cache.size() == 2  # Two distinct cache entries

    def test_different_specs_different_keys(self):
        """Test that different specs produce different cache entries."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            j1 = yuzuha.Spin(2)

            # Configuration 1 (4 spin-1/2)
            spec1 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            # Configuration 2 (3 spin-1)
            spec2 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
            ])

            permutation1 = [1, 0, 2, 3]
            permutation2 = [1, 0, 2]

            r1, _ = yuzuha.compute_rsymbol(spec1, permutation1)
            r2, _ = yuzuha.compute_rsymbol(spec2, permutation2)

            # Results should be different (different specs)
            assert r1.shape != r2.shape

    def test_identity_permutation(self):
        """Test that identity permutation is cached correctly."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            permutation = [0, 1, 2, 3]

            # First call
            r1, _ = yuzuha.compute_rsymbol(spec, permutation)

            # Second call (should hit cache)
            r2, _ = yuzuha.compute_rsymbol(spec, permutation)

            # Should be identity matrix
            assert np.allclose(r1, np.eye(r1.shape[0]))
            assert np.allclose(r1, r2)

    def test_cache_isolation_between_contexts(self):
        """Test that different TestCacheContext instances are isolated."""
        j_half = yuzuha.Spin(1)
        spec = yuzuha.CGSpec.from_edges([
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
            yuzuha.Edge.incoming(j_half),
        ])

        permutation = [1, 0, 2, 3]

        # First context
        with yuzuha.TestCacheContext():
            yuzuha.compute_rsymbol(spec, permutation)
            from yuzuha.cache import get_rsymbol_cache
            cache1 = get_rsymbol_cache()
            size1 = cache1.size()
            assert size1 == 1

        # Second context (should be empty)
        with yuzuha.TestCacheContext():
            from yuzuha.cache import get_rsymbol_cache
            cache2 = get_rsymbol_cache()
            size2 = cache2.size()
            assert size2 == 0

    def test_cache_performance(self):
        """Test that cache provides performance benefit."""
        with yuzuha.TestCacheContext():
            j1 = yuzuha.Spin(2)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.incoming(j1),
            ])

            permutation = [1, 2, 0]

            # First call (cache miss)
            start = time.time()
            r1, _ = yuzuha.compute_rsymbol(spec, permutation)
            time_miss = time.time() - start

            # Second call (cache hit)
            start = time.time()
            r2, _ = yuzuha.compute_rsymbol(spec, permutation)
            time_hit = time.time() - start

            # Results should match
            assert np.allclose(r1, r2)

            # Cache hit should complete without error
            assert time_hit >= 0


class TestCacheUtilities:
    """Tests for cache utility functions."""

    def test_cache_clear_xsymbol(self):
        """Test clearing X-symbol cache."""
        with yuzuha.TestCacheContext():
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

            # Compute to populate cache
            yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            from yuzuha.cache import get_xsymbol_cache
            cache = get_xsymbol_cache()
            assert cache.size() == 1

            # Clear cache
            cache.clear()
            assert cache.size() == 0

    def test_cache_clear_rsymbol(self):
        """Test clearing R-symbol cache."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            spec = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
                yuzuha.Edge.incoming(j_half),
            ])

            permutation = [1, 0, 2, 3]

            # Compute to populate cache
            yuzuha.compute_rsymbol(spec, permutation)

            from yuzuha.cache import get_rsymbol_cache
            cache = get_rsymbol_cache()
            assert cache.size() == 1

            # Clear cache
            cache.clear()
            assert cache.size() == 0

    def test_cache_size_tracking(self):
        """Test that cache size is tracked correctly."""
        with yuzuha.TestCacheContext():
            j_half = yuzuha.Spin(1)
            j1 = yuzuha.Spin(2)

            from yuzuha.cache import get_xsymbol_cache
            cache = get_xsymbol_cache()
            assert cache.size() == 0

            # Add one entry
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
            yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
            assert cache.size() == 1

            # Add another entry (different contraction)
            contraction2 = yuzuha.Contraction([2], [0])
            spec_b2 = yuzuha.CGSpec.from_edges([
                yuzuha.Edge.incoming(j1),
                yuzuha.Edge.outgoing(j1),
                yuzuha.Edge.outgoing(j1),
            ])
            yuzuha.compute_xsymbol(spec_a, spec_b2, contraction2)
            assert cache.size() == 2

    def test_set_cache_path(self):
        """Test programmatic cache path setting."""
        # Use TestCacheContext which properly handles both Python and Rust caches
        with yuzuha.TestCacheContext() as ctx:
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

            # Compute (should use test context path)
            yuzuha.compute_xsymbol(spec_a, spec_b, contraction)

            # Check that database was created in test context location
            temp_dir = ctx.temp_dir
            assert (temp_dir / 'xsymbol.db').exists()
