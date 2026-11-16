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
Cache location and lifecycle management.

`get_cache_dir` consults three sources in priority order: the thread-local
override installed by `TestCacheContext`, the `YUZUHA_CACHE_PATH` environment
variable, and finally `.yuzuha/` in the working directory. The global cache
singletons must notice when that answer changes and reconnect.

The session-wide `isolated_cache` fixture in `conftest.py` sets
`YUZUHA_CACHE_PATH`, so tests that need a different priority level clear it
explicitly with `monkeypatch`.
"""

from pathlib import Path

import numpy as np
import pytest

import yuzuha
from yuzuha import Contraction, Edge, Spin, CGSpec
from yuzuha.cache import (
    clear_all_caches,
    get_cache_dir,
    get_cache_stats,
    get_rsymbol_cache,
    get_xsymbol_cache,
    print_cache_stats,
    reset_caches,
    set_cache_path,
)

# Aliased on import: pytest would otherwise try to collect the context manager
# as a test class, because it matches the configured Test* pattern.
from yuzuha.cache import TestCacheContext as cache_context


@pytest.fixture
def no_env_cache_path(monkeypatch):
    """Remove the session fixture's YUZUHA_CACHE_PATH for one test."""
    monkeypatch.delenv("YUZUHA_CACHE_PATH", raising=False)
    yield
    reset_caches()


def spec(*doubled):
    edges = [Edge.incoming(Spin(d)) for d in doubled[:-1]]
    edges.append(Edge.outgoing(Spin(doubled[-1])))
    return CGSpec.from_edges(edges)


class TestGetCacheDir:
    def test_environment_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(tmp_path))
        assert get_cache_dir() == tmp_path

    def test_default_is_dot_yuzuha(self, no_env_cache_path):
        # With no override of any kind, the cache is relative to the cwd.
        assert get_cache_dir() == Path(".yuzuha")

    def test_thread_local_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(tmp_path))
        with cache_context() as ctx:
            assert get_cache_dir() == ctx.temp_dir
            assert get_cache_dir() != tmp_path
        # Leaving the context restores the environment variable's answer.
        assert get_cache_dir() == tmp_path


class TestSetCachePath:
    def test_sets_and_clears_the_environment_variable(self, no_env_cache_path, tmp_path):
        set_cache_path(str(tmp_path))
        assert get_cache_dir() == tmp_path

        set_cache_path(None)
        assert get_cache_dir() == Path(".yuzuha")

    def test_clearing_an_unset_path_is_a_no_op(self, no_env_cache_path):
        set_cache_path(None)
        set_cache_path(None)
        assert get_cache_dir() == Path(".yuzuha")

    def test_thread_local_context_takes_precedence(self, monkeypatch, tmp_path):
        # Documented behaviour: inside a TestCacheContext, set_cache_path has
        # no visible effect because the thread-local path outranks it.
        with cache_context() as ctx:
            set_cache_path(str(tmp_path))
            assert get_cache_dir() == ctx.temp_dir
        assert get_cache_dir() == tmp_path


class TestSingletonsFollowThePath:
    def test_caches_reconnect_when_the_directory_changes(self, monkeypatch, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"

        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(first))
        reset_caches()
        x_first = get_xsymbol_cache()
        r_first = get_rsymbol_cache()
        assert x_first.db_path == first / "xsymbol.db"
        assert r_first.db_path == first / "rsymbol.db"

        monkeypatch.setenv("YUZUHA_CACHE_PATH", str(second))
        x_second = get_xsymbol_cache()
        r_second = get_rsymbol_cache()
        assert x_second is not x_first
        assert r_second is not r_first
        assert x_second.db_path == second / "xsymbol.db"
        assert r_second.db_path == second / "rsymbol.db"

        reset_caches()

    def test_repeated_calls_reuse_one_instance(self):
        assert get_xsymbol_cache() is get_xsymbol_cache()
        assert get_rsymbol_cache() is get_rsymbol_cache()

    def test_reset_is_idempotent(self):
        reset_caches()
        reset_caches()
        # The next access must rebuild a usable cache.
        assert get_xsymbol_cache().size() >= 0


class TestClearAllCaches:
    def test_empties_both_databases(self):
        with cache_context():
            spec_a, spec_b = spec(1, 1, 2), spec(2, 1, 1)
            yuzuha.compute_xsymbol(spec_a, spec_b, Contraction([2], [0]))

            four = spec(1, 1, 1, 1)
            yuzuha.compute_rsymbol(four, [1, 0, 2, 3])

            assert get_xsymbol_cache().size() > 0
            assert get_rsymbol_cache().size() > 0

            clear_all_caches()

            assert get_xsymbol_cache().size() == 0
            assert get_rsymbol_cache().size() == 0


class TestStatsReporting:
    def test_stats_cover_both_caches(self):
        with cache_context():
            stats = get_cache_stats()
            assert set(stats) == {"xsymbol", "rsymbol"}
            for entry in stats.values():
                assert entry["size"] >= 0
                assert entry["db_size_bytes"] >= 0
                assert "db_path" in entry

    def test_db_size_is_zero_when_the_file_is_gone(self):
        # The open connection keeps working on the unlinked inode, so stats
        # must report a zero size in this scenario.
        with cache_context():
            for cache in (get_xsymbol_cache(), get_rsymbol_cache()):
                assert cache.size() == 0
                cache.db_path.unlink()
                assert cache.stats()["db_size_bytes"] == 0
        reset_caches()

    def test_print_cache_stats_scales_units(self, monkeypatch, capsys):
        huge = 3 * 1024**4
        monkeypatch.setattr(
            yuzuha.cache,
            "get_cache_stats",
            lambda: {
                "xsymbol": {"size": 1, "db_path": "x.db", "db_size_bytes": huge},
                "rsymbol": {"size": 2, "db_path": "r.db", "db_size_bytes": 2048},
            },
        )
        print_cache_stats()
        out = capsys.readouterr().out
        assert "3.0 TB" in out
        assert "2.0 KB" in out

    def test_print_cache_stats_formats_both_sections(self, capsys):
        with cache_context():
            print_cache_stats()
        out = capsys.readouterr().out
        assert "X-symbol cache:" in out
        assert "R-symbol cache:" in out
        assert "Entries:" in out
        assert "Database:" in out
        assert "Size:" in out


class TestCacheRoundTrip:
    def test_store_then_query_returns_the_same_arrays(self):
        with cache_context():
            cache = get_xsymbol_cache()
            spec_a, spec_b = spec(1, 1, 2), spec(2, 1, 1)
            contraction = Contraction([2], [0])

            assert cache.query(spec_a, spec_b, contraction) is None

            x_data, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
            hit = cache.query(spec_a, spec_b, contraction)
            assert hit is not None
            cached_data, cached_spec = hit
            np.testing.assert_allclose(cached_data, x_data)
            assert cached_spec == spec_c

    def test_closing_twice_is_safe(self):
        with cache_context():
            cache = get_xsymbol_cache()
            cache.close()
            cache.close()
            # A closed cache reopens on next use.
            assert cache.size() >= 0
