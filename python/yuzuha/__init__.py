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
Yuzuha - SU(2) Recoupling Theory Library

This package provides efficient computation of SU(2) X-symbols and R-symbols
for quantum recoupling theory, with persistent caching.
"""

from importlib.metadata import version

from .yuzuha import (
    Direction,
    Spin,
    Edge,
    CGSpec,
    Contraction,
    canonical_basis,
    fs_phase_for_spin,
    compute_conjugate,
)
from .symbols import compute_xsymbol, compute_rsymbol
from .database import startup_database
from .cache import (
    set_cache_path,
    reset_caches,
    TestCacheContext,
    clear_all_caches,
    get_cache_stats,
    print_cache_stats,
)

__version__ = version("yuzuha")

__all__ = [
    "Direction",
    "Spin",
    "Edge",
    "CGSpec",
    "Contraction",
    "compute_xsymbol",
    "compute_rsymbol",
    "fs_phase_for_spin",
    "compute_conjugate",
    "canonical_basis",
    "startup_database",
    "set_cache_path",
    "reset_caches",
    "TestCacheContext",
    "clear_all_caches",
    "get_cache_stats",
    "print_cache_stats",
]
