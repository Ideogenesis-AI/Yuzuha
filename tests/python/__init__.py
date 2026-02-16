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
Python test suite for Yuzuha.

This package contains comprehensive tests for the Yuzuha Python bindings,
including tests for:

- Basic type construction (Spin, Edge, CGSpec, Contraction)
- X-symbol computation and properties
- R-symbol computation and properties
- Caching behavior
- Mathematical properties (unitarity, normalization, etc.)

Run tests with pytest from the repository root:
    pytest tests/python/
"""
