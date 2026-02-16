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
Pytest configuration for Yuzuha test suite.

This module configures pytest to use a temporary cache directory for each
test session, ensuring that tests are isolated from each other and from
any pre-existing cache database.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_cache():
    """
    Set up an isolated cache directory for the test session.
    
    This fixture automatically runs for the entire test session and ensures
    that all tests use a temporary cache directory instead of the default
    .yuzuha/ location. This provides test isolation and prevents
    interference from previously cached data.
    
    This covers all cache types:
    - cgbasis.db (canonical basis cache - Rust)
    - xsymbol.db (X-symbol cache - Python)
    - rsymbol.db (R-symbol cache - Python)
    
    All caches now use YUZUHA_CACHE_PATH as a directory and append
    their respective filenames.
    
    The temporary directory is automatically cleaned up after the test
    session completes.
    """
    # Create a unique temporary directory for this test session
    temp_dir = tempfile.mkdtemp(prefix="yuzuha_test_cache_")
    temp_dir_path = Path(temp_dir)
    
    # Set the environment variable to the cache directory
    # All caches (Rust and Python) will append their filenames
    old_cache_path = os.environ.get("YUZUHA_CACHE_PATH")
    os.environ["YUZUHA_CACHE_PATH"] = str(temp_dir_path)
    
    print(f"\n✓ Using isolated test cache directory: {temp_dir_path}")
    
    # Import yuzuha here to ensure cache instances pick up the new path
    import yuzuha
    yuzuha.reset_caches()
    
    yield temp_dir_path
    
    # Cleanup: reset cache connections
    yuzuha.reset_caches()
    
    # Restore the old environment variable
    if old_cache_path is not None:
        os.environ["YUZUHA_CACHE_PATH"] = old_cache_path
    else:
        os.environ.pop("YUZUHA_CACHE_PATH", None)
    
    # Remove the temporary directory and all its contents
    try:
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up test cache: {temp_dir}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not clean up test cache {temp_dir}: {e}")


