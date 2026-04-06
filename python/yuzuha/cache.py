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
SQLite cache for X-symbol and R-symbol data.

This module provides persistent caching of expensive X-symbol and R-symbol
computations using SQLite databases. The cache stores numpy arrays in .npy
format, keyed by the input parameters (CGSpecs, contractions, permutations).
"""

import io
import os
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Thread-local storage for test cache path override
_test_cache_path = threading.local()


def get_cache_dir() -> Path:
    """
    Get the cache directory path.
    
    Priority:
    1. Test override (thread-local)
    2. YUZUHA_CACHE_PATH environment variable
    3. Default: .yuzuha/ in current directory
    
    Returns
    -------
    Path
        Cache directory path
    """
    # Check thread-local test override first
    if hasattr(_test_cache_path, 'path') and _test_cache_path.path is not None:
        return Path(_test_cache_path.path)
    
    # Check environment variable
    if 'YUZUHA_CACHE_PATH' in os.environ:
        return Path(os.environ['YUZUHA_CACHE_PATH'])
    
    # Default
    return Path('.yuzuha')


def set_cache_path(path: Optional[str]) -> None:
    """
    Set the cache directory path programmatically.
    
    Parameters
    ----------
    path : str or None
        Cache directory path. If None, uses default.
    """
    if path is None:
        if 'YUZUHA_CACHE_PATH' in os.environ:
            del os.environ['YUZUHA_CACHE_PATH']
    else:
        os.environ['YUZUHA_CACHE_PATH'] = str(path)


def serialize_cgspec(spec) -> Tuple[bytes, bytes]:
    """
    Serialize a CGSpec to bytes.
    
    Parameters
    ----------
    spec : CGSpec
        The CGSpec to serialize
    
    Returns
    -------
    tuple[bytes, bytes]
        (spins_bytes, dirs_bytes) where:
        - spins_bytes: serialized doubled spin values as i32
        - dirs_bytes: serialized directions as i8 (+1/-1)
    """
    # Use efficient bulk methods that avoid Python loop overhead
    spins = spec.get_spins()
    dirs = spec.get_directions()
    
    spins_bytes = struct.pack(f'{len(spins)}i', *spins)
    dirs_bytes = struct.pack(f'{len(dirs)}b', *dirs)
    
    return spins_bytes, dirs_bytes


def serialize_contraction(contraction) -> Tuple[bytes, bytes]:
    """
    Serialize a Contraction to bytes.
    
    Parameters
    ----------
    contraction : Contraction
        The Contraction to serialize
    
    Returns
    -------
    tuple[bytes, bytes]
        (axes_a_bytes, axes_b_bytes)
    """
    # Get the axes lists from the Contraction object
    axes_a = list(contraction.axes_a)
    axes_b = list(contraction.axes_b)
    
    axes_a_bytes = struct.pack(f'{len(axes_a)}I', *axes_a)
    axes_b_bytes = struct.pack(f'{len(axes_b)}I', *axes_b)
    
    return axes_a_bytes, axes_b_bytes


def serialize_permutation(permutation: list) -> bytes:
    """
    Serialize a permutation list to bytes.
    
    Parameters
    ----------
    permutation : list[int]
        Permutation indices
    
    Returns
    -------
    bytes
        Serialized permutation
    """
    return struct.pack(f'{len(permutation)}I', *permutation)


def serialize_array(array: np.ndarray) -> bytes:
    """
    Serialize a NumPy array to .npy format bytes.
    
    Parameters
    ----------
    array : np.ndarray
        Array to serialize
    
    Returns
    -------
    bytes
        Serialized array in .npy format
    """
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def deserialize_array(data: bytes) -> np.ndarray:
    """
    Deserialize .npy format bytes to a NumPy array.
    
    Parameters
    ----------
    data : bytes
        Serialized array in .npy format
    
    Returns
    -------
    np.ndarray
        Deserialized array
    """
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


def reconstruct_cgspec(spins_bytes: bytes, dirs_bytes: bytes):
    """
    Reconstruct a CGSpec from serialized data.
    
    Parameters
    ----------
    spins_bytes : bytes
        Serialized doubled spin values
    dirs_bytes : bytes
        Serialized directions
    
    Returns
    -------
    CGSpec
        Reconstructed CGSpec
    """
    from .yuzuha import CGSpec, Edge, Spin
    
    # Deserialize spins and directions
    n_spins = len(spins_bytes) // 4  # 4 bytes per i32
    spins = struct.unpack(f'{n_spins}i', spins_bytes)
    
    n_dirs = len(dirs_bytes)
    dirs = struct.unpack(f'{n_dirs}b', dirs_bytes)
    
    # Reconstruct edges
    edges = []
    for spin_val, dir_val in zip(spins, dirs):
        spin = Spin(spin_val)
        if dir_val == 1:
            edge = Edge.incoming(spin)
        else:
            edge = Edge.outgoing(spin)
        edges.append(edge)
    
    return CGSpec.from_edges(edges)


class XSymbolCache:
    """
    Cache manager for X-symbol computations.
    
    This class manages a SQLite database that caches X-symbol results
    to avoid redundant computations.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the X-symbol cache.
        
        Parameters
        ----------
        db_path : Path, optional
            Path to the database file. If None, uses default location.
        """
        if db_path is None:
            cache_dir = get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / 'xsymbol.db'
        
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection (thread-safe)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn
    
    def _init_db(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS xsymbol_cache (
                    spec_a_spins BLOB NOT NULL,
                    spec_a_dirs BLOB NOT NULL,
                    spec_b_spins BLOB NOT NULL,
                    spec_b_dirs BLOB NOT NULL,
                    contraction_axes_a BLOB NOT NULL,
                    contraction_axes_b BLOB NOT NULL,
                    x_data BLOB NOT NULL,
                    spec_c_spins BLOB NOT NULL,
                    spec_c_dirs BLOB NOT NULL,
                    PRIMARY KEY (spec_a_spins, spec_a_dirs, spec_b_spins, spec_b_dirs,
                                 contraction_axes_a, contraction_axes_b)
                )
            """)
            conn.commit()
    
    def query(self, spec_a, spec_b, contraction) -> Optional[Tuple[np.ndarray, object]]:
        """
        Query the cache for an X-symbol result.
        
        Parameters
        ----------
        spec_a : CGSpec
            First CG specification
        spec_b : CGSpec
            Second CG specification
        contraction : Contraction
            Contraction specification
        
        Returns
        -------
        tuple[np.ndarray, CGSpec] or None
            Cached (x_array, spec_c) if found, None otherwise
        """
        try:
            # Serialize keys
            spec_a_spins, spec_a_dirs = serialize_cgspec(spec_a)
            spec_b_spins, spec_b_dirs = serialize_cgspec(spec_b)
            axes_a, axes_b = serialize_contraction(contraction)
            
            with self._lock:
                conn = self._get_connection()
                cursor = conn.execute("""
                    SELECT x_data, spec_c_spins, spec_c_dirs
                    FROM xsymbol_cache
                    WHERE spec_a_spins = ? AND spec_a_dirs = ?
                      AND spec_b_spins = ? AND spec_b_dirs = ?
                      AND contraction_axes_a = ? AND contraction_axes_b = ?
                """, (spec_a_spins, spec_a_dirs, spec_b_spins, spec_b_dirs, axes_a, axes_b))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                x_data_bytes, spec_c_spins, spec_c_dirs = row
                x_array = deserialize_array(x_data_bytes)
                spec_c = reconstruct_cgspec(spec_c_spins, spec_c_dirs)
                
                return x_array, spec_c
        
        except Exception:
            # If any error occurs during cache lookup, return None (cache miss)
            return None
    
    def store(self, spec_a, spec_b, contraction, x_data: np.ndarray, spec_c) -> None:
        """
        Store an X-symbol result in the cache.
        
        Parameters
        ----------
        spec_a : CGSpec
            First CG specification
        spec_b : CGSpec
            Second CG specification
        contraction : Contraction
            Contraction specification
        x_data : np.ndarray
            X-symbol array to cache
        spec_c : CGSpec
            Output CG specification
        """
        try:
            # Serialize all data
            spec_a_spins, spec_a_dirs = serialize_cgspec(spec_a)
            spec_b_spins, spec_b_dirs = serialize_cgspec(spec_b)
            axes_a, axes_b = serialize_contraction(contraction)
            x_data_bytes = serialize_array(x_data)
            spec_c_spins, spec_c_dirs = serialize_cgspec(spec_c)
            
            with self._lock:
                conn = self._get_connection()
                conn.execute("""
                    INSERT OR REPLACE INTO xsymbol_cache
                    (spec_a_spins, spec_a_dirs, spec_b_spins, spec_b_dirs,
                     contraction_axes_a, contraction_axes_b, x_data, spec_c_spins, spec_c_dirs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (spec_a_spins, spec_a_dirs, spec_b_spins, spec_b_dirs,
                      axes_a, axes_b, x_data_bytes, spec_c_spins, spec_c_dirs))
                conn.commit()
        
        except Exception:
            # Silently fail if caching fails (don't break the computation)
            pass
    
    def clear(self) -> None:
        """Clear all cached X-symbol data."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM xsymbol_cache")
            conn.commit()
    
    def _size_unlocked(self) -> int:
        """Return the entry count without acquiring the lock (caller must hold it)."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM xsymbol_cache")
        return cursor.fetchone()[0]

    def size(self) -> int:
        """
        Get the number of cached entries.
        
        Returns
        -------
        int
            Number of cached X-symbol entries
        """
        with self._lock:
            return self._size_unlocked()
    
    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary containing cache statistics:
            - 'size': number of cached entries
            - 'db_path': path to database file
            - 'db_size_bytes': size of database file in bytes
        """
        with self._lock:
            stats = {
                'size': self._size_unlocked(),
                'db_path': str(self.db_path),
            }
            
            # Get database file size if it exists
            if self.db_path.exists():
                stats['db_size_bytes'] = self.db_path.stat().st_size
            else:
                stats['db_size_bytes'] = 0
            
            return stats
    
    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class RSymbolCache:
    """
    Cache manager for R-symbol computations.
    
    This class manages a SQLite database that caches R-symbol results
    to avoid redundant computations.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the R-symbol cache.
        
        Parameters
        ----------
        db_path : Path, optional
            Path to the database file. If None, uses default location.
        """
        if db_path is None:
            cache_dir = get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / 'rsymbol.db'
        
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection (thread-safe)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn
    
    def _init_db(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rsymbol_cache (
                    spec_spins BLOB NOT NULL,
                    spec_dirs BLOB NOT NULL,
                    permutation BLOB NOT NULL,
                    r_data BLOB NOT NULL,
                    spec_permuted_spins BLOB NOT NULL,
                    spec_permuted_dirs BLOB NOT NULL,
                    PRIMARY KEY (spec_spins, spec_dirs, permutation)
                )
            """)
            conn.commit()
    
    def query(self, spec, permutation: list) -> Optional[Tuple[np.ndarray, object]]:
        """
        Query the cache for an R-symbol result.
        
        Parameters
        ----------
        spec : CGSpec
            CG specification to permute
        permutation : list[int]
            Permutation indices
        
        Returns
        -------
        tuple[np.ndarray, CGSpec] or None
            Cached (r_array, spec_permuted) if found, None otherwise
        """
        try:
            # Serialize keys
            spec_spins, spec_dirs = serialize_cgspec(spec)
            perm_bytes = serialize_permutation(permutation)
            
            with self._lock:
                conn = self._get_connection()
                cursor = conn.execute("""
                    SELECT r_data, spec_permuted_spins, spec_permuted_dirs
                    FROM rsymbol_cache
                    WHERE spec_spins = ? AND spec_dirs = ? AND permutation = ?
                """, (spec_spins, spec_dirs, perm_bytes))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                r_data_bytes, spec_permuted_spins, spec_permuted_dirs = row
                r_array = deserialize_array(r_data_bytes)
                spec_permuted = reconstruct_cgspec(spec_permuted_spins, spec_permuted_dirs)
                
                return r_array, spec_permuted
        
        except Exception:
            # If any error occurs during cache lookup, return None (cache miss)
            return None
    
    def store(self, spec, permutation: list, r_data: np.ndarray, spec_permuted) -> None:
        """
        Store an R-symbol result in the cache.
        
        Parameters
        ----------
        spec : CGSpec
            Original CG specification
        permutation : list[int]
            Permutation indices
        r_data : np.ndarray
            R-symbol array to cache
        spec_permuted : CGSpec
            Permuted CG specification
        """
        try:
            # Serialize all data
            spec_spins, spec_dirs = serialize_cgspec(spec)
            perm_bytes = serialize_permutation(permutation)
            r_data_bytes = serialize_array(r_data)
            spec_permuted_spins, spec_permuted_dirs = serialize_cgspec(spec_permuted)
            
            with self._lock:
                conn = self._get_connection()
                conn.execute("""
                    INSERT OR REPLACE INTO rsymbol_cache
                    (spec_spins, spec_dirs, permutation, r_data,
                     spec_permuted_spins, spec_permuted_dirs)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (spec_spins, spec_dirs, perm_bytes, r_data_bytes,
                      spec_permuted_spins, spec_permuted_dirs))
                conn.commit()
        
        except Exception:
            # Silently fail if caching fails (don't break the computation)
            pass
    
    def clear(self) -> None:
        """Clear all cached R-symbol data."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM rsymbol_cache")
            conn.commit()
    
    def _size_unlocked(self) -> int:
        """Return the entry count without acquiring the lock (caller must hold it)."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM rsymbol_cache")
        return cursor.fetchone()[0]

    def size(self) -> int:
        """
        Get the number of cached entries.
        
        Returns
        -------
        int
            Number of cached R-symbol entries
        """
        with self._lock:
            return self._size_unlocked()
    
    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary containing cache statistics:
            - 'size': number of cached entries
            - 'db_path': path to database file
            - 'db_size_bytes': size of database file in bytes
        """
        with self._lock:
            stats = {
                'size': self._size_unlocked(),
                'db_path': str(self.db_path),
            }
            
            # Get database file size if it exists
            if self.db_path.exists():
                stats['db_size_bytes'] = self.db_path.stat().st_size
            else:
                stats['db_size_bytes'] = 0
            
            return stats
    
    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# Global cache instances (lazy initialization)
_xsymbol_cache: Optional[XSymbolCache] = None
_rsymbol_cache: Optional[RSymbolCache] = None
_cache_lock = threading.Lock()


def get_xsymbol_cache() -> XSymbolCache:
    """
    Get the global X-symbol cache instance.
    
    Returns
    -------
    XSymbolCache
        Global X-symbol cache
    """
    global _xsymbol_cache
    # Check if cache path has changed (happens with test isolation)
    current_dir = get_cache_dir()
    if _xsymbol_cache is not None:
        expected_path = current_dir / 'xsymbol.db'
        if _xsymbol_cache.db_path != expected_path:
            # Path has changed, reset cache
            _xsymbol_cache.close()
            _xsymbol_cache = None
    
    if _xsymbol_cache is None:
        with _cache_lock:
            if _xsymbol_cache is None:
                _xsymbol_cache = XSymbolCache()
    return _xsymbol_cache


def get_rsymbol_cache() -> RSymbolCache:
    """
    Get the global R-symbol cache instance.
    
    Returns
    -------
    RSymbolCache
        Global R-symbol cache
    """
    global _rsymbol_cache
    # Check if cache path has changed (happens with test isolation)
    current_dir = get_cache_dir()
    if _rsymbol_cache is not None:
        expected_path = current_dir / 'rsymbol.db'
        if _rsymbol_cache.db_path != expected_path:
            # Path has changed, reset cache
            _rsymbol_cache.close()
            _rsymbol_cache = None
    
    if _rsymbol_cache is None:
        with _cache_lock:
            if _rsymbol_cache is None:
                _rsymbol_cache = RSymbolCache()
    return _rsymbol_cache


def reset_caches() -> None:
    """
    Reset global cache instances.
    
    This is useful for testing to ensure clean state between tests.
    """
    global _xsymbol_cache, _rsymbol_cache
    with _cache_lock:
        if _xsymbol_cache is not None:
            _xsymbol_cache.close()
            _xsymbol_cache = None
        if _rsymbol_cache is not None:
            _rsymbol_cache.close()
            _rsymbol_cache = None


class TestCacheContext:
    """
    Context manager for isolated cache testing.
    
    Creates temporary cache directories for tests to ensure isolation.
    
    Examples
    --------
    >>> with TestCacheContext():
    ...     # All cache operations use temporary directory
    ...     x_array, spec_c = compute_xsymbol(spec_a, spec_b, contraction)
    """
    
    def __init__(self):
        self.temp_dir = None
        self.old_path = None
    
    def __enter__(self):
        import tempfile
        import time
        
        # Create unique temporary directory
        self.temp_dir = Path(tempfile.gettempdir()) / f"yuzuha_test_cache_{os.getpid()}_{time.time_ns()}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Store old path and set new one using thread-local storage
        self.old_path = getattr(_test_cache_path, 'path', None)
        _test_cache_path.path = self.temp_dir
        
        # Reset caches to pick up new path
        reset_caches()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil
        
        # Restore old path
        _test_cache_path.path = self.old_path
        
        # Reset caches
        reset_caches()
        
        # Clean up temporary directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Convenience functions for cache management

def clear_all_caches() -> None:
    """
    Clear all symbol caches (X-symbol and R-symbol).
    
    This removes all cached entries from both databases but keeps
    the database files and schema intact.
    
    Examples
    --------
    >>> import yuzuha
    >>> from yuzuha.cache import clear_all_caches
    >>> clear_all_caches()
    """
    get_xsymbol_cache().clear()
    get_rsymbol_cache().clear()


def get_cache_stats() -> dict:
    """
    Get statistics for all symbol caches.
    
    Returns
    -------
    dict
        Dictionary with keys 'xsymbol' and 'rsymbol', each containing
        cache statistics (size, db_path, db_size_bytes).
    
    Examples
    --------
    >>> import yuzuha
    >>> from yuzuha.cache import get_cache_stats
    >>> stats = get_cache_stats()
    >>> print(f"X-symbol cache has {stats['xsymbol']['size']} entries")
    >>> print(f"R-symbol cache has {stats['rsymbol']['size']} entries")
    """
    return {
        'xsymbol': get_xsymbol_cache().stats(),
        'rsymbol': get_rsymbol_cache().stats(),
    }


def print_cache_stats() -> None:
    """
    Print a formatted summary of cache statistics.
    
    This is a convenience function for quickly inspecting cache status.
    
    Examples
    --------
    >>> import yuzuha
    >>> from yuzuha.cache import print_cache_stats
    >>> print_cache_stats()
    X-symbol cache:
      Entries: 42
      Database: .yuzuha/xsymbol.db
      Size: 1.2 MB
    R-symbol cache:
      Entries: 18
      Database: .yuzuha/rsymbol.db
      Size: 0.5 MB
    """
    stats = get_cache_stats()
    
    def format_size(bytes_):
        """Format bytes as human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_ < 1024.0:
                return f"{bytes_:.1f} {unit}"
            bytes_ /= 1024.0
        return f"{bytes_:.1f} TB"
    
    print("X-symbol cache:")
    print(f"  Entries: {stats['xsymbol']['size']}")
    print(f"  Database: {stats['xsymbol']['db_path']}")
    print(f"  Size: {format_size(stats['xsymbol']['db_size_bytes'])}")
    
    print("\nR-symbol cache:")
    print(f"  Entries: {stats['rsymbol']['size']}")
    print(f"  Database: {stats['rsymbol']['db_path']}")
    print(f"  Size: {format_size(stats['rsymbol']['db_size_bytes'])}")
