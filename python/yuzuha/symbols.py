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
Cached wrapper functions for X-symbol and R-symbol computations.

This module provides caching wrappers around the Rust implementations
of compute_xsymbol and compute_rsymbol. The wrappers transparently
cache results to SQLite databases to avoid redundant computations.
"""

from typing import Tuple

import numpy as np

from .yuzuha import compute_xsymbol as _rust_compute_xsymbol
from .yuzuha import compute_rsymbol as _rust_compute_rsymbol
from .cache import get_xsymbol_cache, get_rsymbol_cache


def compute_xsymbol(spec_a, spec_b, contraction) -> Tuple[np.ndarray, object]:
    """
    Compute X-symbol for contracting two CGTs with caching.
    
    Given two CGTs A and B with a contraction specification, computes the
    X-symbol tensor X^γ_{αβ} representing the coupling to output CGT C.
    Results are cached to avoid redundant computations.
    
    The X-symbol is computed by:
    1. Building canonical basis tensors for A and B
    2. Contracting them over the specified edges (keeping OM indices separate)
    3. Projecting the result onto the canonical basis of C
    
    Parameters
    ----------
    spec_a : CGSpec
        First CG specification
    spec_b : CGSpec
        Second CG specification
    contraction : Contraction
        Specification of which edges to contract
    
    Returns
    -------
    tuple[numpy.ndarray, CGSpec]
        A tuple containing:
        - X-symbol array of shape [om_a, om_b, om_c]
        - Output CGSpec with edges from uncontracted A and B edges
    
    Raises
    ------
    ValueError
        If the contraction is invalid (incompatible edge spins).
    RuntimeError
        If computation fails.
    
    Examples
    --------
    >>> import yuzuha
    >>> j_half = yuzuha.Spin(1)
    >>> j1 = yuzuha.Spin(2)
    >>> spec_a = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.outgoing(j1)
    ... ])
    >>> spec_b = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(j1),
    ...     yuzuha.Edge.outgoing(j_half),
    ...     yuzuha.Edge.outgoing(j_half)
    ... ])
    >>> contraction = yuzuha.Contraction([2], [0])
    >>> x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
    >>> print(x_array.shape)
    (1, 1, 2)
    
    Notes
    -----
    The first call for a given set of parameters will compute the result
    and cache it. Subsequent calls with the same parameters will return
    the cached result, which is typically much faster.
    """
    cache = get_xsymbol_cache()
    
    # Try cache lookup
    cached = cache.query(spec_a, spec_b, contraction)
    if cached is not None:
        return cached
    
    # Cache miss - compute via Rust
    x_array, spec_c = _rust_compute_xsymbol(spec_a, spec_b, contraction)
    
    # Store in cache
    cache.store(spec_a, spec_b, contraction, x_array, spec_c)
    
    return x_array, spec_c


def compute_rsymbol(spec, permutation) -> Tuple[np.ndarray, object]:
    """
    Compute R-symbol for tensor leg permutation with caching.
    
    The R-symbol represents how outer multiplicity indices transform under
    permutation of tensor legs. Results are cached to avoid redundant
    computations.
    
    Parameters
    ----------
    spec : CGSpec
        The CGSpec to permute.
    permutation : list[int]
        Permutation of external edge indices. Must be a valid permutation
        (each index from 0 to num_external-1 appears exactly once).
    
    Returns
    -------
    tuple[numpy.ndarray, CGSpec]
        A tuple containing:
        - R-symbol array of shape [om_original, om_permuted].
        - Permuted CGSpec with edges reordered according to permutation.
    
    Raises
    ------
    ValueError
        If the permutation is invalid.
    RuntimeError
        If computation fails.
    
    Examples
    --------
    >>> import yuzuha
    >>> j_half = yuzuha.Spin(1)
    >>> spec = yuzuha.CGSpec.from_edges([
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.incoming(j_half)
    ... ])
    >>> # Swap first two legs
    >>> permutation = [1, 0, 2, 3]
    >>> r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)
    >>> print(r_array.shape)
    (2, 2)
    
    Notes
    -----
    The first call for a given set of parameters will compute the result
    and cache it. Subsequent calls with the same parameters will return
    the cached result, which is typically much faster.
    """
    cache = get_rsymbol_cache()
    
    # Try cache lookup
    cached = cache.query(spec, permutation)
    if cached is not None:
        return cached
    
    # Cache miss - compute via Rust
    r_array, spec_permuted = _rust_compute_rsymbol(spec, permutation)
    
    # Store in cache
    cache.store(spec, permutation, r_array, spec_permuted)
    
    return r_array, spec_permuted
