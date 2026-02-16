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
Database utilities for CG basis caching.

This module provides functions to pre-build and cache commonly used
CG bases to improve runtime performance.
"""

import logging
from typing import Optional
from itertools import product

from .yuzuha import Spin, Edge, CGSpec, canonical_basis


def startup_database(
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO
) -> None:
    """
    Pre-compute and cache CG bases for commonly used spin configurations.
    
    This function computes and caches:
    1. 3rd order bases: one spin is 1/2 or 1, other two spins up to spin-3
    2. 4th order bases: two spins are 1/2, other two spins up to spin-3
    
    The database is automatically populated when bases are computed for the
    first time. Subsequent calls will use the cached values.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance to use. If None, creates a default stream logger.
    log_level : int, optional
        Logging level to use for the default logger. Default is logging.INFO.
    
    Examples
    --------
    >>> import yuzuha
    >>> yuzuha.startup_database()
    
    >>> # With custom logger
    >>> import logging
    >>> logger = logging.getLogger('my_app')
    >>> yuzuha.startup_database(logger=logger)
    """
    # Setup logger
    if logger is None:
        logger = logging.getLogger('yuzuha.database')
        logger.setLevel(log_level)
        
        # Check if handlers already exist to avoid duplicates
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    logger.info("Starting CG basis database pre-computation")
    
    # Define spin ranges
    # Spin(n) represents j = n/2, so:
    # Spin(1) = j=1/2, Spin(2) = j=1, ..., Spin(6) = j=3
    spin_half_or_one = [Spin(1), Spin(2)]  # j=1/2 and j=1
    spin_half = [Spin(1)]  # j=1/2
    spins_up_to_three = [Spin(i) for i in range(1, 7)]  # j=1/2, 1, 3/2, 2, 5/2, 3
    
    total_count = 0
    success_count = 0
    
    # ========================================================================
    # 3rd order bases: one spin is 1/2 or 1, other two spins up to spin-3
    # ========================================================================
    logger.info("Computing 3rd order bases...")
    logger.info("  Configuration: one spin ∈ {1/2, 1}, two spins ∈ {1/2, ..., 3}")
    
    third_order_configs = set()
    
    # Generate all configurations with one special spin and two general spins
    # We need to consider all permutations since position matters in CGSpec
    for special_spin in spin_half_or_one:
        for spin2 in spins_up_to_three:
            for spin3 in spins_up_to_three:
                # Three positions for the special spin
                third_order_configs.add((special_spin, spin2, spin3))
                third_order_configs.add((spin2, special_spin, spin3))
                third_order_configs.add((spin2, spin3, special_spin))
    
    # Convert to list
    third_order_configs = list(third_order_configs)
    
    logger.info(f"  Total 3rd order configurations: {len(third_order_configs)}")
    
    for idx, (j1, j2, j3) in enumerate(third_order_configs, 1):
        total_count += 1
        try:
            spec = CGSpec.from_edges([
                Edge.incoming(j1),
                Edge.incoming(j2),
                Edge.outgoing(j3),
            ])
            
            basis = canonical_basis(spec)
            success_count += 1
            
            if idx % 10 == 0 or idx == len(third_order_configs):
                logger.info(
                    f"  Progress: {idx}/{len(third_order_configs)} "
                    f"(j1={j1.twice()/2}, j2={j2.twice()/2}, "
                    f"j3={j3.twice()/2}) - shape: {basis.shape}"
                )
        except Exception as e:
            error_msg = str(e)
            if "Invalid CGSpec" in error_msg or "triangle inequality" in error_msg:
                total_count -= 1
                logger.debug(
                    f"  Skipped invalid configuration (j1={j1.twice()/2}, "
                    f"j2={j2.twice()/2}, j3={j3.twice()/2}): {e}"
                )
            else:
                logger.error(
                    f"  Failed for configuration (j1={j1.twice()/2}, "
                    f"j2={j2.twice()/2}, j3={j3.twice()/2}): {e}"
                )
    
    # ========================================================================
    # 4th order bases: two spins are 1/2, other two spins up to spin-3
    # ========================================================================
    logger.info("\nComputing 4th order bases...")
    logger.info("  Configuration: two spins = 1/2, two spins ∈ {1/2, ..., 3}")
    
    fourth_order_configs = set()
    
    # Generate all configurations with two j=1/2 spins and two general spins
    j_half = Spin(1)
    for spin3 in spins_up_to_three:
        for spin4 in spins_up_to_three:
            # All unique positions for two j=1/2 spins in a 4-spin system
            # Positions: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            fourth_order_configs.add((j_half, j_half, spin3, spin4))  # positions 0,1
            fourth_order_configs.add((j_half, spin3, j_half, spin4))  # positions 0,2
            fourth_order_configs.add((j_half, spin3, spin4, j_half))  # positions 0,3
            fourth_order_configs.add((spin3, j_half, j_half, spin4))  # positions 1,2
            fourth_order_configs.add((spin3, j_half, spin4, j_half))  # positions 1,3
            fourth_order_configs.add((spin3, spin4, j_half, j_half))  # positions 2,3
    
    # Convert to list
    fourth_order_configs = list(fourth_order_configs)
    
    logger.info(f"  Total 4th order configurations: {len(fourth_order_configs)}")
    
    for idx, (j1, j2, j3, j4) in enumerate(fourth_order_configs, 1):
        total_count += 1
        try:
            spec = CGSpec.from_edges([
                Edge.incoming(j1),
                Edge.incoming(j2),
                Edge.incoming(j3),
                Edge.outgoing(j4),
            ])
            
            basis = canonical_basis(spec)
            success_count += 1
            
            if idx % 10 == 0 or idx == len(fourth_order_configs):
                logger.info(
                    f"  Progress: {idx}/{len(fourth_order_configs)} "
                    f"(j1={j1.twice()/2}, j2={j2.twice()/2}, "
                    f"j3={j3.twice()/2}, j4={j4.twice()/2}) - "
                    f"shape: {basis.shape}"
                )
        except Exception as e:
            error_msg = str(e)
            if "Invalid CGSpec" in error_msg or "triangle inequality" in error_msg:
                total_count -= 1
                logger.debug(
                    f"  Skipped invalid configuration (j1={j1.twice()/2}, "
                    f"j2={j2.twice()/2}, j3={j3.twice()/2}, "
                    f"j4={j4.twice()/2}): {e}"
                )
            else:
                logger.error(
                    f"  Failed for configuration (j1={j1.twice()/2}, "
                    f"j2={j2.twice()/2}, j3={j3.twice()/2}, "
                    f"j4={j4.twice()/2}): {e}"
                )
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CG basis database pre-computation complete!")
    logger.info(f"  Total configurations attempted: {total_count}")
    logger.info(f"  Successfully computed: {success_count}")
    logger.info(f"  Failed: {total_count - success_count}")
    logger.info("=" * 70)
