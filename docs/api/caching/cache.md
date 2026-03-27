# Cache Management

Utilities for controlling the symbol cache path, inspecting statistics, and
managing cache state.

## Overview

Yuzuha maintains three persistent SQLite databases:

| Database | File | Content |
|----------|------|---------|
| CG basis cache | `.yuzuha/cgbasis.db` | `(edges, alphas) → canonical basis array` |
| X-symbol cache | `.yuzuha/xsymbol.db` | `(spec_a, spec_b, contraction) → (x_array, spec_c)` |
| R-symbol cache | `.yuzuha/rsymbol.db` | `(spec, permutation) → (r_array, spec_permuted)` |

All three databases are created automatically on first use and are thread-safe.

## Cache Path

The cache directory is resolved in the following priority order:

1. **Thread-local override** (set by `TestCacheContext`) — highest priority
2. **Environment variable**: `YUZUHA_CACHE_PATH`
3. **Default**: `.yuzuha/` in the current working directory

### Setting the Cache Path

```python
import yuzuha

# Set via Python
yuzuha.set_cache_path('/path/to/cache')

# Or via environment variable (before importing yuzuha)
# export YUZUHA_CACHE_PATH=/path/to/cache
```

## API Reference

### `set_cache_path`

::: yuzuha.set_cache_path
    options:
      show_source: false
      heading_level: 4

### `reset_caches`

::: yuzuha.reset_caches
    options:
      show_source: false
      heading_level: 4

### `clear_all_caches`

::: yuzuha.clear_all_caches
    options:
      show_source: false
      heading_level: 4

### `get_cache_stats`

::: yuzuha.get_cache_stats
    options:
      show_source: false
      heading_level: 4

### `print_cache_stats`

::: yuzuha.print_cache_stats
    options:
      show_source: false
      heading_level: 4

### `TestCacheContext`

::: yuzuha.TestCacheContext
    options:
      show_source: false
      heading_level: 4

## Usage Examples

### Inspecting Cache State

```python
import yuzuha

# Print a human-readable summary
yuzuha.print_cache_stats()
# X-symbol cache:
#   Entries: 42
#   Database: .yuzuha/xsymbol.db
#   Size: 1.2 MB
# R-symbol cache:
#   Entries: 18
#   Database: .yuzuha/rsymbol.db
#   Size: 0.5 MB

# Or get the raw dict
stats = yuzuha.get_cache_stats()
print(stats['xsymbol']['size'])        # number of cached X-symbols
print(stats['rsymbol']['db_path'])     # path to R-symbol database
```

### Clearing Caches

```python
# Clear all symbol caches (keeps database files, removes entries)
yuzuha.clear_all_caches()
```

### Isolated Cache for Testing

```python
import yuzuha
from yuzuha import TestCacheContext

with TestCacheContext():
    # All cache operations use an isolated temporary directory
    x, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
# Temporary cache is cleaned up automatically on exit
```

## See Also

- [Database](database.md): Pre-populate the canonical basis cache
- [X-Symbol](../symbols/xsymbol.md): Uses the X-symbol cache transparently
- [R-Symbol](../symbols/rsymbol.md): Uses the R-symbol cache transparently
- [Yuzuha Protocol — Caching Requirements](../../protocol/caching.md): Formal
  specification of the caching interface

## Notes

`reset_caches()` closes and destroys the in-memory cache objects so that the next
call to `compute_xsymbol` or `compute_rsymbol` opens a fresh connection. It does
**not** delete the database files. Use `clear_all_caches()` to remove cached entries
while preserving the file and schema.
