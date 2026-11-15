# Cache Management

Caching in Yuzuha is transparent — nothing on this page is needed to get correct results.
These utilities exist for the cases where the cache's location or contents matter to you:
pointing it at a writable directory in a deployment, inspecting how much has accumulated,
and isolating it so that a test run neither reads nor pollutes real data.

## Description

Yuzuha maintains three persistent SQLite databases, each keyed on the inputs of the
function whose results it holds:

| Database | File | Content |
|----------|------|---------|
| CG basis cache | `.yuzuha/cgbasis.db` | `external spins → canonical basis array` |
| X-symbol cache | `.yuzuha/xsymbol.db` | `(spec_a, spec_b, contraction) → (x_array, spec_c)` |
| R-symbol cache | `.yuzuha/rsymbol.db` | `(spec, permutation) → (r_array, spec_permuted)` |

All three are created on first use and are thread-safe. The CG basis cache is managed by
the Rust layer; the two symbol caches are managed by the Python layer, so the
cache-clearing utilities below reach the latter two only.

### Cache Path

All three databases live in one directory, resolved in this priority order:

1. **Thread-local override**, set by `TestCacheContext` — highest priority
2. **Environment variable** `YUZUHA_CACHE_PATH`
3. **Default**: `.yuzuha/` in the current working directory

`set_cache_path` works by setting the environment variable, so it takes effect for the
whole process but is still overridden inside a `TestCacheContext` block:

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

`print_cache_stats` is the quick human-readable view; `get_cache_stats` returns the same
information as a dict of `size`, `db_path`, and `db_size_bytes` per cache, for logging or
assertions:

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
# Clear the symbol caches (keeps database files, removes entries)
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
- [Yuzuha Protocol — Data Caching](../../protocol/caching.md): Formal
  specification of the caching interface

## Notes

`reset_caches()` and `clear_all_caches()` do different things. The first closes and
discards the in-memory cache objects so that the next symbol computation opens a fresh
connection, leaving the stored data untouched. The second deletes the entries from the
X- and R-symbol databases while preserving the files and their schema. Neither clears
`cgbasis.db`; to discard cached canonical bases, delete that file.
