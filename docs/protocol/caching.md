# Data Caching

Caching is **optional** in the Yuzuha Protocol, but if implemented it must satisfy the
requirements on this page. A non-caching implementation may return freshly computed
values on every call; a caching implementation must behave identically to a
non-caching one from the caller's perspective.

---

## 1. Correctness (Determinism)

**Requirement**: A cached result must be bitwise-equal to the value that would be
returned by a fresh computation with no cache present.

Rationale: Callers must be able to rely on the cache being a pure performance
optimization with no effect on results. This rules out approximate or compressed
caches.

---

## 2. Key Spaces

Each cached function must use the following minimum key spaces. Additional
implementation-specific key components (e.g. a format version number) are permitted
but must not affect the correctness guarantee.

### Canonical Basis Cache

The cache key for `canonical_basis(spec)` must uniquely determine the result. A
sufficient key is:

```
key = (
    tuple((edge.rep, edge.dir) for edge in spec.edges),
    tuple(spec.alphas)           # internal representation sequence
)
```

For SU(2), this reduces to the tuple of `(twice_spin, direction_sign)` pairs for
external edges plus the tuple of doubled internal spins.

### X-Symbol Cache

The cache key for `compute_x_symbol(spec_a, spec_b, contraction)` must uniquely
determine both the returned array and the returned `spec_c`. A sufficient key is:

```
key = (
    tuple((e.rep, e.dir) for e in spec_a.edges),
    tuple(spec_a.alphas),
    tuple((e.rep, e.dir) for e in spec_b.edges),
    tuple(spec_b.alphas),
    tuple(contraction.axes_a),
    tuple(contraction.axes_b),
)
```

### R-Symbol Cache

The cache key for `compute_r_symbol(spec, permutation)` must uniquely determine both
the returned array and the returned `spec_permuted`. A sufficient key is:

```
key = (
    tuple((e.rep, e.dir) for e in spec.edges),
    tuple(spec.alphas),
    tuple(permutation),
)
```

---

## 3. Thread Safety

**Requirement**: All cache read and write operations must be **thread-safe**. Concurrent
calls from different threads with the same or different keys must not produce data
races, corruption, or incorrect results.

Acceptable strategies include:

- Per-cache mutex protecting all database operations
- Connection pool with per-connection serialization
- Immutable (write-once) on-disk stores

---

## 4. Persistence

**Recommendation**: Caches should be **persistent** across Python sessions (i.e. stored
on disk rather than in memory only). This is a recommendation, not a hard requirement.

If a persistent cache is implemented:

- The cache file format must be versioned or include a format identifier so that
  incompatible format changes can be detected and the old cache discarded safely.
- The implementation must handle a corrupted or missing cache file gracefully,
  falling back to fresh computation.

---

## 5. Cache Invalidation

The Yuzuha Protocol does **not** specify when a cache must be invalidated. In practice,
the cache is valid as long as the implementation of `canonical_basis`,
`compute_x_symbol`, and `compute_r_symbol` does not change in a way that would alter
their outputs for the same inputs.

If an implementation update changes outputs, it must either:
- Increment a format version tag in the cache key space, or
- Provide a documented procedure for clearing the old cache (e.g. deleting the database
  file or calling a `clear_caches` function)

---

## 6. Isolation for Testing

**Requirement**: If a caching layer is implemented, the implementation must provide a
mechanism for test code to use an isolated, temporary cache so that tests do not
interfere with production data or with each other.

In the SU(2) reference implementation, this is provided by `TestCacheContext`:

```python
with yuzuha.TestCacheContext():
    # All cache operations use a unique temporary directory
    x, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
# Temporary cache cleaned up on exit
```

---

## 7. Cache Management API

If a persistent cache is implemented, the following utility functions are **recommended**
(present in the SU(2) reference implementation):

| Function | Description |
|----------|-------------|
| `set_cache_path(path)` | Override the cache directory location |
| `reset_caches()` | Close in-memory cache connections and reset state |
| `clear_all_caches()` | Delete all cached entries (keep schema) |
| `get_cache_stats() → dict` | Return entry counts and file sizes per cache |
| `print_cache_stats()` | Print a human-readable cache summary |

---

## 8. SU(2) Reference Implementation Summary

| Cache | Storage | File | Key | Value |
|-------|---------|------|-----|-------|
| Canonical basis | SQLite (Rust) | `cgbasis.db` | (external edges, internal spins) | `ndarray` in `.npy` format |
| X-symbol | SQLite (Python) | `xsymbol.db` | (spec_a, spec_b, contraction) | `(ndarray, spec_c)` |
| R-symbol | SQLite (Python) | `rsymbol.db` | (spec, permutation) | `(ndarray, spec_permuted)` |

See [Database](../api/caching/database.md) and [Cache Management](../api/caching/cache.md)
for the full Python API.
