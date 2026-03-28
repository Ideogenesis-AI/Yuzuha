# API Reference

Welcome to the Yuzuha API Reference. This section documents all types, functions, and
modules available in the `yuzuha` Python package, organized by functionality.

## Core Types

The fundamental building blocks for specifying CG tensors:

| Type | Description |
|------|-------------|
| [Spin](core/spin.md) | SU(2) irreducible representation label (half-integer \(j\)) |
| [Direction](core/direction.md) | Arrow direction: `Incoming` or `Outgoing` |
| [Edge](core/edge.md) | Directed representation — one external edge of a CG tensor |
| [CGSpec](core/cgspec.md) | Full CG tensor specification: edges + internal fusion tree |
| [Contraction](core/contraction.md) | Edge-pair specification for X-symbol computation |

## Symbols

The primary outputs of Yuzuha — recoupling coefficients as NumPy arrays:

| Function | Returns | Description |
|----------|---------|-------------|
| [compute_xsymbol](symbols/xsymbol.md) | `(ndarray, CGSpec)` | Recoupling coefficients for contracting two CG tensors; shape `(om_a, om_b, om_c)` |
| [compute_rsymbol](symbols/rsymbol.md) | `(ndarray, CGSpec)` | Permutation transformation on the OM space; shape `(om_in, om_out)` |
| [canonical_basis](symbols/canonical-basis.md) | `ndarray` | Canonical CG basis tensor; shape `(*external_dims, om_dim)` |

### Auxiliary Symbol Functions

| Function | Description |
|----------|-------------|
| `fs_phase_for_spin(spin)` | Frobenius-Schur indicator \((-1)^{2j}\) |
| `compute_conjugate(spec)` | Canonical basis with all arrows reversed |

## Primitives

Internal mathematical utilities (Rust layer, not directly Python-callable):

| Topic | Description |
|-------|-------------|
| [Clebsch-Gordan](primitives/cg.md) | Real CG coefficients in the Condon-Shortley convention |
| [Metric Tensor](primitives/metric.md) | Arrow-reversal transformation \(g^{(j)}_{m,m'} = (-1)^{j-m}\delta_{m,-m'}\) |
| [Triangle Rules](primitives/triangle.md) | Triangle inequality and admissible spin range utilities |

## Caching

Utilities for managing persistent symbol caches:

| Topic | Description |
|-------|-------------|
| [Database](caching/database.md) | `startup_database()` — pre-populate canonical basis cache |
| [Cache Management](caching/cache.md) | Path control, statistics, test isolation |

### Caching Functions

| Function | Description |
|----------|-------------|
| `startup_database()` | Pre-compute common 3rd- and 4th-order CG bases |
| `set_cache_path(path)` | Override the cache directory |
| `reset_caches()` | Close and reset in-memory cache connections |
| `clear_all_caches()` | Delete all entries from X- and R-symbol databases |
| `get_cache_stats()` | Return cache statistics dict |
| `print_cache_stats()` | Print a formatted cache summary |
| `TestCacheContext` | Context manager for isolated test caching |

## Quick Links by Task

### I want to...

**Compute a contraction (X-symbol)**
→ [compute_xsymbol](symbols/xsymbol.md), [Contraction](core/contraction.md)

**Compute a permutation (R-symbol)**
→ [compute_rsymbol](symbols/rsymbol.md)

**Build a canonical CG basis**
→ [canonical_basis](symbols/canonical-basis.md), [CGSpec](core/cgspec.md)

**Reverse an edge arrow**
→ [Metric Tensor](primitives/metric.md), `compute_conjugate`

**Pre-populate the cache at startup**
→ [startup_database](caching/database.md)

**Inspect or clear the cache**
→ [Cache Management](caching/cache.md)

**Understand the formal interface contract**
→ [Yuzuha Protocol](../protocol/index.md)

## API Design Principles

Yuzuha follows these design principles:

- **All outputs are NumPy arrays**: Symbols are returned as `float64` `ndarray` objects,
  compatible with standard scientific Python tools
- **Immutable specs**: `CGSpec` objects are never mutated; `with_inverted_axes` returns
  a new object
- **Transparent caching**: Cache lookups and stores are invisible to the caller; the
  same result is returned regardless of whether the cache was hit
- **Fail loudly**: Invalid inputs (bad permutations, incompatible contractions, triangle
  violations) raise `ValueError` rather than silently returning wrong results
- **Thread-safe caching**: All database operations are protected by mutexes

## See Also

- [Core Concepts](../getting-started/core-concepts.md): Mathematical background
- [Yuzuha Protocol](../protocol/index.md): Formal specification of the SU(N) interface
