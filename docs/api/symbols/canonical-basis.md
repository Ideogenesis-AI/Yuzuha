# Canonical Basis

The canonical CG basis tensor for a given specification.

::: yuzuha.canonical_basis
    options:
      show_source: false
      heading_level: 2

## Description

`canonical_basis(spec)` computes the canonical basis tensor for a `CGSpec`. The result
is a real-valued NumPy array of shape `(*external_dims, om_dim)`, where:

- `external_dims` is the tuple `(2j_0+1, 2j_1+1, ..., 2j_{n-1}+1)` — one factor per
  external edge, equal to its representation dimension
- `om_dim` is `spec.om_dimension()` — the number of valid internal spin paths

Each slice `basis[..., mu]` is the CG tensor for OM index \(\mu\), constructed via
the left-associative fusion tree (see [Core Concepts](../../getting-started/core-concepts.md)).

### Usage

```python
import yuzuha
import numpy as np

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])

basis = yuzuha.canonical_basis(spec)
print(basis.shape)    # (2, 2, 3, 1) — (d_j1, d_j2, d_j3, om_dim)
print(basis.dtype)    # float64

# Normalization: each OM slice is normalized to 1 (in the CG sense)
# The squared norm over magnetic indices equals 1/om_dim for each slice
```

### Normalization Convention

The canonical basis is normalized with respect to the OM space. For a 3rd-order CG
tensor with a single OM configuration (\(\mathrm{om\_dim} = 1\)), each basis element
acquires a scaling factor relative to the standard Clebsch-Gordan coefficient:

$$\langle B_\mu | B_\nu \rangle = \delta_{\mu\nu}$$

where the inner product sums over all magnetic quantum number indices weighted by the
appropriate fusion metric.

### Caching

Canonical bases are cached in a Rust-managed SQLite database inside the Yuzuha
installation. The cache is populated on first use and persists across Python sessions.
Use `startup_database()` to pre-populate the cache for common spin configurations.

### Auxiliary Functions

| Function | Description |
|----------|-------------|
| `fs_phase_for_spin(spin)` | Frobenius-Schur indicator \((-1)^{2j}\) for a spin |
| `compute_conjugate(spec)` | Canonical basis for the conjugated spec (arrow-reversed) |

## See Also

- [X-Symbol](xsymbol.md): Built from pairs of canonical bases
- [R-Symbol](rsymbol.md): Built by comparing canonical bases before and after permutation
- [CGSpec](../core/cgspec.md): Input specification
- [Database](../caching/database.md): Pre-population of the canonical basis cache
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Formal normalization
  requirement

## Notes

`canonical_basis` is thread-safe. Concurrent calls with different specs are safe; calls
with identical specs share the same cached result.

The function raises `ValueError` if the spec has fewer than 2 edges or if no valid
internal spin configuration exists.
