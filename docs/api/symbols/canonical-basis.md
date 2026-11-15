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

# Every OM slice is a unit vector over the magnetic indices
print(np.linalg.norm(basis[..., 0]))   # 1.0
```

### Normalization Convention

The slices form an **orthonormal** set with respect to the inner product that sums
over all magnetic quantum number indices:

$$\langle B_\mu | B_\nu \rangle = \sum_{m_0, \ldots, m_{n-1}}
B_\mu(m_0, \ldots, m_{n-1})\, B_\nu(m_0, \ldots, m_{n-1}) = \delta_{\mu\nu}$$

For a 3rd-order tensor, where the OM dimension is always 1, the single basis element
carries a factor of \(1/\sqrt{2j_3 + 1}\) relative to the bare Clebsch-Gordan
coefficient:

$$B_0(m_1, m_2, m_3) = \frac{1}{\sqrt{2j_3 + 1}}\, C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2}$$

### Caching

Canonical bases are cached by the Rust layer in `.yuzuha/cgbasis.db`, keyed on the
external edges and the internal spin sequence. The cache is populated on first use
and persists across Python sessions.
Use `startup_database()` to pre-populate the cache for common spin configurations.

### Auxiliary Functions

| Function | Description |
|----------|-------------|
| `fs_phase_for_spin(spin)` | Frobenius-Schur indicator \((-1)^{2j}\) for a spin |
| `compute_conjugate(spec)` | Conjugate phase and arrow-reversed spec for a given spec |

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
