# Canonical Basis

The canonical basis is the explicit CG tensor a [CGSpec](../core/cgspec.md) stands for,
written out in the magnetic-quantum-number basis. It is the object from which both
symbols are derived — X-symbols by contracting and projecting a pair of bases, R-symbols
by overlapping a basis with its permuted counterpart — so most users reach for it only
when they need the raw coupling tensor itself.

::: yuzuha.canonical_basis
    options:
      show_source: false
      heading_level: 2

## Description

`canonical_basis(spec)` returns a real-valued NumPy array of shape
`(*external_dims, om_dim)`, where:

- `external_dims` is the tuple \((2j_0+1, \ldots, 2j_{n-1}+1)\) — one factor per external
  edge, equal to its representation dimension
- `om_dim` is `spec.om_dimension()` — the number of valid internal spin paths

Each slice `basis[..., mu]` is the CG tensor for OM index \(\mu\), built by contracting
Clebsch-Gordan coefficients along the left-associative fusion tree described in
[Core Concepts](../../getting-started/core-concepts.md). Edges whose direction deviates
from the canonical pattern are corrected by contracting with the invariant metric, so the
array returned is always the basis for the spec as given.

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

The slices form an **orthonormal** set with respect to the inner product that sums over
all magnetic quantum number indices:

$$\langle B_\mu | B_\nu \rangle = \sum_{m_0, \ldots, m_{n-1}}
B_\mu(m_0, \ldots, m_{n-1})\, B_\nu(m_0, \ldots, m_{n-1}) = \delta_{\mu\nu}$$

This is not the normalization of the bare Clebsch-Gordan coefficients. For a 3rd-order
tensor, where the OM dimension is always 1, the single basis element carries a factor of
\(1/\sqrt{2j_3 + 1}\) relative to the bare coefficient:

$$B_0(m_1, m_2, m_3) = \frac{1}{\sqrt{2j_3 + 1}}\, C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2}$$

### Caching

Canonical bases are cached by the Rust layer in `.yuzuha/cgbasis.db`, keyed on the
external spins alone — arrow directions are stripped, since a deviating spec is derived
from the stored tensor by applying the metric. The cache is populated on first use and
persists across Python sessions; `startup_database()` pre-computes it for the spin
configurations most commonly needed, trading startup time for predictable latency later.

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

`canonical_basis` is thread-safe. Concurrent calls with different specs are safe, and
calls with identical specs share the same cached result.

The function raises `ValueError` if the spec has fewer than 2 edges or if no valid
internal spin configuration exists.
