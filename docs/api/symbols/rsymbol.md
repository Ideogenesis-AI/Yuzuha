# R-Symbol

The R-symbol is the transformation a CG tensor's outer-multiplicity index undergoes when
its external edges are permuted. Permuting the axes of a symmetry-constrained tensor is
not a pure reindexing — the fusion tree is rebuilt in a different order — and the
R-symbol is the matrix that carries the reduced blocks from the old OM basis to the new
one.

::: yuzuha.compute_rsymbol
    options:
      show_source: false
      heading_level: 2

## Description

The R-symbol \(R^{\mu'}_{\mu}\) is the overlap between the permuted and the original
canonical basis:

$$R^{\mu'}_{\mu} = \langle \text{permuted},{\mu'} | \text{original},\mu \rangle$$

The returned array has shape `(om_original, om_permuted)`. For OM dimension 1 it degenerates
to a \(1 \times 1\) matrix holding the permutation phase \(\pm 1\); for higher OM
dimensions it is a genuine orthogonal rotation of the multiplicity space.

### Permutation Convention

The permutation is given as a list of **source** indices: `permutation[i]` is the position
in the **original** spec whose edge moves to position `i` of the permuted spec, so that
`spec_permuted.edges[i] == spec.edges[permutation[i]]`. This is the same convention
NumPy's `transpose` uses, which makes the two safe to apply in tandem to a tensor and its
reduced blocks.

### Usage

```python
import yuzuha
import numpy as np

jhalf = yuzuha.Spin(1)   # j = 1/2

spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(yuzuha.Spin(3)),  # j = 3/2
])

# Swap the first two edges
r_array, spec_permuted = yuzuha.compute_rsymbol(spec, [1, 0, 2, 3])

print(r_array.shape)   # (om_original, om_permuted)
print(r_array.dtype)   # float64

# The identity permutation gives the identity matrix
r_id, _ = yuzuha.compute_rsymbol(spec, [0, 1, 2, 3])
assert np.allclose(r_id, np.eye(spec.om_dimension()))
```

### Algebraic Properties

The R-symbol is always **unitary**, and since it is real this amounts to orthogonality:

$$R^\top R = R R^\top = I$$

R-symbols also compose under composition of permutations, and the inverse permutation
gives the transpose:

$$R(\sigma \circ \pi) = R(\sigma)\, R(\pi), \qquad R(\pi^{-1}) = R(\pi)^\top$$

Together these make a sequence of axis permutations collapsible into a single matrix,
which is the practical reason for caching R-symbols to avoid recomputing them per
permutation step:

```python
r, _ = yuzuha.compute_rsymbol(spec, [1, 0, 2, 3])
assert np.allclose(r.T @ r, np.eye(r.shape[1]))
```

### Caching

The first call for a given `(spec, permutation)` pair stores the result in
`.yuzuha/rsymbol.db`. Subsequent calls return the cached value immediately.

## See Also

- [X-Symbol](xsymbol.md): Contraction recoupling coefficients
- [Canonical Basis](canonical-basis.md): The basis from which R-symbols are derived
- [CGSpec](../core/cgspec.md): Input specification type
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Formal specification
  of the R-symbol interface

## Notes

The permuted spec is returned alongside the matrix for convenience, because its internal
fusion tree is re-derived from the permuted edge order and will not in general match the
internal spins of the original.

An invalid permutation raises `ValueError`: wrong length, an out-of-range index, or a
repeated index.
