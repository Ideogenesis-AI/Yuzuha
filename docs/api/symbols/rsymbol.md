# R-Symbol

Permutation transformation on the outer-multiplicity space of a CG tensor.

::: yuzuha.compute_rsymbol
    options:
      show_source: false
      heading_level: 2

## Description

The R-symbol \(R^{\mu'}_{\mu}\) is a real unitary matrix that encodes how the
outer-multiplicity (OM) index transforms when the external edges of a CG tensor are
permuted:

$$R^{\mu'}_{\mu} = \langle \text{permuted},{\mu'} | \text{original},\mu \rangle$$

The shape of the returned array is `(om_original, om_permuted)`.

For OM dimension 1, the R-symbol is a \(1 \times 1\) matrix whose single entry is
\(\pm 1\) (the permutation phase). For higher OM dimensions it is a proper unitary
matrix.

### Usage

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2

spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(yuzuha.Spin(3)),  # j = 3/2
])

# Swap the first two edges
permutation = [1, 0, 2, 3]
r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)

print(r_array.shape)   # (om_original, om_permuted)
print(r_array.dtype)   # float64

# Identity permutation gives the identity matrix
identity_perm = [0, 1, 2, 3]
r_id, _ = yuzuha.compute_rsymbol(spec, identity_perm)
import numpy as np
assert np.allclose(r_id, np.eye(spec.om_dimension()))
```

### Unitarity

The R-symbol is always **unitary**:

$$R^T R = R R^T = I$$

This can be verified for any spec and permutation:

```python
import numpy as np
r, _ = yuzuha.compute_rsymbol(spec, [1, 0, 2, 3])
assert np.allclose(r.T @ r, np.eye(r.shape[1]))
```

### Composition

R-symbols compose correctly under permutation composition:

$$R(\sigma \circ \pi) = R(\sigma)\, R(\pi)$$

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

The permutation is specified as a list of **destination** indices: `permutation[i]` is
the index in the **original** spec that moves to position `i` in the permuted spec.
That is, `spec_permuted.edge_at(i) == spec.edge_at(permutation[i])`.

An invalid permutation (wrong length, out-of-range indices, or repeated indices) raises
`ValueError`.
