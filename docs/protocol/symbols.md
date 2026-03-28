# Symbol Interface

This page specifies the required function signatures, mathematical properties, and
error conditions for the three core symbol functions of the Yuzuha Protocol.

---

## `canonical_basis`

### Signature

```
canonical_basis(spec: Spec) → Array[*shape, om_dim]
```

Where:
- `spec.shape` = `(dim_0, dim_1, ..., dim_{n-1})` — one factor per external edge
- `spec.om_dimension()` = `om_dim` — the OM dimension
- The return type is a real-valued \(n+1\)-dimensional array of `float64`

### Required Properties

1. **Real-valued**: All entries are real numbers.

2. **Orthonormality**: Let \(B_\mu\) denote the slice `basis[..., mu]`. Then:

    $$\sum_{m_0,\ldots,m_{n-1}} B_\mu(m_0,\ldots,m_{n-1})\, B_\nu(m_0,\ldots,m_{n-1}) = \delta_{\mu\nu}$$

3. **Charge conservation**: For every magnetic-number index tuple
   \((m_0, \ldots, m_{n-1})\) where the charge-conservation law is violated:

    $$B_\mu(m_0, \ldots, m_{n-1}) = 0 \quad \forall\, \mu$$

4. **Self-consistent fusion tree**: The basis must be constructed from a fixed,
   documented coupling order applied consistently across all calls. The SU(2) reference
   implementation uses left-associative coupling; see
   [Fusion Conventions — Left-Associative Fusion Trees](conventions/fusion.md#left-associative-fusion-trees).

5. **Self-consistent OM ordering**: OM indices must be assigned consistently across all
   calls for the same spec. The SU(2) reference implementation uses lexicographic ordering
   of internal representations; see
   [Fusion Conventions — OM Ordering](conventions/fusion.md#canonical-ordering-of-internal-representations).

### Error Conditions

| Condition | Exception |
|-----------|-----------|
| `spec.num_external() < 2` | `ValueError` |
| No valid internal coupling exists | `ValueError` |

### SU(2) Reference

```python
import yuzuha
basis = yuzuha.canonical_basis(spec)
# shape: (*spec.shape, spec.om_dimension())
# dtype: float64
```

---

## `compute_x_symbol`

### Signature

```
compute_x_symbol(spec_a: Spec, spec_b: Spec, contraction: Contraction)
    → (Array[om_a, om_b, om_c], Spec)
```

Where:
- `om_a = spec_a.om_dimension()`
- `om_b = spec_b.om_dimension()`
- `om_c = spec_c.om_dimension()` where `spec_c` is the returned output spec

### Semantics

The X-symbol \(X^{\gamma}_{\alpha\beta}\) is defined by:

$$X^{\gamma}_{\alpha\beta} = \sum_{\text{contracted indices}} \langle C_\gamma \mid A_\alpha \otimes B_\beta \rangle$$

where:
- \(A_\alpha = \text{canonical_basis}(\text{spec_a})[..., \alpha]\)
- \(B_\beta = \text{canonical_basis}(\text{spec_b})[..., \beta]\)
- \(C_\gamma = \text{canonical_basis}(\text{spec_c})[..., \gamma]\)
- The sum is over all magnetic-number index combinations for the contracted edges

### Output Spec

The output `spec_c` has external edges in the following order:

1. The **uncontracted** edges of spec A, in their original order
2. The **uncontracted** edges of spec B, in their original order

The internal fusion tree of `spec_c` is determined by the left-associative convention
applied to this concatenated edge list.

### Required Properties

1. **Real-valued**: All entries of the returned array are real numbers.

2. **Basis decomposition**: The X-symbol encodes the expansion of the contracted
   basis product in terms of the output canonical basis:

    $$A_\alpha \otimes_{\text{contracted}} B_\beta = \sum_\gamma X^\gamma_{\alpha\beta}\, C_\gamma$$

    In general \(\mathrm{om}_a \cdot \mathrm{om}_b \neq \mathrm{om}_c\), so \(\mathbf{X}\)
    is a rectangular tensor; no general unitarity or orthonormality is guaranteed.

3. **Arrow consistency**: Contracted edge pairs must satisfy the direction condition
   (one incoming, one outgoing). The X-symbol automatically incorporates any metric
   factors needed for the Frobenius-Schur phase.

4. **Bond inversion phase**: If the direction of contracted pair \((i, k)\) with
   representation \(j\) is flipped, the X-symbol is multiplied by \((-1)^{2j}\).

### Error Conditions

| Condition | Exception |
|-----------|-----------|
| `len(contraction.axes_a) != len(contraction.axes_b)` | `ValueError` |
| Any index out of range | `ValueError` |
| Representation mismatch on a pair | `ValueError` |
| Direction mismatch on a pair (both same direction) | `ValueError` |
| No valid output coupling | `ValueError` |

### SU(2) Reference

```python
import yuzuha

x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
# x_array.shape: (om_a, om_b, om_c)
# x_array.dtype: float64
```

---

## `compute_r_symbol`

### Signature

```
compute_r_symbol(spec: Spec, permutation: list[int])
    → (Array[om_original, om_permuted], Spec)
```

Where:
- `om_original = spec.om_dimension()`
- `om_permuted = spec_permuted.om_dimension()` where `spec_permuted` is the returned spec

### Semantics

The R-symbol \(R^{\mu'}_\mu\) is defined by:

$$R^{\mu'}_\mu = \langle B'_{\mu'} \mid B_\mu \rangle
= \sum_{m_0,\ldots,m_{n-1}} B'_{\mu'}(m_{\pi(0)},\ldots,m_{\pi(n-1)})\, B_\mu(m_0,\ldots,m_{n-1})$$

where \(\pi\) is the permutation and \(B', B\) are the canonical bases of
`spec_permuted` and `spec` respectively.

### Permutation Convention

The permutation list `p` of length `n` is interpreted as: position \(i\) in the
**permuted** spec receives the edge from position `p[i]` in the **original** spec.
That is, `spec_permuted.edge_at(i) == spec.edge_at(p[i])`.

### Required Properties

1. **Real-valued**: All entries of the returned array are real numbers.

2. **Unitarity**:

    $$(R^\dagger)^\mu{}_{\mu'}\, R^{\mu'}{}_\nu = \delta^\mu_\nu$$

    $$R^{\mu'}{}_\mu\, (R^\dagger)^\mu{}_{\mu''} = \delta^{\mu'}_{\mu''}$$

    Since \(R\) is real, \(R^\dagger = R^\top\), so \((R^\dagger)^\mu{}_{\mu'} = R^{\mu'}{}_\mu\).

3. **Identity permutation**: For the identity permutation, the returned array is
   the identity matrix \(I_{\text{om}}\).

4. **Composition**: For permutations \(\pi\) and \(\sigma\):

    $$R(\sigma \circ \pi) = R(\sigma) \cdot R(\pi)$$

5. **Inverse**: \(R(\pi^{-1}) = R(\pi)^\top = R(\pi)^{-1}\)

### Error Conditions

| Condition | Exception |
|-----------|-----------|
| `len(permutation) != spec.num_external()` | `ValueError` |
| Any index out of range `[0, n)` | `ValueError` |
| Repeated index in permutation | `ValueError` |

### SU(2) Reference

```python
import yuzuha

r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)
# r_array.shape: (om_original, om_permuted)
# r_array.dtype: float64
```

---

## Conformance Test Summary

An implementation passes the symbol conformance tests if:

| Test | Checks |
|------|--------|
| Basis orthonormality | \(\langle B_\mu \mid B_\nu \rangle = \delta_{\mu\nu}\) for all specs |
| X-symbol basis decomposition | \(A_\alpha \otimes_{\text{contracted}} B_\beta = \sum_\gamma X^\gamma_{\alpha\beta} C_\gamma\) holds for all \(\alpha, \beta\) |
| R-symbol unitarity | \(R^\top R = I\) and \(R R^\top = I\) for all valid permutations |
| R-symbol composition | \(R(\sigma \circ \pi) = R(\sigma) R(\pi)\) |
| R-symbol inversion | \(R(\pi^{-1}) = R(\pi)^\top\) |
| Bond inversion phase | X-symbol scales by \((-1)^{2j}\) on arrow flip |
| Error raising | `ValueError` for all specified invalid inputs |
