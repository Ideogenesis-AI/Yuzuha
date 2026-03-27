# X-Symbol

Recoupling coefficients for contracting two CG tensors.

::: yuzuha.compute_xsymbol
    options:
      show_source: false
      heading_level: 2

## Description

The X-symbol \(X^{\gamma}_{\alpha\beta}\) is a 3-index real tensor that encodes how
the outer-multiplicity (OM) indices of two CG tensors A and B combine when their shared
edges are contracted to produce an output CG tensor C:

$$X^{\gamma}_{\alpha\beta} = \sum_{\text{contracted edges}} \langle C_\gamma | A_\alpha \otimes B_\beta \rangle$$

The indices \(\alpha, \beta, \gamma\) run over the OM spaces of specs A, B, and C
respectively, so the shape of the returned array is `(om_a, om_b, om_c)`.

### Usage

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

spec_a = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])
spec_b = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(j1),
    yuzuha.Edge.outgoing(jhalf),
    yuzuha.Edge.outgoing(jhalf),
])

# Contract edge 2 of spec_a (outgoing j=1) with edge 0 of spec_b (incoming j=1)
contraction = yuzuha.Contraction([2], [0])

x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
print(x_array.shape)         # (1, 1, 2) — (om_a, om_b, om_c)
print(x_array.dtype)         # float64
print(spec_c.num_external()) # 4 — uncontracted edges from A and B
```

### Output Spec

The returned `spec_c` describes the output CG tensor. Its edges are the **uncontracted**
edges of spec A followed by the **uncontracted** edges of spec B, in the order they
appear in the original specs (contracted edges removed).

### Caching

The first call for a given `(spec_a, spec_b, contraction)` triple computes the result
from scratch via Rust and stores it in `.yuzuha/xsymbol.db`. Subsequent calls with the
same arguments return the cached result immediately.

### Computation Procedure

Internally, `compute_xsymbol` proceeds as follows:

1. Compute the canonical basis for spec A: shape `(*dims_a, om_a)`
2. Compute the canonical basis for spec B: shape `(*dims_b, om_b)`
3. Contract the two basis tensors over the specified magnetic-quantum-number axes
4. Build the canonical basis for output spec C: shape `(*dims_c, om_c)`
5. Project the contracted result onto spec C to extract \(X^{\gamma}_{\alpha\beta}\)

## See Also

- [R-Symbol](rsymbol.md): Permutation transformation on a single CG tensor
- [Canonical Basis](canonical-basis.md): The basis tensors used in the computation
- [CGSpec](../core/cgspec.md): Input specification type
- [Contraction](../core/contraction.md): Edge-pair specification
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Formal specification
  of the X-symbol interface

## Notes

The X-symbol is a real-valued tensor of shape \((\mathrm{om}_A, \mathrm{om}_B, \mathrm{om}_C)\).
It encodes the projection of the contracted basis product \(A_\alpha \otimes_{\text{contracted}} B_\beta\)
in terms of the output basis \(C_\gamma\):

$$A_\alpha \otimes_{\text{contracted}} B_\beta = \sum_\gamma X^\gamma_{\alpha\beta}\, C_\gamma$$

In general \(\mathrm{om}_A \cdot \mathrm{om}_B \neq \mathrm{om}_C\), so \(\mathbf{X}\) is a
rectangular tensor and there is no general unitarity or orthonormality guarantee.

Both the direction of contracted edges and the Frobenius-Schur phase \((-1)^{2j}\) for
arrow-reversed pairs are handled automatically via the metric tensor.
