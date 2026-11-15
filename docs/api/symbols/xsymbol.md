# X-Symbol

The X-symbol is the recoupling coefficient for contracting two CG tensors: it says how
the outer-multiplicity indices of the two inputs combine into the outer-multiplicity
index of the result. It is the symbol a tensor network contraction actually needs, since
contracting the symmetry-constrained parts of two tensors reduces to a dense contraction
of their reduced blocks followed by multiplication with this coefficient.

::: yuzuha.compute_xsymbol
    options:
      show_source: false
      heading_level: 2

## Description

The X-symbol \(X^{\gamma}_{\alpha\beta}\) is a 3-index real tensor giving the expansion of
the contracted product of the input bases in terms of the output basis:

$$A_\alpha \otimes_{\text{contracted}} B_\beta = \sum_\gamma X^\gamma_{\alpha\beta}\, C_\gamma,
\qquad X^{\gamma}_{\alpha\beta} = \langle C_\gamma \mid A_\alpha \otimes_{\text{contracted}} B_\beta \rangle$$

The indices \(\alpha, \beta, \gamma\) run over the OM spaces of specs A, B, and C, so the
returned array has shape `(om_a, om_b, om_c)`. In general
\(\mathrm{om}_A \cdot \mathrm{om}_B \neq \mathrm{om}_C\), which means \(\mathbf{X}\) is a
rectangular tensor with no unitarity or orthonormality property to rely on — unlike the
[R-symbol](rsymbol.md).

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
edges of spec A followed by the **uncontracted** edges of spec B, each group keeping the
relative order it had in its original spec. Its internal fusion tree then follows from
the left-associative convention applied to that concatenated edge list, so the
order of the pairs in the `Contraction` is significant.

### Computation Procedure

The projection onto spec C distinguishes the X-symbol from a plain tensor
contraction, and it requires a third basis to be built:

1. Compute the canonical bases for specs A and B, of shapes `(*dims_a, om_a)` and
   `(*dims_b, om_b)`
2. Contract the two basis tensors over the magnetic-quantum-number axes named by the
   contraction, keeping the OM axes separate
3. Build the canonical basis for the output spec C, of shape `(*dims_c, om_c)`
4. Project the contracted result onto that basis to extract
   \(X^{\gamma}_{\alpha\beta}\)

### Caching

The first call for a given `(spec_a, spec_b, contraction)` triple computes the result
from scratch via Rust and stores it in `.yuzuha/xsymbol.db`. Subsequent calls with the
same arguments return the cached result immediately.

## See Also

- [R-Symbol](rsymbol.md): Permutation transformation on a single CG tensor
- [Canonical Basis](canonical-basis.md): The basis tensors used in the computation
- [CGSpec](../core/cgspec.md): Input specification type
- [Contraction](../core/contraction.md): Edge-pair specification
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Formal specification
  of the X-symbol interface

## Notes

Both the direction of the contracted edges and the Frobenius-Schur phase \((-1)^{2j}\)
for arrow-reversed pairs are handled automatically through the invariant metric. Flipping
the arrows of a contracted pair of spin \(j\) therefore scales the X-symbol by
\((-1)^{2j}\).

An invalid contraction raises `ValueError`: index out of range, mismatched spins on a
pair, or two edges of the same direction paired together.
