# Duality Interface

This page specifies the required function signatures, mathematical properties, and
error conditions for the two duality functions of the Yuzuha Protocol.
These functions formalise how representations and CG tensors behave under arrow
reversal, and underpin the Frobenius-Schur phase bookkeeping needed for physically
consistent dualisation in tensor network contractions.

---

## `fs_phase`

### Signature

```
fs_phase(rep: RepLabel) → float
```

### Semantics

The **Frobenius-Schur (FS) phase** of a representation \(r\) is defined by the
behaviour of the invariant metric \(g^{(r)}\) under self-composition:

$$\left(g^{(r)}\right)^2 = \text{FS}(r)\cdot I$$

This is well-defined for any group whose irreducible representations are **self-dual**
(i.e. every irrep \(r\) is isomorphic to its dual \(\bar{r}\)), since in that case
\(g^{(r)}\) is an invertible scalar-multiple of the identity when squared. The value
\(\text{FS}(r) \in \{+1, -1\}\) distinguishes **real** representations
(\(\text{FS} = +1\), metric symmetric) from **pseudo-real** ones
(\(\text{FS} = -1\), metric antisymmetric). The SU(2)-specific formula is given in
[Arrow Conventions — Frobenius-Schur Indicator](conventions/arrows.md#frobenius-schur-indicator).

### Required Properties

1. **Values**: The return value is always \(+1.0\) or \(-1.0\).
2. **Bosonic representations**: If \(g^{(r)}\) is symmetric (\(g^\top = g\)), then
   \(\text{FS}(r) = +1\).
3. **Fermionic representations**: If \(g^{(r)}\) is antisymmetric (\(g^\top = -g\)),
   then \(\text{FS}(r) = -1\).
4. **Identity representation**: \(\text{FS}(\mathbf{1}) = +1\).
5. **Bond inversion**: Flipping the arrows of a contracted edge pair with
   representation \(r\) multiplies the X-symbol by \(\text{FS}(r)\).

### SU(2) Reference

For SU(2) with \(\text{rep} = \text{Spin}(n)\) (i.e. \(j = n/2\)):

$$\text{FS}(j) = (-1)^{2j} = \begin{cases} +1 & j \in \mathbb{Z} \\ -1 & j \in \mathbb{Z} + \tfrac{1}{2} \end{cases}$$

```python
import yuzuha

yuzuha.fs_phase_for_spin(yuzuha.Spin(1))   # j=1/2 → -1.0
yuzuha.fs_phase_for_spin(yuzuha.Spin(2))   # j=1   → +1.0
```

### Error Conditions

| Condition | Exception |
|-----------|-----------|
| `rep` is not a valid representation label | `ValueError` |

---

## `compute_conjugate`

### Signature

```
compute_conjugate(spec: Spec) → (float, Spec)
```

### Semantics

The **conjugate** of a spec \(S\) is the spec \(\bar{S}\) obtained by reversing all
edge directions while keeping the same representation labels. The function also returns
a scalar **conjugate phase** \(\phi \in \{+1, -1\}\).

The phase encodes the discrepancy between the actual arrow configuration of \(S\) and
the implementation's chosen **canonical conjugate pattern** — a fixed reference
direction assignment used to define the normalised dual tensor. Different implementations
may adopt different canonical patterns; some may always return \(\phi = +1\) by
absorbing the sign into the basis convention. The SU(2)-specific canonical pattern and
phase formula are documented in
[Arrow Conventions — Canonical Conjugate Pattern](conventions/arrows.md#canonical-conjugate-pattern).

### Required Properties

1. **Spec structure**: The returned \(\bar{S}\) has the same representation labels as
   \(S\) on each edge, with all directions flipped. The internal representations
   (alphas) are preserved unchanged.
2. **Phase values**: \(\phi \in \{+1.0, -1.0\}\).
3. **Double conjugation**: Applying `compute_conjugate` twice recovers the original
   spec and a net phase of \(+1\):

    $$\phi(\bar{S}) \cdot \phi(S) = +1 \qquad \text{and} \qquad \overline{\bar{S}} = S$$

4. **Self-consistency**: \(\phi\) must be consistent with `fs_phase` in the sense that
   any bond-inversion phase accumulated by the arrow reversal equals the product of
   \(\text{FS}(r_k)\) over the affected edges.

### SU(2) Reference

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])

phase, conj_spec = yuzuha.compute_conjugate(spec)
# phase: +1.0 or -1.0
# conj_spec: CGSpec with all edge directions flipped
```

### Error Conditions

| Condition | Exception |
|-----------|-----------|
| The conjugated edge configuration admits no valid coupling | `ValueError` |

---

## Relationship Between the Two Functions

`compute_conjugate` is built on top of `fs_phase`: the accumulated phase it returns
is the product of `fs_phase(rep)` over the deviating edges. Implementations may expose
`fs_phase` as a standalone utility so that callers can compute bond-inversion phases
for individual edges without constructing a full conjugated spec.

---

## Conformance Test Summary

| Test | Checks |
|------|--------|
| FS phase values | `fs_phase(rep) ∈ {+1.0, -1.0}` for all valid representations |
| FS phase identity | `fs_phase(identity_rep) == +1.0` |
| Double conjugation spec | `compute_conjugate(compute_conjugate(spec)[1])[1] == spec` |
| Double conjugation phase | `compute_conjugate(spec)[0] * compute_conjugate(compute_conjugate(spec)[1])[0] == +1.0` |
| Phase consistency | Returned phase consistent with `fs_phase` over arrow-reversed edges |
