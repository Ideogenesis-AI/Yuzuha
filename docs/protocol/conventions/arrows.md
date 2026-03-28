# Arrow Conventions

## Sign Convention for Incoming vs. Outgoing

The `Direction` type assigns \(+1\) to `Incoming` and \(-1\) to `Outgoing` — a
protocol-level requirement defined in the [Type System](../types.md#direction).
In the abstract index notation used throughout this section, **incoming edges carry
upper (contravariant) indices** and **outgoing edges carry lower (covariant) indices**.

The signed sum of all magnetic quantum numbers must vanish for any non-zero basis
element:

$$\sum_{\text{incoming}} m^i - \sum_{\text{outgoing}} m_j = 0$$

Incoming edges contribute \(+m\) and outgoing edges contribute \(-m\) to the
conservation law.

## Invariant Metric

The **invariant metric** \(g^{(j)}\) implements the canonical isomorphism between
an irreducible representation and its dual. It must satisfy:

$$g^{(j)}_{m,\,m'} = (-1)^{j - m}\,\delta_{m,\,-m'}$$

This is a \((2j+1) \times (2j+1)\) real matrix. Its key properties are:

- **Orthogonality**: \(g^{(j)} (g^{(j)})^\top = I\)
- **Square**: \((g^{(j)})^2 = (-1)^{2j}\, I\)
- **Transpose**: \((g^{(j)})^\top = (-1)^{2j}\, g^{(j)}\) (symmetric for integer spins, antisymmetric for half-integer)
- **Inverse**: \((g^{(j)})^{-1} = (-1)^{2j}\, g^{(j)}\)

## Arrow Reversal

To convert an **incoming** (upper) edge to an **outgoing** (lower) edge, lower the
corresponding index with \(g_{ab}\):

$$\tilde{T}_{\ldots m'} = g^{(j)}_{m'm}\, T^{\ldots m}$$

To convert an **outgoing** (lower) edge to an **incoming** (upper) edge, raise the
corresponding index with \(g^{ab}\):

$$\tilde{T}^{m'}_{\ldots} = g^{(j)m'm}\, T_{\ldots m} = T_{\ldots m} {[g^{(j)\top}]}^{mm;}$$

The two operations are exact inverses: \(g_{m'k}\, g^{km} = \delta_{m'}^{m}\).

## Frobenius-Schur Indicator

The general definition of the Frobenius-Schur (FS) phase is given in
[Duality Interface — `fs_phase`](../duality.md#fs_phase). For SU(2) with spin \(j\)
the formula specialises to:

$$\text{FS}(j) = (-1)^{2j}$$

It equals \(+1\) for integer spin (bosonic) representations and \(-1\) for
half-integer spin (fermionic) representations. The FS indicator governs the phase
accumulated when a contracted edge pair has its arrows flipped (bond inversion).

## Bond Inversion

A **bond** is a contracted pair of edges between two tensors. The canonical pairing
contracts an incoming (upper) edge of one tensor against an outgoing (lower) edge of
the other:

$$\langle T, S \rangle = T^m\, S_m$$

When both edges of a bond share the same direction — for example, two leading edges
(both incoming) are contracted together — a metric must be inserted to form a valid
pairing. Contracting \(T\) and \(S\) along a bond with a metric insertion on \(S\):

$$\langle T,\, g S \rangle = T^m\, g_{mn}\, S^n$$

**Bond inversion** means moving the metric insertion from one side of the bond to the
other. Using the transpose property \(g^\top = (-1)^{2j}\, g\):

$$\langle T,\, g S \rangle = \langle g^\top T,\, S \rangle = (-1)^{2j}\, \langle g T,\, S \rangle$$

Each time the metric is transferred across the bond, the contraction acquires a factor
of \(\text{FS}(j) = (-1)^{2j}\). In practice, this means that computing an X-symbol
with a bond whose arrows differ from the canonical configuration (e.g. both leading
edges contracted, or both terminal edges contracted) produces a result that differs from
the canonical-direction X-symbol by a factor of \(\text{FS}(j)\) per inverted bond.

## Canonical Conjugate Pattern

The general contract for `compute_conjugate` is given in
[Duality Interface — `compute_conjugate`](../duality.md#compute_conjugate). For the SU(2)
reference implementation, the **canonical conjugate pattern** is the direction-reversed
counterpart of the canonical basis direction ([Basis Conventions — Canonical Basis Direction](basis.md#canonical-basis-direction)):
all **leading edges outgoing** and the **terminal edge incoming**.

This is the unique pattern consistent with the left-associative fusion tree when all
arrows are reversed. An edge \(k\) of the input spec \(S\) is **deviating** if its
direction differs from this canonical pattern. Each deviating edge contributes a factor
of \(\text{FS}(r_k)\) to the accumulated conjugate phase:

$$\phi = \prod_{k\,\in\,\text{deviating edges}} \text{FS}(r_k) \in \{+1, -1\}$$

If \(S\) already matches the canonical conjugate pattern, then \(\phi = +1\).

### Origin of the conjugate phase

The phase \(\phi\) is fixed by requiring that the full trace of \(\overline{T}\) against
\(T\) yields the identity:

$$\langle \overline{T}_\mu, T_\nu \rangle = \delta_{\mu\nu}$$

For the canonical basis direction (leading edges incoming, terminal edge outgoing), the
canonical conjugate \(\overline{T}^{\,\text{canon}}_\mu\) has all arrows reversed, and
the trace pairs each upper index of \(T\) with the corresponding lower index of
\(\overline{T}\) — no metric insertions are needed, so \(\phi = +1\).

When spec \(S\) deviates from the canonical basis direction on edge \(k\), the basis
tensor \(T^S\) is obtained from the canonical tensor \(T^{\,\text{canon}}\) by applying
\(g^{(r_k)}\) on edge \(k\):

$$T^S = g^{(r_k)}_k\, T^{\,\text{canon}}$$

For the trace \(\langle \overline{T}^S, T^S \rangle = \langle \overline{T}^S,\,
g^{(r_k)}_k\, T^{\,\text{canon}} \rangle\) to equal the canonical trace, the extra
metric factor on \(T^S\) must be transferred to \(\overline{T}^S\). By the bond
inversion identity:

$$\langle \overline{T},\, g S \rangle = (-1)^{2j}\, \langle g\, \overline{T},\, S \rangle$$

moving \(g^{(r_k)}\) from \(T^S\) to \(\overline{T}^S\) costs a factor of
\(\text{FS}(r_k) = (-1)^{2j_k}\). Each deviating edge contributes one such factor,
giving the accumulated phase \(\phi = \prod_{k\,\in\,\text{deviating}} \text{FS}(r_k)\).
