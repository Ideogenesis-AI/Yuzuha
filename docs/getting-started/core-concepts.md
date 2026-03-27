# Core Concepts

Before using Yuzuha, it helps to understand the mathematical objects and conventions
the library is built around. This page introduces spins, edges, CG specifications,
fusion trees, the outer-multiplicity index, and the caching model.

## Spin Representations

In SU(2) theory, every irreducible representation is labeled by a non-negative
half-integer spin \(j \in \{0, \frac{1}{2}, 1, \frac{3}{2}, \ldots\}\). The dimension
of the representation is \(2j + 1\).

Yuzuha represents spins internally as **doubled integer values** to avoid floating-point
arithmetic. `Spin(n)` corresponds to \(j = n/2\):

```python
import yuzuha

j0   = yuzuha.Spin(0)   # j = 0   (singlet)
jhalf = yuzuha.Spin(1)  # j = 1/2 (doublet)
j1   = yuzuha.Spin(2)   # j = 1   (triplet)
j32  = yuzuha.Spin(3)   # j = 3/2 (quartet)

print(jhalf.value())      # 0.5
print(jhalf.dimension())  # 2  (= 2j+1)
print(jhalf.twice())      # 1  (the internal integer 2j)
```

## Arrow Directions

Tensor network edges carry a **direction** — incoming or outgoing — that determines
whether the edge transforms in the primal or dual representation. Arrow direction is
crucial for computing the correct coupling coefficients and for arrow-reversal
transformations via the invariant metric.

```python
from yuzuha import Direction

d_in  = Direction.Incoming   # primal representation
d_out = Direction.Outgoing   # dual representation
```

Arrow directions can be flipped using the invariant metric \(g\):

$$g^{(j)}_{m,m'} = (-1)^{j-m}\,\delta_{m,-m'}$$

See [Metric Tensor](../api/primitives/metric.md) for the functions that implement
this transformation.

## Edges

An **Edge** pairs a spin with a direction, representing one external edge of a tensor:

```python
import yuzuha

j = yuzuha.Spin(1)  # j = 1/2

e_in  = yuzuha.Edge.incoming(j)   # spin-1/2 incoming edge
e_out = yuzuha.Edge.outgoing(j)   # spin-1/2 outgoing edge
```

## CG Specifications

A **CGSpec** (Clebsch-Gordan specification) fully describes a CG tensor — the canonical
basis element of a tensor network node. It records:

- The list of **edges** (spin + direction for each external edge)
- The list of **internal spins** (\(\alpha\) values) — one per intermediate fusion step

For an \(n\)th-order tensor there are \(n - 2\) internal spins, determined by the
left-associative fusion tree structure (see below).

```python
import yuzuha

jhalf = yuzuha.Spin(1)  # j = 1/2
j1    = yuzuha.Spin(2)  # j = 1

# 3rd-order spec: two spin-1/2 incoming, one spin-1 outgoing
spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])

print(spec.num_external())  # 3
print(spec.om_dimension())  # 1  (only one valid internal coupling)
print(spec.shape)           # (2, 2, 3)  — product of (2j+1) per edge
```

The `from_edges` constructor automatically enumerates all valid internal spin
configurations and stores them as the `alphas` field.

## Left-Associative Fusion Trees

Yuzuha represents multi-spin couplings using **left-associative fusion trees**. For
\(n\) external spins \((j_1, j_2, \ldots, j_n)\), the coupling proceeds from left to
right:

$$\bigl(\cdots\bigl((j_1 \otimes j_2) \otimes j_3\bigr) \otimes \cdots\bigr) \otimes j_n$$

Each step produces an intermediate spin \(\alpha_k\). For \(n\) edges there are \(n - 2\)
intermediate spins \(\alpha_1, \ldots, \alpha_{n-2}\):

- **3 edges**: \(j_1 \otimes j_2 \to \alpha_1 = j_3\) — no free internal spin
- **4 edges**: \((j_1 \otimes j_2 \to \alpha_1) \otimes j_3 \to j_4\) — one free internal spin

## Outer Multiplicity (OM) Index

When the same total spin \(j\) can be reached via multiple internal spin paths
\((\alpha_1, \ldots, \alpha_{n-2})\), those paths are distinguished by the **outer
multiplicity (OM) index** \(\mu\). The OM dimension is the number of such valid paths.

The OM index appears as the last axis of canonical basis tensors and as axes of
X-symbol and R-symbol matrices.

```python
# 4th-order spec with non-trivial OM
j1    = yuzuha.Spin(2)   # j = 1
jhalf = yuzuha.Spin(1)   # j = 1/2

spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(j1),
    yuzuha.Edge.incoming(j1),
    yuzuha.Edge.incoming(j1),
    yuzuha.Edge.outgoing(j1),
])
print(spec.om_dimension())  # > 1 for certain spin combinations
```

## Canonical CG Bases

The **canonical basis** for a CGSpec is a real-valued tensor of shape
\((2j_1+1, \ldots, 2j_n+1, \mathrm{om\_dim})\) that expresses the CG tensor in the
magnetic-quantum-number basis. It is computed from the CG coefficients using the
left-associative fusion tree and normalized with respect to the OM space.

```python
basis = yuzuha.canonical_basis(spec)
print(basis.shape)  # (*external_dims, om_dim)
```

The canonical basis is the fundamental object from which X-symbols and R-symbols
are derived.

## X-Symbols and R-Symbols

**X-symbols** are the recoupling coefficients for contracting two CG tensors:

$$X^{\gamma}_{\alpha\beta} = \langle C_\gamma | A_\alpha \otimes_{\text{contracted}} B_\beta \rangle$$

They form a 3-index tensor of shape \((\mathrm{om}_A, \mathrm{om}_B, \mathrm{om}_C)\).

**R-symbols** are the transformation matrices for permuting edges of a CG tensor:

$$R^{\mu'}_{\mu} = \langle \text{permuted},{\mu'} | \text{original},\mu \rangle$$

They form a 2-index unitary matrix of shape `(om_original, om_permuted)`.

See [X-Symbol](../api/symbols/xsymbol.md) and [R-Symbol](../api/symbols/rsymbol.md) for
the full API.

## SQLite Caching

Yuzuha uses a two-level caching strategy:

1. **CG basis cache** (Rust, in-memory + SQLite): Canonical bases are cached inside
   the Rust layer in a SQLite database. This cache survives between Python sessions.
2. **Symbol cache** (Python, SQLite): X-symbol and R-symbol results are cached in
   separate SQLite databases in the `.yuzuha/` directory.

Both caches are thread-safe and transparent — you never need to manage them explicitly.
See [Database](../api/caching/database.md) and [Cache Management](../api/caching/cache.md)
for cache control utilities.

## Next Steps

- **[API Reference](../api/index.md)**: Complete documentation for all types and functions
- **[Yuzuha Protocol](../protocol/index.md)**: Formal specification of the SU(N) interface
