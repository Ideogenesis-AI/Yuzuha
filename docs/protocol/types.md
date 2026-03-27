# Type System

This page specifies the abstract types that every Yuzuha Protocol implementation must
provide. Each type is described by its required properties (not its representation),
so implementations are free to choose any internal encoding that satisfies the
constraints.

## `RepLabel`

A **representation label** uniquely identifies an irreducible representation of the
symmetry group.

### Requirements

- **Equality**: Two labels are equal if and only if they denote the same irreducible
  representation.
- **Total order**: Labels admit a total order consistent with equality, enabling
  deterministic sorting of internal representations in canonical sequences.
- **Hashability**: Labels can be used as dictionary keys and placed in sets.
- **Identity representation**: There exists a distinguished label `zero` (or equivalent)
  for the trivial (identity) representation. Coupling any label with `zero` yields the
  original label.

### SU(2) Implementation

| Concept | SU(2) realization |
|---------|------------------|
| Irreducible representations | Half-integer spins \(j \in \{0, \tfrac{1}{2}, 1, \tfrac{3}{2}, \ldots\}\) |
| `RepLabel` type | `Spin`, stored as doubled integer \(2j\) |
| Identity representation | `Spin(0)` (\(j = 0\)) |
| Total order | Lexicographic on doubled integer value |
| Fusion rule | Triangle inequality: \(j_3 \in [\lvert j_1 - j_2 \rvert,\; j_1 + j_2]\) with integer-sum constraint |

### Generalization Notes

For SU(N) with \(N > 2\), the representation label would typically be a highest-weight
vector \(\lambda = (\lambda_1, \ldots, \lambda_{N-1}) \in \mathbb{Z}_{\geq 0}^{N-1}\).
The total order would be lexicographic on this tuple. The fusion rule is determined
by the Littlewood-Richardson rule.

---

## `Direction`

**Direction** is a two-valued enumeration specifying whether an edge belongs to
the primal (incoming) or dual (outgoing) representation space.

### Requirements

- **Values**: Exactly two values: `Incoming` and `Outgoing`.
- **Sign**: `Incoming` carries sign \(+1\) and `Outgoing` carries sign \(-1\). This
  assignment is fixed by the protocol and must be consistent across all implementations.
- **Complement**: There exists an operation `flip` (or equivalent) that maps each
  value to the other.
- **Equality**: Two `Direction` values are equal if and only if they are the same enum
  variant.

### SU(2) Implementation

| Value | Meaning | Sign | Magnetic number contribution |
|-------|---------|------|------------------------------|
| `Incoming` | Primal / ket-type | \(+1\) | \(+m\) |
| `Outgoing` | Dual / bra-type | \(-1\) | \(-m\) |

### Generalization Notes

The two-valued `Direction` type is universal across all compact Lie groups. The
distinction between primal and dual is tied to the existence of a non-degenerate
invariant bilinear form (metric) on each irreducible representation. All compact Lie
groups possess such a form.

---

## `Edge`

An **Edge** combines a `RepLabel` with a `Direction`, representing one external edge
of a tensor.

### Requirements

- **Fields**: An edge has exactly two attributes:
  - `rep: RepLabel` — the representation label
  - `dir: Direction` — the arrow direction
- **Equality**: Two edges are equal if and only if both their representation labels
  and directions are equal.
- **Constructors**: Implementations must provide at least `incoming(rep)` and
  `outgoing(rep)` constructors.
- **Dimension**: Each edge carries an implicit dimension equal to the dimension of
  the representation (the number of magnetic quantum number states).

### SU(2) Implementation

| Concept | SU(2) realization |
|---------|------------------|
| `rep` field | `Spin` value |
| `dir` field | `Direction` enum |
| Dimension | \(2j + 1\) |
| `incoming(j)` | `Edge.incoming(Spin(2j))` |
| `outgoing(j)` | `Edge.outgoing(Spin(2j))` |

### Generalization Notes

For SU(N), the dimension of representation \(\lambda\) is given by the Weyl dimension
formula. The `Edge` structure is otherwise identical.

---

## `Spec`

A **Spec** (specification) completely describes a CG tensor by recording its external
edges and the internal representations of its left-associative fusion tree.

### Requirements

- **External edges**: An ordered list of `Edge` values; length \(n \geq 2\).
- **Internal representations**: An ordered list of \(n - 2\) `RepLabel` values
  \((\alpha_1, \ldots, \alpha_{n-2})\) specifying one internal coupling path. A
  `Spec` object may represent a single path or a collection of all valid paths for
  a given set of external edges.
- **Derived quantities**:
  - `num_external() → int`: number of external edges
  - `om_dimension() → int`: number of valid internal coupling paths (OM dimension)
  - `shape → tuple[int, ...]`: tuple of representation dimensions, one per edge
- **Mutation**: `with_inverted_axes(axes) → Spec` returns a new spec with the
  direction of the specified external edges flipped; internal representations are
  unchanged.
- **Validity**: A spec is **valid** if for every consecutive pair of internal
  representations, the triangle (or equivalent fusion) inequality is satisfied.

### SU(2) Implementation

| Concept | SU(2) realization |
|---------|------------------|
| External edges | List of `Edge` objects |
| Internal representations | List of `Spin` values (one per fusion step) |
| `from_edges(edges)` | Enumerates all valid internal spin configurations |
| OM dimension | Number of valid spin paths through the fusion tree |

### Generalization Notes

For SU(N), the internal representations would be highest-weight vectors. The
enumeration of valid paths is governed by the Littlewood-Richardson rule rather than
the triangle inequality. The rest of the `Spec` interface is identical.

---

## `Contraction`

A **Contraction** specifies which external edges of two specs to contract together.

### Requirements

- **Fields**:
  - `axes_a: list[int]` — indices into spec A's edge list
  - `axes_b: list[int]` — indices into spec B's edge list
- **Length equality**: `len(axes_a) == len(axes_b)`
- **Pair validity**: For each pair \((i, k)\):
  - The representation of edge \(i\) in spec A equals the representation of edge
    \(k\) in spec B
  - The direction of edge \(i\) in spec A is opposite to the direction of edge \(k\)
    in spec B (one incoming, one outgoing)
- **No duplicates**: Each index appears at most once in `axes_a` and at most once
  in `axes_b`.

### SU(2) Implementation

| Concept | SU(2) realization |
|---------|------------------|
| `Contraction([i, ...], [k, ...])` | Contracts edge \(i\) of spec A with edge \(k\) of spec B |
| Spin matching | `spin_a[i] == spin_b[k]` |
| Direction matching | One `Incoming`, one `Outgoing` |

### Generalization Notes

For SU(N), the representation matching condition uses the SU(N) `RepLabel` equality.
The direction condition remains identical.

---

## Summary Table

| Type | SU(2) Python class | Protocol role |
|------|-------------------|---------------|
| `RepLabel` | `Spin` | Identifies irreps |
| `Direction` | `Direction` | Arrow direction |
| `Edge` | `Edge` | Directed edge |
| `Spec` | `CGSpec` | Full tensor specification |
| `Contraction` | `Contraction` | Edge-pair specification |
