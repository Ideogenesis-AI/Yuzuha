# Fusion Conventions

## Left-Associative Fusion Trees

For an \(n\)th-order tensor with external representations \((j_1, j_2, \ldots, j_n)\),
the canonical fusion tree is **left-associative**:

$$\bigl(\cdots\bigl((j_1 \otimes j_2) \otimes j_3\bigr) \otimes \cdots\bigr) \otimes j_n$$

This uniquely determines the structure of the internal representation sequence
\((\alpha_1, \ldots, \alpha_{n-2})\):

- \(\alpha_1\) is an admissible representation in \(j_1 \otimes j_2\)
- \(\alpha_2\) is an admissible representation in \(\alpha_1 \otimes j_3\)
- \(\vdots\)
- \(\alpha_{n-2}\) is an admissible representation in \(\alpha_{n-3} \otimes j_{n-1}\),
  and must equal \(j_n\) (fixed by the terminal edge)

Any alternative fusion-tree ordering (e.g. right-associative, binary-tree) is **not**
conformant with this convention and will produce incompatible symbols.

## Canonical Ordering of Internal Representations

For a given set of external edges, the valid internal representation sequences are
enumerated in **lexicographic order** of the sequence \((\alpha_1, \ldots, \alpha_{n-2})\),
where each \(\alpha_k\) is compared using the total order defined by the `RepLabel`
type. The OM index \(\mu\) maps to this ordering: \(\mu = 0\) is the lexicographically
smallest valid sequence.
