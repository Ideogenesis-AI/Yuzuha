# Triangle Rules

The triangle rules decide which spin triplets can couple at all, and so they decide the
shape of everything downstream: they allow `CGSpec.from_edges` to enumerate internal
spin paths, they fix a spec's outer-multiplicity dimension, and they are what a
`Contraction` is validated against. Like the CG coefficients, they live in the Rust layer
and have no Python binding.

## Description

Three spins satisfy the **triangle inequality** if and only if they can be the sides of a
triangle in the usual sense, with the extra requirement that their sum be an integer:

$$|j_1 - j_2| \leq j_3 \leq j_1 + j_2 \quad \text{and} \quad j_1 + j_2 + j_3 \in \mathbb{Z}$$

For a fixed pair \(j_1, j_2\), the admissible coupled spins therefore form a contiguous
ladder in unit steps,

$$j_3 \in \{|j_1 - j_2|,\; |j_1 - j_2| + 1,\; \ldots,\; j_1 + j_2\},$$

whose length is \(N = j_1 + j_2 - |j_1 - j_2| + 1 = 2\min(j_1, j_2) + 1\). That count is
the branching factor at one node of a fusion tree, and the product of such factors over
the tree bounds the outer multiplicity of a spec.

### Enumeration in CGSpec Construction

`CGSpec.from_edges` walks the left-associative fusion tree, applying the ladder above at
each step and discarding paths that dead-end:

1. The first internal spin \(\alpha_1\) ranges over the admissible spins of
   \((j_1, j_2)\)
2. Each subsequent \(\alpha_k\) ranges over the admissible spins of
   \((\alpha_{k-1}, j_{k+1})\)
3. The last coupling must land exactly on the terminal edge's spin \(j_n\)

Step 3 is why a spec can have an OM dimension smaller than the naive product of branching
factors, and why an edge list can be rejected outright: if no path survives, `from_edges`
raises `ValueError`.

### Internal Rust Functions

The crate exposes these in `yuzuha::primitives`:

| Function | Description |
|----------|-------------|
| `triangle_check(j1, j2, j3)` | Returns `Ok(())` or `Err(YuzuhaError)` |
| `allowed_triangle(j1, j2)` | Iterator over valid \(j_3\) values |
| `min_coupled_spin(j1, j2)` | Minimum allowed \(j_3 = \lvert j_1 - j_2 \rvert\) |
| `max_coupled_spin(j1, j2)` | Maximum allowed \(j_3 = j_1 + j_2\) |
| `num_allowed_spins(j1, j2)` | Number of valid \(j_3\) values |

## See Also

- [Clebsch-Gordan Coefficients](cg.md): CG coefficients are zero for triplets violating
  the triangle inequality
- [CGSpec](../core/cgspec.md): Uses the triangle inequality to enumerate internal spins
- [Yuzuha Protocol — Fusion Conventions](../../protocol/conventions/fusion.md): Triangle
  inequality as a conformance requirement

## Notes

The integer-sum condition \(j_1 + j_2 + j_3 \in \mathbb{Z}\) rarely has to be checked
separately in practice. It holds automatically whenever the three spins are all integers
or contain exactly two half-integers, which is the case for every triplet a valid SU(2)
fusion tree produces.
