# Triangle Rules

Triangle inequality and admissible spin range utilities.

## Description

The **triangle inequality** constrains which triplets \((j_1, j_2, j_3)\) can appear
as valid CG coupling arguments. Three spins satisfy the triangle inequality if and only
if they can be the sides of a triangle (in the usual sense), with the additional
requirement that their sum is an integer:

$$|j_1 - j_2| \leq j_3 \leq j_1 + j_2 \quad \text{and} \quad j_1 + j_2 + j_3 \in \mathbb{Z}$$

Yuzuha uses these rules internally to enumerate valid internal spin configurations when
constructing a `CGSpec` from a list of edges, and to validate `Contraction` inputs.

### Internal Rust Functions

The following functions are available in the Rust crate (`yuzuha::primitives`):

| Function | Description |
|----------|-------------|
| `triangle_check(j1, j2, j3)` | Returns `Ok(())` or `Err(YuzuhaError)` |
| `allowed_triangle(j1, j2)` | Iterator over valid \(j_3\) values |
| `min_coupled_spin(j1, j2)` | Minimum allowed \(j_3 = \lvert j_1 - j_2 \rvert\) |
| `max_coupled_spin(j1, j2)` | Maximum allowed \(j_3 = j_1 + j_2\) |
| `num_allowed_spins(j1, j2)` | Number of valid \(j_3\) values |

These are not directly exposed to Python but govern all coupling operations.

### Admissible Spin Range

For two spins \(j_1\) and \(j_2\), the range of valid coupled spins \(j_3\) is:

$$j_3 \in \{|j_1 - j_2|,\; |j_1 - j_2| + 1,\; \ldots,\; j_1 + j_2\}$$

The number of valid values is:

$$N = j_1 + j_2 - |j_1 - j_2| + 1 = 2\min(j_1, j_2) + 1$$

## Triangle Inequality in CGSpec Construction

When `CGSpec.from_edges` enumerates internal spin configurations:

1. The first internal spin \(\alpha_1\) must satisfy the triangle inequality for
   \((j_1, j_2)\): \(\alpha_1 \in [\,|j_1-j_2|,\; j_1+j_2\,]\)
2. Each subsequent \(\alpha_k\) must satisfy \(\alpha_k \in [\,|\alpha_{k-1}-j_{k+1}|,\;
   \alpha_{k-1}+j_{k+1}\,]\)
3. The final internal spin must equal the output spin \(j_n\) (fixed by the last edge)

## See Also

- [Clebsch-Gordan Coefficients](cg.md): CG coefficients are zero for triplets violating
  the triangle inequality
- [CGSpec](../core/cgspec.md): Uses the triangle inequality to enumerate `alphas`
- [Yuzuha Protocol — SU(2) Conventions](../../protocol/conventions.md): Triangle
  inequality as a conformance requirement

## Notes

The integer-sum condition \(j_1 + j_2 + j_3 \in \mathbb{Z}\) is automatically
satisfied when all spins are either all integers or contain an even number of
half-integers, which is always the case for valid SU(2) fusion trees.
