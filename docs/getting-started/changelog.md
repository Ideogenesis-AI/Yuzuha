# Changelog

## 0.1.4 — March 28, 2026

### What's New

#### Full MkDocs Documentation Site

A comprehensive documentation site is now published, covering the Getting Started
guide, full API Reference, and the formal Yuzuha Protocol specification. The site
is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and
includes MathJax for rendered equations, syntax-highlighted code blocks with copy
buttons, a custom landing page with a full-viewport hero image, and a three-option
light/dark/system colour palette.

The Protocol section includes a dedicated SU(2) Conventions sub-site covering fusion
conventions, canonical basis construction (with the Racah formula), and arrow
conventions — including derivations of the FS phase from bond inversion and the origin
of the conjugate phase.

#### Python Type Stubs (`yuzuha.pyi`)

A complete `.pyi` stub file for the Rust-compiled extension module has been added at
`python/yuzuha/yuzuha.pyi`, enabling full IDE auto-completion, inline parameter hints,
and static type-checking for all Rust-exposed types and functions.

#### Terminology: "Leg" → "Edge"

All occurrences of *leg* have been renamed to *edge* throughout the codebase for
consistency with standard tensor network literature. Two Rust error variants were
renamed accordingly:

| Old name | New name |
|---|---|
| `LegNotFound` | `EdgeNotFound` |
| `IncompatibleLegs` | `IncompatibleEdges` |

### Statistics

- **349 Python tests** (unchanged)
- **29 commits** since v0.1.3
- 1 new file: `python/yuzuha/yuzuha.pyi`

### API Changes

No new public API methods. Error variant renames are non-breaking — they are not
exposed through the Python bindings.

---

## 0.1.3 — March 26, 2026

### What's New

#### CGSpec Axis Inversion (`CGSpec.with_inverted_axes`)

A new method `with_inverted_axes(axes)` on `CGSpec` returns a copy of the spec with
the edge directions at the specified axis indices flipped (`Incoming ↔ Outgoing`). The
OM configurations (alphas) are reused unchanged because they depend only on spin values,
not edge directions.

- `CGSpec.with_inverted_axes(axes)` — returns a new `CGSpec` with the given edges
  direction-flipped; raises `ValueError` if any axis index is out of bounds
- Available in both the Rust API and the Python bindings

#### Bond Inversion Scheme in X-Symbol Tests

The X-symbol test suite has been substantially reorganised around a systematic bond
inversion scheme. Each test case now constructs a contracted tensor network, flips one
or more contracted edge pairs, and verifies that the resulting X-symbol agrees with the
original up to the expected Frobenius-Schur phase factor \((-1)^{2j}\) accumulated over
the flipped pairs.

#### Expanded Test Suite

26 net tests added (17 Python + 9 Rust):

| Suite | v0.1.2 | v0.1.3 | Change |
|---|---|---|---|
| `test_invert_axes` | — | 17 | +17 |
| `test_xsymbol` | 87 | 87 | 0 (refactored) |
| *(other Python suites unchanged)* | 245 | 245 | — |
| **Python total** | **332** | **349** | **+17** |
| `test_cgspec` (Rust) | 27 | 36 | +9 |
| *(other Rust suites unchanged)* | 84 | 84 | — |
| **Rust total** | **111** | **120** | **+9** |
| **Grand total** | **443** | **469** | **+26** |

### Statistics

- **469 tests** (349 Python + 120 Rust)
- **6 commits** since v0.1.2
- 1 new test file: `test_invert_axes.py`

### API Changes

One new method added to `CGSpec`; no breaking changes.

| Symbol | Kind | Change |
|---|---|---|
| `CGSpec.with_inverted_axes` | method | added — returns a new CGSpec with selected edge directions flipped |

---

## 0.1.2 — March 14, 2026

### What's New

#### CGSpec Dualization (`dualize` module)

A new `dualize` module implements conjugation of CGSpecs. The canonical conjugate
convention flips all edge directions relative to the original spec (first \(n-1\) edges
become Outgoing, last edge becomes Incoming), with an accumulated FS phase correction
for any edge deviating from that pattern.

- `compute_conjugate(spec)` — returns `(phase, conj_spec)`: the conjugated CGSpec with
  all edge directions flipped, and the cumulated FS phase factor (`+1.0` or `−1.0`)
- `fs_phase_for_spin(j)` — returns \((-1)^{2j}\) for a single spin

Both functions are exposed in the Python bindings and added to `__all__`.

#### FS Phase Removed from X-Symbol Computation

The integrated FS phase bookkeeping has been removed from `compute_xsymbol`. X-symbols
now return bare recoupling coefficients, consistent with the mathematical definition.
Users requiring FS phase correction should apply `compute_conjugate` or
`fs_phase_for_spin` explicitly.

The `fs_phase` module has been removed; its functionality is superseded by `dualize`.

#### Expanded Test Suite

59 net Python tests added (99 new, 40 removed); Rust suite unchanged:

| Suite | v0.1.1 | v0.1.2 | Change |
|---|---|---|---|
| `test_conjugate` | — | 60 | +60 |
| `test_xr_consistency` | — | 39 | +39 |
| `test_canonical_basis` | 97 | 57 | −40 |
| *(other Python suites unchanged)* | 176 | 176 | — |
| **Python total** | **273** | **332** | **+59** |
| Rust (all suites unchanged) | 111 | 111 | — |
| **Grand total** | **384** | **443** | **+59** |

### Statistics

- **443 tests** (332 Python + 111 Rust)
- **12 commits** since v0.1.1
- 1 new source file: `src/builders/dualize.rs`
- 1 removed source file: `src/builders/fs_phase.rs`
- 2 new test files: `test_conjugate.py`, `test_xr_consistency.py`

### API Changes

| Symbol | Kind | Change |
|---|---|---|
| `compute_conjugate` | function | added — conjugate CGSpec with FS phase |
| `fs_phase_for_spin` | function | added — standalone per-spin FS phase |
| `compute_fs_phase` | function | removed — superseded by `compute_conjugate` |

---

## 0.1.1 — February 26, 2026

### What's New

#### Frobenius-Schur Phase Bookkeeping

A new `fs_phase` module implements the FS phase factor \((-1)^{2j}\) that arises when
contracting a pair of edges in opposite canonical directions. The phase is accumulated
over all contracted pairs and integrated directly into `compute_xsymbol`:

- **First-region pairs** (both axes `< n−1`): phase applies when directions are
  `(Incoming, Outgoing)`
- **Last-edge pairs** (both axes `== n−1`): phase applies when directions are
  `(Outgoing, Incoming)`

`compute_fs_phase(spec_a, spec_b, contraction)` is also exposed as a standalone Python
function for inspection and testing.

#### Direction Class in Python

The `Direction` enum (`Incoming` / `Outgoing`) is now exported from the Python package,
making edge direction explicit in Python-level code.

#### Canonical Basis Generalised to n ≥ 2 Edges

The canonical basis builder and OM tensor construction have been refactored to correctly
handle all configurations with two or more external edges.

#### Expanded Test Suite

165 new tests added across Python and Rust. New test categories include:

- `Direction` class construction and behaviour
- X-symbol 2-edge and 3-edge consistency and inversion checks
- R-symbol two-edge configuration tests
- Canonical basis two-edge identity and cross-Gram orthogonality
- Canonical basis stress tests for large edge counts

### Statistics

- **384 tests** (273 Python + 111 Rust)
- **23 commits** since v0.1.0
- 1 new source file: `src/builders/fs_phase.rs`

### API Changes

| Symbol | Kind | Notes |
|---|---|---|
| `Direction` | type | `Incoming` / `Outgoing` enum, now exported to Python |
| `compute_fs_phase` | function | standalone FS phase computation |

---

## 0.1.0 — February 16, 2026

Initial stable release.

### Core Features

- **X-symbols** (`compute_xsymbol`): tensor contraction coefficients \(X^\gamma_{\alpha\beta}\)
  coupling two CG tensors into an output CG tensor, with full outer multiplicity support
- **R-symbols** (`compute_rsymbol`): tensor permutation coefficients encoding the effect
  of transposing edges in a CG tensor
- **Clebsch-Gordan coefficients** with Condon-Shortley phase convention
- Left-associative fusion-tree canonical basis for arbitrary edge configurations
- Full outer multiplicity (OM) enumeration via `CGSpec`
- Persistent SQLite caching for canonical basis data
- In-memory caching for X-symbol and R-symbol results with full lifecycle control
- PyO3-based Python bindings with ABI3 compatibility (Python ≥ 3.11)
- NumPy integration — all symbol tensors returned as `numpy.ndarray`

### Statistics

- **219 tests** (119 Python + 100 Rust)
- **62 commits** across 2 feature branches
- GPL-3.0 licence

### Initial API Surface

**Core types:** `Spin`, `Edge`, `CGSpec`, `Contraction`

**Computation:** `canonical_basis`, `compute_xsymbol`, `compute_rsymbol`

**Database & Cache:** `startup_database`, `set_cache_path`, `reset_caches`,
`clear_all_caches`, `get_cache_stats`, `print_cache_stats`, `TestCacheContext`
