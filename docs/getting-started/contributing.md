# Contributing

We welcome contributions from the community! Whether you are fixing bugs, adding features,
improving algorithms, or enhancing documentation, your help is appreciated.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please
[open an issue](https://github.com/Ideogenesis-AI/Yuzuha/issues) on GitHub. When reporting
a bug, include:

- Your Python and NumPy versions
- A minimal reproducible example
- The expected and actual output

### Submitting Pull Requests

1. Fork the repository on GitHub
2. Create a branch for your change
3. Make your changes, adding tests where appropriate
4. Run the test suite and ensure it passes
5. Open a pull request with a clear description of your changes

### Development Setup

Building from source requires Rust 1.85 or higher. The [installation guide](installation.md) covers the toolchain setup.

```bash
git clone https://github.com/Ideogenesis-AI/Yuzuha.git
cd Yuzuha

# Install dev dependencies
pip install -e ".[dev]"

# Build the Rust extension
maturin develop --release

# Run tests
pytest
```

### Running Tests

Yuzuha has two test suites:

```bash
# Python tests (pytest)
pytest

# Rust tests (cargo). The integration suite needs the test-utils feature,
# which exposes TestCacheGuard for cache isolation.
cargo test --features test-utils
```

The Python suite covers the full API including integration tests. The Rust suite covers
internal algorithms like CG coefficient computation and CGSpec logic.

### Where to Place a Rust Test

Rust tests live in two places, and which one you pick is determined by what the test
asserts, not by what it happens to call.

**Inline `#[cfg(test)] mod tests`, at the bottom of the source file**, holds unit tests of
a single function: constructors, accessors, `Display` and operator impls, and error
branches. Anything that touches a private item *must* go here, because `tests/rust/` can
only reach the public API. `get_db_path`, `init_db`, `serialize_spins` and
`reset_connection` in `src/builders/cache.rs`, and `build_permuted_spec` in
`src/builders/rsymbol.rs`, are only testable from inline modules.

**`tests/rust/`** holds tests of mathematical properties that span several functions:
orthonormality of the canonical basis, unitarity of R-symbols, F-symbol and X-symbol
consistency between equivalent contraction paths, Clebsch-Gordan values checked against
published tables, and concurrency behaviour. These tests exercise the crate the way a
downstream user would, through `yuzuha::` imports only.

Shared fixtures for the integration suite belong in `tests/rust/common.rs`. New
integration files must be registered in `tests/rust.rs`, which is the single entry point
Cargo compiles.

### Test Coverage

The project holds *production line coverage* above 95% for the Rust library, the Python
bindings, and the Python package independently. Production coverage excludes lines inside
inline `#[cfg(test)]` modules, so a file cannot raise its score by testing its own test
code.

```bash
# One-off tooling setup; Coverage.py comes with the dev extras
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview

# Rust library
cargo llvm-cov --features test-utils --summary-only

# Python package
coverage run --source yuzuha -m pytest && coverage report -m
```

Install `llvm-tools-preview` rather than relying on a system LLVM: the coverage format is
versioned with the compiler, and an `llvm-cov` from Xcode or Homebrew will refuse to read
profiles produced by `rustc`.

Two caveats when reading those numbers. `cargo llvm-cov` counts inline test modules, which
are themselves almost fully covered, so its figure is optimistic relative to the metric
above; use `--lcov` output and discard the lines at or after each file's first
`#[cfg(test)]` to reproduce the production figure. And `src/bindings.rs` is absent from
the report entirely, because it only compiles under the `python` feature — measuring it
means building an extension with `RUSTFLAGS="-C instrument-coverage"`, running pytest
against that build with `LLVM_PROFILE_FILE` set, and reporting the resulting profiles with
`llvm-cov`. Remember to reinstall a clean release build afterwards, since an instrumented
extension is slow and writes profile data on every run.

When adding code, check the uncovered lines rather than the percentage. A few lines are
genuinely unreachable and are expected to stay uncovered: the `?` arms on the infallible
`build_cg3` in `src/builders/builders.rs`, and the `InvalidMagneticNumber`,
`TriangleViolation` and `InvalidContraction` conversion arms in `src/bindings.rs`, which
nothing in the crate ever constructs on a path reachable from Python.

## Code Style

- **Rust**: Follow `rustfmt` formatting (`cargo fmt`)
- **Python**: Follow PEP 8

## Areas for Contribution

- **Performance**: Algorithmic improvements to CG basis construction or OM enumeration
- **Testing**: Additional edge cases or stress tests
- **Documentation**: Corrections, clarifications, or additional examples
- **Feature requests**: New utility functions, additional cache backends, etc.

## Contact

Yuzuha is created and maintained by [Changkai Zhang](https://chx-zh.cc) as part of the
Ideogenesis-AI effort. If you have questions about contributing or are interested in
collaboration, please open an issue or contact the maintainer directly.
