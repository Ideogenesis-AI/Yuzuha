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

# Rust tests (cargo)
cargo test
```

The Python suite covers the full API including integration tests. The Rust suite covers
internal algorithms like CG coefficient computation and CGSpec logic.

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
