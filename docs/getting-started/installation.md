# Installation

Yuzuha is a Python library with a Rust core, distributed as a binary wheel on PyPI.
This guide covers installation options for both end users and contributors.

## Installing from PyPI

The recommended way to install Yuzuha is via pip:

```bash
pip install yuzuha
```

This installs the pre-compiled Rust extension and the Python package. The only runtime
dependency is NumPy 2.0 or higher.

### Requirements

- **Python**: 3.11 or higher
- **NumPy**: 2.0 or higher

## Installing from Source

If you need the latest development version or want to contribute to Yuzuha, build from
source using [Maturin](https://www.maturin.rs/).

### Prerequisites

Install Rust (1.70 or higher) and Maturin:

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Maturin
pip install maturin
```

### Building and Installing

```bash
# Clone the repository
git clone https://github.com/Ideogenesis-AI/Yuzuha.git
cd Yuzuha

# Build and install in development mode (editable install)
maturin develop --release

# Or install with test dependencies
pip install -e ".[dev]"
maturin develop --release
```

The `--release` flag enables Rust optimizations; omit it for faster compile times during
development (at the cost of runtime performance).

### Running Tests

```bash
pytest
```

## Verifying Installation

Verify that Yuzuha is installed and functioning correctly:

```python
import yuzuha

# Create a spin-1/2 representation
j = yuzuha.Spin(1)          # Spin(n) represents j = n/2, so Spin(1) = j=1/2
print(j.value())            # 0.5

# Build a 3rd-order CG specification
spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(j),
    yuzuha.Edge.incoming(j),
    yuzuha.Edge.outgoing(yuzuha.Spin(2)),  # j=1
])
print(spec.num_external())  # 3
print(spec.om_dimension())  # 1

# Compute the canonical basis
import numpy as np
basis = yuzuha.canonical_basis(spec)
print(basis.shape)          # (2, 2, 3, 1)
```

## Cache Directory

Yuzuha automatically creates a `.yuzuha/` directory in the current working directory to
store the SQLite databases for X-symbol, R-symbol, and canonical-basis caches. You can
override this location:

```bash
# Via environment variable
export YUZUHA_CACHE_PATH=/path/to/cache

# Or programmatically
import yuzuha
yuzuha.set_cache_path('/path/to/cache')
```

See [Cache Management](../api/caching/cache.md) for details.

## Troubleshooting

### Python Version

Ensure you are using Python 3.11 or higher:

```bash
python --version
```

### NumPy Version

Yuzuha requires NumPy 2.0 or higher for the ABI-stable extension interface:

```bash
pip install --upgrade numpy
```

### Rust Build Errors

If you see linker or compilation errors when building from source, ensure your Rust
toolchain is up to date:

```bash
rustup update stable
```

## Next Steps

Once installed, continue with:

- **[Core Concepts](core-concepts.md)**: Understand spins, edges, CG specifications, and
  the OM index
- **[API Reference](../api/index.md)**: Detailed documentation for all types and functions
