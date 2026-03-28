<h1 align="center">
  <img src="docs/images/yuzuha.png" alt="Yuzuha SU(2) Protocol" width="300">
</h1>

<p align="center">
  <a href="https://pypi.org/project/yuzuha/"><img src="https://img.shields.io/pypi/v/yuzuha?color=red" alt="PyPI Version"></a>
  <a href="https://github.com/Ideogenesis-AI/Yuzuha/blob/stable/LICENSE"><img src="https://img.shields.io/github/license/Ideogenesis-AI/Yuzuha?color=orange" alt="License"></a>
  <a href="https://ideogenesis-ai.github.io/Yuzuha/"><img src="https://img.shields.io/badge/docs-github.io-c9a400" alt="Documentation"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/yuzuha?color=228b22" alt="Python Version"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.70+-blue?logo=rust&logoColor=white" alt="Rust"></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-%3E95%25-9400d3" alt="Coverage"></a>
  <a href="https://pypi.org/project/yuzuha/"><img src="https://img.shields.io/pypi/status/yuzuha?color=4b0082" alt="Status"></a>
</p>

Yuzuha is a high-performance library for computing SU(2) recoupling coefficients in tensor network applications, providing efficient implementations of X-symbols (recoupling coefficients for arbitrary tensor network contractions) and R-symbols (axes permutation). The library features automatic database caching for computed canonical bases, enabling significant performance improvements for repeated calculations. Available for both **Rust** and **Python**, and optimized as the recoupling engine for the [**Nicole**](https://github.com/Ideogenesis-AI/Nicole) tensor library.


## Features

- **X-symbols**: Complete computation for arbitrary tensor network contractions
- **R-symbols**: Tensor axis permutation transformations
- **Clebsch-Gordan (CG) coefficients**: Real-valued Condon-Shortley convention with automatic caching
- **Invariant metric**: Full support for arbitrary arrow directions via metric tensor
- **Outer multiplicity (OM) enumeration**: Systematic enumeration of internal spin configurations
- **CG database**: SQLite database for CG bases with automatic connection management
- **Python bindings**: High-performance Python interface with NumPy integration
- **Type safety**: Extensive use of Rust's type system for correctness
- **Comprehensive tests**: >95% code coverage with unit and integration tests


## Database Caching

Yuzuha automatically caches computed canonical basis data in an SQLite database for significant performance improvements on repeated calculations.

- **Persistent storage**: Cached data survives between program runs
- **Thread-safe**: Automatic connection management with mutex protection
- **Automatic**: No explicit connection management needed
- **Efficient**: Subsequent computations with same spin configurations are near-instantaneous

## Conventions

### Clebsch-Gordan Coefficients

Yuzuha uses the **Condon-Shortley convention** for Clebsch-Gordan (CG) coefficients, which describe the coupling of two angular momentum states:

$$C^{j_3 m_3}_{j_1 m_1, j_2 m_2} \equiv \langle j_1 m_1, j_2 m_2 | j_3 m_3 \rangle$$

These coefficients satisfy several important properties:

- **Real-valued**: All CG coefficients are real numbers in the Condon-Shortley convention
- **Orthonormality**: The coefficients form an orthonormal basis:
  $$\sum_{m_1, m_2} \langle j_1 m_1, j_2 m_2 | j_3 m_3 \rangle^2 = 1$$
- **Selection rules**: Non-zero coefficients require:
  - Magnetic quantum number conservation: $m_1 + m_2 = m_3$
  - Triangle inequality: $|j_1 - j_2| \leqslant j_3 \leqslant j_1 + j_2$
  - Integer total spin: $j_1 + j_2 + j_3 \in \mathbb{Z}$

The library automatically caches computed CG coefficients in an SQLite database for efficient reuse across calculations.

### Arrow Directions and the Invariant Metric

In tensor network diagrams, tensor edges can point either inward (incoming) or outward (outgoing), corresponding to primal and dual spaces, respectively. The **invariant metric** $g$ implements arrow reversal transformations:

$$g^{(j)}_{m,m'} = (-1)^{j-m} \delta_{m,-m'}$$

This metric tensor allows seamless conversion between incoming and outgoing edges while maintaining the correct sign conventions. Arrow inversion from incoming to outgoing uses the metric tensor $g$, while inversion from outgoing to incoming uses the inverse transformation $g^{-1}$.

### Fusion Trees

Yuzuha represents tensor network contractions using **left-associative fusion trees**, which provide a systematic way to couple multiple angular momenta:

- **Left-associative structure**: For $n$ spins, the coupling proceeds sequentially from left to right:
  $$(((j_1 \otimes j_2) \otimes j_3) \otimes \cdots) \otimes j_n$$
  
- **Internal spins** ($\alpha$): Each intermediate fusion step produces an internal spin quantum number. For $n$ external edges, there are $n-2$ internal spins that characterize the coupling path.

- **Outer Multiplicity (OM) index** ($\mu$): When multiple internal spin configurations (fusion patterns) lead to the same total coupling, they are distinguished by the OM index placed at the end of the canonical bases.

- **CG bases normalization**: The canonical basis is normalized with respect to the OM space. As a result, 3rd order CG bases acquire a scaling factor relative to the standard CG coefficients.

The left-associative structure ensures a unique canonical form for each tensor network, enabling efficient computation and caching of recoupling coefficients.


## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving algorithms, or enhancing documentation, your help is appreciated. You can also contribute by requesting new features or reporting performance issues.

Yuzuha is created and maintained by [Changkai Zhang](https://chx-zh.cc) as part of the Ideogenesis-AI effort in studying quantum many-body systems. If you have questions about contributing to the project or are interested in collaboration opportunities, please feel free to open an issue on GitHub or contact the maintainer directly.


## License

Yuzuha is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Yuzuha grow and benefit all users. See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html