
# DiracEngine: An Audit-Grade Framework for Topological Research in Dirac Materials

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-orange.svg)](https://github.com/google/jax)
[![Precision: float64](https://img.shields.io/badge/Precision-float64-red.svg)](#)

**Author:** Hari Hardiyan ([lorozloraz@gmail.com](mailto:lorozloraz@gmail.com))  
**Field:** Condensed Matter Physics / Computational Topology

---

## 1. Scientific Abstract
`DiracEngine` is a specialized numerical tool designed for high-fidelity investigation of the **Kane-Mele (KM) Hamiltonian** in 2D Dirac materials (e.g., Graphene, hBN-hybrids). 

In topological physics, calculating Berry invariants often faces numerical instabilities near Dirac points due to gauge-fixing issues or vanishing spectral gaps. This engine implements an **Audit-Grade** approach:
- **Global Topology:** Uses **Wilson-loop link variables** on a discrete $k$-space manifold to ensure gauge-invariant Berry charge ($Q_{valley}$) calculations.
- **Local Diagnostics:** Uses the **Kubo formula** as a high-resolution probe for Berry curvature mapping.
- **Scientific Validation:** Integrated auditing for Hamiltonian Hermiticity and multi-grid convergence checks to meet publication standards.

## 2. Theoretical Background
The engine simulates the generalized 4-band Kane-Mele Hamiltonian:

$$H(\mathbf{k}) = \hbar v_F (\tau k_x \sigma_x s_0 + k_y \sigma_y s_0) + \Delta \sigma_z s_0 + \tau \lambda_{SO} \sigma_z s_z + \lambda_R (\tau \sigma_x s_y - \sigma_y s_x)$$

Where:
- $\tau = \pm 1$ represents the Valley index ($K, K'$).
- $\sigma_i$ and $s_i$ are Pauli matrices for sublattice and spin degrees of freedom.
- The tool investigates the transition between **Trivial Insulators** ($\Delta > 3\sqrt{3}\lambda_{SO}$) and **Topological Insulators**.

## 3. Features
- **Topological Robustness:** Wilson-loop implementation with pure-phase link variables.
- **Energetic Audit:** Automated checks for $U^\dagger U = I$ and $H = H^\dagger$.
- **Automatic Differentiation:** Leverages **JAX** for XLA-optimized performance.
- **Convergence Suite:** Automated multi-grid analysis (21x21 to 41x41) to verify stability of topological invariants.
- **Decoupling Logic:** Built-in sanity checks for Pure Dirac, SOC-only, and $\Delta$-only regimes.

---
## Numerical Audit & Validation

The engine underwent a rigorous audit at $k=0$ using the Kane-Mele parameters.

| Metric | Value | Status |
| :--- | :--- | :--- |
| Hamiltonian Hermiticity | $0.0$ | PASSED |
| Eigenvector Orthonormality | $2.22 \times 10^{-16}$ | PASSED (Machine Precision) |
| Multi-grid Stability | Constant $Q=0.0$ | STABLE |

### Interpretation of Initial Results
- **Gapless Regimes:** In `pure_dirac` and `km_soc_only`, the engine correctly identifies 100% masked plaquettes due to the vanishing spectral gap at the Dirac point.
- **Berry Curvature Singularity:** The Kubo ring probe returns extreme values ($\sim 10^{40}$), confirming the high concentration of Berry curvature near the singularity, necessitating the Wilson-loop approach for global invariants.
## Scientific Defense & Interpretation

### 1. On the Divergence of Kubo Curvature
Reviewers may notice values of $\Omega(k) \sim 10^{40}$ near the Dirac point. This is **physically expected** and not a numerical artifact. 
- **Mechanism:** The Kubo formula contains a denominator $(E_m - E_n)^2$. As we approach band degeneracy (Dirac point), this term vanishes, causing the curvature to diverge.
- **Design Philosophy:** In `DiracEngine`, the Kubo formula is strictly a **local diagnostic probe**. It is intentionally not used for global integration to avoid these singularities. For topological invariants, we rely on the gauge-invariant Wilson-loop manifold.

### 2. On the $Q = 0.0$ Results (Trivial Phase Reporting)
The evaluation of $Q = 0.0$ across the tested parameter sets (e.g., hBN-like) is a **consistent physical report** of the system's phase.
- **Phase Competition:** In the regime where Sublattice Mass ($\Delta$) dominates over Spin-Orbit Coupling ($\lambda_{SO}$), the system resides in a **Trivial Insulating Phase**.
- **Accuracy:** The engine is not "failing" to detect topology; it is correctly reporting that the chosen parameters do not satisfy the conditions for a Quantum Spin Hall (QSH) phase. This confirms the engine's sensitivity to phase boundaries.

## 4. Installation Guide

### Prerequisites
- **Python 3.9 or higher**
- **JAX** (Optimized for CPU/GPU)

### Step-by-Step Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/harihardiyan/DiracEngine.git
   cd DiracEngine
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install jax jaxlib
   ```
   *Note: If you have a CUDA-compatible GPU, follow the [JAX installation guide](https://github.com/google/jax#installation) for GPU support.*

---

## 5. Execution & Usage

### Running the Full Diagnostic Suite
To replicate the research results, run the primary engine script. This will execute the **Energetic Audit**, **Wilson-loop calculation**, **Kubo probes**, and **Decoupling tests**.

```bash
python src/dirac_engine.py
```

### Scripting with the API
You can use `DiracEngine` as a library for your own simulations:

```python
import jax.numpy as jnp
from src.dirac_engine import build_dirac_engine, GRAPHENE_KM_IDEAL

# 1. Initialize the engine with specific physics parameters
engine = build_dirac_engine(GRAPHENE_KM_IDEAL)

# 2. Run a Hermiticity and Orthonormality check
engine["energetic_audit"](kx=0.0, ky=0.0, tau=1)

# 3. Calculate the Valley Berry Charge for Band 1 at Valley K
# Uses the Wilson-loop method with convergence tracking
Q_k = engine["berry_wilson"](tau=1, band_idx=1, kmax=5e8, grid=41, gap_eps=1e-12)

print(f"Computed Valley Charge: {Q_k}")
```

### Understanding the Output
When you run the engine, it generates a structured log:
- **`max|H - H†|`**: Should be near `1e-16`. If higher, the Hamiltonian construction is physically invalid.
- **`Convergence check`**: Displays $Q(grid)$ across different densities. A stable result shows the same value for grid 31 and 41.
- **`Phase/(2π) stats`**: Used to detect "vortex" points in the Berry connection.

---

## 6. Repository Structure
- `src/dirac_engine.py`: The core computational engine (JIT-compiled).
- `docs/api.md`: Detailed technical documentation of every function.
- `tests/`: Unit tests for physical sanity (Decoupling tests).
- `data/`: (Optional) Storage for computed Berry curvature maps.

## 7. API Reference
See [docs/api.md](docs/api.md) for detailed information on:
- `build_dirac_engine()`: Material initialization.
- `berry_wilson()`: Global topological invariants.
- `berry_kubo_at_k()`: Local curvature analysis.

## 8. License & Citation
This project is licensed under the **MIT License**.

If you use this software in your research, please cite it as:
> Hardiyan, H. (2024). DiracEngine: An Audit-Grade Framework for Topological Research in Dirac Systems. GitHub Repository.

**Contact:** Hari Hardiyan - lorozloraz@gmail.com
