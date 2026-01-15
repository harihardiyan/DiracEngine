# API Technical Reference

## Hamiltonian Construction
The engine uses a 4x4 basis ordered as:
`|A, up>, |B, up>, |A, down>, |B, down>`

### `velocity_HF(kx, ky, tau)`
Calculates the Hellmann-Feynman velocity. This is useful for checking the group velocity $v_g = \frac{1}{\hbar} \nabla_k E$.

## Wilson Loop Integration
The Berry charge $Q$ is calculated using:
$$W = \prod_{i} \langle u(k_i) | u(k_{i+1}) \rangle$$
Our implementation uses **Pure Phase Links** to avoid numerical instability when the overlap magnitude is small but the phase is well-defined.

### Stability Metrics
- `gap_eps`: If the spectral gap between bands is smaller than this value, the plaquette is "masked" (skipped) to prevent $0/0$ errors, and a warning is logged.
