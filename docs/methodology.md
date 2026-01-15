# Methodological Rigor

## Wilson Loop Robustness
The Wilson loop approach is chosen because it remains well-defined even when the Berry connection is singular. By using link variables $U_{ij} = \exp(-i \phi)$, we maintain topological stability.

## Broadening ($\eta$) in Kubo Probe
We implement a small imaginary broadening $\eta = 10^{-10}$ eV. It is important to note that:
1. $\eta$ is used only for numerical stability of the diagnostic probe.
2. It does not define the topology of the system.
3. It is chosen to be several orders of magnitude smaller than the physical gaps being investigated.

## Multi-Grid Convergence (MGC)
Every $Q$ value reported is subjected to MGC. If $Q$ remains $0.0$ across grid sizes 21 to 41, we conclude with high confidence that the result is not a discretization error, but a converged physical state.
