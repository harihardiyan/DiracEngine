#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiracEngine (AUDIT-GRADE, WILSON-ONLY TOPOLOGY, KUBO DIAGNOSTIC)

Main upgrades compared to the previous version:
- Global valley charge is computed *only* via the Wilson loop (berry_wilson).
- The Kubo formula is used *only* as a local diagnostic probe, with a small eta broadening.
- The k-space patch and ring radius are chosen to remain within the linear Dirac regime
  (|k| << 1/a for graphene) while avoiding the exact k=0 singularity.

Additional audit-oriented features:
  * Energetic audit:
      - Hermiticity of H
      - Orthonormality of eigenvectors
  * Wilson patch diagnostics:
      - Minimum band gap on the patch
      - Number and fraction of masked plaquettes (where the gap is too small)
      - Basic statistics of phase/(2π) over plaquettes (min, max, mean)
  * Multi-grid convergence check for the Wilson valley charge:
      - Evaluate Q on grids 21, 25, 31, and a chosen grid (default 41)
  * Decoupling sanity tests in run_decoupling_tests:
      - Turn off Δ, λ_SO, λ_R in various combinations to check expected behavior.
"""

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass

# --------------------------------------------------------------------
# Constants and Pauli matrices
# --------------------------------------------------------------------
hbar     = 1.054571817e-34
eV_to_J  = 1.602176634e-19

def pauli_x():
    return jnp.array([[0.0, 1.0],
                      [1.0, 0.0]], dtype=jnp.complex128)

def pauli_y():
    return jnp.array([[0.0, -1.0j],
                      [1.0j, 0.0]], dtype=jnp.complex128)

def pauli_z():
    return jnp.array([[1.0, 0.0],
                      [0.0, -1.0]], dtype=jnp.complex128)

def pauli_0():
    return jnp.eye(2, dtype=jnp.complex128)

σx = pauli_x()
σy = pauli_y()
σz = pauli_z()
σ0 = pauli_0()

sx = σx
sy = σy
sz = σz
s0 = σ0

# --------------------------------------------------------------------
# Spec (energies in eV)
# --------------------------------------------------------------------
@dataclass(frozen=True)
class DiracGrapheneSpec:
    name: str
    vF: float        # Fermi velocity in m/s
    Delta: float     # Sublattice mass term in eV
    lambda_SO: float # Intrinsic spin-orbit coupling in eV
    lambda_R: float  # Rashba spin-orbit coupling in eV

    def __repr__(self):
        return (f"DiracGrapheneSpec(name='{self.name}', vF={self.vF:.1e} m/s, "
                f"Delta={self.Delta:.2e} eV, lambda_SO={self.lambda_SO:.2e} eV, "
                f"lambda_R={self.lambda_R:.2e} eV)")

GRAPHENE_KM_IDEAL = DiracGrapheneSpec(
    name       = "graphene_kane_mele_ideal",
    vF         = 1.0e6,
    Delta      = 0.0,
    lambda_SO  = 1.0e-3,
    lambda_R   = 5.0e-3,
)

HBN_LIKE_KM = DiracGrapheneSpec(
    name       = "graphene_on_hBN_like",
    vF         = 0.9e6,
    Delta      = 50e-3,
    lambda_SO  = 5.0e-3,
    lambda_R   = 5.0e-3,
)

# --------------------------------------------------------------------
# Engine builder
# --------------------------------------------------------------------
def build_dirac_engine(spec: DiracGrapheneSpec):
    """
    Build a Dirac-Kane-Mele engine with:
    - H(k) 4x4 Hamiltonian near each valley τ = ±1
    - energies(kx, ky, tau): band energies
    - eigsys(kx, ky, tau): eigenvalues and eigenvectors
    - velocity_HF: Hellmann-Feynman band velocities
    - berry_wilson: Wilson-loop Berry charge on a k-space patch (global, topological)
    - berry_kubo_at_k: local Berry curvature via Kubo formula (diagnostic only)
    - probe_kubo_on_ring / probe_kubo_along_line: local sampling of Ω_n(k)
    - energetic_audit: checks hermiticity and orthonormality at a given k, τ
    """
    vF  = jnp.asarray(spec.vF, dtype=jnp.float64)
    Δ   = jnp.asarray(spec.Delta * eV_to_J, dtype=jnp.float64)
    lSO = jnp.asarray(spec.lambda_SO * eV_to_J, dtype=jnp.float64)
    lR  = jnp.asarray(spec.lambda_R * eV_to_J, dtype=jnp.float64)

    # Prebuilt 4x4 matrices in orbital ⊗ spin space
    σx_s0 = jnp.kron(σx, s0)
    σy_s0 = jnp.kron(σy, s0)
    σz_s0 = jnp.kron(σz, s0)
    σz_sz = jnp.kron(σz, sz)
    σx_sy = jnp.kron(σx, sy)
    σy_sx = jnp.kron(σy, sx)

    # Hellmann-Feynman velocity operators (dH/dk)
    dHx = hbar * vF * σx_s0
    dHy = hbar * vF * σy_s0

    # Kubo velocity operators
    vx_base = vF * σx_s0   # v_x = τ * vx_base
    vy_op   = vF * σy_s0   # v_y independent of τ

    # Small broadening for Kubo stability (diagnostic only, does not define topology)
    eta_kubo = 1e-10 * eV_to_J

    # ---------------- H, energies, eigsys ----------------
    @jit
    def H(kx, ky, tau):
        """
        Dirac-Kane-Mele Hamiltonian near valley τ = ±1.

        Terms:
        - H0: linear Dirac kinetic term ~ vF (τ kx σ_x + ky σ_y)
        - HΔ: sublattice mass term Δ σ_z
        - HSO: intrinsic SOC τ λ_SO σ_z ⊗ s_z
        - HR: Rashba SOC with a specific sign convention

        Rashba sign convention:
        HR = τ λ_R σ_x ⊗ s_y - λ_R σ_y ⊗ s_x

        This convention is chosen to be consistent with a standard Kane-Mele-like
        form and to preserve the expected valley and spin symmetries. The exact
        sign convention is not unique in the literature, but once fixed, it is
        used consistently throughout this engine.
        """
        H0  = hbar * vF * (tau * kx * σx_s0 + ky * σy_s0)
        HΔ  = Δ   * σz_s0
        HSO = tau * lSO * σz_sz
        HR  = tau * lR  * σx_sy - lR * σy_sx
        return (H0 + HΔ + HSO + HR).astype(jnp.complex128)

    @jit
    def energies(kx, ky, tau):
        """Return the band energies (eigenvalues) of H(k, τ)."""
        return jnp.linalg.eigvalsh(H(kx, ky, tau))

    @jit
    def eigsys(kx, ky, tau):
        """Return eigenvalues and eigenvectors of H(k, τ)."""
        return jnp.linalg.eigh(H(kx, ky, tau))

    # ---------------- HF velocity ----------------
    band_indices = jnp.arange(4)

    @jit
    def velocity_HF(kx, ky, tau):
        """
        Compute band velocities via the Hellmann-Feynman theorem:
        v_n = (1/ħ) ⟨u_n | ∂H/∂k | u_n⟩.

        Returns:
        - evals: band energies
        - v: array of shape (4, 2) with [v_x, v_y] for each band.
        """
        evals, evecs = eigsys(kx, ky, tau)

        def vel(n):
            u = evecs[:, n]
            vx = jnp.real(jnp.vdot(u, dHx @ u)) / hbar
            vy = jnp.real(jnp.vdot(u, dHy @ u)) / hbar
            return jnp.array([vx, vy])

        return evals, vmap(vel)(band_indices)

    # ---------------- Energetic audit ----------------
    def energetic_audit(kx=0.0, ky=0.0, tau=+1):
        """
        Energetic audit at a given k and valley τ:
        - Check hermiticity of H: max|H - H†|
        - Check orthonormality of eigenvectors: max|U†U - I|

        This is a structural sanity check to ensure:
        - No hidden non-Hermitian artifacts
        - Eigenvectors form an orthonormal basis
        """
        Hk = H(kx, ky, tau)
        evals, evecs = eigsys(kx, ky, tau)

        # Hermiticity check
        herm_diff = jnp.max(jnp.abs(Hk - jnp.conj(Hk.T)))

        # Orthonormality check: U^† U = I
        U = evecs
        UdU = jnp.conj(U.T) @ U
        ortho_diff = jnp.max(jnp.abs(UdU - jnp.eye(4, dtype=jnp.complex128)))

        print("Energetic audit at k=({:.3e},{:.3e}), tau={:+d}".format(kx, ky, tau))
        print("  max|H - H†|      =", float(herm_diff))
        print("  max|U†U - I|     =", float(ortho_diff))

    # ---------------- Wilson Berry (NO JIT) ----------------
    def berry_wilson(tau, band_idx, kmax, grid, gap_eps):
        """
        Wilson-loop valley Berry charge on a rectangular patch around k=0.

        This is the *only* estimator used for the global valley Berry charge
        in this engine. It is gauge-sensitive but topologically robust when
        the band is well isolated.

        Parameters:
        - tau: valley index (+1 or -1)
        - band_idx: band index (0..3)
        - kmax: half-width of the square patch in k-space
        - grid: number of k-points along each direction (grid x grid)
        - gap_eps: minimum allowed band gap (in Joules) to trust the band isolation

        Implementation details:
        - Uses a Wilson loop over plaquettes in k-space.
        - Each plaquette contributes a phase from the product of link variables.
        - The phase is converted to a winding number via rounding of phase/(2π).
        - Plaquettes where the band gap is below gap_eps are masked (contribute 0).
        - Diagnostics:
            * Minimum band gap on the patch
            * Number and fraction of masked plaquettes
            * Statistics of phase/(2π) over all plaquettes
        """
        ks = jnp.linspace(-kmax, kmax, grid)
        KX, KY = jnp.meshgrid(ks, ks, indexing="ij")

        # Compute eigensystem on the grid
        def row(kx_row, ky_row):
            return vmap(lambda kx, ky: eigsys(kx, ky, tau))(kx_row, ky_row)

        evals_grid, evecs_grid = vmap(row, in_axes=(0,0))(KX, KY)

        # Band-aware gap: minimum gap to neighboring bands
        gaps_below = jnp.where(
            band_idx > 0,
            evals_grid[..., band_idx] - evals_grid[..., band_idx-1],
            jnp.inf
        )
        gaps_above = jnp.where(
            band_idx < 3,
            evals_grid[..., band_idx+1] - evals_grid[..., band_idx],
            jnp.inf
        )
        gap_rel = jnp.minimum(gaps_below, gaps_above)

        # Global minimum gap on the patch (for audit)
        min_gap_patch = float(jnp.min(gap_rel))
        print(f"  [Wilson] min gap on patch (band {band_idx}) = {min_gap_patch:.3e} J")

        n = grid
        I = jnp.arange(n-1)
        J = jnp.arange(n-1)
        II, JJ = jnp.meshgrid(I, J, indexing="ij")

        # Collect phase/(2π) for diagnostics (Python list for convenience)
        phase_over_2pi_list = []

        def plaquette(i, j):
            """
            Compute the Wilson loop phase and winding for a single plaquette,
            and determine whether it should be masked based on the local gap.
            """
            gmin = jnp.min(jnp.array([
                gap_rel[i,j], gap_rel[i+1,j],
                gap_rel[i,j+1], gap_rel[i+1,j+1]
            ]))

            u00 = evecs_grid[i,   j,   :, band_idx]
            u10 = evecs_grid[i+1, j,   :, band_idx]
            u01 = evecs_grid[i,   j+1, :, band_idx]
            u11 = evecs_grid[i+1, j+1, :, band_idx]

            def link(a, b):
                ov = jnp.vdot(a, b)
                phase = jnp.angle(ov)
                return jnp.exp(-1j * phase)  # pure phase link

            W = link(u00, u10) * link(u10, u11) * jnp.conj(link(u01, u11)) * jnp.conj(link(u00, u01))
            phase = jnp.angle(W)
            phase_over_2pi = phase / (2 * jnp.pi)

            # Store for diagnostics (outside JAX tracing)
            phase_over_2pi_list.append(float(phase_over_2pi))

            # Winding number via rounding; robust if phase is near multiples of 2π
            winding = jnp.round(phase_over_2pi)
            return jnp.where(gmin < gap_eps, 0.0, winding), gmin < gap_eps

        # Manual loop over plaquettes to accumulate windings and mask flags
        windings = []
        masked_flags = []
        for i in range(n-1):
            for j in range(n-1):
                w, masked = plaquette(i, j)
                windings.append(w)
                masked_flags.append(bool(masked))

        windings = jnp.array(windings)
        masked_flags = jnp.array(masked_flags)

        # Diagnostics: number of plaquettes, masked fraction, phase statistics
        num_plaquettes = (n-1)*(n-1)
        num_masked = int(jnp.sum(masked_flags))
        frac_masked = num_masked / num_plaquettes if num_plaquettes > 0 else 0.0

        phase_over_2pi_arr = jnp.array(phase_over_2pi_list)
        phase_min = float(jnp.min(phase_over_2pi_arr))
        phase_max = float(jnp.max(phase_over_2pi_arr))
        phase_mean = float(jnp.mean(phase_over_2pi_arr))

        print(f"  [Wilson] plaquettes total     = {num_plaquettes}")
        print(f"  [Wilson] plaquettes masked    = {num_masked} ({frac_masked:.3f} of total)")
        print(f"  [Wilson] phase/(2π) stats     = min {phase_min:.3f}, max {phase_max:.3f}, mean {phase_mean:.3f}")

        # Sum of windings gives the Berry charge on the patch
        return float(jnp.sum(windings))

    # ---------------- Kubo Berry (LOCAL ONLY) ----------------
    @jit
    def berry_kubo_at_k(kx, ky, tau, band_idx, gap_eps_kubo):
        """
        Local Berry curvature Ω_n(k) via the Kubo formula (diagnostic only).

        This is *not* used for global integration or topological classification.
        It is only used as a local probe to inspect the behavior of Ω_n(k)
        away from degeneracies.

        Formula (schematically):
        Ω_n(k) ~ -2 Im Σ_{m≠n} [ ⟨n|v_x|m⟩ ⟨m|v_y|n⟩ / (E_m - E_n)^2 ]

        Implementation details:
        - A small imaginary broadening eta_kubo is added in the denominator
          to avoid numerical instabilities near exact degeneracies.
        - States with |E_m - E_n| < gap_eps_kubo are masked out.
        """
        evals, evecs = eigsys(kx, ky, tau)

        vx = tau * vx_base
        vy = vy_op

        U = evecs
        Ud = jnp.conj(U.T)

        Vx = Ud @ (vx @ U)
        Vy = Ud @ (vy @ U)

        En = evals[band_idx]
        dE = evals - En

        mask = (jnp.arange(4) != band_idx) & (jnp.abs(dE) > gap_eps_kubo)

        num = Vx[band_idx,:] * Vy[:,band_idx]
        denom = dE**2 + 1j * eta_kubo  # small broadening to avoid exact zero denominators

        return -2 * jnp.imag(jnp.sum(jnp.where(mask, num / denom, 0)))

    # ---------------- Kubo probes (optional helpers) ----------------
    def probe_kubo_on_ring(k0, tau, band_idx,
                           radius=5e8, n_theta=128,
                           gap_eps_kubo=1e-3 * eV_to_J):
        """
        Sample Ω_n(k) along a ring of fixed radius around a center k0.

        Parameters:
        - k0: center in k-space, array-like of shape (2,) (kx0, ky0)
        - tau: valley index (+1 or -1)
        - band_idx: band index (0..3)
        - radius: |k - k0|, chosen such that:
            * |k| remains within the linear Dirac regime (|k| << 1/a),
            * but not exactly at k=0 to avoid the Dirac point singularity.
        - n_theta: number of angular samples along the ring
        - gap_eps_kubo: minimum energy separation for including states in the Kubo sum

        Returns:
        - thetas: angles along the ring
        - omegas: sampled Ω_n(k) values along the ring
        """
        k0 = jnp.asarray(k0, dtype=jnp.float64)
        thetas = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
        ks = k0[None,:] + radius * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=1)

        def omega_at_theta(k):
            return berry_kubo_at_k(k[0], k[1], tau, band_idx, gap_eps_kubo)

        omegas = vmap(omega_at_theta)(ks)
        return thetas, omegas

    def probe_kubo_along_line(k_start, k_end, tau, band_idx,
                              n_points=200,
                              gap_eps_kubo=1e-3 * eV_to_J):
        """
        Sample Ω_n(k) along a straight line segment in k-space.

        Parameters:
        - k_start: starting point in k-space, shape (2,)
        - k_end: ending point in k-space, shape (2,)
        - tau: valley index (+1 or -1)
        - band_idx: band index (0..3)
        - n_points: number of sampling points along the line
        - gap_eps_kubo: minimum energy separation for including states in the Kubo sum

        Returns:
        - ks: array of shape (n_points, 2) with sampled k-points
        - omegas: array of shape (n_points,) with Ω_n(k) along the line
        """
        k_start = jnp.asarray(k_start, dtype=jnp.float64)
        k_end   = jnp.asarray(k_end, dtype=jnp.float64)
        ts = jnp.linspace(0.0, 1.0, n_points)
        ks = (1.0 - ts)[:,None] * k_start[None,:] + ts[:,None] * k_end[None,:]

        def omega_at_k(k):
            return berry_kubo_at_k(k[0], k[1], tau, band_idx, gap_eps_kubo)

        omegas = vmap(omega_at_k)(ks)
        return ks, omegas

    return {
        "spec": spec,
        "H": H,
        "energies": energies,
        "eigsys": eigsys,
        "velocity_HF": velocity_HF,
        "berry_wilson": berry_wilson,
        "berry_kubo_at_k": berry_kubo_at_k,
        "probe_kubo_on_ring": probe_kubo_on_ring,
        "probe_kubo_along_line": probe_kubo_along_line,
        "energetic_audit": energetic_audit,
    }

# --------------------------------------------------------------------
# Unified valley_berry (Wilson only, with multi-grid convergence)
# --------------------------------------------------------------------
def valley_berry(engine,
                 tau,
                 band_idx,
                 kmax=5e8,
                 grid=41,
                 gap_eps=1e-6*eV_to_J):
    """
    Valley Berry charge on a k-space patch via the Wilson loop only.

    This is the *only* estimator for the valley Berry charge Q_valley in this engine.

    Convergence strategy:
    - Evaluate Q on several grids: 21, 25, 31, and a chosen grid (default 41).
    - Print a small table of Q(grid) and |Q(grid) - Q_ref|, where Q_ref is the
      value at the finest grid.
    - This provides a simple but explicit convergence audit for the Wilson result.
    """
    grids_to_check = [21, 25, 31, grid]
    Qs = []

    print(f"Convergence check for valley_berry (tau={tau}, band={band_idx}):")
    for g in grids_to_check:
        print(f"  -- grid = {g} --")
        Qg = engine["berry_wilson"](tau, band_idx, kmax, g, gap_eps)
        Qs.append(Qg)

    Qs = jnp.array(Qs)
    Q_ref = Qs[-1]

    print("  Grid   Q(grid)    |Q(grid) - Q_ref|")
    for g, Qg in zip(grids_to_check, Qs):
        diff = abs(float(Qg - Q_ref))
        print(f"  {g:4d}  {float(Qg):+7.4f}      {diff:7.4f}")

    return float(Q_ref)

# --------------------------------------------------------------------
# Decoupling sanity tests
# --------------------------------------------------------------------
def run_decoupling_tests():
    """
    Sanity checks by decoupling different terms in the Hamiltonian.

    The idea is to construct simplified models and see whether the Wilson
    valley charges behave qualitatively as expected:

    - Pure Dirac (Δ = λ_SO = λ_R = 0):
        * Gapless Dirac cones, symmetric patch around k=0.
        * Expectation: Q ≈ 0 for each valley on a symmetric patch (no net Berry charge).

    - KM-like with SOC only (Δ = 0, λ_SO ≠ 0, λ_R = 0):
        * Intrinsic SOC opens a topological gap.
        * Expectation: nontrivial valley/spin structure; Q changes sign under τ → -τ.

    - Δ-only (Δ ≠ 0, λ_SO = λ_R = 0):
        * Sublattice mass opens a trivial gap.
        * Expectation: valley charges may cancel in a way consistent with a trivial insulator.

    Here we only print the results; you can later add explicit target values
    or assertions if you want to turn this into a formal test suite.
    """
    base = GRAPHENE_KM_IDEAL

    # Pure Dirac: no mass, no SOC, no Rashba
    pure_dirac = DiracGrapheneSpec(
        name       = "pure_dirac",
        vF         = base.vF,
        Delta      = 0.0,
        lambda_SO  = 0.0,
        lambda_R   = 0.0,
    )

    # KM-like (SOC only): intrinsic SOC, no mass, no Rashba
    km_soc_only = DiracGrapheneSpec(
        name       = "km_soc_only",
        vF         = base.vF,
        Delta      = 0.0,
        lambda_SO  = base.lambda_SO,
        lambda_R   = 0.0,
    )

    # Δ-only: sublattice mass, no SOC, no Rashba
    delta_only = DiracGrapheneSpec(
        name       = "delta_only",
        vF         = base.vF,
        Delta      = 50e-3,
        lambda_SO  = 0.0,
        lambda_R   = 0.0,
    )

    specs = [pure_dirac, km_soc_only, delta_only]

    for spec in specs:
        print(f"\n=== Decoupling test: {spec.name} ===")
        print(spec)
        engine = build_dirac_engine(spec)
        # Energetic audit at k=0, τ=+1
        engine["energetic_audit"](0.0, 0.0, +1)

        QK  = valley_berry(engine, +1, 1, kmax=5e8, grid=41)
        QKp = valley_berry(engine, -1, 1, kmax=5e8, grid=41)
        print("  Wilson valley charges (band 1):")
        print("    K  :", QK)
        print("    K':", QKp)
        print("    sum =", QK+QKp)

# --------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------
def run_all():
    """
    Main demonstration routine:
    - Builds engines for two representative specs:
        * GRAPHENE_KM_IDEAL
        * HBN_LIKE_KM
    - Runs an energetic audit at k=0, τ=+1
    - Computes valley Berry charges at K and K' using the Wilson method
      with a multi-grid convergence check
    - Probes the local Kubo Berry curvature on a ring around k=0
    - Finally, runs decoupling sanity tests
    """
    for spec in [GRAPHENE_KM_IDEAL, HBN_LIKE_KM]:
        print(f"\n=== {spec.name} ===")
        print(spec)  # uses __repr__
        engine = build_dirac_engine(spec)

        # Energetic audit at k=0, τ=+1
        engine["energetic_audit"](0.0, 0.0, +1)

        QK  = valley_berry(engine, +1, 1, kmax=5e8, grid=41)
        QKp = valley_berry(engine, -1, 1, kmax=5e8, grid=41)
        print("Wilson valley charges (band 1):")
        print("  K :", QK)
        print("  K':", QKp)
        print("  sum =", QK+QKp)

        # Example of local Kubo probe around k=0, τ=+1
        k0 = jnp.array([0.0, 0.0])
        thetas, omega_ring = engine["probe_kubo_on_ring"](k0, tau=+1, band_idx=1,
                                                          radius=5e8, n_theta=64)
        mean_omega = float(jnp.mean(omega_ring))
        max_abs_omega = float(jnp.max(jnp.abs(omega_ring)))
        print("  Kubo ring probe (mean Ω on ring):", mean_omega)
        print("  Kubo ring probe (max |Ω| on ring):", max_abs_omega)

    # Run decoupling sanity tests as part of the global audit
    print("\n=== Running decoupling sanity tests ===")
    run_decoupling_tests()

if __name__ == "__main__":
    run_all()
