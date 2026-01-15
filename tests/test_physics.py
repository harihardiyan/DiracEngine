import pytest
import jax.numpy as jnp
from src.engine import build_dirac_engine, DiracGrapheneSpec

def test_pure_dirac_topology():
    """Test: Pure Dirac (no SOC, no Delta) must have zero Berry charge."""
    spec = DiracGrapheneSpec(
        name="test_pure", vF=1e6, Delta=0.0, lambda_SO=0.0, lambda_R=0.0
    )
    engine = build_dirac_engine(spec)
    # Band 1, Valley K
    Q = engine["berry_wilson"](tau=1, band_idx=1, kmax=1e8, grid=21, gap_eps=1e-12)
    # Harusnya sangat dekat dengan 0
    assert abs(Q) < 1e-5

def test_hermiticity():
    """Test: Hamiltonian must be Hermitian at arbitrary k."""
    spec = DiracGrapheneSpec(
        name="test_herm", vF=1e6, Delta=0.01, lambda_SO=0.01, lambda_R=0.01
    )
    engine = build_dirac_engine(spec)
    Hk = engine["H"](kx=123.45, ky=67.89, tau=1)
    diff = jnp.max(jnp.abs(Hk - jnp.conj(Hk.T)))
    assert diff < 1e-12
