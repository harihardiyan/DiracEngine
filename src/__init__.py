# Mengangkat fungsi utama dari engine.py ke level package
from .engine import (
    build_dirac_engine,
    DiracGrapheneSpec,
    GRAPHENE_KM_IDEAL,
    HBN_LIKE_KM,
    valley_berry
)

# Metadata riset
__version__ = "1.0.0"
__author__ = "Hari Hardiyan"

# Ini opsional, fungsinya agar saat orang menulis 'from src import *', 
# hanya fungsi ini yang terambil.
__all__ = [
    "build_dirac_engine",
    "DiracGrapheneSpec",
    "GRAPHENE_KM_IDEAL",
    "HBN_LIKE_KM",
    "valley_berry"
]
