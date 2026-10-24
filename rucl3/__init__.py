"""Quantum-classical crossover for the neutron spectrum of alpha-RuCl3."""

import os

# The matrix-free Hamiltonian runs on numba threads; idle BLAS threads that
# spin after small matrix products in the Krylov code slow it down by ~2x.
# This only takes effect if numpy has not been imported yet, so scripts
# import rucl3 first.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
