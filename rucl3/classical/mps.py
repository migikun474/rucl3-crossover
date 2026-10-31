"""Matrix-product-state baseline (TeNPy): DMRG ground state and two-site TDVP
for <S_i^a(t) S_j^a(0)> on honeycomb cylinders.

The MPO is built from exactly the bond list used by the exact solver (via
:class:`rucl3.model.lattice.Cylinder`), so the two can be compared term by
term.  The cost that matters for the crossover is how the bond dimension chi
needed for a given accuracy grows with simulated time and with the
circumference, and the time per TDVP step at that chi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..model.hamiltonian import SpinModel
from ..model.lattice import Cylinder
from ..model.params import ParameterSet

AX = "xyz"


def tenpy_model(cyl: Cylinder, params: ParameterSet):
    """TeNPy MPO model with the same couplings as SpinModel(cyl, params)."""
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    bonds = SpinModel(cyl, params).bonds

    class _Bonds(CouplingMPOModel):
        def init_sites(self, opts):
            return SpinHalfSite(conserve=None)

        def init_lattice(self, opts):
            return Chain(cyl.N, self.init_sites(opts), bc="open", bc_MPS="finite")

        def init_terms(self, opts):
            for i, j, _, J in bonds:
                if i > j:  # TeNPy wants i < j; S_i^a S_j^b = S_j^b S_i^a
                    i, j, J = j, i, J.T
                for a in range(3):
                    for b in range(3):
                        if abs(J[a, b]) > 1e-14:
                            self.add_coupling_term(J[a, b], i, j, "S" + AX[a], "S" + AX[b])

    return _Bonds({"verbose": 0})


def ground_state(M, chi_max: int = 256, seed: int = 0):
    """DMRG ground state (random product start)."""
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS

    rng = np.random.default_rng(seed)
    N = M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), list(rng.integers(0, 2, N)), bc="finite")
    t0 = time.perf_counter()
    info = dmrg.run(psi, M, {"mixer": True, "max_E_err": 1e-10, "max_sweeps": 40,
                             "trunc_params": {"chi_max": chi_max, "svd_min": 1e-10}})
    return psi, float(info["E"]), time.perf_counter() - t0


@dataclass
class MPSCorrelations:
    """C[i, k] = <0| S_i^a(t_k) S_j^a(0) |0> for one reference site j and one
    component a, with bond-dimension history and timings."""
    times: np.ndarray
    C: np.ndarray
    chi: np.ndarray  # max bond dimension reached at each time
    seconds: np.ndarray  # cumulative wall-clock at each time
    j: int
    a: str


def correlations(M, psi0, E0: float, j: int, times: np.ndarray, chi_max: int,
                 a: str = "z", substeps: int = 2, svd_min: float = 1e-10) -> MPSCorrelations:
    """Two-site TDVP of S_j^a|0>, overlaps with S_i^a|0> for all i on the grid."""
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    from tenpy.networks.mps import MPSEnvironment

    op = "S" + a
    phi = psi0.copy()
    phi.apply_local_op(j, op, unitary=False)
    dt = (times[1] - times[0]) / substeps
    eng = TwoSiteTDVPEngine(phi, M, {"dt": dt, "N_steps": substeps,
                                     "trunc_params": {"chi_max": chi_max, "svd_min": svd_min}})
    N = psi0.L
    C = np.zeros((N, len(times)), dtype=complex)
    chi = np.zeros(len(times), dtype=int)
    secs = np.zeros(len(times))
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        if k:
            eng.run()
        env = MPSEnvironment(psi0, phi)
        # the environment already carries the norm of phi (= |S_j|0>|)
        C[:, k] = np.exp(1j * E0 * t) * np.asarray(env.expectation_value(op))
        chi[k] = max(phi.chi)
        secs[k] = time.perf_counter() - t0
    return MPSCorrelations(np.asarray(times), C, chi, secs, j, a)
