import numpy as np
import pytest
import scipy.linalg as sla

from rucl3.classical.exact import correlations, lanczos_ground_state
from rucl3.model import params as P
from rucl3.model.hamiltonian import SpinModel, pauli_matrix
from rucl3.model.lattice import best_cluster
from rucl3.quantum.trotter import (TrotterCircuit, _rotate, rotations_per_step,
                                   trotter_correlations)

GENERIC = P.ParameterSet("generic", J1=-1.3, K=-15.1, G=10.1, Gp=-0.12, J3=0.0, verified=True)


@pytest.mark.parametrize("ops", [{0: "X"}, {1: "Z", 3: "Y"}, {0: "Y", 2: "Y"}, {2: "X", 3: "Z"},
                                 {0: "Z", 4: "Z"}])
def test_rotation_kernel(ops):
    N = 5
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2, 1 << N)) + 1j * rng.standard_normal((2, 1 << N))
    P_ = pauli_matrix(N, ops)
    xm = sum(1 << k for k, a in ops.items() if a in "XY")
    zm = sum(1 << k for k, a in ops.items() if a in "YZ")
    ny = sum(a == "Y" for a in ops.values())
    want = (sla.expm(-1j * 0.37 * P_) @ v.T).T
    _rotate(v, xm, zm, ny, 0.37)
    assert np.allclose(v, want)


@pytest.mark.parametrize("order,slope", [(1, 1), (2, 2), (4, 4)])
def test_trotter_error_order(order, slope):
    m = SpinModel(best_cluster(8, include_j3=False), GENERIC)
    H = m.sparse().toarray()
    rng = np.random.default_rng(1)
    v = rng.standard_normal(m.dim) + 1j * rng.standard_normal(m.dim)
    v /= np.linalg.norm(v)
    t = 0.2
    exact = sla.expm(-1j * H * t) @ v
    errs = []
    steps = [8, 16]
    for r in steps:
        s = v[None].copy()
        TrotterCircuit(m, order).evolve(s, t, r)
        errs.append(np.linalg.norm(s[0] - exact))
    measured = np.log2(errs[0] / errs[1])
    assert measured == pytest.approx(slope, abs=0.35)


def test_trotter_correlations_converge_to_exact():
    m = SpinModel(best_cluster(8, include_j3=False), GENERIC)
    H = m.operator()
    gs = lanczos_ground_state(H)
    times = np.linspace(0, 0.6, 4)
    ex = correlations(H, gs, m.N, times)
    tr = trotter_correlations(m, gs, times, dt=0.002, order=2)
    assert np.abs(tr.C - ex.C).max() < 1e-4


def test_rotation_counts():
    assert rotations_per_step(10, 1) == 10
    assert rotations_per_step(10, 2) == 18
    assert rotations_per_step(10, 4) == 90
