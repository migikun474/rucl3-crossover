import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit.quantum_info import Statevector  # noqa: E402

from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import Flake  # noqa: E402
from rucl3.quantum.hardware import (exact_magnetization, neel_state,  # noqa: E402
                                    quench_circuit, trotter_magnetization)
from rucl3.quantum.trotter import TrotterCircuit  # noqa: E402

GENERIC = P.ParameterSet("generic", J1=-1.3, K=-15.1, G=10.1, Gp=-0.12, J3=0.9, verified=True)


def test_flake_geometry():
    fl = Flake(3, 2)  # 3 cells around, 2 along: 12 sites
    assert fl.N == 12
    deg = np.zeros(fl.N, int)
    for b in fl.nn_bonds:
        deg[b.i] += 1
        deg[b.j] += 1
    assert deg.max() == 3 and len(fl.nn_bonds) == 13


def test_qiskit_circuit_matches_statevector_trotter():
    """The Qiskit circuit is exactly the rotation sequence of TrotterCircuit
    (all nine Pauli products appear for this parameter set)."""
    m = SpinModel(Flake(2, 2), GENERIC)
    qc = quench_circuit(m, t=0.3, steps=2, measure=False)
    ours = neel_state(m)[None].copy()
    TrotterCircuit(m, 2).evolve(ours, 0.3, 2)
    theirs = Statevector(qc).data
    assert abs(abs(np.vdot(ours[0], theirs)) - 1) < 1e-10


def test_trotter_magnetization_converges():
    m = SpinModel(Flake(2, 2), GENERIC)
    times = np.linspace(0, 0.4, 5)
    ex = exact_magnetization(m, times)
    tr = trotter_magnetization(m, times, steps_per_point=20)
    assert np.allclose(ex[0], 1 - 2 * np.array(m.cluster.sublattice))
    assert np.abs(ex - tr).max() < 1e-3
