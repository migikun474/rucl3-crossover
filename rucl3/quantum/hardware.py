"""Small hardware experiment: quench dynamics with the same Trotter circuit.

On a present-day device the Hadamard-test correlators of the full protocol
(ground-state preparation, ancilla) are far too deep, so the hardware check
is the standard quench: prepare the Neel product state, apply the Trotter
circuit of :mod:`rucl3.quantum.trotter` for time t, and measure every spin in
the z basis, giving m_i(t) = <sigma^z_i(t)> for all sites from one circuit.
Exact and noiseless-Trotter references come from the state-vector code.
Qiskit qubit k is site k (both little-endian), and bit 0 is spin up.
"""

from __future__ import annotations

import numpy as np

from ..model.hamiltonian import SpinModel
from .trotter import TrotterCircuit


def neel_bits(model: SpinModel) -> np.ndarray:
    """1 (spin down) on the B sublattice, 0 (up) on A."""
    return np.array(model.cluster.sublattice, dtype=int)


def pauli_rotation(qc, i, j, pa, pb, theta):
    """Append exp(-i theta sigma^pa_i sigma^pb_j) using basis changes and RZZ."""
    def to_z(q, pauli):
        if pauli == "X":
            qc.h(q)
        elif pauli == "Y":  # S^dag then H maps Y to Z
            qc.sdg(q)
            qc.h(q)

    def from_z(q, pauli):
        if pauli == "X":
            qc.h(q)
        elif pauli == "Y":
            qc.h(q)
            qc.s(q)

    to_z(i, pa)
    to_z(j, pb)
    qc.rzz(2 * theta, i, j)
    from_z(i, pa)
    from_z(j, pb)


def quench_circuit(model: SpinModel, t: float, steps: int, order: int = 2, measure=True):
    """Neel state, `steps` Trotter steps up to time t, z measurement of all spins."""
    from qiskit import QuantumCircuit
    tc = TrotterCircuit(model, order)
    qc = QuantumCircuit(model.N, model.N if measure else 0)
    for q, b in enumerate(neel_bits(model)):
        if b:
            qc.x(q)
    dt = t / steps if steps else 0.0
    for _ in range(steps):
        for k, f in tc.seq:
            term = tc.terms[k]
            pauli_rotation(qc, term.i, term.j, term.pa, term.pb, term.coeff * dt * f)
    if measure:
        qc.measure(range(model.N), range(model.N))
    return qc


def magnetization_from_counts(counts: dict, N: int) -> np.ndarray:
    """<sigma^z_i> from a counts dict keyed by bitstrings (qubit 0 rightmost)."""
    m = np.zeros(N)
    total = 0
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]
        m += c * np.array([1 - 2 * int(b) for b in bits[:N]])
        total += c
    return m / total


def neel_state(model: SpinModel) -> np.ndarray:
    idx = int(sum(int(b) << q for q, b in enumerate(neel_bits(model))))
    psi = np.zeros(model.dim, dtype=complex)
    psi[idx] = 1.0
    return psi


def exact_magnetization(model: SpinModel, times: np.ndarray) -> np.ndarray:
    """m_i(t) from exact Krylov evolution of the Neel state."""
    from ..classical.exact import KrylovPropagator, apply_pauli
    H = model.operator()
    prop = KrylovPropagator(H, m=16, tol=1e-11)
    psi = neel_state(model)
    out = np.zeros((len(times), model.N))
    for k, t in enumerate(times):
        if k:
            psi = prop.evolve(psi, t - times[k - 1])
        out[k] = [np.vdot(psi, apply_pauli(psi, i, "Z")).real for i in range(model.N)]
    return out


def trotter_magnetization(model: SpinModel, times: np.ndarray, steps_per_point: int,
                          order: int = 2) -> np.ndarray:
    """m_i(t) from the noiseless Trotter circuit (state vector)."""
    from ..classical.exact import apply_pauli
    tc = TrotterCircuit(model, order)
    psi = neel_state(model)[None].copy()
    out = np.zeros((len(times), model.N))
    for k, t in enumerate(times):
        if k:
            tc.evolve(psi, t - times[k - 1], steps_per_point)
        v = psi[0]
        out[k] = [np.vdot(v, apply_pauli(v, i, "Z")).real for i in range(model.N)]
    return out
