"""Trotterised evolution at the level of Pauli rotations.

The circuit that would run on hardware is a sequence of rotations
exp(-i theta P) with P a two-qubit Pauli string (one per nonzero J_ab on
each bond, see :meth:`SpinModel.pauli_terms`).  We simulate exactly that
sequence on a state vector, so the measured Trotter error belongs to the
same circuit whose gates are counted in :mod:`rucl3.quantum.resources`.

Correlation functions are what the Hadamard test measures,

    C_r(t) = <0| U_r(t)^dag S_i^a U_r(t) S_j^b |0>,

with U_r the Trotter circuit; both |0> and S_j^b|0> are evolved by U_r.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numba as nb
import numpy as np

from ..classical.exact import Correlations, GroundState, apply_pauli, pauli_overlap
from ..model.hamiltonian import PauliTerm, SpinModel

# Suzuki's fourth-order coefficient
P4 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


@nb.njit(inline="always")
def _parity(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


@nb.njit(parallel=True, cache=True)
def _rotate(psi, xmask, zmask, ny, theta):
    """In place: psi <- exp(-i theta P) psi for every row of psi (2D array)."""
    nvec, dim = psi.shape
    c = np.cos(theta)
    s = np.sin(theta)
    iy = (1j) ** ny
    if xmask == 0:
        e_plus = complex(c, -s)
        e_minus = complex(c, s)
        for b in nb.prange(dim):
            f = e_minus if _parity(b & zmask) else e_plus
            for v in range(nvec):
                psi[v, b] *= f
        return
    top = 1 << (63 - _clz(xmask))
    for b in nb.prange(dim):
        if b & top:
            continue
        b2 = b ^ xmask
        ph_b = iy * (1 - 2 * _parity(b & zmask))  # P|b> = ph_b |b2>
        ph_b2 = iy * (1 - 2 * _parity(b2 & zmask))  # P|b2> = ph_b2 |b>
        for v in range(nvec):
            x1 = psi[v, b]
            x2 = psi[v, b2]
            psi[v, b] = c * x1 - 1j * s * ph_b2 * x2
            psi[v, b2] = c * x2 - 1j * s * ph_b * x1


@nb.njit(inline="always")
def _clz(x):
    n = 0
    for k in range(63, -1, -1):
        if (x >> k) & 1:
            return n
        n += 1
    return 64


@dataclass
class Rotation:
    xmask: int
    zmask: int
    ny: int
    coeff: float  # angle per unit time: theta = coeff * dt * weight


def ordered_terms(model: SpinModel) -> list[PauliTerm]:
    """Pauli terms grouped by bond family (x, y, z bonds, then J3)."""
    order = {"x": 0, "y": 1, "z": 2, "3": 3}
    kinds = {(i, j): kind for i, j, kind, _ in model.bonds}
    return sorted(model.pauli_terms(), key=lambda t: order[kinds[(t.i, t.j)]])


def step_sequence(terms: list[PauliTerm], order: int) -> list[tuple[int, float]]:
    """One Trotter step as a list of (term index, fraction of dt)."""
    L = len(terms)
    if order == 1:
        return [(k, 1.0) for k in range(L)]
    if order == 2:
        fwd = [(k, 0.5) for k in range(L)]
        return fwd + fwd[::-1]
    if order == 4:
        seq = []
        for w in (P4, P4, 1 - 4 * P4, P4, P4):
            seq += [(k, f * w) for k, f in step_sequence(terms, 2)]
        return seq
    raise ValueError(order)


def merge(seq: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Fuse neighbouring rotations about the same Pauli string."""
    out = []
    for k, f in seq:
        if out and out[-1][0] == k:
            out[-1] = (k, out[-1][1] + f)
        else:
            out.append((k, f))
    return out


def rotations_per_step(n_terms: int, order: int) -> float:
    """Rotations per Trotter step in a long run, after fusing neighbours
    (including across step boundaries)."""
    terms = [None] * n_terms
    one = len(merge(step_sequence(terms, order)))
    two = len(merge(step_sequence(terms, order) * 2))
    return two - one


class TrotterCircuit:
    def __init__(self, model: SpinModel, order: int = 2):
        self.model = model
        self.order = order
        self.terms = ordered_terms(model)
        self.seq = merge(step_sequence(self.terms, order))
        self.rot = [Rotation(t.xmask, t.zmask, t.ny, t.coeff) for t in self.terms]
        self.nrot = 0

    def evolve(self, states: np.ndarray, t: float, steps: int) -> np.ndarray:
        """Apply U_r(t) with r = steps Trotter steps to every row of `states` (in place)."""
        dt = t / steps
        for _ in range(steps):
            for k, f in self.seq:
                r = self.rot[k]
                _rotate(states, r.xmask, r.zmask, r.ny, r.coeff * dt * f)
                self.nrot += 1
        return states


def trotter_correlations(model: SpinModel, gs: GroundState, times: np.ndarray,
                         dt: float, order: int = 2, ref_sites=(0, 1)) -> Correlations:
    """Correlations measured with the Trotter circuit, step size <= dt."""
    times = np.asarray(times, dtype=float)
    t0 = time.perf_counter()
    N = model.N
    circ = TrotterCircuit(model, order)
    vecs = [gs.vector]
    labels = []
    for s, j in enumerate(ref_sites):
        for b, pb in enumerate("XYZ"):
            vecs.append(apply_pauli(gs.vector, j, pb))
            labels.append((s, b))
    states = np.array(vecs)
    C = np.zeros((3, 3, N, len(ref_sites), len(times)), dtype=complex)
    for k, t in enumerate(times):
        if k:
            span = t - times[k - 1]
            circ.evolve(states, span, max(1, int(np.ceil(span / dt - 1e-9))))
        for n, (s, b) in enumerate(labels):
            for i in range(N):
                for a, pa in enumerate("XYZ"):
                    C[a, b, i, s, k] = pauli_overlap(states[0], states[n + 1], i, pa) / 4
    return Correlations(times, np.array(ref_sites), C, circ.nrot, time.perf_counter() - t0)
