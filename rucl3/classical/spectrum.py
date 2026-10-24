"""Dynamical structure factor from time-domain correlation functions.

    S^{ab}(q, omega) = 1/(2 pi) int_{-T}^{T} dt w(t) e^{i omega t} S^{ab}(q, t)
    S^{ab}(q, t)     = 1/N sum_ij e^{i q.(r_i - r_j)} <S_i^a(t) S_j^b(0)>

With a Gaussian window this equals the Lehmann sum broadened by a
normalised Gaussian of width 1/tau, i.e. the spectrum as seen through the
instrument resolution (see :mod:`rucl3.target`).  Note the 1/(2 pi), which
makes int S d omega the equal-time structure factor; the plan's formula
omits it.
"""

from __future__ import annotations

import numpy as np

from ..model.lattice import Cluster
from .exact import Correlations


def momentum_correlations(cluster: Cluster, corr: Correlations, q: np.ndarray) -> np.ndarray:
    """S^{ab}(q, t) for momenta q of shape (nq, 2); returns (nq, 3, 3, nt).

    Uses translation invariance: every site j is a translate of one of the
    reference sites, so (1/N) sum_ij -> (1/2) sum_{ref s} sum_i.
    """
    q = np.atleast_2d(q)
    r = cluster.positions
    out = np.zeros((len(q), 3, 3, len(corr.times)), dtype=complex)
    nref = len(corr.ref_sites)
    if nref != 2 or set(cluster.sublattice[corr.ref_sites]) != {0, 1}:
        raise ValueError("need one reference site per sublattice")
    for s, j in enumerate(corr.ref_sites):
        phase = np.exp(1j * (r - r[j]) @ q.T)  # (N, nq)
        out += np.einsum("iq,abit->qabt", phase, corr.C[:, :, :, s, :]) / nref
    return out


def fourier(times: np.ndarray, Sqt: np.ndarray, window: np.ndarray,
            omegas: np.ndarray) -> np.ndarray:
    """Windowed transform of S^{ab}(q, t) (shape (..., 3, 3, nt)) using
    S^{ab}(q, -t) = conj(S^{ba}(q, t)).  Returns (..., 3, 3, nomega), real
    on the diagonal."""
    dt = times[1] - times[0]
    wts = window * dt
    wts[0] *= 0.5
    wts[-1] *= 0.5
    ph = np.exp(1j * np.outer(times, omegas))  # (nt, nw)
    fwd = (Sqt * wts) @ ph
    bwd = (np.conj(np.swapaxes(Sqt, -3, -2)) * wts) @ np.conj(ph)
    return (fwd + bwd) / (2 * np.pi)


def lehmann_dense(H: np.ndarray, cluster: Cluster, q: np.ndarray, omegas: np.ndarray,
                  sigma: float, ground: int = 0) -> np.ndarray:
    """Reference S^{ab}(q, omega) from full diagonalisation (tiny N only),
    broadened with a normalised Gaussian of standard deviation ``sigma``."""
    from ..model.hamiltonian import pauli_matrix
    E, U = np.linalg.eigh(H)
    psi0 = U[:, ground]
    N = cluster.N
    q = np.atleast_2d(q)
    out = np.zeros((len(q), 3, 3, len(omegas)), dtype=complex)
    S = {a: [pauli_matrix(N, {i: a}) / 2 for i in range(N)] for a in "XYZ"}
    gauss = np.exp(-0.5 * ((omegas[None, :] - (E - E[0])[:, None]) / sigma) ** 2) / (
        np.sqrt(2 * np.pi) * sigma)
    for iq, qq in enumerate(q):
        ph = np.exp(-1j * cluster.positions @ qq) / np.sqrt(N)
        Sq = {a: sum(ph[i] * S[a][i] for i in range(N)) for a in "XYZ"}
        amp = {a: U.conj().T @ (Sq[a] @ psi0) for a in "XYZ"}  # <n|S_q^a|0>
        for ia, a in enumerate("XYZ"):
            for ib, b in enumerate("XYZ"):
                w = np.conj(amp[a]) * amp[b]  # <0|S_{-q}^a|n><n|S_q^b|0>
                out[iq, ia, ib] = w @ gauss
    return out
