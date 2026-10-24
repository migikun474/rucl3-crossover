"""Exact state-vector baseline: Lanczos ground state, Krylov time evolution,
and the spin correlation functions <S_i^a(t) S_j^b(0)>.

Memory is kept to a handful of 2^N vectors so that N = 24 (268 MB per
complex vector) fits in a laptop's RAM: the ground state uses a two-pass
Lanczos that does not store its basis, and the propagator stores only its
``m`` Krylov vectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla


# ----------------------------------------------------------------------
# single-site Pauli operators on a state vector (bit k = site k)
# ----------------------------------------------------------------------
def _split(psi: np.ndarray, i: int) -> np.ndarray:
    """View psi as (high bits, bit i, low bits)."""
    return psi.reshape(-1, 2, 1 << i)


def apply_pauli(psi: np.ndarray, i: int, a: str) -> np.ndarray:
    """Return sigma^a_i |psi>."""
    v = _split(psi, i)
    out = np.empty_like(v)
    if a == "Z":
        out[:, 0] = v[:, 0]
        out[:, 1] = -v[:, 1]
    elif a == "X":
        out[:, 0] = v[:, 1]
        out[:, 1] = v[:, 0]
    elif a == "Y":
        out[:, 0] = -1j * v[:, 1]
        out[:, 1] = 1j * v[:, 0]
    else:
        raise ValueError(a)
    return out.reshape(-1)


def pauli_overlap(bra: np.ndarray, ket: np.ndarray, i: int, a: str) -> complex:
    """<bra| sigma^a_i |ket> without allocating sigma|ket>."""
    u, v = _split(bra, i), _split(ket, i)
    if a == "Z":
        return np.vdot(u[:, 0], v[:, 0]) - np.vdot(u[:, 1], v[:, 1])
    if a == "X":
        return np.vdot(u[:, 0], v[:, 1]) + np.vdot(u[:, 1], v[:, 0])
    if a == "Y":
        return -1j * np.vdot(u[:, 0], v[:, 1]) + 1j * np.vdot(u[:, 1], v[:, 0])
    raise ValueError(a)


# ----------------------------------------------------------------------
# ground state
# ----------------------------------------------------------------------
@dataclass
class GroundState:
    energy: float
    vector: np.ndarray
    residual: float
    matvecs: int
    seconds: float
    ritz: np.ndarray = field(repr=False)  # low Ritz values of the last run


def _lanczos_coeffs(H, v0, maxiter, tol, check_every=10):
    """First Lanczos pass: tridiagonal coefficients only."""
    alpha, beta = [], []
    v_prev = np.zeros_like(v0)
    v = v0 / np.linalg.norm(v0)
    b = 0.0
    e_old = np.inf
    w = np.empty_like(v)
    for k in range(maxiter):
        H.matvec(v, w)
        a = np.vdot(v, w).real
        w -= a * v + b * v_prev
        alpha.append(a)
        b = np.linalg.norm(w)
        beta.append(b)
        if b < 1e-12:
            break
        if (k + 1) % check_every == 0:
            e = sla.eigvalsh_tridiagonal(np.array(alpha), np.array(beta[:-1]),
                                         select="i", select_range=(0, 0))[0]
            if abs(e - e_old) < tol:
                break
            e_old = e
        v_prev, v = v, w / b
        w = np.empty_like(v)
    return np.array(alpha), np.array(beta)


def lanczos_ground_state(H, v0: np.ndarray | None = None, tol: float = 1e-10,
                         maxiter: int = 400, restarts: int = 5,
                         seed: int = 0) -> GroundState:
    """Lowest eigenpair of a Hermitian operator H (with ``H.matvec(v, out)``).

    Two-pass Lanczos without stored basis; restarted from the current Ritz
    vector until the residual ||H psi - E psi|| < sqrt(tol).
    """
    t0 = time.perf_counter()
    n0 = getattr(H, "nmatvec", 0)
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(H.dim) + 1j * rng.standard_normal(H.dim)
    psi = v0 / np.linalg.norm(v0)
    for _ in range(restarts):
        alpha, beta = _lanczos_coeffs(H, psi, maxiter, tol)
        m = len(alpha)
        evals, evecs = sla.eigh_tridiagonal(alpha, beta[: m - 1])
        y = evecs[:, 0]
        # second pass: rebuild the basis on the fly and accumulate the Ritz vector
        out = y[0] * psi
        v_prev = np.zeros_like(psi)
        v = psi.copy()
        w = np.empty_like(psi)
        for k in range(m - 1):
            H.matvec(v, w)
            w -= alpha[k] * v + (beta[k - 1] if k else 0.0) * v_prev
            v_prev, v = v, w / beta[k]
            w = np.empty_like(psi)
            out += y[k + 1] * v
        psi = out / np.linalg.norm(out)
        Hpsi = H.matvec(psi)
        E = np.vdot(psi, Hpsi).real
        res = np.linalg.norm(Hpsi - E * psi)
        if res < np.sqrt(tol):
            break
    return GroundState(E, psi, res, getattr(H, "nmatvec", 0) - n0,
                       time.perf_counter() - t0, evals[:6])


# ----------------------------------------------------------------------
# time evolution
# ----------------------------------------------------------------------
class KrylovPropagator:
    """exp(-i H t) |v> by short-iterative Lanczos with adaptive sub-steps.

    Each Krylov basis of dimension ``m`` is reused for the largest sub-step
    whose a-posteriori error estimate beta_m |c_{m-1}| stays below ``tol``.
    """

    def __init__(self, H, m: int = 16, tol: float = 1e-10):
        self.H = H
        self.m = m
        self.tol = tol

    def evolve(self, v: np.ndarray, t: float) -> np.ndarray:
        remaining = t
        v = v.copy()
        while remaining > 1e-14:
            v, used = self._substep(v, remaining)
            remaining -= used
        return v

    def _substep(self, v, tmax):
        nrm = np.linalg.norm(v)
        m = self.m
        V = np.empty((m, v.size), dtype=complex)
        alpha = np.zeros(m)
        beta = np.zeros(m)
        V[0] = v / nrm
        k_end = m
        for k in range(m):
            w = self.H.matvec(V[k])
            alpha[k] = np.vdot(V[k], w).real
            w -= alpha[k] * V[k]
            if k:
                w -= beta[k - 1] * V[k - 1]
            # full reorthogonalisation inside the small basis
            w -= V[: k + 1].T @ np.conj(V[: k + 1] @ np.conj(w))
            beta[k] = np.linalg.norm(w)
            if beta[k] < 1e-12:  # invariant subspace: exact
                k_end = k + 1
                break
            if k + 1 < m:
                V[k + 1] = w / beta[k]
        Tm = np.diag(alpha[:k_end]) + np.diag(beta[: k_end - 1], 1) + np.diag(beta[: k_end - 1], -1)
        evals, evecs = np.linalg.eigh(Tm)
        dt = tmax
        while True:
            c = evecs @ (np.exp(-1j * evals * dt) * evecs[0].conj())
            err = beta[k_end - 1] * abs(c[-1]) if k_end == m else 0.0
            if err <= self.tol or dt < 1e-8 * tmax:
                break
            dt *= 0.5
        return nrm * (c @ V[:k_end]), dt


# ----------------------------------------------------------------------
# correlation functions
# ----------------------------------------------------------------------
@dataclass
class Correlations:
    """C[a, b, i, s, t] = <0| S_i^a(t) S_{j_s}^b(0) |0> for the reference sites
    j_s of each sublattice, a, b in (x, y, z) and spin-1/2 operators S."""
    times: np.ndarray
    ref_sites: np.ndarray
    C: np.ndarray
    matvecs: int
    seconds: float


def correlations(H, gs: GroundState, N: int, times: np.ndarray,
                 ref_sites=(0, 1), m: int = 16, tol: float = 1e-10,
                 progress: bool = False) -> Correlations:
    """Spin correlations on a uniform time grid ``times`` (starting at 0).

    Translation invariance means the reference sites (one per sublattice)
    give every pair (i, j); see :func:`rucl3.classical.spectrum.structure_factor`.
    """
    times = np.asarray(times, dtype=float)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at 0 and increase")
    t0 = time.perf_counter()
    n0 = getattr(H, "nmatvec", 0)
    prop = KrylovPropagator(H, m=m, tol=tol)
    psi0, E0 = gs.vector, gs.energy
    C = np.zeros((3, 3, N, len(ref_sites), len(times)), dtype=complex)
    for s, j in enumerate(ref_sites):
        for b, pb in enumerate("XYZ"):
            phi = apply_pauli(psi0, j, pb)
            for k, t in enumerate(times):
                if k:
                    phi = prop.evolve(phi, t - times[k - 1])
                phase = np.exp(1j * E0 * t) / 4  # S = sigma/2 on both operators
                for i in range(N):
                    for a, pa in enumerate("XYZ"):
                        C[a, b, i, s, k] = phase * pauli_overlap(psi0, phi, i, pa)
            if progress:
                print(f"  ref site {j}, S^{pb}: done, {time.perf_counter() - t0:.1f}s", flush=True)
    return Correlations(times, np.array(ref_sites), C,
                        getattr(H, "nmatvec", 0) - n0, time.perf_counter() - t0)
