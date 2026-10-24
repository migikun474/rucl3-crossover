"""The J1-K-Gamma-Gamma'-J3 Hamiltonian on a periodic honeycomb cluster.

Conventions
-----------
* Basis states are integers ``idx`` with bit ``k`` the spin of site ``k``;
  bit 0 is spin up (sigma^z = +1).
* Two-site matrices act on the pair (i, j) in the basis ``2*b_i + b_j``.
* Spin operators are S = sigma / 2; energies are in meV and hbar = 1, so
  times are in meV^-1 (multiply by 0.6582 to get picoseconds).

The model has no U(1) symmetry, so the full 2^N space is used.  Three ways
of applying H are provided: a scipy sparse matrix (small N, tests), a
matrix-free numba kernel (production, N up to ~26), and the Pauli-string
decomposition used by the Trotter circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .lattice import Cluster
from .params import ParameterSet

SIGMA = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
AXES = "xyz"
PAULI_OF_AXIS = {"x": "X", "y": "Y", "z": "Z"}


def bond_coupling(kind: str, p: ParameterSet) -> np.ndarray:
    """3x3 exchange matrix J_ab so that the bond energy is sum_ab J_ab S_i^a S_j^b."""
    if kind == "3":
        return p.J3 * np.eye(3)
    g = AXES.index(kind)
    a, b = [k for k in range(3) if k != g]
    J = p.J1 * np.eye(3)
    J[g, g] += p.K
    J[a, b] += p.G
    J[b, a] += p.G
    for k in (a, b):
        J[g, k] += p.Gp
        J[k, g] += p.Gp
    return J


def two_site_matrix(J: np.ndarray) -> np.ndarray:
    """4x4 matrix of sum_ab J_ab S^a (x) S^b."""
    M = np.zeros((4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            if J[a, b]:
                M += J[a, b] / 4 * np.kron(SIGMA["XYZ"[a]], SIGMA["XYZ"[b]])
    return M


@dataclass
class PauliTerm:
    """coeff * sigma^{pa}_i sigma^{pb}_j (coeff real)."""
    coeff: float
    i: int
    j: int
    pa: str
    pb: str

    @property
    def xmask(self) -> int:
        m = 0
        if self.pa in "XY":
            m |= 1 << self.i
        if self.pb in "XY":
            m |= 1 << self.j
        return m

    @property
    def zmask(self) -> int:
        m = 0
        if self.pa in "YZ":
            m |= 1 << self.i
        if self.pb in "YZ":
            m |= 1 << self.j
        return m

    @property
    def ny(self) -> int:
        return (self.pa == "Y") + (self.pb == "Y")


class SpinModel:
    def __init__(self, cluster: Cluster, params: ParameterSet):
        self.cluster = cluster
        self.params = params
        self.N = cluster.N
        self.dim = 1 << self.N

        self.bonds = []  # (i, j, kind, J 3x3)
        for b in cluster.nn_bonds:
            self.bonds.append((b.i, b.j, b.kind, bond_coupling(b.kind, params)))
        if params.J3:
            for b in cluster.j3_bonds:
                self.bonds.append((b.i, b.j, b.kind, bond_coupling("3", params)))

        self.bond_i = np.array([b[0] for b in self.bonds], dtype=np.int64)
        self.bond_j = np.array([b[1] for b in self.bonds], dtype=np.int64)
        self.bond_M = np.array([two_site_matrix(b[3]) for b in self.bonds])
        self._diag = None

    # ------------------------------------------------------------------
    def pauli_terms(self) -> list[PauliTerm]:
        """H = sum coeff * P, one term per nonzero J_ab on each bond."""
        terms = []
        for i, j, kind, J in self.bonds:
            for a in range(3):
                for b in range(3):
                    if abs(J[a, b]) > 1e-15:
                        terms.append(PauliTerm(J[a, b] / 4, i, j, "XYZ"[a], "XYZ"[b]))
        return terms

    def diagonal(self) -> np.ndarray:
        """Diagonal of H in the computational basis (float64, length 2^N)."""
        if self._diag is None:
            idx = np.arange(self.dim, dtype=np.int64)
            d = np.zeros(self.dim)
            for (i, j), M in zip(zip(self.bond_i, self.bond_j), self.bond_M):
                row = 2 * ((idx >> i) & 1) + ((idx >> j) & 1)
                d += M.diagonal().real[row]
            self._diag = d
        return self._diag

    def sparse(self) -> sp.csr_matrix:
        """Sparse matrix of H.  Memory grows like 2^N * (number of bonds);
        use for N <= 18."""
        if self.N > 20:
            raise MemoryError("use the matrix-free operator for N > 20")
        idx = np.arange(self.dim, dtype=np.int64)
        H = sp.diags(self.diagonal()).tocsr().astype(complex)
        for i, j, M in zip(self.bond_i, self.bond_j, self.bond_M):
            bi, bj = (idx >> i) & 1, (idx >> j) & 1
            row = 2 * bi + bj
            base = idx & ~((1 << i) | (1 << j))
            for c in range(4):
                vals = M[row, c]
                vals = np.where(row == c, 0, vals)  # diagonal already included
                keep = vals != 0
                if not keep.any():
                    continue
                cols = base | ((c >> 1) << i) | ((c & 1) << j)
                H = H + sp.csr_matrix((vals[keep], (idx[keep], cols[keep])),
                                      shape=(self.dim, self.dim))
        return H.tocsr()

    def operator(self, threads: int | None = None) -> "MatrixFreeH":
        return MatrixFreeH(self, threads)

    def apply_pauli_dense(self) -> np.ndarray:
        """Dense H built from the Pauli terms (tiny N only, for tests)."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for t in self.pauli_terms():
            H += t.coeff * pauli_matrix(self.N, {t.i: t.pa, t.j: t.pb})
        return H


def pauli_matrix(N: int, ops: dict[int, str]) -> np.ndarray:
    """Dense matrix of a Pauli string; site k is bit k (site 0 = least significant)."""
    out = np.array([[1.0 + 0j]])
    for k in reversed(range(N)):
        out = np.kron(out, SIGMA[ops.get(k, "I")])
    return out


# ----------------------------------------------------------------------
# matrix-free application with numba
# ----------------------------------------------------------------------
try:
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def _apply_h(psi, out, diag, bi, bj, M):
        dim = psi.shape[0]
        nbonds = bi.shape[0]
        for idx in nb.prange(dim):
            acc = diag[idx] * psi[idx]
            for b in range(nbonds):
                i = bi[b]
                j = bj[b]
                si = (idx >> i) & 1
                sj = (idx >> j) & 1
                row = 2 * si + sj
                base = idx & ~((1 << i) | (1 << j))
                for c in range(4):
                    if c != row:
                        m = M[b, row, c]
                        if m != 0:
                            k = base | ((c >> 1) << i) | ((c & 1) << j)
                            acc += m * psi[k]
            out[idx] = acc

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


class MatrixFreeH:
    """H as a callable / scipy LinearOperator without storing the matrix."""

    def __init__(self, model: SpinModel, threads: int | None = None):
        self.model = model
        self.dim = model.dim
        self.shape = (self.dim, self.dim)
        self.dtype = np.dtype(complex)
        self.nmatvec = 0
        if HAVE_NUMBA:
            if threads:
                nb.set_num_threads(threads)
            self._diag = model.diagonal()
            self._M = np.ascontiguousarray(model.bond_M)
        else:
            self._H = model.sparse()

    def matvec(self, psi: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        self.nmatvec += 1
        psi = np.ascontiguousarray(psi, dtype=complex).reshape(-1)
        if not HAVE_NUMBA:
            res = self._H @ psi
            if out is not None:
                out[:] = res
                return out
            return res
        if out is None:
            out = np.empty_like(psi)
        _apply_h(psi, out, self._diag, self.model.bond_i, self.model.bond_j, self._M)
        return out

    __call__ = matvec

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.shape, matvec=self.matvec, dtype=complex)
