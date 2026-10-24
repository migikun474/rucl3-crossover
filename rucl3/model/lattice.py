"""Periodic honeycomb clusters with Kitaev bond labels.

Geometry (nearest-neighbour distance 1):

    delta_z = (0, 1),  delta_x = (-sqrt3/2, -1/2),  delta_y = (sqrt3/2, -1/2)

are the vectors from an A site to its three B neighbours.  The Bravais
vectors are a1 = delta_z - delta_x and a2 = delta_z - delta_y, so that

    A(R) -z- B(R),   A(R) -x- B(R - a1),   A(R) -y- B(R - a2).

Third neighbours of A(R) sit across each hexagon at
B(R - a1 - a2), B(R + a1 - a2), B(R - a1 + a2).

A torus is defined by an integer 2x2 matrix T whose rows give the supercell
vectors L_k = T[k,0] a1 + T[k,1] a2.  The cluster has N = 2 |det T| sites.

Bonds are generated per unit cell by translation, so every bond of the
infinite lattice maps to exactly one term on the torus.  Small tori can map
two different bonds onto the same pair of sites (or onto an existing
nearest-neighbour pair); :func:`Cluster.check` reports this, and
:func:`best_cluster` only returns clusters where it does not happen.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

SQ3 = np.sqrt(3.0)
DELTA = {
    "x": np.array([-SQ3 / 2, -0.5]),
    "y": np.array([SQ3 / 2, -0.5]),
    "z": np.array([0.0, 1.0]),
}
A1 = DELTA["z"] - DELTA["x"]
A2 = DELTA["z"] - DELTA["y"]
BRAVAIS = np.array([A1, A2])  # rows

# (offset of the B cell relative to the A cell, in units of a1, a2), bond label
NN_BONDS = [((0, 0), "z"), ((-1, 0), "x"), ((0, -1), "y")]
J3_BONDS = [(-1, -1), (1, -1), (-1, 1)]


@dataclass
class Bond:
    i: int
    j: int
    kind: str  # "x", "y", "z" for nearest neighbours, "3" for third neighbours


@dataclass
class Cluster:
    T: np.ndarray
    cells: np.ndarray = field(init=False)  # (Nc, 2) integer cell coordinates
    positions: np.ndarray = field(init=False)  # (N, 2) site positions
    sublattice: np.ndarray = field(init=False)  # 0 = A, 1 = B
    cell_of_site: np.ndarray = field(init=False)
    nn_bonds: list[Bond] = field(init=False)
    j3_bonds: list[Bond] = field(init=False)

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=int)
        det = int(round(np.linalg.det(self.T)))
        if det == 0:
            raise ValueError("degenerate supercell")
        if det < 0:  # keep a right-handed supercell
            self.T = self.T[::-1].copy()
            det = -det
        self._Tinv = np.linalg.inv(self.T.astype(float))
        self.ncells = det
        self.cells = self._enumerate_cells()
        self._index = {tuple(c): k for k, c in enumerate(self.cells)}

        nc = self.ncells
        # site index: A of cell k -> 2k, B of cell k -> 2k + 1
        self.sublattice = np.tile([0, 1], nc)
        self.cell_of_site = np.repeat(np.arange(nc), 2)
        R = self.cells @ BRAVAIS
        self.positions = np.empty((2 * nc, 2))
        self.positions[0::2] = R
        self.positions[1::2] = R + DELTA["z"]

        self.nn_bonds = []
        self.j3_bonds = []
        for k, c in enumerate(self.cells):
            for off, label in NN_BONDS:
                self.nn_bonds.append(Bond(2 * k, 2 * self.cell(c + off) + 1, label))
            for off in J3_BONDS:
                self.j3_bonds.append(Bond(2 * k, 2 * self.cell(c + np.array(off)) + 1, "3"))

    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return 2 * self.ncells

    @property
    def supercell(self) -> np.ndarray:
        """Supercell vectors L1, L2 (rows) in Cartesian coordinates."""
        return self.T @ BRAVAIS

    def _reduce(self, n) -> tuple[int, int]:
        n = np.asarray(n, dtype=float)
        f = n @ self._Tinv
        f -= np.floor(f + 1e-9)
        r = np.rint(f @ self.T).astype(int)
        return int(r[0]), int(r[1])

    def _enumerate_cells(self) -> np.ndarray:
        span = int(np.abs(self.T).sum()) + 1
        seen = set()
        for n1 in range(-span, span + 1):
            for n2 in range(-span, span + 1):
                seen.add(self._reduce((n1, n2)))
        cells = np.array(sorted(seen), dtype=int)
        if len(cells) != self.ncells:
            raise RuntimeError("cell enumeration failed")
        return cells

    def cell(self, n) -> int:
        return self._index[self._reduce(n)]

    def translate(self, shift) -> np.ndarray:
        """Site permutation p with p[s] = image of site s under translation by
        `shift` (integer cell coordinates)."""
        shift = np.asarray(shift, dtype=int)
        p = np.empty(self.N, dtype=int)
        for k, c in enumerate(self.cells):
            k2 = self.cell(c + shift)
            p[2 * k] = 2 * k2
            p[2 * k + 1] = 2 * k2 + 1
        return p

    def momenta(self) -> np.ndarray:
        """Momenta q allowed by the torus, one per unit cell, as (Nc, 2)."""
        L = self.supercell
        G = 2 * np.pi * np.linalg.inv(L).T  # rows g_k with g_k . L_l = 2 pi delta_kl
        B = 2 * np.pi * np.linalg.inv(BRAVAIS).T  # primitive reciprocal vectors
        qs = {}
        span = int(np.abs(self.T).sum()) + 1
        for m1 in range(-span, span + 1):
            for m2 in range(-span, span + 1):
                q = m1 * G[0] + m2 * G[1]
                # fold into the first Brillouin zone's parallelogram for dedup
                f = q @ np.linalg.inv(B)
                key = tuple(np.round((f - np.floor(f + 1e-9)) * 1e6).astype(int))
                if key not in qs or np.linalg.norm(q) < np.linalg.norm(qs[key]):
                    qs[key] = q
        out = np.array(list(qs.values()))
        if len(out) != self.ncells:
            raise RuntimeError("momentum enumeration failed")
        return out[np.lexsort((out[:, 1], out[:, 0]))]

    # ------------------------------------------------------------------
    def check(self, include_j3: bool = True) -> list[str]:
        """Return a list of problems that make the torus unfaithful.

        With ``include_j3=False`` only the nearest-neighbour bonds are checked,
        which is enough for parameter sets with J3 = 0."""
        problems = []
        nn = [frozenset((b.i, b.j)) for b in self.nn_bonds]
        j3 = [frozenset((b.i, b.j)) for b in self.j3_bonds]
        if any(len(p) < 2 for p in nn + (j3 if include_j3 else [])):
            problems.append("self bond")
        if len(set(nn)) < len(nn):
            problems.append("repeated nearest-neighbour pair")
        if not include_j3:
            return problems
        if len(set(j3)) < len(j3):
            problems.append("repeated third-neighbour pair")
        if set(nn) & set(j3):
            problems.append("third-neighbour pair coincides with nearest-neighbour pair")
        return problems

    def min_image_distance(self) -> float:
        """Length of the shortest nonzero supercell lattice vector."""
        L = self.supercell
        best = np.inf
        for m1, m2 in itertools.product(range(-4, 5), repeat=2):
            if (m1, m2) != (0, 0):
                best = min(best, np.linalg.norm(m1 * L[0] + m2 * L[1]))
        return best

    def has_c3(self) -> bool:
        """True if the supercell lattice is invariant under 120-degree rotation."""
        rot = np.array([[np.cos(2 * np.pi / 3), -np.sin(2 * np.pi / 3)],
                        [np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)]])
        L = self.supercell
        Linv = np.linalg.inv(L)
        for v in L:
            c = (rot @ v) @ Linv
            if not np.allclose(c, np.rint(c), atol=1e-8):
                return False
        return True


def best_cluster(N: int, include_j3: bool = True, search: int | None = None) -> Cluster:
    """Most compact faithful torus with N sites.

    Every supercell lattice of index N/2 has a unique Hermite normal form
    [[a, b], [0, d]] with a d = N/2 and 0 <= b < d, so all of them are
    enumerated exactly.  Among those that pass :meth:`Cluster.check`, pick the
    one with the largest minimum image distance, preferring C3-symmetric
    supercells, then the most equilateral reduced basis.  Faithful
    third-neighbour bonds need N >= 14; pass ``include_j3=False`` for smaller
    test clusters.  (``search`` is accepted for backward compatibility.)
    """
    if N % 2:
        raise ValueError("honeycomb clusters have an even number of sites")
    n = N // 2
    best, best_key = None, None
    for a in range(1, n + 1):
        if n % a:
            continue
        d = n // a
        for b in range(d):
            T = gauss_reduce(np.array([[a, b], [0, d]]))
            cl = Cluster(T)
            if cl.check(include_j3):
                continue
            L = cl.supercell
            key = (round(cl.min_image_distance(), 6), cl.has_c3(),
                   -round(abs(np.linalg.norm(L[0]) - np.linalg.norm(L[1])), 6))
            if best_key is None or key > best_key:
                best, best_key = cl, key
    if best is None:
        raise ValueError(f"no faithful {N}-site cluster found")
    return best


def gauss_reduce(T: np.ndarray) -> np.ndarray:
    """Lagrange-Gauss reduction of the supercell basis (rows of T, in units of
    a1, a2) so that the rows are the two shortest independent vectors."""
    G = BRAVAIS @ BRAVAIS.T
    u, v = T[0].astype(int), T[1].astype(int)

    def norm2(x):
        return float(x @ G @ x)

    if norm2(u) > norm2(v):
        u, v = v, u
    while True:
        m = int(round(float(u @ G @ v) / norm2(u)))
        v = v - m * u
        if norm2(v) >= norm2(u):
            break
        u, v = v, u
    R = np.array([u, v])
    if np.linalg.det(R) < 0:
        R[1] = -R[1]
    return R
