"""Inputs for the ground-state preparation line item: the spectral gap and the
overlap of the exact ground state with the best simple product state.

    python scripts/gs_prep_data.py --N 14 16 18 --params winter2017

Trial states are product states |n, s_i> with every spin along +-n, the sign
s_i following a zigzag pattern (three orientations), the Neel pattern, or a
ferromagnet; n is optimised on the sphere.  Rows go to data/gs_prep.jsonl.
On a finite cluster the ground state is a symmetric superposition of the
symmetry-related ordered states, so the overlap with a single product state
is suppressed by about sqrt(number of partners); both the raw value and the
best value found are recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import Cluster, best_cluster  # noqa: E402


def patterns(cl: Cluster) -> dict[str, np.ndarray]:
    """Sign patterns compatible with the torus."""
    n1, n2 = cl.cells[:, 0], cl.cells[:, 1]
    out = {"ferro": np.ones(cl.N)}
    neel = np.empty(cl.N)
    neel[0::2], neel[1::2] = 1, -1
    out["neel"] = neel
    # zigzag-gamma: antiparallel across gamma bonds, parallel across the others.
    # A(R)-B(R) is z, A(R)-B(R-a1) is x, A(R)-B(R-a2) is y.
    for label, (c1, c2, flip) in {"zigzag_z": (1, 1, -1), "zigzag_x": (1, 0, 1),
                                  "zigzag_y": (0, 1, 1)}.items():
        # chi(R) = (-1)^(c1 n1 + c2 n2); s_A = chi, s_B = flip * chi
        if any((c1 * row[0] + c2 * row[1]) % 2 for row in cl.T):
            continue  # pattern does not fit on this torus
        chi = (-1.0) ** (c1 * n1 + c2 * n2)
        s = np.empty(cl.N)
        s[0::2], s[1::2] = chi, flip * chi
        out[label] = s
    return out


def spinor(theta, phi, sign):
    """|+n> for sign = +1, |-n> for sign = -1 (basis: up = bit 0)."""
    if sign > 0:
        return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.array([-np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)])


def overlap(psi, N, signs, theta, phi):
    v = psi.reshape([2] * N)  # axis 0 = highest bit = site N-1
    for site in range(N - 1, -1, -1):
        v = np.tensordot(spinor(theta, phi, signs[site]).conj(), v, axes=(0, 0))
    return abs(complex(v))


def best_overlap(psi, N, signs, starts=6, seed=0):
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(starts):
        x0 = [np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)]
        r = minimize(lambda x: -overlap(psi, N, signs, *x), x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-8})
        best = max(best, -r.fun)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--params", default="winter2017")
    a = ap.parse_args(argv)
    p = P.get(a.params)
    for N in a.N:
        cl = best_cluster(N, include_j3=bool(p.J3))
        H = SpinModel(cl, p).operator()
        op = LinearOperator((H.dim, H.dim), matvec=H.matvec, dtype=complex)
        E, V = eigsh(op, k=4, which="SA", tol=1e-10)
        order = np.argsort(E)
        E, V = E[order], V[:, order]
        psi = V[:, 0]
        ov = {name: best_overlap(psi, N, s) for name, s in patterns(cl).items()}
        row = {"N": N, "params": a.params, "E": E.tolist(), "gap": float(E[1] - E[0]),
               "overlaps": ov, "best": max(ov.values())}
        print(json.dumps(row), flush=True)
        with open(ROOT / "data" / "gs_prep.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
