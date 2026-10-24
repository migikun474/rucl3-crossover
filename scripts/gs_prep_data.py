"""Inputs for the ground-state preparation line item: the gap above the
ground level and the overlap of the ground level with the best product state.

    python scripts/gs_prep_data.py --N 14 16 18 --params winter2017

The trial state is the product state |phi_1 ... phi_N> maximising
sum_k |<phi|psi_k>|^2 over the (possibly degenerate) ground manifold
{psi_k}, found by alternating single-site updates: with all other spins
fixed the best phi_j is the top eigenvector of a 2x2 matrix.  Starts are
random plus the ordered patterns (zigzag, Neel, ferro) that fit the torus;
zigzag needs an even number of cells along its stripes, so it does not fit
the 14- and 18-site tori.  The ground level and gap come from deflated
Lanczos (ARPACK with small k undercounts degeneracies).  Rows go to
data/gs_prep.jsonl.
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

from rucl3.classical.exact import ground_manifold  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import Cluster, best_cluster  # noqa: E402


def patterns(cl: Cluster) -> dict[str, np.ndarray]:
    """Collinear sign patterns compatible with the torus."""
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


def _contract_except(psi, N, phis, k):
    """u_s = <phi_others (x) e_s at site k | psi>."""
    v = psi.reshape([2] * N)  # axis a <-> site N-1-a
    for ax in range(N - 1, -1, -1):  # drop trailing axes so earlier ones keep their index
        site = N - 1 - ax
        if site != k:
            v = np.tensordot(v, phis[site].conj(), axes=(ax, 0))
    return v


def product_overlap(ground, N, phis):
    v = [psi.reshape([2] * N) for psi in ground]
    tot = 0.0
    for t in v:
        for ax in range(N - 1, -1, -1):
            t = np.tensordot(t, phis[N - 1 - ax].conj(), axes=(ax, 0))
        tot += abs(complex(t)) ** 2
    return float(np.sqrt(tot))


def best_product(ground, N, phis, sweeps=30, tol=1e-9):
    phis = [p / np.linalg.norm(p) for p in phis]
    last = -1.0
    for _ in range(sweeps):
        for k in range(N):
            M = np.zeros((2, 2), dtype=complex)
            for psi in ground:
                u = _contract_except(psi, N, phis, k)
                M += np.outer(u, u.conj())
            w, V = np.linalg.eigh(M)
            phis[k] = V[:, -1]
            val = float(np.sqrt(max(w[-1], 0.0)))
        if val - last < tol:
            break
        last = val
    return val, phis


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--random-starts", type=int, default=4)
    a = ap.parse_args(argv)
    p = P.get(a.params)
    rng = np.random.default_rng(0)
    for N in a.N:
        cl = best_cluster(N, include_j3=bool(p.J3))
        gm = ground_manifold(SpinModel(cl, p).operator())
        ground = [s.vector for s in gm.states]
        starts = {}
        for name, s in patterns(cl).items():
            # seed the ordered pattern along a few directions
            for d, (th, ph) in enumerate([(0.0, 0.0), (np.pi / 2, 0.0), (np.arccos(1 / np.sqrt(3)), np.pi / 4)]):
                starts[f"{name}_{d}"] = [spinor(th, ph, s[i]) for i in range(N)]
        for r in range(a.random_starts):
            starts[f"random_{r}"] = list(rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2)))
        results = {}
        for name, phis in starts.items():
            seed_val = product_overlap(ground, N, [x / np.linalg.norm(x) for x in phis])
            val, _ = best_product(ground, N, phis)
            results[name] = (seed_val, val)
        best_name = max(results, key=lambda k: results[k][1])
        row = {"N": N, "params": a.params, "cluster": cl.T.tolist(), "E0": gm.energy,
               "degeneracy": gm.degeneracy, "gap": gm.gap,
               "patterns_fit": sorted(patterns(cl)),
               "seed_overlaps": {k: v[0] for k, v in results.items()},
               "best": results[best_name][1], "best_from": best_name}
        print(json.dumps(row), flush=True)
        with open(ROOT / "data" / "gs_prep.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
