"""MPS (TDVP) baseline: correlations on honeycomb cylinders at several bond
dimensions, timings, and (for small cylinders) the exact reference.

    python scripts/mps_scan.py --W 2 --L 4 --chi 16 32 64 128 --exact
    python scripts/mps_scan.py --W 3 --L 3 --chi 32 64 --exact

One component (S^z) from one bulk reference site is evolved; the full
spectrum needs 3 components x 2 sublattices, i.e. 6 such runs, which the cost
model accounts for.  Each (W, chi) run is saved to data/mps/ as soon as it
finishes, so a run stopped by a time limit loses nothing else.  The ground
state is computed once per cylinder at --chi-gs and cached.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401
import numpy as np  # noqa: E402
from rucl3.classical import mps  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.lattice import Cylinder  # noqa: E402
from rucl3.target import DEFAULT  # noqa: E402

OUT = ROOT / "data" / "mps"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--chi", type=int, nargs="+", required=True)
    ap.add_argument("--chi-gs", type=int, default=256)
    ap.add_argument("--exact", action="store_true", help="also compute the exact reference")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    p = P.get(a.params)
    cyl = Cylinder(a.W, a.L)
    tag = f"{a.params}_W{a.W}_L{a.L}"
    M = mps.tenpy_model(cyl, p)
    times = DEFAULT.times()
    j = 2 * ((a.L // 2) * a.W)  # A site in the middle ring

    gs_file = OUT / f"gs_{tag}.pkl"
    if gs_file.exists():
        psi, E0, gs_s = pickle.loads(gs_file.read_bytes())
    else:
        psi, E0, gs_s = mps.ground_state(M, chi_max=a.chi_gs)
        gs_file.write_bytes(pickle.dumps((psi, E0, gs_s)))
    print(f"{tag}: N={cyl.N} MPO chi={max(M.H_MPO.chi)} E0={E0:.8f} gs {gs_s:.0f}s "
          f"(chi {max(psi.chi)})", flush=True)

    if a.exact:
        from rucl3.classical.exact import (KrylovPropagator, apply_pauli, ground_manifold,
                                           pauli_overlap)
        from rucl3.model.hamiltonian import SpinModel
        H = SpinModel(cyl, p).operator()
        gm = ground_manifold(H)
        if gm.degeneracy > 1:
            print(f"  WARNING: {gm.degeneracy}-fold ground level; DMRG picks one state", flush=True)
        t0 = time.perf_counter()
        prop = KrylovPropagator(H, m=16, tol=1e-10)
        C = np.zeros((cyl.N, len(times)), dtype=complex)
        for g in gm.states:  # only the S^z component of the reference site, as the MPS run
            phi = apply_pauli(g.vector, j, "Z")
            for k, t in enumerate(times):
                if k:
                    phi = prop.evolve(phi, t - times[k - 1])
                ph = np.exp(1j * g.energy * t) / (4 * gm.degeneracy)
                C[:, k] += [ph * pauli_overlap(g.vector, phi, i, "Z") for i in range(cyl.N)]
        np.savez_compressed(OUT / f"exact_{tag}.npz", times=times, C=C, E0=gm.energy,
                            degeneracy=gm.degeneracy, gap=gm.gap)
        print(f"  exact: E0={gm.energy:.8f} deg={gm.degeneracy} gap={gm.gap:.4f} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    for chi in a.chi:
        mc = mps.correlations(M, psi, E0, j, times, chi_max=chi)
        np.savez_compressed(OUT / f"mps_{tag}_chi{chi}.npz", times=times, C=mc.C,
                            chi=mc.chi, seconds=mc.seconds, j=j)
        row = {"params": a.params, "W": a.W, "L": a.L, "N": cyl.N, "chi": chi,
               "mpo_chi": int(max(M.H_MPO.chi)), "E0": E0, "seconds": float(mc.seconds[-1]),
               "n_times": len(times), "substeps": 2}
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  chi={chi}: {mc.seconds[-1]:.0f}s", flush=True)


if __name__ == "__main__":
    main()
