"""Exact classical baseline for one cluster size: ground state, correlations
on the experiment-defined time grid, S(q, omega), and timings.

    python scripts/run_exact.py --N 18 --params winter2017
    python scripts/run_exact.py --N 24 --params winter2017 --threads 8   # laptop / Colab

Writes data/exact_<params>_N<N>.npz and prints a one-line JSON summary.
Memory at N = 24: about (m + 6) x 268 MB with the default Krylov m = 12,
i.e. ~5 GB.  Lower --krylov-m if RAM is short (more matvecs, same answer).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401  (must precede numpy: pins BLAS threads)
import numpy as np  # noqa: E402
from rucl3.classical.exact import correlations, ground_manifold  # noqa: E402
from rucl3.classical.spectrum import fourier, momentum_correlations  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import best_cluster  # noqa: E402
from rucl3.target import AccuracyTarget, DEFAULT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--dE", type=float, default=DEFAULT.dE, help="resolution FWHM, meV")
    ap.add_argument("--eps", type=float, default=DEFAULT.eps)
    ap.add_argument("--omega-max", type=float, default=DEFAULT.omega_max)
    ap.add_argument("--T", type=float, default=None, help="override total time (meV^-1)")
    ap.add_argument("--krylov-m", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ground-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args(argv)

    p = P.get(a.params)
    target = AccuracyTarget(dE=a.dE, eps=a.eps, omega_max=a.omega_max)
    cl = best_cluster(a.N, include_j3=bool(p.J3))
    model = SpinModel(cl, p)
    H = model.operator(threads=a.threads)
    print(f"N={a.N} T={cl.T.tolist()} C3={cl.has_c3()} bonds={len(model.bonds)} "
          f"T_sim={target.T:.2f}/meV ({target.T_ps:.2f} ps) n_t={target.n_times}", flush=True)

    t0 = time.perf_counter()
    H.matvec(np.ones(model.dim, dtype=complex))  # compile + warm up
    t_mv = time.perf_counter()
    for _ in range(3):
        H.matvec(np.ones(model.dim, dtype=complex))
    t_mv = (time.perf_counter() - t_mv) / 3
    print(f"matvec {t_mv:.3f}s (setup {time.perf_counter() - t0:.1f}s)", flush=True)

    gs = ground_manifold(H)
    print(f"E0={gs.energy:.8f} E0/N={gs.energy / a.N:.6f} degeneracy={gs.degeneracy} "
          f"gap={gs.gap:.5f} res={max(s.residual for s in gs.states):.1e} "
          f"matvecs={gs.matvecs} {gs.seconds:.1f}s", flush=True)
    summary = {
        "N": a.N, "params": a.params, "cluster": cl.T.tolist(), "E0": gs.energy,
        "degeneracy": gs.degeneracy, "gap": gs.gap, "gs_seconds": gs.seconds,
        "gs_matvecs": gs.matvecs, "matvec_seconds": t_mv,
        "dE": a.dE, "T": target.T, "n_times": target.n_times,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.ground_only:
        times = target.times() if a.T is None else np.linspace(0, a.T, int(a.T / target.dt) + 1)
        corr = correlations(H, gs, a.N, times, m=a.krylov_m, tol=a.tol, progress=True)
        q = cl.momenta()
        Sqt = momentum_correlations(cl, corr, q)
        omegas = np.linspace(0, target.omega_max, 601)
        Sqw = fourier(times, Sqt, target.window(times), omegas)
        summary |= {
            "corr_seconds": corr.seconds, "corr_matvecs": corr.matvecs,
            # per ground state, so sizes with different degeneracy compare
            "seconds_per_unit_time": corr.seconds / times[-1] / gs.degeneracy,
            "S_peak": float(np.einsum("qaaw->qw", Sqw).real.max()),
        }
        np.savez_compressed(out / f"exact_{a.params}_N{a.N}.npz", times=times, C=corr.C,
                            q=q, omegas=omegas, Sqw=Sqw, T=cl.T, E0=gs.energy,
                            positions=cl.positions, summary=json.dumps(summary))
    summary["max_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(json.dumps(summary), flush=True)
    with open(out / "exact_runs.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
