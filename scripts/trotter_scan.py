"""Trotter error in S(q, omega) against the exact baseline.

    python scripts/trotter_scan.py --N 14 16 18 --params winter2017 --order 2 4

For each N, Trotter step dt and resolution dE (each dE sets its own window and
T, all within the longest T simulated), records

    eps = max_{q, omega} |S_trot - S_exact| / max S_exact,  S = sum_a S^aa,

and appends rows to data/trotter_scan.jsonl.  Reuses the exact correlations
from scripts/run_exact.py when data/exact_<params>_N<N>.npz exists.
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
from rucl3.classical.exact import Correlations, correlations, lanczos_ground_state  # noqa: E402
from rucl3.classical.spectrum import fourier, momentum_correlations  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import best_cluster  # noqa: E402
from rucl3.quantum.trotter import trotter_correlations  # noqa: E402
from rucl3.target import AccuracyTarget, DEFAULT  # noqa: E402

# Trotter steps per sampling interval (dt = sampling interval / k)
DEFAULT_K = {2: [2, 3, 4, 6, 8], 4: [1, 2, 3, 4]}


def spectrum_total(cl, corr, target, omegas):
    """sum_a S^aa(q, omega) on the time grid truncated to target.T."""
    n = int(np.searchsorted(corr.times, target.T + 1e-9))
    c = Correlations(corr.times[:n], corr.ref_sites, corr.C[..., :n], 0, 0)
    Sqt = momentum_correlations(cl, c, cl.momenta())
    S = fourier(c.times, Sqt, target.window(c.times), omegas)
    return np.einsum("qaaw->qw", S).real


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--order", type=int, nargs="+", default=[2])
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="Trotter steps per sampling interval")
    ap.add_argument("--dE", type=float, nargs="+", default=[DEFAULT.dE, 1.0, 2.0],
                    help="resolutions to evaluate; the smallest sets the simulated time")
    ap.add_argument("--omega-max", type=float, default=DEFAULT.omega_max)
    a = ap.parse_args(argv)

    p = P.get(a.params)
    targets = [AccuracyTarget(dE=d, eps=DEFAULT.eps, omega_max=a.omega_max) for d in a.dE]
    longest = max(targets, key=lambda t: t.T)
    omegas = np.linspace(0, a.omega_max, 601)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)

    for N in a.N:
        cl = best_cluster(N, include_j3=bool(p.J3))
        model = SpinModel(cl, p)
        H = model.operator()
        gs = lanczos_ground_state(H)
        times = longest.times()
        f = out / f"exact_{a.params}_N{N}.npz"
        saved = np.load(f) if f.exists() else None
        if (saved is not None and saved["times"].shape == times.shape
                and np.allclose(saved["times"], times) and np.array_equal(saved["T"], cl.T)):
            ex = Correlations(times, np.array([0, 1]), saved["C"], 0, 0)
            print(f"N={N}: exact correlations from {f.name}", flush=True)
        else:
            ex = correlations(H, gs, N, times)
            print(f"N={N}: exact correlations {ex.seconds:.0f}s", flush=True)
        ref = {t.dE: spectrum_total(cl, ex, t, omegas) for t in targets}
        for order in a.order:
            for k in (a.steps or DEFAULT_K[order]):
                dt = (times[1] - times[0]) / k
                tr = trotter_correlations(model, gs, times, dt=dt * (1 + 1e-9), order=order)
                for t in targets:
                    S = spectrum_total(cl, tr, t, omegas)
                    eps = float(np.abs(S - ref[t.dE]).max() / ref[t.dE].max())
                    row = {"N": N, "params": a.params, "order": order, "dt": dt,
                           "dE": t.dE, "T": t.T, "eps": eps,
                           "corr_err": float(np.abs(tr.C - ex.C).max()),
                           "seconds": tr.seconds, "rotations": tr.matvecs}
                    print(json.dumps(row), flush=True)
                    with open(out / "trotter_scan.jsonl", "a") as fh:
                        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
