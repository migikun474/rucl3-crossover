"""Fit the measured data and compute the crossover N* (first-cut, Thu 29 Oct).

    python scripts/crossover.py --params winter2017

Reads data/exact_runs.jsonl, data/trotter_scan.jsonl, data/gs_prep.jsonl;
writes data/crossover_<params>.json (curves, N*, fits, India gap factors).
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
from rucl3.analysis.crossover import crossover, india_gap  # noqa: E402
from rucl3.analysis.fits import fit_classical, fit_trotter, read_jsonl  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.quantum.resources import GroundStatePrep  # noqa: E402
from rucl3.target import AccuracyTarget, DEFAULT  # noqa: E402


def prep_model(rows):
    """Gap: mean over sizes (finite-size gaps do not extrapolate reliably).
    Overlap: gamma(N) = gamma0 exp(-kappa N), fitted to the best product state."""
    if len(rows) < 2:
        return None, {}
    N = np.array([r["N"] for r in rows], float)
    lg = np.log([r["best"] for r in rows])
    kappa, lng0 = np.polyfit(N, lg, 1)
    gap = float(np.mean([r["gap"] for r in rows]))
    info = {"gap_mean": gap, "gamma0": float(np.exp(lng0)), "kappa": float(-kappa)}
    return (lambda n: GroundStatePrep(gap=gap, overlap=min(1.0, float(np.exp(lng0 + kappa * n))))), info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--dE", type=float, default=DEFAULT.dE)
    ap.add_argument("--eps", type=float, default=DEFAULT.eps)
    ap.add_argument("--Nmax", type=int, default=100)
    a = ap.parse_args(argv)
    p = P.get(a.params)
    target = AccuracyTarget(dE=a.dE, eps=a.eps)
    data = ROOT / "data"

    exact = [r for r in read_jsonl(data / "exact_runs.jsonl") if r["params"] == a.params]
    cfit = fit_classical(exact)
    S_peak = max(exact, key=lambda r: r["N"])["S_peak"]
    scan = [r for r in read_jsonl(data / "trotter_scan.jsonl") if r["params"] == a.params]
    prep_of_N, prep_info = prep_model(
        [r for r in read_jsonl(data / "gs_prep.jsonl") if r["params"] == a.params])

    out = {"params": a.params, "dE": a.dE, "eps": a.eps, "T": target.T, "S_peak": S_peak,
           "classical_fit": cfit.__dict__, "prep": prep_info, "curves": []}
    Ns = range(16, a.Nmax + 1, 4)
    for order in (2, 4):
        try:
            tf = fit_trotter(scan, order)
        except ValueError as e:
            print(f"order {order}: {e}")
            continue
        out[f"trotter_fit_{order}"] = {**tf.spec.__dict__, "beta": tf.beta,
                                       "rms": tf.rms_log_residual, "n": tf.n_points}
        for perr in (1e-3, 1e-4):
            res = crossover(p, target, tf.spec, cfit, S_peak, Ns, perr, prep_of_N)
            k = (np.nonzero(res.N == res.N_star)[0] if res.N_star else [])
            gap = india_gap(int(res.physical_qubits[k[0]]), perr) if len(k) else None
            curve = {"order": order, "physical_error": perr, "N": res.N.tolist(),
                     "classical_s": res.classical_s.tolist(), "quantum_s": res.quantum_s.tolist(),
                     "quantum_prep_s": res.quantum_prep_s.tolist(),
                     "physical_qubits": res.physical_qubits.tolist(),
                     "N_star": res.N_star, "N_star_prep": res.N_star_prep,
                     "india": gap.__dict__ if gap else None}
            out["curves"].append(curve)
            print(f"order {order} p={perr:g}: N*={res.N_star} (with prep {res.N_star_prep})",
                  flush=True)
    path = data / f"crossover_{a.params}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
