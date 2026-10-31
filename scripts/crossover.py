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
from rucl3.analysis.mps_fit import fit_mps  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.quantum.resources import GroundStatePrep  # noqa: E402
from rucl3.target import AccuracyTarget, DEFAULT  # noqa: E402


def inelastic_peak(data, params, dE):
    """Reference spectral weight for the accuracy target: the smallest, over
    the measured cluster sizes, of the peak of sum_a S^aa(q, omega) at
    omega > 2 dE.  The quasi-elastic weight below that (on tori that fit the
    zigzag order it piles up at the M point) is excluded, as in an INS
    analysis of the inelastic spectrum; taking the smallest size-wise peak is
    conservative (more shots)."""
    by_N = {}
    for f in sorted(data.glob(f"exact_{params}_N*.npz")):
        d = np.load(f)
        S = np.einsum("qaaw->qw", d["Sqw"]).real
        by_N[int(f.stem.split("_N")[-1])] = float(S[:, d["omegas"] > 2 * dE].max())
    if not by_N:
        raise FileNotFoundError(f"no exact spectra for {params}")
    return min(by_N.values()), by_N


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
    S_peak, S_peak_by_N = inelastic_peak(data, a.params, a.dE)
    scan = [r for r in read_jsonl(data / "trotter_scan.jsonl") if r["params"] == a.params]
    prep_of_N, prep_info = prep_model(
        [r for r in read_jsonl(data / "gs_prep.jsonl") if r["params"] == a.params])

    mps_model = None
    if (data / "mps" / "runs.jsonl").exists():
        try:
            mps_model = fit_mps(data / "mps", a.params, a.eps, a.dE)
        except (ValueError, FileNotFoundError) as e:
            print(f"MPS: {e}")
    out = {"params": a.params, "dE": a.dE, "eps": a.eps, "T": target.T, "S_peak": S_peak,
           "S_peak_by_N": S_peak_by_N,
           "mps_fit": mps_model.__dict__ if mps_model else None,
           "classical_fit": cfit.__dict__, "prep": prep_info, "curves": []}
    Ns = range(16, a.Nmax + 1, 2)
    for order in (2, 4):
        try:
            tf = fit_trotter(scan, order, conservative=True)
            free = fit_trotter(scan, order) if len({r["N"] for r in scan}) > 1 else None
        except ValueError as e:
            print(f"order {order}: {e}")
            continue
        out[f"trotter_fit_{order}"] = {**tf.spec.__dict__, "beta": tf.beta,
                                       "rms": tf.rms_log_residual, "n": tf.n_points,
                                       "free_alpha": free.spec.alpha if free else None}
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
            if mps_model:
                mps_s = mps_model.seconds(res.N, target)
                below = np.nonzero(res.quantum_s < mps_s)[0]
                curve["mps_s"] = mps_s.tolist()
                curve["N_star_mps"] = int(res.N[below[0]]) if len(below) else None
            out["curves"].append(curve)
            print(f"order {order} p={perr:g}: N*={res.N_star} (with prep {res.N_star_prep})"
                  + (f", vs MPS {curve['N_star_mps']}" if mps_model else ""), flush=True)
    suffix = "" if (a.eps == DEFAULT.eps and a.dE == DEFAULT.dE) else f"_dE{a.dE:g}_eps{a.eps:g}"
    path = data / f"crossover_{a.params}{suffix}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
