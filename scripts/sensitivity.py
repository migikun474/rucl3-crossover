"""Sensitivity sweeps of the crossover N*.

    python scripts/sensitivity.py

Varies one assumption at a time around the baseline (Winter et al. set,
dE = 0.5 meV, eps = 5 %, second-order Trotter, gate-based qubits with
nanosecond gates and p = 1e-4, one classical workstation) and also runs the
full grid over parameter sets x dE x eps x qubit model.  Writes
data/sensitivity.json and paper/table2.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401
import numpy as np  # noqa: E402
from rucl3.analysis.crossover import crossover  # noqa: E402
from rucl3.analysis.fits import fit_classical, fit_trotter, read_jsonl  # noqa: E402
from rucl3.classical.exact import Correlations  # noqa: E402
from rucl3.classical.spectrum import fourier, momentum_correlations  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.lattice import Cluster  # noqa: E402
from rucl3.target import AccuracyTarget  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from crossover import prep_model  # noqa: E402

DATA = ROOT / "data"
YEAR = 3.156e7
SETS = ["winter2017", "ran2017", "laurell2020"]
LABEL = {"winter2017": "Winter", "ran2017": "Ran", "laurell2020": "Laurell--Okamoto"}
DES = [0.5, 1.0, 2.0]
EPSS = [0.01, 0.02, 0.05, 0.10]
QMODELS = {"ns, $p=10^{-4}$": 1e-4, "ns, $p=10^{-3}$": 1e-3,
           "$\\mu$s, $p=10^{-4}$": "qubit_gate_us_e4", "$\\mu$s, $p=10^{-3}$": "qubit_gate_us_e3"}
SPEEDUPS = [1.0, 1e2, 1e4]
NS = range(16, 121, 2)


def inelastic_peak(params, dE):
    """As scripts/crossover.py, but recomputing each spectrum for window dE."""
    peaks = []
    for f in sorted(DATA.glob(f"exact_{params}_N*.npz")):
        d = np.load(f)
        cl = Cluster(d["T"])
        tg = AccuracyTarget(dE=dE, eps=0.05)
        n = int(np.searchsorted(d["times"], tg.T + 1e-9))
        c = Correlations(d["times"][:n], np.array([0, 1]), d["C"][..., :n], 0, 0)
        S = fourier(c.times, momentum_correlations(cl, c, cl.momenta()), tg.window(c.times),
                    d["omegas"])
        S = np.einsum("qaaw->qw", S).real
        peaks.append(float(S[:, d["omegas"] > 2 * dE].max()))
    return min(peaks)


def first_below(N, q, c):
    idx = np.nonzero(q < c)[0]
    return int(N[idx[0]]) if len(idx) else None


def main():
    rows = []
    for params in SETS:
        p = P.get(params)
        cfit = fit_classical([r for r in read_jsonl(DATA / "exact_runs.jsonl")
                              if r["params"] == params])
        scan = [r for r in read_jsonl(DATA / "trotter_scan.jsonl") if r["params"] == params]
        prep_rows = [r for r in read_jsonl(DATA / "gs_prep.jsonl") if r["params"] == params]
        prep_of_N = prep_model(prep_rows)[0] if len(prep_rows) >= 3 else None
        orders = [o for o in (2, 4) if sum(r["order"] == o for r in scan) >= 3]
        for dE in DES:
            S_peak = inelastic_peak(params, dE)
            for eps in EPSS:
                target = AccuracyTarget(dE=dE, eps=eps)
                for order in orders:
                    spec = fit_trotter(scan, order, conservative=True).spec
                    for qname, qm in QMODELS.items():
                        res = crossover(p, target, spec, cfit, S_peak, NS, qm, prep_of_N)
                        for f in SPEEDUPS:
                            Ns = first_below(res.N, res.quantum_s, res.classical_s / f)
                            i = list(res.N).index(Ns) if Ns else None
                            rows.append({
                                "params": params, "dE": dE, "eps": eps, "order": order,
                                "qubits": qname, "speedup": f, "N_star": Ns,
                                "N_star_prep": (first_below(res.N, res.quantum_prep_s,
                                                            res.classical_s / f)
                                                if prep_of_N else None),
                                "runtime_yr": res.quantum_s[i] / YEAR if Ns else None,
                                "phys_qubits": int(res.physical_qubits[i]) if Ns else None,
                            })
                print(f"{params} dE={dE} done", flush=True)
    (DATA / "sensitivity.json").write_text(json.dumps(rows, indent=0))
    write_table(rows)


def pick(rows, **kw):
    base = dict(params="winter2017", dE=0.5, eps=0.05, order=2,
                qubits="ns, $p=10^{-4}$", speedup=1.0)
    base.update(kw)
    hits = [r for r in rows if all(r[k] == v for k, v in base.items())]
    return hits[0] if hits else None


def fmt_time(yr):
    if yr is None:
        return "--"
    return f"{yr:.0f}~yr" if yr >= 1 else f"{yr * 365:.0f}~d"


def write_table(rows):
    lines = []

    def add(label, r, key="N_star"):
        if r is None:
            return
        n = r[key]
        q = f"{r['phys_qubits'] / 1e6:.2g}\\,M" if r["phys_qubits"] else "--"
        t = fmt_time(r["runtime_yr"])
        if key != "N_star":  # runtime and qubits were recorded at the plain N* only
            q = t = "--"
        lines.append(f"{label} & {n if n else '$>$120'} & {t} & {q} \\\\")

    add("Baseline (\\S\\ref{sec:results})", pick(rows))
    lines.append("\\midrule")
    for eps in (0.01, 0.02, 0.10):
        add(f"Accuracy $\\varepsilon={100 * eps:.0f}\\%$", pick(rows, eps=eps))
    for dE in (1.0, 2.0):
        add(f"Resolution $\\delta E={dE:g}$\\,meV", pick(rows, dE=dE))
    lines.append("\\midrule")
    add("Physical error $p=10^{-3}$", pick(rows, qubits="ns, $p=10^{-3}$"))
    add("$\\mu$s gates, $p=10^{-4}$", pick(rows, qubits="$\\mu$s, $p=10^{-4}$"))
    add("$\\mu$s gates, $p=10^{-3}$", pick(rows, qubits="$\\mu$s, $p=10^{-3}$"))
    add("Fourth-order Trotter", pick(rows, order=4))
    add("With ground-state prep.", pick(rows), key="N_star_prep")
    lines.append("\\midrule")
    add("Classical $10^2\\times$ faster", pick(rows, speedup=1e2))
    add("Classical $10^4\\times$ faster", pick(rows, speedup=1e4))
    lines.append("\\midrule")
    for params in SETS[1:]:
        add(f"{LABEL[params]} parameters", pick(rows, params=params))
    with open(ROOT / "paper" / "table2.tex", "w") as f:
        f.write("% generated by scripts/sensitivity.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n")
        f.write("Variation & $N^*$ & Runtime & Phys.\\ qubits \\\\\n\\midrule\n")
        f.write("\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # grid extremes for the text
    one = [r for r in rows if r["speedup"] == 1.0]
    ns = [r["N_star"] for r in one if r["N_star"]]
    fast = [r["N_star"] for r in one if r["N_star"] and isinstance(QMODELS[r["qubits"]], float)]
    macros = {"gridCount": str(len(one)), "gridMin": str(min(ns)), "gridMax": str(max(ns)),
              "gridFastMax": str(max(fast)), "gridNone": str(sum(r["N_star"] is None for r in one))}
    with open(ROOT / "paper" / "sensitivity_numbers.tex", "w") as f:
        f.write("% generated by scripts/sensitivity.py -- do not edit by hand\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("grid (one workstation):", macros)
    print(open(ROOT / "paper" / "table2.tex").read())


if __name__ == "__main__":
    if "--table-only" in sys.argv:  # rebuild the table from data/sensitivity.json
        write_table(json.loads((DATA / "sensitivity.json").read_text()))
    else:
        main()
