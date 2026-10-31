"""Write paper/numbers.tex (LaTeX macros) and paper/table1.tex from the data.

    python scripts/make_numbers.py --params winter2017 ran2017

Every number quoted in the paper text comes from here, so the abstract,
figures and table cannot drift apart.  Run scripts/crossover.py first.
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
from rucl3.analysis.crossover import _terms, india_gap, NQM_QUBITS, QPIAI_TWO_QUBIT_FIDELITY  # noqa: E402
from rucl3.analysis.fits import fit_trotter, read_jsonl  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.quantum.resources import quantum_cost, step_profile  # noqa: E402
from rucl3.target import AccuracyTarget  # noqa: E402

YEAR = 3.156e7
LABEL = {"winter2017": "Winter", "ran2017": "Ran", "laurell2020": "Laurell", "kimkee2016": "KimKee"}


WORDS = {8: "Eight", 12: "Twelve", 14: "Fourteen", 16: "Sixteen", 18: "Eighteen",
         20: "Twenty", 24: "TwentyFour", 32: "ThirtyTwo", 64: "SixtyFour",
         128: "OneTwentyEight", 256: "TwoFiftySix"}


def sig(x, n=2):
    """Number with n significant figures, LaTeX-friendly (x10^k for large/small)."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e < 4:
        return f"{x:.{max(n - 1 - e, 0)}f}"
    m = x / 10 ** e
    return f"{m:.{n - 1}f}\\times10^{{{e}}}"


def human_time(s):
    for unit, v in [("yr", YEAR), ("d", 86400), ("h", 3600), ("min", 60)]:
        if s >= v:
            return f"{sig(s / v)}~{unit}"
    return f"{sig(s)}~s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", default=["winter2017"])
    ap.add_argument("--order", type=int, default=2)
    a = ap.parse_args(argv)
    data = ROOT / "data"
    out = ROOT / "paper"
    out.mkdir(exist_ok=True)
    macros = {}
    rows = []

    def m(name, value):
        macros[name] = value

    for pname in a.params:
        d = json.loads((data / f"crossover_{pname}.json").read_text())
        p = P.get(pname)
        target = AccuracyTarget(dE=d["dE"], eps=d["eps"])
        scan = [r for r in read_jsonl(data / "trotter_scan.jsonl") if r["params"] == pname]
        tf = fit_trotter(scan, a.order, conservative=True)
        L = LABEL[pname]
        exact = {r["N"]: r for r in read_jsonl(data / "exact_runs.jsonl") if r["params"] == pname}
        for N, r in exact.items():
            m(f"tExact{L}N{WORDS[N]}", human_time(r["corr_seconds"]))
            m(f"degen{L}N{WORDS[N]}", str(r["degeneracy"]))
        m(f"exactSizes{L}", ", ".join(str(n) for n in sorted(exact)))
        m(f"Speak{L}", sig(d["S_peak"]))
        m(f"trotA{L}", sig(tf.spec.A))
        m(f"trotP{L}", f"{tf.spec.p:.2f}")
        m(f"trotNpts{L}", str(tf.n_points))
        fa = d.get(f"trotter_fit_{a.order}", {}).get("free_alpha")
        m(f"trotFreeAlpha{L}", f"{fa:.2f}" if fa is not None else "--")
        mf = d.get("mps_fit")
        if mf:
            from rucl3.analysis.mps_fit import MPSCostModel
            mm = MPSCostModel(**mf)
            for W in (2, 3):
                m(f"mpsS{L}W{WORDS.get(W, W) if W > 3 else ['', '', 'Two', 'Three'][W]}",
                  f"{mf['s'][0] + mf['s'][1] * W:.2f}")
            for pt in mf["points"]:
                if pt["T_reach"] > 0:
                    m(f"mpsReach{L}W{['', '', 'Two', 'Three'][pt['W']]}Chi{WORDS.get(pt['chi'], 'X')}",
                      f"{pt['T_reach']:.1f}")
            Nm = None
            for c in d["curves"]:
                if c["order"] == a.order and c.get("N_star_mps"):
                    Nm = c["N_star_mps"]
            if Nm:
                W = np.sqrt(Nm / 2)
                m(f"mpsChiAtNstar{L}", sig(mm.chi_needed(W, target.T)))
                m(f"mpsWAtNstar{L}", f"{W:.1f}")
            m(f"mpsChiWTwo{L}", sig(mm.chi_needed(2, target.T)))
            m(f"mpsChiWThree{L}", sig(mm.chi_needed(3, target.T)))
        cf = d["classical_fit"]
        m(f"classBase{L}", f"{cf['base_free']:.2f}")
        if d["prep"]:
            m(f"prepGap{L}", sig(d["prep"]["gap_mean"]))
            m(f"prepKappa{L}", sig(d["prep"]["kappa"]))
        for c in d["curves"]:
            if c["order"] != a.order:
                continue
            tag = "A" if c["physical_error"] == 1e-3 else "B"  # A: 1e-3, B: 1e-4
            Ns = c["N_star"]
            m(f"Nstar{L}{tag}", str(Ns) if Ns else "$>$100")
            if "N_star_mps" in c:
                m(f"NstarMPS{L}{tag}", str(c["N_star_mps"]) if c["N_star_mps"] else "$>$100")
            # classical machine faster by a factor f (cluster instead of one workstation)
            Narr, cs, qs = np.array(c["N"]), np.array(c["classical_s"]), np.array(c["quantum_s"])
            for f, word in [(1e2, "Hundred"), (1e4, "TenThousand")]:
                below = np.nonzero(qs < cs / f)[0]
                m(f"NstarFast{word}{L}{tag}", str(int(Narr[below[0]])) if len(below) else "$>$100")
            m(f"NstarPrep{L}{tag}", str(c["N_star_prep"]) if c["N_star_prep"] else "$>$100")
            if not Ns:
                continue
            i = c["N"].index(Ns)
            terms = list(_terms(p, Ns))
            qc = quantum_cost(Ns, terms, target, tf.spec, d["S_peak"], c["physical_error"])
            last = qc.per_shot[-1]
            rot, depth = step_profile(terms, a.order)
            g = india_gap(qc.physical_qubits, c["physical_error"])
            m(f"tQuantum{L}{tag}", human_time(c["quantum_s"][i]))
            m(f"tClassical{L}{tag}", human_time(c["classical_s"][i]))
            m(f"physQ{L}{tag}", sig(qc.physical_qubits / 1e6) + "\\,M")
            m(f"gapNQM{L}{tag}", f"{g.qubit_gap_vs_nqm:.0f}")
            m(f"gapQpiAI{L}{tag}", f"{g.error_gap_vs_qpiai:.0f}")
            m(f"shots{L}", sig(qc.shots))
            m(f"circuits{L}{tag}", sig(qc.circuits))
            m(f"steps{L}{tag}", str(last["steps"]))
            m(f"rotPerStep{L}{tag}", str(rot))
            rows.append({
                "set": L, "p": c["physical_error"], "N": Ns,
                "logical": last["logical_qubits"], "steps": last["steps"],
                "tstates": last["t_states"], "d": last["code_distance"],
                "phys": qc.physical_qubits, "shot_s": last["runtime_s"],
                "total": c["quantum_s"][i], "total_prep": c["quantum_prep_s"][i],
                "classical": c["classical_s"][i],
            })
    first = LABEL[a.params[0]]
    for pname in a.params:  # tighter accuracy, if computed
        for f in sorted(data.glob(f"crossover_{pname}_dE*_eps*.json")):
            d2 = json.loads(f.read_text())
            if abs(d2["dE"] - target.dE) > 1e-9:
                continue
            for c in d2["curves"]:
                if c["order"] == a.order:
                    tag = "A" if c["physical_error"] == 1e-3 else "B"
                    pct = f"{100 * d2['eps']:.0f}"
                    word = {"1": "One", "2": "Two", "10": "Ten"}.get(pct, "X")
                    m(f"NstarEps{word}{LABEL[pname]}{tag}", str(c["N_star"]) if c["N_star"] else "$>$100")
    m("dE", sig(target.dE))
    m("epsPct", f"{100 * target.eps:.0f}")
    m("Tmev", f"{target.T:.1f}")
    m("Tps", f"{target.T_ps:.1f}")
    m("TroughPs", f"{target.T_rough_ps:.1f}")
    m("nTimes", str(target.n_times))
    m("omegaMax", f"{target.omega_max:.0f}")
    m("nqmMax", str(NQM_QUBITS[1]))
    m("nqmMin", str(NQM_QUBITS[0]))
    m("qpiaiFid", f"{100 * QPIAI_TWO_QUBIT_FIDELITY:.0f}")
    m("mainSet", first)

    # ranges over the two error rates (B = 1e-4 first), collapsed when equal,
    # and over all parameter sets
    def rng(vals):
        nums = sorted({int(v) for v in vals if v.isdigit()})
        if not nums:
            return "$>$100"
        return str(nums[0]) if len(nums) == 1 else f"{nums[0]}\\text{{--}}{nums[-1]}"  # works in math too

    for kind in ("Nstar", "NstarPrep", "NstarEpsOne", "NstarFastHundred", "NstarFastTenThousand",
                 "NstarMPS"):
        every = []
        for pname in a.params:
            L = LABEL[pname]
            vals = [macros[k] for k in (f"{kind}{L}B", f"{kind}{L}A") if k in macros]
            if vals:
                m(f"{kind}Range{L}", rng(vals))
                every += vals
        if every:
            m(f"{kind}RangeAll", rng(every))
    m("nSets", {1: "one", 2: "two", 3: "three", 4: "four"}[len(a.params)])
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        raise ValueError(f"LaTeX macro names must be letters only: {bad}")
    with open(out / "numbers.tex", "w") as f:
        f.write("% generated by scripts/make_numbers.py -- do not edit by hand\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    with open(out / "table1.tex", "w") as f:
        f.write("% generated by scripts/make_numbers.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n")
        f.write("Set & $p$ & $N^*$ & $r$ & T states & $d$ & Phys.\\ qubits & Runtime \\\\\n")
        f.write(" & & & (steps) & per shot & & & at $N^*$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['set']} & $10^{{{int(np.log10(r['p']))}}}$ & {r['N']} & {r['steps']} & "
                    f"${sig(r['tstates'])}$ & {r['d']} & ${sig(r['phys'] / 1e6)}$\\,M & "
                    f"{human_time(r['total'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out / 'numbers.tex'} ({len(macros)} macros) and table1.tex ({len(rows)} rows)")


if __name__ == "__main__":
    main()
