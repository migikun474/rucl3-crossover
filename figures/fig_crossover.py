"""Fig. 3 (headline): wall-clock to compute S(q, omega) against N.

    python figures/fig_crossover.py --params winter2017

Reads data/crossover_<params>.json, writes figures/fig3_crossover_<params>.pdf/.png.
Solid lines: without ground-state preparation; dashed: with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# categorical slots 1-3 of the reference palette (light mode, print)
COLOR = {"classical": "#2a78d6", 1e-3: "#eb6834", 1e-4: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
YEAR = 3.156e7


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--order", type=int, default=2)
    a = ap.parse_args(argv)
    d = json.loads((ROOT / "data" / f"crossover_{a.params}.json").read_text())
    curves = [c for c in d["curves"] if c["order"] == a.order]

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    c0 = curves[0]
    ax.semilogy(c0["N"], c0["classical_s"], color=COLOR["classical"], lw=2,
                label="classical (exact)")
    for c in curves:
        col = COLOR[c["physical_error"]]
        ax.semilogy(c["N"], c["quantum_s"], color=col, lw=2,
                    label=f"quantum, $p$ = {c['physical_error']:g}")
        if any(x != y for x, y in zip(c["quantum_prep_s"], c["quantum_s"])):
            ax.semilogy(c["N"], c["quantum_prep_s"], color=col, lw=1.5, ls="--",
                        label=f"  + ground-state prep")
        if c["N_star"]:
            i = c["N"].index(c["N_star"])
            ax.plot(c["N_star"], c["quantum_s"][i], "o", ms=5, color=col,
                    mec="white", mew=1.5, zorder=5)
    stars = sorted({c["N_star"] for c in curves if c["N_star"]})
    if stars:
        i = c0["N"].index(stars[0])
        lab = f"$N^*$ = {stars[0]}" + (f"–{stars[-1]}" if len(stars) > 1 else "")
        ax.annotate(lab, (stars[0], c0["classical_s"][i]), xytext=(10, -14),
                    textcoords="offset points", color=INK, fontsize=7)
    xmax = c0["N"][-1]
    for v, lab in [(3600, "1 h"), (86400 * 30, "1 month"), (YEAR * 1e3, "1000 yr")]:
        ax.axhline(v, color=GRID, lw=0.8, zorder=0)
        ax.text(xmax, v * 1.6, lab, color=INK2, fontsize=6, ha="right", va="bottom")
    ax.set_xlabel("number of spins $N$")
    ax.set_ylabel("wall-clock time (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig3_crossover_{a.params}.{ext}")
    print("wrote figures/fig3_crossover_%s.{pdf,png}" % a.params)


if __name__ == "__main__":
    main()
