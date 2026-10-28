"""Figures 1, 2 and 4 of the paper (Fig. 3 is fig_crossover.py).

    python figures/make_figures.py --params winter2017 --N 18

Fig. 1: pipeline from neutron resolution to crossover, with the cluster.
Fig. 2: computed S(q, omega) at the symmetry-distinct momenta of one cluster.
Fig. 4: hardware needed at N* against India's targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401
import numpy as np  # noqa: E402
from rucl3.analysis.crossover import NQM_QUBITS, QPIAI_TWO_QUBIT_FIDELITY  # noqa: E402
from rucl3.model.lattice import BRAVAIS, best_cluster  # noqa: E402

# reference palette, light mode (categorical slots 1-4), ink and grid
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3f0"
BOND = {"x": C1, "y": C2, "z": C3}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=300)
    print(f"wrote figures/{name}.{{pdf,png}}")


# ----------------------------------------------------------------------
def fig1(N=24):
    cl = best_cluster(N)
    fig = plt.figure(figsize=(3.4, 2.3))
    ax = fig.add_axes([0.0, 0.0, 0.58, 1.0])
    ax.axis("off")
    steps = [
        ("Neutron resolution $\\delta E$", "window $w(t)$, time $T$, accuracy $\\varepsilon$"),
        ("Classical cost", "exact Lanczos/Krylov, measured $t(N)$"),
        ("Quantum cost", "Trotter error vs exact $\\to r$; QEC estimate"),
        ("Crossover $N^*$", "+ India hardware gap"),
    ]
    y = 0.9
    for k, (head, sub) in enumerate(steps):
        box = FancyBboxPatch((0.04, y - 0.13), 0.92, 0.17, boxstyle="round,pad=0.01,rounding_size=0.03",
                             fc=SURF, ec=GRID, lw=0.8, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(0.08, y - 0.005, head, transform=ax.transAxes, fontsize=7.5, color=INK,
                weight="bold", va="center")
        ax.text(0.08, y - 0.075, sub, transform=ax.transAxes, fontsize=6, color=INK2, va="center")
        if k < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.19), xytext=(0.5, y - 0.135), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
        y -= 0.245

    ax2 = fig.add_axes([0.6, 0.08, 0.4, 0.8])
    ax2.set_aspect("equal")
    ax2.axis("off")
    L = cl.supercell
    f = cl.positions @ np.linalg.inv(L)
    f -= np.floor(f + 0.05)  # fractional coordinates in [-0.05, 0.95)
    r = f @ L
    for bd in cl.nn_bonds:
        d = r[bd.j] - r[bd.i]
        if np.linalg.norm(d) < 1.01:
            ax2.plot(*np.c_[r[bd.i], r[bd.j]], color=BOND[bd.kind], lw=1.6,
                     solid_capstyle="round", zorder=1)
        else:  # bond leaves the cell: draw half-length stubs at both ends
            best = min((d + m1 * L[0] + m2 * L[1] for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)),
                       key=np.linalg.norm)
            for site, vec in ((bd.i, best), (bd.j, -best)):
                ax2.plot(*np.c_[r[site], r[site] + 0.45 * vec], color=BOND[bd.kind], lw=1.6,
                         solid_capstyle="round", zorder=1)
    ax2.scatter(r[:, 0], r[:, 1], s=9, c=[INK if s == 0 else "white" for s in cl.sublattice],
                edgecolors=INK, linewidths=0.6, zorder=2)
    o = np.array([-0.05, -0.05]) @ L
    corners = np.array([o, o + L[0], o + L[0] + L[1], o + L[1], o])
    ax2.plot(corners[:, 0], corners[:, 1], color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
    for k, lab in enumerate("xyz"):
        ax2.text(0.02 + 0.3 * k, -0.02, f"{lab} bond", color=INK, fontsize=6.5,
                 transform=ax2.transAxes)
        ax2.plot([0.02 + 0.3 * k - 0.0, 0.02 + 0.3 * k + 0.22], [-0.05, -0.05], color=BOND[lab],
                 lw=1.6, transform=ax2.transAxes, clip_on=False)
    ax2.set_title(f"{N}-site cluster", fontsize=7.5, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig1_pipeline.{ext}", dpi=300)
    print("wrote figures/fig1_pipeline.{pdf,png}")


# ----------------------------------------------------------------------
def label_q(q):
    B = 2 * np.pi * np.linalg.inv(BRAVAIS).T
    f = q @ np.linalg.inv(B)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return "$\\Gamma$"
    K = 4 * np.pi / (3 * np.sqrt(3))
    M = 2 * np.pi / 3
    if abs(n - K) < 1e-6:
        return "K"
    if abs(n - M) < 1e-6:
        return "M"
    if abs(f[0] * f[1]) < 1e-9 or abs(f[0] - f[1]) < 1e-9:  # along Gamma-M
        frac = {0.5: "$\\frac{1}{2}$", 0.67: "$\\frac{2}{3}$", 0.33: "$\\frac{1}{3}$"}.get(
            round(n / M, 2), f"{n / M:.2f}")
        return f"{frac}$\\Gamma$M"
    return f"|q|={n:.2f}"


def fig2(params, N):
    d = np.load(ROOT / "data" / f"exact_{params}_N{N}.npz")
    S = np.einsum("qaaw->qw", d["Sqw"]).real
    q, w = d["q"], d["omegas"]
    groups = {}
    for k in range(len(q)):
        key = round(float(S[k].max()), 6)
        groups.setdefault(key, []).append(k)
    order = sorted(groups.values(), key=lambda ks: np.linalg.norm(q[ks[0]]))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for col, ks in zip([C1, C2, C3, C4], order):
        k = ks[0]
        ax.plot(w, S[k], color=col, lw=1.8, label=f"{label_q(q[k])} ({len(ks)}$\\times$)")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, None)
    ax.set_xlabel("energy transfer $\\hbar\\omega$ (meV)")
    ax.set_ylabel("$S(\\mathbf{q},\\omega)$ (meV$^{-1}$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, loc="upper right", title=f"$N$ = {N}, multiplicity", title_fontsize=6)
    save(fig, f"fig2_spectrum_{params}_N{N}")


# ----------------------------------------------------------------------
def fig4(params, order=2):
    d = json.loads((ROOT / "data" / f"crossover_{params}.json").read_text())
    curves = {c["physical_error"]: c for c in d["curves"] if c["order"] == order}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 2.1), gridspec_kw={"width_ratios": [1.5, 1]})
    # (a) physical qubits needed at N* against what India targets
    labels, vals, cols = [], [], []
    for perr, col in [(1e-3, C2), (1e-4, C3)]:
        c = curves.get(perr)
        if c and c["N_star"]:
            i = c["N"].index(c["N_star"])
            labels.append(f"$p$={perr:g}\n$N^*$={c['N_star']}")
            vals.append(c["physical_qubits"][i])
            cols.append(col)
    y = np.arange(len(vals))
    a.barh(y, vals, color=cols, height=0.5, zorder=2)
    for yy, v in zip(y, vals):
        a.text(v * 1.3, yy, f"{v / 1e6:.2g} M", va="center", fontsize=6, color=INK)
    a.axvspan(*NQM_QUBITS, color=C1, alpha=0.18, lw=0, zorder=1)
    a.text(NQM_QUBITS[1] * 1.3, len(vals) - 0.45, "NQM\ntarget", ha="left",
           va="bottom", fontsize=6, color=INK)
    a.axvline(25, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    a.text(20, len(vals) - 0.62, "QpiAI 25 q", rotation=90, ha="right", va="bottom", fontsize=5.5,
           color=INK2)
    a.set_xscale("log")
    a.set_xlim(3, 3e8)
    a.set_xticks([1e1, 1e3, 1e5, 1e7])
    a.set_ylim(-0.6, len(vals) + 0.2)
    a.set_yticks(y, labels, fontsize=6)
    a.set_xlabel("physical qubits")
    a.spines[["top", "right"]].set_visible(False)
    a.set_title("(a) qubits at $N^*$", fontsize=7, loc="left", color=INK)
    # (b) two-qubit error: claimed vs assumed
    qp = 1 - QPIAI_TWO_QUBIT_FIDELITY
    b.axhline(1e-2, color=GRID, lw=1.0, zorder=0)
    b.text(1.55, 0.8e-2, "threshold ~1%", fontsize=5.5, color=INK2, va="top", ha="right")
    b.scatter([0], [qp], s=28, c=[INK2], zorder=3)
    b.scatter([1, 1], [1e-3, 1e-4], s=28, c=[C2, C3], zorder=3)
    b.text(1.12, 1e-3, "$10^{-3}$", fontsize=6, va="center", color=INK)
    b.text(1.12, 1e-4, "$10^{-4}$", fontsize=6, va="center", color=INK)
    b.annotate("", xy=(0.92, 1.1e-3), xytext=(0.08, qp * 0.9),
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7))
    b.text(0.42, np.sqrt(qp * 1e-3) * 0.7, f"{qp / 1e-3:.0f}$\\times$", fontsize=6.5, color=INK,
           ha="right", va="top")
    b.set_yscale("log")
    b.set_ylim(3e-5, 2e-1)
    b.set_xlim(-0.5, 1.6)
    b.set_xticks([0, 1], ["QpiAI\n(claimed)", "assumed\nat $N^*$"], fontsize=6)
    b.set_ylabel("two-qubit error")
    b.spines[["top", "right"]].set_visible(False)
    b.set_title("(b) error rate", fontsize=7, loc="left", color=INK)
    save(fig, f"fig4_india_{params}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--N", type=int, default=18, help="cluster for the spectrum figure")
    a = ap.parse_args(argv)
    fig1()
    fig2(a.params, a.N)
    fig4(a.params)


if __name__ == "__main__":
    main()
