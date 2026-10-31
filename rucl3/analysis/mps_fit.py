"""From MPS runs to a classical cost model.

1. Error.  For a resolution dE' (window w, time T') the error of a run is
       eps = max_omega sum_i |F_ij - F_ij^ref|(omega) / max_omega |F_jj^ref(omega)|,
   with F_ij(omega) = (1/pi) int_0^T' w(t) e^{i omega t} C_ij(t) dt.  The sum
   over i bounds the error of S(q, omega) at every q, so this is more
   conservative than the momentum-resolved metric used for Trotter errors.
2. Reach.  For each chi the longest T' with eps <= target; entanglement grows
   linearly in time, so log chi_needed = c_W + s_W T, fitted per
   circumference W, with s_W and c_W linear in W.
3. Cost.  Seconds per sample interval = kappa * N * D(W) * chi^3, kappa from
   the largest-chi runs, MPO bond dimension D(W) linear in W; six evolutions
   (3 components x 2 sublattices) on a cylinder with L = W, i.e. N = 2 W^2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..target import AccuracyTarget


def one_sided(times, C, target: AccuracyTarget, omegas):
    """F(omega) for each row of C (sites x times), truncated at target.T."""
    n = int(np.searchsorted(times, target.T + 1e-9))
    t = times[:n]
    w = target.window(t) * (t[1] - t[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return (C[:, :n] * w) @ np.exp(1j * np.outer(t, omegas)) / np.pi


def mps_error(times, C, C_ref, j, target, omegas):
    F = one_sided(times, C, target, omegas)
    Fr = one_sided(times, C_ref, target, omegas)
    return float(np.abs(F - Fr).sum(axis=0).max() / np.abs(Fr[j]).max())


def reach(times, C, C_ref, j, eps, dEs, omegas, cutoff=1e-3):
    """Longest T' (over resolutions dEs) whose error stays below eps."""
    best = 0.0
    for dE in sorted(dEs, reverse=True):  # short windows first
        tg = AccuracyTarget(dE=dE, eps=eps, cutoff=cutoff)
        if tg.T > times[-1] + 1e-9:
            continue
        if mps_error(times, C, C_ref, j, tg, omegas) <= eps:
            best = max(best, tg.T)
        else:
            break
    return best


@dataclass
class MPSCostModel:
    s: tuple  # s_W = s[0] + s[1] W   (per meV^-1)
    c: tuple  # c_W = c[0] + c[1] W
    kappa: float
    D: tuple  # D(W) = D[0] + D[1] W
    points: list = field(default_factory=list)

    def chi_needed(self, W, T):
        return float(np.exp(self.c[0] + self.c[1] * W + (self.s[0] + self.s[1] * W) * T))

    def seconds(self, N, target: AccuracyTarget, n_evolutions=6):
        W = np.sqrt(np.asarray(N, dtype=float) / 2)
        chi = np.vectorize(self.chi_needed)(W, target.T)
        D = self.D[0] + self.D[1] * W
        return n_evolutions * target.n_times * self.kappa * np.asarray(N) * D * chi ** 3


def fit_mps(data_dir: Path, params: str, eps: float, dE: float,
            dEs=None) -> MPSCostModel:
    data_dir = Path(data_dir)
    runs = [json.loads(line) for line in (data_dir / "runs.jsonl").read_text().splitlines()
            if line.strip()]
    runs = [r for r in runs if r["params"] == params]
    dEs = dEs if dEs is not None else list(np.geomspace(dE, 16 * dE, 25))
    omegas = np.linspace(0, 30, 301)
    points, sW, cW, kap, Ds = [], [], [], [], []
    for W in sorted({r["W"] for r in runs}):
        rs = sorted((r for r in runs if r["W"] == W), key=lambda r: r["chi"])
        tag = f"{params}_W{W}_L{rs[0]['L']}"
        ex = data_dir / f"exact_{tag}.npz"
        if ex.exists():
            ref = np.load(ex)["C"]
            ref_label, test = "exact", rs
        else:  # converge against the largest bond dimension
            ref = np.load(data_dir / f"mps_{tag}_chi{rs[-1]['chi']}.npz")["C"]
            ref_label, test = f"chi={rs[-1]['chi']}", rs[:-1]
        xs, ys = [], []
        for r in test:
            d = np.load(data_dir / f"mps_{tag}_chi{r['chi']}.npz")
            T_reach = reach(d["times"], d["C"], ref, int(d["j"]), eps, dEs, omegas)
            points.append({"W": W, "chi": r["chi"], "T_reach": T_reach, "ref": ref_label})
            if 0 < T_reach < d["times"][-1] - 1e-9:  # censored points carry no slope
                xs.append(T_reach)
                ys.append(np.log(r["chi"]))
        if len(xs) >= 2:
            s, c = np.polyfit(xs, ys, 1)
            sW.append((W, s))
            cW.append((W, c))
        big = rs[-1]
        per_sample = big["seconds"] / (big["n_times"] - 1)
        kap.append(per_sample / (big["N"] * big["mpo_chi"] * big["chi"] ** 3))
        Ds.append((W, big["mpo_chi"]))

    def line(pts):
        if len(pts) == 1:
            return (pts[0][1], 0.0)
        b, a = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        return (float(a), float(b))

    if not sW:
        raise ValueError("no width with two uncensored reach points")
    return MPSCostModel(line(sW), line(cW), float(np.max(kap)), line(Ds), points)
