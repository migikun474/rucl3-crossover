"""Fits that turn measured small-N data into cost models.

* Trotter:   eps = A * N^alpha * T^beta * dt^p           (log-linear least squares)
* Classical: seconds per unit simulated time = c * N^2 * 2^N
             (matvec cost ~ N 2^N, matvecs per unit time ~ ||H|| ~ N),
             with a free-exponent fit c' * b^N reported alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..quantum.resources import TrotterSpec


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class TrotterFit:
    spec: TrotterSpec
    beta: float
    rms_log_residual: float
    n_points: int


def fit_trotter(rows: list[dict], order: int, eps_max: float = 0.3,
                fix_beta: float | None = 1.0, conservative: bool = False) -> TrotterFit:
    """Fit eps(N, T, dt) from trotter_scan rows of one order.

    Only points with eps < eps_max are used (asymptotic regime).  With
    ``fix_beta`` the T exponent is fixed (default 1, the linear growth of
    Trotter error with time); pass None to fit it.

    ``conservative=True`` fits a common dt exponent p with one prefactor per
    cluster size and returns the largest prefactor with alpha = 0.  On the
    small clusters available the size dependence is dominated by
    finite-size effects (it is not monotonic), so a power law in N is not
    trustworthy; the error of an intensive, resolution-limited quantity is
    expected to saturate with N.
    """
    pts = [r for r in rows if r["order"] == order and 0 < r["eps"] < eps_max]
    if len(pts) < 3:
        raise ValueError(f"need at least 3 points for order {order}, have {len(pts)}")
    y = np.log([r["eps"] for r in pts])
    lnN = np.log([r["N"] for r in pts])
    lnT = np.log([r["T"] for r in pts])
    lndt = np.log([r["dt"] for r in pts])
    if conservative:
        beta = 1.0 if fix_beta is None else fix_beta
        sizes = sorted(set(r["N"] for r in pts))
        X = np.column_stack([np.array([r["N"] == n for r in pts], float) for n in sizes] + [lndt])
        coef, *_ = np.linalg.lstsq(X, y - beta * lnT, rcond=None)
        lnA, alpha, p = float(coef[:-1].max()), 0.0, coef[-1]
        pred = X @ coef + beta * lnT
        rms = float(np.sqrt(np.mean((y - pred) ** 2)))
        return TrotterFit(TrotterSpec(order, float(np.exp(lnA)), alpha, float(p)),
                          beta, rms, len(pts))
    if fix_beta is None:
        X = np.column_stack([np.ones_like(y), lnN, lnT, lndt])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        lnA, alpha, beta, p = coef
    else:
        X = np.column_stack([np.ones_like(y), lnN, lndt])
        coef, *_ = np.linalg.lstsq(X, y - fix_beta * lnT, rcond=None)
        lnA, alpha, p = coef
        beta = fix_beta
    pred = lnA + alpha * lnN + beta * lnT + p * lndt
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    # TrotterSpec assumes beta = 1; fold any other beta into the prefactor at T = 1
    return TrotterFit(TrotterSpec(order, float(np.exp(lnA)), float(alpha), float(p)),
                      float(beta), rms, len(pts))


@dataclass
class ClassicalFit:
    c: float  # seconds per unit time = c * N^2 * 2^N
    c_free: float
    base_free: float  # seconds per unit time = c_free * base_free^N
    gs_c: float  # ground-state seconds = gs_c * N * 2^N

    def seconds(self, N, T, free: bool = False):
        N = np.asarray(N, dtype=float)
        evo = (self.c_free * self.base_free ** N if free else self.c * N ** 2 * 2.0 ** N) * T
        return evo + self.gs_c * N * 2.0 ** N

    @staticmethod
    def memory_bytes(N, vectors: int = 20):
        return vectors * 16 * 2.0 ** np.asarray(N, dtype=float)


def fit_classical(rows: list[dict], params: str | None = None) -> ClassicalFit:
    pts = [r for r in rows if "seconds_per_unit_time" in r and (params is None or r["params"] == params)]
    # keep the latest run for each N
    byN = {r["N"]: r for r in pts}
    N = np.array(sorted(byN), dtype=float)
    if len(N) < 2:
        raise ValueError("need exact runs at two or more sizes")
    s = np.array([byN[n]["seconds_per_unit_time"] for n in N])
    g = np.array([byN[n]["gs_seconds"] for n in N])
    # fix the form, fit the prefactor in log space (weights the largest N most)
    w = N / N.max()
    c = float(np.exp(np.average(np.log(s / (N ** 2 * 2 ** N)), weights=w)))
    gs_c = float(np.exp(np.average(np.log(g / (N * 2 ** N)), weights=w)))
    slope, icpt = np.polyfit(N, np.log(s), 1)
    return ClassicalFit(c, float(np.exp(icpt)), float(np.exp(slope)), gs_c)
