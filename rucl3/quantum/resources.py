"""Fault-tolerant cost of measuring S(q, omega) with Trotter circuits.

Pipeline (all choices are explicit arguments so the sensitivity sweeps can
vary them):

1. Logical circuit per Hadamard-test shot at time t: N system qubits + 1
   ancilla; r(t) Trotter steps of Pauli rotations; two controlled Paulis
   (Clifford, ignored).  Rotation depth comes from greedy layering of the
   rotations onto qubits.
2. Microsoft's resource estimator (``qdk.estimator``) turns the logical
   counts into physical qubits and wall-clock time per shot for a surface
   code on gate-based qubits with physical error 1e-3 or 1e-4.  Rotation
   synthesis is done inside the estimator.
3. Shots: the shot noise of every correlator is propagated through the
   windowed Fourier transform (see :func:`shots_per_circuit`), so the shot
   count follows from the same eps as the Trotter error.
4. Ground-state preparation is a separate line item
   (:func:`ground_state_prep`), reported with and without.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..target import AccuracyTarget
from .trotter import merge, step_sequence

QUBIT_MODELS = {1e-3: "qubit_gate_ns_e3", 1e-4: "qubit_gate_ns_e4"}


# ----------------------------------------------------------------------
# logical counts
# ----------------------------------------------------------------------
def step_profile(terms, order: int) -> tuple[int, int]:
    """(rotations, rotation depth) per Trotter step in a long run.

    Depth is from greedy as-soon-as-possible scheduling of two-qubit
    rotations; consecutive rotations about the same string are fused.
    """
    def schedule(seq):
        ready = {}
        depth = 0
        for k, _ in seq:
            q = (terms[k].i, terms[k].j)
            layer = max(ready.get(q[0], 0), ready.get(q[1], 0)) + 1
            ready[q[0]] = ready[q[1]] = layer
            depth = max(depth, layer)
        return depth

    one = merge(step_sequence(terms, order))
    two = merge(step_sequence(terms, order) * 2)
    return len(two) - len(one), schedule(two) - schedule(one)


@dataclass
class TrotterSpec:
    """Steps needed to reach a Trotter error eps_trot at time T.

    Model: eps_trot = A * N**alpha * T * dt**p (fitted in
    :mod:`rucl3.analysis.trotter_fit`), so dt = (eps / (A N^alpha T))^(1/p).
    """
    order: int
    A: float
    alpha: float
    p: float

    def dt(self, N: int, T: float, eps: float) -> float:
        return (eps / (self.A * N ** self.alpha * T)) ** (1.0 / self.p)

    def steps(self, N: int, t: float, T: float, eps: float) -> int:
        return max(1, int(np.ceil(t / self.dt(N, T, eps))))


# ----------------------------------------------------------------------
# shots
# ----------------------------------------------------------------------
def shots_per_circuit(N: int, target: AccuracyTarget, S_peak: float, eps_shot: float,
                      z: float = 2.0) -> float:
    """Shots per Hadamard-test circuit so that shot noise in S(q, omega) is
    below eps_shot * S_peak at z standard deviations.

    Each shot gives +-1, so the estimate of a correlator component of
    S = sigma/2 operators has variance <= 1/(16 n).  The total spectrum
    sum_a S^aa(q, omega) = (1/pi) sum_k w_k dt Re[e^{i w t_k} S(q, t_k)] with
    S(q, t) = (1/2) sum_{s,i} phase * C_{i s}(t) then has variance
    <= 3 N sum_k (w_k dt)^2 / (32 pi^2 n).
    """
    times = target.times()
    w = target.window(times) * target.dt
    var_unit = 3 * N * np.sum(w ** 2) / (32 * np.pi ** 2)
    return z ** 2 * var_unit / (eps_shot * S_peak) ** 2


def circuits_per_time(N: int) -> int:
    """Distinct circuits per time point: i in N sites, one reference site per
    sublattice, a in x, y, z, real and imaginary parts."""
    return N * 2 * 3 * 2


# ----------------------------------------------------------------------
# ground state preparation (separate line item)
# ----------------------------------------------------------------------
@dataclass
class GroundStatePrep:
    """Phase-estimation projection from a trial state with overlap gamma.

    Evolution time per attempt ~ c * log(1/delta) / gap, controlled
    evolution costs ``control_factor`` times the uncontrolled one, and
    1/gamma^2 attempts are needed on average (no amplitude amplification).
    """
    gap: float  # meV
    overlap: float  # |<trial|0>|
    delta: float = 1e-2
    c: float = 2 * np.pi
    control_factor: float = 2.0

    @property
    def evolution_time(self) -> float:
        return self.c * np.log(1 / self.delta) / self.gap

    @property
    def attempts(self) -> float:
        return 1.0 / self.overlap ** 2


# ----------------------------------------------------------------------
# estimator
# ----------------------------------------------------------------------
def estimate_physical(n_qubits: int, rotations: float, rotation_depth: float,
                      physical_error: float = 1e-3, error_budget: float = 1e-2) -> dict:
    """Physical qubits and runtime (seconds) for one shot, from qdk.estimator."""
    from qdk.estimator import LogicalCounts
    counts = LogicalCounts({
        "numQubits": int(n_qubits),
        "tCount": 0,
        "rotationCount": int(rotations),
        "rotationDepth": int(rotation_depth),
        "cczCount": 0,
        "measurementCount": 1,
    })
    r = counts.estimate({"qubitParams": {"name": QUBIT_MODELS[physical_error]},
                         "qecScheme": {"name": "surface_code"},
                         "errorBudget": error_budget})
    pc = r["physicalCounts"]
    return {
        "physical_qubits": pc["physicalQubits"],
        "runtime_s": pc["runtime"] * 1e-9,
        "t_states": pc["breakdown"]["numTstates"],
        "code_distance": r["logicalQubit"]["codeDistance"],
        "logical_qubits": pc["breakdown"]["algorithmicLogicalQubits"],
    }


@dataclass
class QuantumCost:
    N: int
    per_shot: list = field(repr=False)  # estimator output at each sampled time
    shots: float
    circuits: int
    runtime_s: float  # S(q, omega) measurement, excluding ground-state prep
    prep_runtime_s: float  # extra time for ground-state prep (all shots)
    physical_qubits: int
    rotations_max: float

    @property
    def total_runtime_s(self) -> float:
        return self.runtime_s + self.prep_runtime_s


def quantum_cost(N: int, terms, target: AccuracyTarget, spec: TrotterSpec,
                 S_peak: float, physical_error: float = 1e-3,
                 prep: GroundStatePrep | None = None, n_time_samples: int = 6,
                 error_budget: float = 1e-2, eps_split: float = 0.5) -> QuantumCost:
    """Total wall-clock to measure S(q, omega) to target.eps.

    eps is split into Trotter (eps_split) and shot noise (1 - eps_split).
    Runtime per shot grows linearly with t; it is estimated at a few times
    and interpolated over the full time grid.
    """
    rot_step, depth_step = step_profile(terms, spec.order)
    T = target.T
    eps_trot = eps_split * target.eps
    eps_shot = (1 - eps_split) * target.eps
    shots = shots_per_circuit(N, target, S_peak, eps_shot)

    prep_rot = prep_depth = 0.0
    if prep is not None:
        r_prep = spec.steps(N, prep.evolution_time, T, eps_trot)
        prep_rot = prep.control_factor * rot_step * r_prep
        prep_depth = prep.control_factor * depth_step * r_prep

    sample_t = np.linspace(T / n_time_samples, T, n_time_samples)
    per_shot = []
    for t in sample_t:
        r = spec.steps(N, t, T, eps_trot)
        est = estimate_physical(N + 1, rot_step * r, depth_step * r, physical_error, error_budget)
        per_shot.append({"t": t, "steps": r, **est})

    # runtime per shot is linear in t; interpolate over the full time grid
    times = target.times()
    st = np.r_[0.0, [p["t"] for p in per_shot]]
    rt = np.interp(times, st, np.r_[0.0, [p["runtime_s"] for p in per_shot]])
    nc = circuits_per_time(N)
    run = nc * shots * rt.sum()

    prep_extra = 0.0
    if prep is not None:
        # every shot needs a fresh ground state: 1/gamma^2 prep attempts each
        pe = estimate_physical(N + 1, prep_rot, prep_depth, physical_error, error_budget)
        prep_extra = nc * shots * len(times) * prep.attempts * pe["runtime_s"]
    qubits = max(p["physical_qubits"] for p in per_shot)
    return QuantumCost(N, per_shot, shots, nc * len(times), run, prep_extra, qubits,
                       rot_step * per_shot[-1]["steps"])
