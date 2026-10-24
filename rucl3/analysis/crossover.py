"""Crossover size N* and the hardware it needs.

N* is the smallest N at which the fault-tolerant wall-clock to measure
S(q, omega) to the experiment's accuracy beats the classical wall-clock at
the same T and eps.  Both sides are extrapolated from measured small-N data;
every assumption is an argument of :func:`crossover`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.hamiltonian import SpinModel
from ..model.lattice import best_cluster
from ..model.params import ParameterSet
from ..quantum.resources import TrotterSpec, quantum_cost
from ..quantum.trotter import ordered_terms
from ..target import AccuracyTarget
from .fits import ClassicalFit


@dataclass
class CrossoverResult:
    N: np.ndarray
    classical_s: np.ndarray
    quantum_s: np.ndarray  # without ground-state preparation
    quantum_prep_s: np.ndarray  # with it
    physical_qubits: np.ndarray
    N_star: int | None
    N_star_prep: int | None


def _first_below(N, q, c):
    idx = np.nonzero(q < c)[0]
    return int(N[idx[0]]) if len(idx) else None


def crossover(params: ParameterSet, target: AccuracyTarget, spec: TrotterSpec,
              classical: ClassicalFit, S_peak: float, Ns=range(24, 81, 4),
              physical_error: float = 1e-3, prep_of_N=None,
              classical_free_exponent: bool = False) -> CrossoverResult:
    """Classical and quantum wall-clock against N.

    ``prep_of_N(N) -> GroundStatePrep`` supplies the extrapolated gap and
    trial-state overlap; without it ground-state preparation is omitted.
    S_peak (peak of sum_a S^aa) is taken as N-independent (intensive).
    """
    Ns = np.array([n for n in Ns if n % 2 == 0])
    cl_s, q_s, qp_s, qubits = [], [], [], []
    for N in Ns:
        cl = best_cluster(int(N), include_j3=bool(params.J3))
        terms = ordered_terms(SpinModel(cl, params))
        prep = prep_of_N(N) if prep_of_N else None
        qc = quantum_cost(int(N), terms, target, spec, S_peak, physical_error, prep)
        cl_s.append(float(classical.seconds(N, target.T, free=classical_free_exponent)))
        q_s.append(qc.runtime_s)
        qp_s.append(qc.total_runtime_s)
        qubits.append(qc.physical_qubits)
    cl_s, q_s, qp_s = map(np.array, (cl_s, q_s, qp_s))
    return CrossoverResult(Ns, cl_s, q_s, qp_s, np.array(qubits),
                           _first_below(Ns, q_s, cl_s), _first_below(Ns, qp_s, cl_s))


# ----------------------------------------------------------------------
# India mapping
# ----------------------------------------------------------------------
# National Quantum Mission 8-year target and QpiAI-Indus claim.  STATUS: take
# the exact wording and numbers from the official NQM document and the
# QpiAI press release before quoting them.
NQM_QUBITS = (50, 1000)
QPIAI_TWO_QUBIT_FIDELITY = 0.96


@dataclass
class GapFactors:
    qubits_needed: int
    error_needed: float
    qubit_gap_vs_nqm: float  # needed / NQM upper target
    error_gap_vs_qpiai: float  # QpiAI two-qubit error / needed error


def india_gap(physical_qubits: int, physical_error: float) -> GapFactors:
    qpiai_err = 1 - QPIAI_TWO_QUBIT_FIDELITY
    return GapFactors(int(physical_qubits), physical_error,
                      physical_qubits / NQM_QUBITS[1], qpiai_err / physical_error)
