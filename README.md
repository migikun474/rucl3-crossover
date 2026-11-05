# rucl3-crossover

Code for the SQST 2027 paper *When Will Quantum Computers Beat Classical
Simulation of a Real Quantum Magnet? A Quantum–Classical Crossover for the
Neutron Spectrum of α-RuCl₃*.

One observable, the dynamical structure factor S(q, ω) of the
J₁–K–Γ–Γ′–J₃ honeycomb model, is costed two ways at the same
experiment-defined accuracy:

```
neutron resolution dE ──► window w(t), total time T, accuracy eps
                               │
          ┌────────────────────┴────────────────────┐
   classical (exact, MPS)                    quantum (Trotter + Hadamard test)
   measured wall-clock, fit 2^N              Trotter error measured vs exact, fit r(N,T,eps)
          │                                  logical counts → qdk resource estimator
          └──────────────► crossover N* ◄────────────┘
                                │
                     India hardware gap factors
```

## Layout

| Path | Contents |
| --- | --- |
| `rucl3/model/` | honeycomb tori (`lattice.py`), parameter sets (`params.py`), Hamiltonian (`hamiltonian.py`) |
| `rucl3/target.py` | resolution dE → Gaussian window, T, time grid |
| `rucl3/classical/` | exact Lanczos + Krylov baseline (`exact.py`), MPS/TDVP baseline (`mps.py`), S(q, ω) (`spectrum.py`) |
| `rucl3/quantum/` | Pauli-rotation Trotter circuits (`trotter.py`), fault-tolerant cost (`resources.py`), 12-qubit hardware check (`hardware.py`) |
| `rucl3/analysis/` | Trotter and classical fits (`fits.py`), crossover and India mapping (`crossover.py`) |
| `scripts/` | runnable entry points (below) |
| `tests/` | unit tests (`pytest`) |
| `figures/`, `paper/` | figure scripts and IEEE LaTeX (to come) |
| `data/` | outputs (`*.jsonl` summaries are tracked, `*.npz` arrays are not) |

The plan's top-level `model/ classical/ quantum/ analysis/` folders live
inside the `rucl3` package so they import cleanly.

## Setup

```bash
pip install -e ".[dev,quantum]"      # numpy scipy numba matplotlib pytest qdk
pip install -e ".[mps]"              # TeNPy, for the MPS baseline
pytest -q                            # ~1 min
```

## Running

```bash
# exact baseline at one size: ground state, correlations, S(q, ω), timings
python scripts/run_exact.py --N 18 --params winter2017

# Trotter error in S(q, ω) vs step size, against the exact data
python scripts/trotter_scan.py --N 14 16 --params winter2017 --order 2 4

# gap and trial-state overlap for the ground-state-preparation line item
python scripts/gs_prep_data.py --N 14 16 18 --params winter2017

# everything that fits on a 4-CPU machine
./scripts/batch_small.sh winter2017
```

### 20- and 24-spin runs (laptop or Colab)

```bash
git clone https://github.com/migikun474/rucl3-crossover && cd rucl3-crossover
pip install -e .
python scripts/run_exact.py --N 24 --params winter2017 --ground-only   # ground state only, < 2 GB
python scripts/run_exact.py --N 20 --params winter2017
python scripts/run_exact.py --N 24 --params winter2017                 # long; ~5 GB RAM
```

Each run appends a JSON line to `data/exact_runs.jsonl`; commit that file
(and keep the `.npz` if you want the N = 24 spectrum figure).  On a machine
with less than 8 GB RAM add `--krylov-m 8`.

### 12-qubit IBM hardware check

```bash
pip install -e ".[hardware]"
python scripts/ibm_check.py --backend fake_torino          # noisy emulation, no account
export IBM_QUANTUM_TOKEN=...                              # your token; never commit it
python scripts/ibm_check.py --backend ibm_torino --shots 4000
python figures/make_figures.py                            # Fig. 5 from data/ibm_<backend>.json
```

Fig. 5 and the numbers in the text use `data/ibm_fake_torino.json`; after a
device run, point `fig5()` in `figures/make_figures.py` at the device file.

## Conventions

* Spin-1/2 operators S = σ/2; energies in meV, ħ = 1, so times are in
  meV⁻¹ (× 0.6582 for ps).
* Basis state bit k is site k; bit 0 is spin up.
* S(q, ω) = (1/2π) ∫ dt w(t) e^{iωt} (1/N) Σᵢⱼ e^{iq·(rᵢ−rⱼ)} ⟨Sᵢ(t)·Sⱼ(0)⟩,
  with the 1/2π that the plan's formula omits.
* With a Gaussian window of width τ = 2√(2 ln 2)/dE the result is exactly
  the spectrum convolved with the instrument resolution; the signal is cut
  at T where w(T) = 10⁻³ (T ≈ 11.5 ps for dE = 0.5 meV, against the plan's
  rough 2πħ/dE ≈ 8.3 ps).
* Third-neighbour bonds are only faithful on tori with N ≥ 14, so sets with
  J₃ ≠ 0 use N = 14, 16, 18, 20, 24 (the plan's 8 and 12 are used only for
  J₃ = 0 tests).  `best_cluster` picks the most compact faithful torus;
  N = 14, 18, 24 are C₃ symmetric.

* **Degenerate ground states.** C₃-symmetric tori have a degenerate ground
  level (6-fold for Winter et al. at N = 14: three zigzag orientations ×
  time reversal). `ground_manifold` finds it by deflated Lanczos (ARPACK
  with small k undercounts it) and all correlations are averaged over it.
  Classical timings are reported per ground state.

## Submission status (SQST 2027, deadline 5 Nov 2026 23:59 AoE)

`paper/sqst2027_submission.pdf` is the submission build: 6 pages, anonymous,
fonts embedded, no unresolved notes (`touch paper/final.flag` before
building makes LaTeX stop on any remaining `\verify`).
`paper/edas_abstract.txt` is the abstract for the EDAS form.

* Parameter sets: Ran et al. confirmed from the abstract; Winter et al. and
  Laurell--Okamoto values confirmed against the tables of
  Maksimov & Chernyshev (PRR 2020) and arXiv:2602.10190.
* Resolution dE = 0.5 meV and accuracy eps = 5% are stated in the paper as
  representative assumptions; Table II covers 1-2 meV and 1-10%.
* Not in the submission: N = 20/24 exact runs, a real IBM device run (Fig. 5
  is an emulation with the ibm_torino noise calibration; `ibm_check.py`
  with a token switches it automatically), an anonymous code mirror (the
  paper says code is released upon acceptance).
