"""12-qubit hardware check: Neel-state quench on a 12-spin honeycomb flake.

Emulated (noise model from a published IBM calibration snapshot, no account
needed):
    python scripts/ibm_check.py --backend fake_torino

Real device (needs an IBM Quantum account; the token is read from the
IBM_QUANTUM_TOKEN environment variable, never stored in the repository):
    export IBM_QUANTUM_TOKEN=...
    python scripts/ibm_check.py --backend ibm_torino --shots 4000

For each time t_k = k * tmax / (npts - 1) a separate circuit applies k * spp
second-order Trotter steps to the Neel state and measures all spins in z.
The measured magnetizations m_i(t) are compared with the exact evolution and
with the noiseless Trotter circuit.  Writes data/ibm_<backend>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rucl3  # noqa: E402,F401
import numpy as np  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import Flake  # noqa: E402
from rucl3.quantum.hardware import (exact_magnetization, magnetization_from_counts,  # noqa: E402
                                    quench_circuit, trotter_magnetization)

FAKES = {"fake_torino": "FakeTorino", "fake_fez": "FakeFez", "fake_marrakesh": "FakeMarrakesh",
         "fake_sherbrooke": "FakeSherbrooke", "fake_brisbane": "FakeBrisbane",
         "fake_kyiv": "FakeKyiv"}


def get_backend(name):
    if name in FAKES:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider
        fake = getattr(fake_provider, FAKES[name])()
        return fake, AerSimulator.from_backend(fake), True
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token) if token
               else QiskitRuntimeService())
    backend = service.backend(name)
    return backend, backend, False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake_torino")
    ap.add_argument("--params", default="winter2017")
    ap.add_argument("--tmax", type=float, default=1.0, help="meV^-1")
    ap.add_argument("--npts", type=int, default=9)
    ap.add_argument("--spp", type=int, default=1, help="Trotter steps per time point")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    model = SpinModel(Flake(3, 2), P.get(a.params))
    times = np.linspace(0, a.tmax, a.npts)
    target_backend, runner, emulated = get_backend(a.backend)

    from qiskit import transpile
    circuits = [quench_circuit(model, t, k * a.spp) for k, t in enumerate(times)]
    tqc = transpile(circuits, backend=target_backend, optimization_level=3,
                    seed_transpiler=a.seed)
    two_q = [sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")) for c in tqc]
    depth = [c.depth() for c in tqc]
    print(f"{a.backend}: N={model.N}, two-qubit gates per circuit {two_q}", flush=True)

    t0 = time.perf_counter()
    if emulated:
        result = runner.run(tqc, shots=a.shots, seed_simulator=a.seed).result()
        counts = [result.get_counts(i) for i in range(len(tqc))]
        job_id = None
    else:
        from qiskit_ibm_runtime import SamplerV2
        job = SamplerV2(mode=runner).run(tqc, shots=a.shots)
        job_id = job.job_id()
        print(f"submitted job {job_id}; waiting for results", flush=True)
        counts = [r.data.c.get_counts() for r in job.result()]
    run_s = time.perf_counter() - t0

    m_dev = np.array([magnetization_from_counts(c, model.N) for c in counts])
    m_ex = exact_magnetization(model, times)
    m_tr = trotter_magnetization(model, times, a.spp)
    err_dev = np.abs(m_dev - m_ex).max(axis=1)
    err_tr = np.abs(m_tr - m_ex).max(axis=1)
    shot_noise = 2 / np.sqrt(a.shots)  # 2 sigma on <Z> at most
    print("t      max|dev-exact|  max|trotter-exact|  2q gates")
    for k, t in enumerate(times):
        print(f"{t:5.2f}  {err_dev[k]:14.3f}  {err_tr[k]:18.4f}  {two_q[k]}")
    out = {"backend": a.backend, "emulated": emulated, "job_id": job_id, "params": a.params,
           "N": model.N, "shots": a.shots, "spp": a.spp, "times": times.tolist(),
           "two_qubit_gates": two_q, "depth": depth, "m_device": m_dev.tolist(),
           "m_exact": m_ex.tolist(), "m_trotter": m_tr.tolist(),
           "err_device": err_dev.tolist(), "err_trotter": err_tr.tolist(),
           "shot_noise_2sigma": shot_noise, "run_seconds": run_s}
    (ROOT / "data" / f"ibm_{a.backend}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
