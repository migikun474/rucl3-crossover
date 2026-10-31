import numpy as np
import pytest

pytest.importorskip("tenpy")

from rucl3.classical import mps  # noqa: E402
from rucl3.classical.exact import correlations, ground_manifold  # noqa: E402
from rucl3.model import params as P  # noqa: E402
from rucl3.model.hamiltonian import SpinModel  # noqa: E402
from rucl3.model.lattice import Cylinder  # noqa: E402

GENERIC = P.ParameterSet("generic", J1=-0.5, K=-5.0, G=2.5, J3=0.5, verified=True)


def test_cylinder_bonds():
    cyl = Cylinder(3, 4)
    labels = {s: [] for s in range(cyl.N)}
    for b in cyl.nn_bonds:
        labels[b.i].append(b.kind)
        labels[b.j].append(b.kind)
    # bulk sites have one bond of each kind; only the open ends lose bonds
    interior = [s for s in range(cyl.N) if 1 <= s // (2 * cyl.W) < cyl.L - 1]
    assert all(sorted(labels[s]) == ["x", "y", "z"] for s in interior)
    assert len(cyl.nn_bonds) == 3 * cyl.ncells - cyl.W  # one bond per ring crosses an end


def test_mps_matches_exact_on_small_cylinder():
    cyl = Cylinder(2, 2)
    H = SpinModel(cyl, GENERIC).operator()
    gm = ground_manifold(H)
    assert gm.degeneracy == 1
    M = mps.tenpy_model(cyl, GENERIC)
    psi, E0, _ = mps.ground_state(M, chi_max=16)
    assert E0 == pytest.approx(gm.energy, abs=1e-9)
    times = np.linspace(0, 1.5, 7)
    j = 2
    ex = correlations(H, gm, cyl.N, times, ref_sites=(j, j + 1))
    mc = mps.correlations(M, psi, E0, j, times, chi_max=16)  # 16 = exact for 8 sites
    assert np.abs(mc.C - ex.C[2, 2, :, 0, :]).max() < 1e-7
