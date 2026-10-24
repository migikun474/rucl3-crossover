import numpy as np
import pytest

from rucl3.model import params as P
from rucl3.model.hamiltonian import SpinModel, pauli_matrix
from rucl3.model.lattice import Cluster, best_cluster

GENERIC = P.ParameterSet("generic", J1=-1.3, K=-15.1, G=10.1, Gp=-0.12, J3=0.9, verified=True)


def min_image(cl, i, j):
    L = cl.supercell
    d = cl.positions[j] - cl.positions[i]
    return min(np.linalg.norm(d + m1 * L[0] + m2 * L[1])
               for m1 in range(-2, 3) for m2 in range(-2, 3))


@pytest.mark.parametrize("N", [14, 16, 18, 20, 24])
def test_cluster_geometry(N):
    cl = best_cluster(N)
    assert cl.N == N and not cl.check()
    labels = {s: [] for s in range(N)}
    for b in cl.nn_bonds:
        labels[b.i].append(b.kind)
        labels[b.j].append(b.kind)
        assert min_image(cl, b.i, b.j) == pytest.approx(1.0)
    for s in range(N):
        assert sorted(labels[s]) == ["x", "y", "z"]
    for b in cl.j3_bonds:
        assert min_image(cl, b.i, b.j) == pytest.approx(2.0)
    assert len(cl.momenta()) == N // 2


def test_translation_maps_bonds_to_bonds():
    cl = best_cluster(18)
    bonds = {(b.i, b.j, b.kind) for b in cl.nn_bonds + cl.j3_bonds}
    for shift in [(1, 0), (0, 1)]:
        p = cl.translate(shift)
        assert {(p[i], p[j], k) for i, j, k in bonds} == bonds


def test_momenta_are_compatible_with_torus():
    cl = best_cluster(24)
    phases = np.exp(1j * cl.momenta() @ cl.supercell.T)
    assert np.allclose(phases, 1)


def test_c3_cluster_for_24_sites():
    assert best_cluster(24).has_c3()


@pytest.mark.parametrize("N", [8, 12])
def test_three_representations_agree(N):
    m = SpinModel(best_cluster(N, include_j3=False), GENERIC.scaled(J3=0.0))
    Hs = m.sparse().toarray()
    assert np.allclose(Hs, Hs.conj().T)
    if N <= 8:  # dense Pauli sum is slow beyond this
        assert np.allclose(Hs, m.apply_pauli_dense())
    rng = np.random.default_rng(1)
    v = rng.standard_normal(m.dim) + 1j * rng.standard_normal(m.dim)
    assert np.allclose(m.operator()(v), Hs @ v)


def test_j3_terms_on_faithful_cluster():
    m = SpinModel(best_cluster(14), GENERIC)
    rng = np.random.default_rng(2)
    v = rng.standard_normal(m.dim) + 1j * rng.standard_normal(m.dim)
    assert np.allclose(m.operator()(v), m.sparse() @ v)
    assert len(m.bonds) == 2 * 3 * 7


def test_pauli_term_count():
    m = SpinModel(best_cluster(14), GENERIC)
    # 9 Pauli strings per nearest-neighbour bond (generic J1, K, G, Gp), 3 per J3 bond
    assert len(m.pauli_terms()) == 21 * 9 + 21 * 3


def test_heisenberg_conserves_total_spin():
    m = SpinModel(best_cluster(8, include_j3=False), P.HEISENBERG)
    H = m.sparse().toarray()
    N = m.N
    for a in "XYZ":
        Stot = sum(pauli_matrix(N, {i: a}) for i in range(N)) / 2
        assert np.allclose(H @ Stot, Stot @ H)


def test_bond_couplings_have_c3_structure():
    """Under the cyclic permutation x->y->z of spin axes and bond labels the
    Hamiltonian maps to itself (bond-level check)."""
    from rucl3.model.hamiltonian import bond_coupling
    perm = [2, 0, 1]  # new axis k comes from old axis perm[k]: x<-z, y<-x, z<-y
    for src, dst in [("x", "y"), ("y", "z"), ("z", "x")]:
        Js, Jd = bond_coupling(src, GENERIC), bond_coupling(dst, GENERIC)
        assert np.allclose(Jd, Js[np.ix_(perm, perm)])
