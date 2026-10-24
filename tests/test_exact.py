import numpy as np
import pytest
import scipy.linalg as sla
from scipy.sparse.linalg import eigsh

from rucl3.classical.exact import (KrylovPropagator, apply_pauli, correlations,
                                   lanczos_ground_state, pauli_overlap)
from rucl3.classical.spectrum import fourier, lehmann_dense, momentum_correlations
from rucl3.model import params as P
from rucl3.model.hamiltonian import SpinModel, pauli_matrix
from rucl3.model.lattice import best_cluster
from rucl3.target import AccuracyTarget

GENERIC = P.ParameterSet("generic", J1=-1.3, K=-15.1, G=10.1, Gp=-0.12, J3=0.0, verified=True)


def rand_state(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("a", "XYZ")
def test_pauli_helpers_match_dense(a):
    N = 5
    v, u = rand_state(1 << N, 1), rand_state(1 << N, 2)
    for i in range(N):
        P_ = pauli_matrix(N, {i: a})
        assert np.allclose(apply_pauli(v, i, a), P_ @ v)
        assert np.isclose(pauli_overlap(u, v, i, a), np.vdot(u, P_ @ v))


def test_lanczos_matches_eigsh():
    m = SpinModel(best_cluster(14), GENERIC.scaled(J3=0.9))
    gs = lanczos_ground_state(m.operator())
    e_ref = eigsh(m.sparse(), k=1, which="SA")[0][0]
    assert gs.energy == pytest.approx(e_ref, abs=1e-8)
    assert gs.residual < 1e-4


def test_krylov_matches_expm():
    m = SpinModel(best_cluster(8, include_j3=False), GENERIC)
    H = m.sparse().toarray()
    v = rand_state(m.dim)
    prop = KrylovPropagator(m.operator(), m=12, tol=1e-12)
    for t in [0.05, 0.7, 2.3]:
        assert np.allclose(prop.evolve(v, t), sla.expm(-1j * H * t) @ v, atol=1e-9)


def hexagon(cl, cell):
    """Sites of one hexagon and the spin component of each site's outward bond."""
    c = np.array(cell)
    a1, a2 = np.array([1, 0]), np.array([0, 1])
    return [(2 * cl.cell(c), "Y"), (2 * cl.cell(c) + 1, "X"),
            (2 * cl.cell(c + a2), "Z"), (2 * cl.cell(c + a2 - a1) + 1, "Y"),
            (2 * cl.cell(c + a2 - a1), "X"), (2 * cl.cell(c - a1) + 1, "Z")]


def test_kitaev_plaquettes_are_conserved():
    cl = best_cluster(12, include_j3=False)
    H = SpinModel(cl, P.KITAEV).sparse()
    Hh = SpinModel(cl, P.HEISENBERG).sparse()
    for cell in cl.cells:
        W = pauli_matrix(cl.N, dict(hexagon(cl, cell)))
        assert abs(H @ W - W @ H).max() < 1e-12
    assert abs(Hh @ W - W @ Hh).max() > 0.1  # sanity: not a trivial identity


def test_kitaev_correlations_are_ultra_short_ranged():
    """Baskaran, Mandal, Shankar (2007): in the Kitaev model <S_i^a S_j^b> vanishes
    unless i = j or (i, j) is a gamma bond with a = b = gamma.  The same holds
    for the dynamical correlations, which we check at a nonzero time."""
    cl = best_cluster(16)
    m = SpinModel(cl, P.KITAEV)
    H = m.operator()
    # Small tori have flux sectors degenerate with the flux-free one; start
    # Lanczos inside the flux-free sector (W_p = +1 on every hexagon).
    v0 = rand_state(m.dim)
    for cell in cl.cells:
        Wv = v0
        for site, a in hexagon(cl, cell):
            Wv = apply_pauli(Wv, site, a)
        v0 = (v0 + Wv) / 2
    gs = lanczos_ground_state(H, v0=v0)
    # Kitaev's thermodynamic-limit value is -0.19682 |K| per site
    assert gs.energy / cl.N == pytest.approx(-0.19682, rel=0.02)
    corr = correlations(H, gs, cl.N, np.array([0.0, 0.8]))
    nn = {}
    for b in cl.nn_bonds:
        nn[(b.i, b.j)] = nn[(b.j, b.i)] = "xyz".index(b.kind)
    for s, j in enumerate(corr.ref_sites):
        for i in range(cl.N):
            for a in range(3):
                for b in range(3):
                    val = np.abs(corr.C[a, b, i, s]).max()
                    allowed = (i == j and a == b) or (nn.get((i, j)) == a == b)
                    if not allowed:
                        assert val < 1e-6, (i, j, a, b, val)
    # energy check: E0 = sum over bonds of K <S^g S^g>
    e_bond = sum(corr.C[nn[(i, j)], nn[(i, j)], i, s, 0].real
                 for s, j in enumerate(corr.ref_sites) for i in range(cl.N)
                 if (i, j) in nn)
    # each bond is counted once from its A end and once from its B end
    assert gs.energy == pytest.approx(P.KITAEV.K * e_bond * cl.ncells / 2, rel=1e-6)


def test_time_domain_spectrum_matches_lehmann():
    cl = best_cluster(8, include_j3=False)
    m = SpinModel(cl, GENERIC)
    H = m.operator()
    gs = lanczos_ground_state(H)
    E = np.linalg.eigvalsh(m.sparse().toarray())
    assert E[1] - E[0] > 1e-3  # nondegenerate ground state
    target = AccuracyTarget(dE=4.0, eps=0.05, cutoff=1e-8, omega_max=120.0)
    times = target.times()
    corr = correlations(H, gs, cl.N, times)
    q = cl.momenta()
    Sqt = momentum_correlations(cl, corr, q)
    omegas = np.linspace(0, 60, 301)
    S_t = fourier(times, Sqt, target.window(times), omegas)
    S_ref = lehmann_dense(m.sparse().toarray(), cl, q, omegas, sigma=1 / target.tau)
    scale = np.abs(S_ref).max()
    assert np.abs(S_t - S_ref).max() < 1e-4 * scale
