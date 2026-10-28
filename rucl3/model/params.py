"""Published exchange parameter sets for alpha-RuCl3 (energies in meV).

Every entry is written for spin-1/2 operators S = sigma/2 and the Hamiltonian

    H = sum_<ij>_gamma [ J1 S_i.S_j + K S_i^g S_j^g + G (S_i^a S_j^b + S_i^b S_j^a)
                         + Gp (S_i^g S_j^a + S_i^g S_j^b + S_i^a S_j^g + S_i^b S_j^g) ]
        + J3 sum_<<<ij>>> S_i.S_j

STATUS: the numbers below were entered from memory and are NOT yet checked
against the papers.  Before any result is frozen, open each PDF, confirm the
values, the sign conventions and whether the paper uses S or sigma, then set
``verified=True``.  Sets with ``verified=False`` print a warning when used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSet:
    name: str
    J1: float = 0.0
    K: float = 0.0
    G: float = 0.0
    Gp: float = 0.0
    J3: float = 0.0
    reference: str = ""
    verified: bool = False
    note: str = ""

    def scaled(self, **overrides) -> "ParameterSet":
        d = self.__dict__ | overrides
        return ParameterSet(**d)


PARAMETER_SETS: dict[str, ParameterSet] = {
    "winter2017": ParameterSet(
        "winter2017", J1=-0.5, K=-5.0, G=2.5, Gp=0.0, J3=0.5,
        reference="S. M. Winter et al., Nat. Commun. 8, 1152 (2017)",
        note="ab initio + fitted to INS; check whether Gp is quoted",
    ),
    "ran2017": ParameterSet(
        "ran2017", J1=0.0, K=-6.8, G=9.5, Gp=0.0, J3=0.0,
        reference="K. Ran et al., Phys. Rev. Lett. 118, 107203 (2017)",
        note="K-Gamma model fitted to spin waves; K and Gamma confirmed from the abstract "
             "(28 Oct 2026); the paper calls J small, set to 0 here; check the article number",
    ),
    "laurell2020": ParameterSet(
        "laurell2020", J1=-1.3, K=-15.1, G=10.1, Gp=-0.12, J3=0.9,
        reference="P. Laurell and S. Okamoto, npj Quantum Mater. 5, 2 (2020)",
        note="fitted to specific heat and INS with ED",
    ),
    # Kim & Kee give several sets depending on the assumed crystal structure;
    # pick one on 25 Oct and fill it in.
    "kimkee2016": ParameterSet(
        "kimkee2016",
        reference="H.-S. Kim and H.-Y. Kee, Phys. Rev. B 93, 155143 (2016)",
        note="TODO: values not entered; choose the C2/m set",
    ),
}

# Idealised limits used in tests and sanity checks (no warning).
KITAEV = ParameterSet("kitaev", K=-1.0, verified=True, note="pure ferromagnetic Kitaev")
HEISENBERG = ParameterSet("heisenberg", J1=1.0, verified=True, note="antiferromagnetic Heisenberg")


def get(name: str) -> ParameterSet:
    p = PARAMETER_SETS[name]
    if not any((p.J1, p.K, p.G, p.Gp, p.J3)):
        raise ValueError(f"parameter set {name!r} has no values yet: {p.note}")
    if not p.verified:
        warnings.warn(f"parameter set {name!r} is not verified against {p.reference}",
                      stacklevel=2)
    return p
