"""Accuracy target set by the neutron experiment.

An instrument with Gaussian energy resolution of full width at half maximum
dE measures the true S(q, omega) convolved with that Gaussian.  In the time
domain this is exactly a Gaussian window

    w(t) = exp(-t^2 / (2 tau^2)),   tau = 2 sqrt(2 ln 2) / dE   (hbar = 1),

so the correlation function is only needed out to the time T at which the
window has decayed below a cutoff: T = tau sqrt(2 ln(1/cutoff)).  This
replaces the rough estimate T ~ 2 pi hbar / dE of the plan by an exact
statement; both are reported.

The accuracy eps is the largest error in the resolution-convolved spectrum
allowed, as a fraction of the peak spectral weight; it should sit below the
experiment's relative error bars.

STATUS: dE and the error bars must be taken from the alpha-RuCl3 INS papers
(Banerjee et al., Science 2017; Do et al., Nat. Phys. 2017).  The default
values below are placeholders, not measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HBAR_MEV_PS = 0.6582119569  # hbar in meV ps
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


@dataclass(frozen=True)
class AccuracyTarget:
    dE: float  # instrument resolution, FWHM in meV
    eps: float  # allowed error relative to peak spectral weight
    cutoff: float = 1e-3  # window value at which the time signal is truncated
    omega_max: float = 30.0  # highest energy transfer to resolve (meV)
    source: str = "placeholder"

    @property
    def tau(self) -> float:
        """Gaussian window width in meV^-1."""
        return FWHM_PER_SIGMA / self.dE

    @property
    def T(self) -> float:
        """Longest simulated time, in meV^-1."""
        return self.tau * np.sqrt(2 * np.log(1 / self.cutoff))

    @property
    def T_ps(self) -> float:
        return self.T * HBAR_MEV_PS

    @property
    def T_rough_ps(self) -> float:
        """The plan's estimate 2 pi hbar / dE, for comparison."""
        return 2 * np.pi * HBAR_MEV_PS / self.dE

    @property
    def dt(self) -> float:
        """Sampling interval that resolves energies up to omega_max (Nyquist)."""
        return np.pi / self.omega_max

    @property
    def n_times(self) -> int:
        return int(np.ceil(self.T / self.dt)) + 1

    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.n_times)

    def window(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (np.asarray(t) / self.tau) ** 2)


# Placeholders until the INS papers are checked (decide on 25 Oct).
DEFAULT = AccuracyTarget(dE=0.5, eps=0.05)
