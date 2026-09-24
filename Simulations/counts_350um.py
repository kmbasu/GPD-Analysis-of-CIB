"""
counts_350um.py — the adopted 350 µm number-count models and instrument specs
==============================================================================

Shared configuration module for `main_sims_unlensed.py` and
`main_sims_lensed.py`.  Everything that both scripts must agree on lives
here: the three fitted dN/dS models, the three instrument specifications,
and the faint-end bookkeeping that makes the Monte Carlo affordable.

The three count models
----------------------
Best-fit parameters from `num_count_fits/fitting_results_number_counts.md`,
Sec. 7 ("Final adopted fits"), fitted to Bèthermin et al. (2012) Table 3 at
350 µm over S = 6.0–94.6 mJy with the GOODS-N stacked points excluded:

  Schechter  dN/dS = (n*/S*)(S/S*)^alpha exp(-S/S*)
             alpha = -1.890, n* = 7014 deg^-2, S* = 19.01 mJy   chi2/nu = 0.19
  SPL        dN/dS = N0 (S/S0)^beta,  S0 = 2.2 mJy fixed
             N0 = 1.053e5 mJy^-1 deg^-2, beta = -3.279           chi2/nu = 12.6
  DPL        dN/dS = (phi*/S*)[(S/S*)^alpha + (S/S*)^beta]^-1
             alpha = 1.082, beta = 3.928, S* = 11.70 mJy,
             phi* = 1.33e4 deg^-2                                chi2/nu = 3.93

The Schechter and DPL re-use `analysis_modules.counts` classes directly (same
(phi*/S*) normalisation convention), so they are drop-in compatible with the
rest of the pipeline.  The SPL needs a new class, supplied below.

The faint end: why S_MIN = 0.1 mJy, and why it matters unequally
-----------------------------------------------------------------
None of the three fits is calibrated below 6 mJy, but at S_min = 6 mJy the
source density is only N_beam ~ 0.2-0.6 at 15-25", i.e. there is no Gaussian
confusion core at all and the whole P(D) picture collapses.  We therefore
extrapolate all three models down to a common S_MIN = 0.1 mJy, roughly the
depth of the deepest 350 µm counts.  Measured consequence at 25":

    model      sigma_c(S_min=6)   sigma_c(S_min=0.1)   N_beam(S_min=0.1)
    Schechter      7.10               8.06                    44
    DPL            7.09               7.55                     4
    SPL            6.72              14.91                  6368

The Schechter and DPL are faint-end convergent, so the extrapolation is
nearly immaterial for them (<15% in sigma_c).  The SPL has beta = -3.279,
i.e. a second moment that formally diverges as S_min -> 0: its sigma_c
doubles and its source density explodes.  That is not a numerical nuisance
but a physical statement — a bare power law is not a viable confusion model —
and it is reported rather than hidden.

The faint-end Gaussian substitution
------------------------------------
Injecting 6368 sources per beam (SPL at 25") or 917000 (SPL at Planck's 5")
is not feasible.  It is also unnecessary: a population with N_beam >> 1
contributes a Gaussian by the central limit theorem and cannot produce tail
events on its own.  `faint_split` therefore picks a split flux S_split such
that N_beam(>S_split) = N_EXPLICIT_MAX, injects sources explicitly only above
it, and replaces everything in [S_MIN, S_split] by a beam-correlated Gaussian
of exactly the matching variance,

    sigma_faint^2 = (Omega_beam / 2) * int_{S_MIN}^{S_split} S^2 (dN/dS) dS,

added in quadrature to the instrument noise.  Because sigma_c^2 is additive
in disjoint flux intervals, the total confusion variance is preserved
exactly.  The substitution is validated in `main_sims_unlensed.py` Sec. 5 by
comparing the MC pixel histogram against the analytic P(D), which uses the
full, unsplit counts and knows nothing about the split.

Instrument specifications
--------------------------
  Planck    5' beam.  sigma_N = 29.3 mJy/beam measured from (odd-even)/2 in
            `Planck_analysis/Planck_analysis_summary.md`; against the model
            sigma_c ~ 90-97 that is f_N ~ 0.3.  S_cut = 650 mJy, the Planck
            857 GHz point-source mask limit (S/N > 5) recorded in the same
            memo.
  Herschel  25" (SPIRE 350 µm native).  Confusion dominated, f_N = 0.5.
            S_cut = 100 mJy, the top of the fitted range.  Cross-check: the
            models give sigma_c = 7.6-8.1 mJy/beam against the measured
            SPIRE 350 µm confusion noise of 6.3 mJy/beam (Nguyen et al. 2010).
  CCAT      15".  f_N = 0.5, S_cut = 100 mJy.

Units: mJy, mJy/beam, arcsec (beam), arcmin (aperture).
"""

# %% Imports
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.join(os.path.dirname(_HERE), "analysis_modules")
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from counts import (BaseCounts, Schechter, DoublePowerLaw,          # noqa: E402
                    FWHM_TO_SIGMA, DEG2_TO_ARCSEC2)
from gpd_tail import confusion_sigma                                 # noqa: E402

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# %% The single power law (not in analysis_modules.counts)
class SinglePowerLaw(BaseCounts):
    """
    Single power law in the Patanchon et al. (2009) parametrisation,

        dN/dS = N0 (S / S0)^beta ,      beta < 0,

    with S0 a FIXED reference flux (chosen to decorrelate N0 from beta) and
    N0 in mJy^-1 deg^-2.

    CAUTION: for beta < -3 the second flux moment q2 = int S^2 (dN/dS) dS
    diverges as S_min -> 0, so `confusion_sigma` on this model is a strong
    function of s_min.  The fitted beta = -3.279 is in that regime.  The
    closed form used for the cross-check in `main_sims_unlensed.py` is

        q2 = N0 S0^-beta (S_max^(3+beta) - S_min^(3+beta)) / (3+beta).
    """

    def __init__(self, N0, beta, S0=2.2, s_min=1e-3, s_max=1e4):
        self.N0, self.beta, self.S0 = float(N0), float(beta), float(S0)
        self.s_min, self.s_max = float(s_min), float(s_max)

    def dnds(self, S):
        S = np.asarray(S, float)
        with np.errstate(divide="ignore", over="ignore"):
            out = self.N0 * (S / self.S0) ** self.beta
        return np.where(S > 0, out, 0.0)

    def q2_closed_form(self, s_min=None, s_max=None):
        """Analytic second moment, for validating the numerical integrator."""
        a = self.s_min if s_min is None else s_min
        b = self.s_max if s_max is None else s_max
        p = 3.0 + self.beta
        return self.N0 * self.S0 ** (-self.beta) * (b ** p - a ** p) / p


# %% The adopted fits (memo Sec. 7)
FIT_SCHECHTER = dict(alpha=-1.8896771723921697,
                     nstar_deg2=7013.971101909618,
                     sstar_mJy=19.01179957743825)
FIT_SPL = dict(N0=105305.00693815576, beta=-3.278939371109816, S0_mJy=2.2)
FIT_DPL = dict(alpha=1.0815396709630054, beta=3.9279212310741896,
               sstar_mJy=11.703690776354527, phistar_deg2=13315.943621569337)

S_MIN_DEFAULT = 0.1          # mJy — common faint-end extrapolation limit
FIT_RANGE_MJY = (6.0, 94.6)  # where the fits are actually calibrated


def make_models(s_min=S_MIN_DEFAULT, s_max=1e4):
    """The three adopted 350 µm models, as `BaseCounts` instances."""
    return {
        "Schechter": Schechter(alpha=FIT_SCHECHTER["alpha"],
                               log_sstar=np.log10(FIT_SCHECHTER["sstar_mJy"]),
                               log_phistar=np.log10(FIT_SCHECHTER["nstar_deg2"]),
                               s_min=s_min, s_max=s_max),
        "SPL": SinglePowerLaw(N0=FIT_SPL["N0"], beta=FIT_SPL["beta"],
                              S0=FIT_SPL["S0_mJy"], s_min=s_min, s_max=s_max),
        "DPL": DoublePowerLaw(alpha=FIT_DPL["alpha"], beta=FIT_DPL["beta"],
                              log_sstar=np.log10(FIT_DPL["sstar_mJy"]),
                              log_phistar=np.log10(FIT_DPL["phistar_deg2"]),
                              s_min=s_min, s_max=s_max),
    }


MODEL_COLORS = {"Schechter": "C0", "SPL": "C2", "DPL": "C3"}
MODEL_ORDER = ["Schechter", "SPL", "DPL"]


# %% Instrument specifications
class Instrument:
    """
    One experimental configuration.

    beam_fwhm_arcsec, f_N (= sigma_N / sigma_c), s_cut [mJy], and the map
    geometry.  `pix_arcsec` is set to FWHM/5 so the beam is comfortably
    oversampled (the declustering peak-density calibration of
    `GPD_beam_and_declustering_v2.md` Sec. 3.5 holds for FWHM/pix ~ 4-10).
    """

    def __init__(self, name, beam_fwhm_arcsec, f_N, s_cut,
                 npix_unlensed=512, theta_in=None, theta_out=None,
                 sigma_c_quoted=None, note=""):
        self.name = name
        self.beam = float(beam_fwhm_arcsec)
        self.pix = self.beam / 5.0
        self.f_N = float(f_N)
        self.s_cut = float(s_cut)
        self.npix_unlensed = int(npix_unlensed)
        self.theta_in = theta_in            # arcmin, lensing aperture
        self.theta_out = theta_out
        self.sigma_c_quoted = sigma_c_quoted
        self.note = note

    @property
    def om_beam_arcmin2(self):
        return 1.133 * (self.beam / 60.0) ** 2

    def npix_lensed(self, pad=2.6, minimum=64):
        """Cutout size that comfortably contains the analysis aperture."""
        need = pad * 2.0 * self.theta_out * 60.0 / self.pix
        n = 1 << int(np.ceil(np.log2(max(need, minimum))))
        return int(n)

    def __repr__(self):
        return (f"<{self.name}: {self.beam:.0f}\" beam, {self.pix:.1f}\" pix, "
                f"f_N={self.f_N}, S_cut={self.s_cut:.0f} mJy>")


INSTRUMENTS = [
    Instrument("Planck", 300.0, f_N=0.3, s_cut=650.0, npix_unlensed=256,
               theta_in=5.0, theta_out=12.0, sigma_c_quoted=90.0,
               note="sigma_N = 29.3 mJy/beam from (odd-even)/2; S_cut is the "
                    "857 GHz point-source mask (S/N>5, ~600-700 mJy)"),
    Instrument("Herschel", 25.0, f_N=0.5, s_cut=100.0, npix_unlensed=512,
               theta_in=0.4, theta_out=1.5, sigma_c_quoted=6.3,
               note="SPIRE 350 um native beam; quoted sigma_c is the measured "
                    "Nguyen et al. (2010) confusion limit"),
    Instrument("CCAT", 15.0, f_N=0.5, s_cut=100.0, npix_unlensed=512,
               theta_in=0.3, theta_out=1.2, sigma_c_quoted=3.0,
               note="CCAT-prime 350 um class resolution"),
]
INSTRUMENT_BY_NAME = {i.name: i for i in INSTRUMENTS}


# %% Faint-end split
N_EXPLICIT_MAX = 200.0       # sources per beam injected individually


def faint_split(cnts, beam_fwhm_arcsec, s_cut, s_min=S_MIN_DEFAULT,
                n_explicit_max=N_EXPLICIT_MAX):
    """
    Split the counts into an explicitly-injected bright part and a Gaussian
    faint part.

    Returns dict with
      s_split       : flux below which sources are replaced by a Gaussian
      sigma_faint   : rms [mJy/beam] of that Gaussian
      sigma_c_total : sigma_c of the FULL population over [s_min, s_cut]
      sigma_c_bright: sigma_c of the explicitly injected part
      n_beam_total, n_beam_bright
    `s_split == s_min` (and `sigma_faint == 0`) when no split is needed.

    The split flux is found by bisection on N_beam(>S) = n_explicit_max.
    Since sigma_c^2 is additive over disjoint flux intervals,
    sigma_c_total^2 = sigma_c_bright^2 + sigma_faint^2 by construction; the
    caller should assert this.
    """
    full = confusion_sigma(cnts, beam_fwhm_arcsec, s_lo=s_min, s_cut=s_cut)
    if full["n_beam"] <= n_explicit_max:
        return dict(s_split=s_min, sigma_faint=0.0,
                    sigma_c_total=full["sigma_c"],
                    sigma_c_bright=full["sigma_c"],
                    n_beam_total=full["n_beam"],
                    n_beam_bright=full["n_beam"])

    lo, hi = s_min, s_cut
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if confusion_sigma(cnts, beam_fwhm_arcsec, s_lo=mid,
                           s_cut=s_cut)["n_beam"] > n_explicit_max:
            lo = mid
        else:
            hi = mid
    s_split = float(np.sqrt(lo * hi))
    bright = confusion_sigma(cnts, beam_fwhm_arcsec, s_lo=s_split, s_cut=s_cut)
    faint = confusion_sigma(cnts, beam_fwhm_arcsec, s_lo=s_min, s_cut=s_split)
    return dict(s_split=s_split, sigma_faint=float(faint["sigma_c"]),
                sigma_c_total=float(full["sigma_c"]),
                sigma_c_bright=float(bright["sigma_c"]),
                n_beam_total=float(full["n_beam"]),
                n_beam_bright=float(bright["n_beam"]))


def noise_budget(cnts, inst, s_min=S_MIN_DEFAULT):
    """
    Full noise bookkeeping for one (model, instrument) pair.

    sigma_N is defined against the TOTAL confusion noise of the model at that
    beam, sigma_N = f_N * sigma_c_total, so the instrument specification is a
    property of the instrument and not of the faint-end split.  The Gaussian
    actually handed to the simulator is
        sigma_gauss = hypot(sigma_faint, sigma_N),
    beam-correlated (`noise_mode="beam"`).
    """
    sp = faint_split(cnts, inst.beam, inst.s_cut, s_min=s_min)
    sigma_n = inst.f_N * sp["sigma_c_total"]
    sp.update(sigma_n=sigma_n,
              sigma_gauss=float(np.hypot(sp["sigma_faint"], sigma_n)),
              sigma_tot=float(np.hypot(sp["sigma_c_total"], sigma_n)))
    return sp


# %% Self-test
if __name__ == "__main__":
    print(__doc__.split("Units:")[0][-1:])
    models = make_models()
    print("SPL q2 closed form vs numerical integrator:")
    spl = models["SPL"]
    num = spl.moment(2, 0.1, 100.0)
    cf = spl.q2_closed_form(0.1, 100.0)
    print(f"  numerical {num:.6e}   closed form {cf:.6e}   "
          f"rel. diff {abs(num / cf - 1):.2e}")

    print(f"\n{'instrument':>10s} {'model':>10s} {'s_split':>8s} "
          f"{'sig_faint':>10s} {'sig_c':>8s} {'sig_N':>8s} {'Sig_tot':>8s} "
          f"{'N_beam':>10s} {'N_bright':>9s} {'quoted sig_c':>13s}")
    for inst in INSTRUMENTS:
        for nm in MODEL_ORDER:
            c = make_models(s_max=inst.s_cut)[nm]
            b = noise_budget(c, inst)
            chk = abs(np.hypot(b["sigma_c_bright"], b["sigma_faint"])
                      / b["sigma_c_total"] - 1)
            assert chk < 1e-6, f"variance split not additive: {chk}"
            print(f"{inst.name:>10s} {nm:>10s} {b['s_split']:8.3f} "
                  f"{b['sigma_faint']:10.2f} {b['sigma_c_total']:8.2f} "
                  f"{b['sigma_n']:8.2f} {b['sigma_tot']:8.2f} "
                  f"{b['n_beam_total']:10.1f} {b['n_beam_bright']:9.1f} "
                  f"{inst.sigma_c_quoted:13.1f}")
    print("\nvariance split is additive to <1e-6 for every combination.")
