"""
sim_core.py — shared engine for the three instrument-specific simulation modules
=================================================================================

Common machinery for `simulate_planck.py`, `simulate_herschel.py` and
`simulate_ccat.py`.  Everything that is *not* instrument-specific lives here:
the benchmark lens, the noise bookkeeping, the peak/GPD estimators, the
bootstrap, and the cluster-count forecast.  Each driver script imports this
module, fills in one `Instrument` record, and runs.

WHY A SHARED MODULE RATHER THAN THREE COPIES
--------------------------------------------
The three drivers remain fully standalone from the *user's* point of view —
`python simulate_herschel.py` runs the whole Herschel arm end to end and writes
its own figures, `.npz` and log.  But the estimator code exists once, so a fix
to the GPD scan or the bootstrap cannot silently apply to two experiments and
not the third.  If you would rather have three literally self-contained files,
this module is small enough to paste into each; the instrument configuration
blocks in the drivers are already written to be the only thing you would need
to keep distinct.

WHAT CHANGED RELATIVE TO `main_sims_unlensed.py` / `main_sims_lensed.py`
-------------------------------------------------------------------------
Every item below is a recommendation of `instrument_consistency_audit.md`
(2026-08-15).  Section numbers refer to that memo.

  R1  S_min = 1.0 mJy, not 0.1.  Matches `herschel_unlensed_v3.py:386` and
      `planck_unlensed_pd_v2.py:266`.  Costs <=1.6% in sigma_c for Schechter
      and DPL, removes a 49-55% inconsistency in the SPL, and takes the
      Schechter's predicted 857 GHz monopole from 2.7x to 1.6x the measured
      background.  See `S_MIN`.

  R2  sigma_N is an ABSOLUTE measured number per instrument, not f_N*sigma_c.
      The old parametrisation made the detector noise a function of the count
      model, which is wrong in principle and was 34% low at Herschel and
      1.3-2x low at CCAT.  See `Instrument.sigma_n`.

  R3  Planck carries an explicit non-Poisson Gaussian of 159.2 mJy/beam
      (clustered CIB + residual cirrus), so its simulated core reproduces the
      measured 185.4 mJy/beam instead of 101.3.  See `Instrument.sigma_nonp`.

  R4  The count models are used EXACTLY as fitted.  No renormalisation of
      n_star anywhere, despite the 11-25% sigma_c offset against Nguyen et al.
      (2010) — that offset is 1.1 sigma of the joint uncertainty and the cut
      dependence is reproduced to 4%, so it is an amplitude, not a shape.

  R6  Threshold grids are anchored in ABSOLUTE mJy/beam against each
      experiment's measured core, not in k*Sigma_tot of the simulation's own
      (possibly wrong) core.  See `Instrument.u_grid`.

  R8  The lens is a single fixed benchmark, and the lensed deliverable is
      (theory, per-cluster noise, clusters-needed) rather than a stacked
      Delta_xi that would silently imply a sample.

THE BENCHMARK LENS
------------------
One NFW halo, identical for all three experiments:

    M_500 = 1.0e15 Msun,  z_l = 0.5,  z_s = 2.0,  Planck18 cosmology

converted to the virial mass the lens model wants by matching the NFW enclosed
mass at R_500, with Duffy et al. (2008) concentration and Bryan & Norman (1998)
overdensity — the same conversion `Revised_Planck_analysis.md` Sec. 4.2 applies
to PSZ2.  This gives

    R_500 = 1.293 Mpc,  theta_500 = 3.425',  M_vir = 1.794e15,  c_vir = 3.503

The point of a *common* benchmark is that the three experiments then differ
only in beam, noise and survey geometry, so the comparison isolates what the
instrument does rather than what its cluster catalogue happens to contain.

THE FORECAST, AND WHY IT IS NOT A SINGLE-CLUSTER Delta_xi
----------------------------------------------------------
One cluster yields, in the r < 1.5 theta_500 aperture and at the declustered
rate of one peak per ~4 beam solid angles,

    Planck    2.9 beams  ->   0.7 peaks
    Herschel  417 beams  -> 104   peaks
    CCAT     1172 beams  -> 293   peaks

A GPD cannot be fitted to 0.7 exceedances, and even 293 peaks gives an error on
xi of order (1+xi)/sqrt(N_exc) ~ 0.1 against a predicted signal two orders of
magnitude smaller.  So a single-cluster Delta_xi is not an observable for any
of the three.  What *is* well defined, and is what these modules deliver:

  1. `delta_xi_theory` — the exact analytic Delta_xi(u) of the benchmark lens.
     Deterministic, no Monte Carlo noise, a genuine single-cluster statement.
  2. `sigma_per_cluster` — the Monte Carlo error on Delta_xi for ONE cluster,
     obtained by running N_MC identical copies and scaling the cluster-level
     bootstrap width by sqrt(N_MC).
  3. `clusters_for_detection` — N = (n_sigma * sigma_1 / Delta_xi_theory)^2,
     the number of such clusters a survey would need.  This is the number that
     answers "is it worth pursuing".

Item 3 carries a documented factor-2 inflation option (`CLUSTERING_INFLATION`)
because `full_simulation_summary_v2.md` Sec. 10.6 measures simulation-derived
exceedance-count scatter to be ~2x too small in rms when source clustering is
omitted, which is the case for the Poisson injector used here.

CONVENTIONS INHERITED UNCHANGED
--------------------------------
* Declustering: local maxima over a one-FWHM square window
  (`gpd_tail.beam_declustered_peaks`), on the WHOLE cutout before the radius
  cut, so a retained peak still had to beat neighbours outside the aperture
  (`GPD_beam_and_declustering_v2.md` Sec. 5.4).
* Instrument noise is BEAM-CORRELATED (`noise_mode="beam"`).  White pixel noise
  inflates the retained peak count ~4x.
* The faint population below `S_split` is replaced by an exact beam-correlated
  Gaussian of matching variance (`counts_350um.faint_split`); sigma_c^2 is
  additive over disjoint flux intervals, so the total is preserved exactly.
* Bootstraps resample the MAP (unlensed) or the CLUSTER (lensed), never the
  individual peak, because peaks within one realisation share its large-scale
  modes.
* Delta_xi intervals come from the difference distribution of PAIRED bootstrap
  replicates, not from adding two half-widths in quadrature.

Units: mJy, mJy/beam, arcsec for beams and pixels, arcmin for apertures.

Requires: numpy, scipy, astropy, matplotlib.
"""

# %% Section 1 — imports and paths
import os
import sys
import time
import pickle
import hashlib
import warnings
import zlib

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import brentq

import matplotlib
if not int(os.environ.get("CIB_SHOW", "0")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18
import astropy.units as as_u

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "analysis_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from counts_350um import (make_models, MODEL_ORDER, MODEL_COLORS,     # noqa: E402
                          faint_split, FIT_RANGE_MJY)
from counts import MixtureCounts               # noqa: E402,F401
from pofd_analytic import PofD                                        # noqa: E402
from simulate_maps import CIBMapSimulator                             # noqa: E402
from lens_model import NFWLens, bryan_norman_delta_c, duffy_cvir      # noqa: E402
from gpd_tail import (confusion_sigma, fit_gpd,                       # noqa: E402
                      xi_of_threshold_analytic)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10})

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# %% Section 2 — global configuration (audit R1, R8)
S_MIN = 1.0                  # mJy — R1.  Matches BOTH data pipelines.
Z_SOURCE = 2.0               # background source redshift, as in H2 and M5-v2
MU_MAX = 100.0               # magnification clip; results shown mu_max-insensitive

# --- the benchmark lens -----------------------------------------------------
M500_BENCHMARK = 1.0e15      # Msun
Z_LENS = 0.5

# --- forecast bookkeeping ---------------------------------------------------
CLUSTERING_INFLATION = 2.0   # v2 memo Sec. 10.6: Poisson sims under-scatter ~2x
PEAKS_PER_BEAM = 0.25        # declustered rate, 1 per ~4 beam solid angles

# --- estimator settings -----------------------------------------------------
N_BOOT_DEFAULT = 300
MIN_EXCEED_DEFAULT = 80      # (1+xi)^2/dxi^2 ceiling of the manual, Sec. 7b
#  Cap on the pooled sample per arm, for speed.  `blocked_scan` thins
#  uniformly above it.  Thinning is unbiased -- it is an i.i.d. subsample of
#  the same marginal -- but it costs precision, and precision loss turns into
#  BIAS wherever it drops a threshold near `min_exceed`, because the
#  surviving fits are then those that happened to draw a heavy tail.
#  Raise it with CIB_MAX_POOLED when the exceedance budget is tight; the
#  2026-08-18 audit runs the production unlensed arms unthinned.
MAX_POOLED_PEAKS = int(os.environ.get("CIB_MAX_POOLED", 60_000))

QUICK_TEST = bool(int(os.environ.get("CIB_QUICK_TEST", "0")))
USE_CACHE = bool(int(os.environ.get("CIB_CACHE", "0")))
SHOW = bool(int(os.environ.get("CIB_SHOW", "0")))


# %% Section 3 — the Instrument record
class Instrument:
    """
    One experiment, specified by MEASURED quantities wherever they exist.

    Parameters
    ----------
    name : str
    beam_arcsec : float
        Gaussian FWHM.  For Herschel use the Condon-equivalent 25.15" from
        the HELP `Matchedfilter` header, not the nominal 24.9".
    pix_arcsec : float
        Map pixel.  Set to the DATA's pixel where data exist, so the
        declustering statistics are those of the real analysis.
    sigma_n : float
        Absolute instrument noise, mJy/beam (audit R2).  NOT f_N * sigma_c.
    sigma_nonp : float
        Additional beam-correlated Gaussian for sky components the Poisson
        injector does not contain (clustered CIB, residual cirrus).  Planck
        only; 0.0 elsewhere (audit R3).
    s_cut : float
        Source-plane flux above which sources are taken to be individually
        detected and removed; the counts are truncated there.
    u_grid : array
        Threshold ladder in ABSOLUTE mJy/beam (audit R6).
    core_measured : float or None
        The measured post-baseline core width, for reporting only.
    apertures_theta500 : sequence
        Filled-disc aperture radii in units of theta_500 for the lensed arm.
    npix_unlensed, n_maps_unlensed : int
        Unlensed Monte Carlo geometry.
    n_clusters : int
        Lensed Monte Carlo ensemble size (identical copies of the benchmark).
    note : str
        Provenance string, printed in the log and stored in the .npz.
    """

    def __init__(self, name, beam_arcsec, pix_arcsec, sigma_n, s_cut, u_grid,
                 sigma_nonp=0.0, core_measured=None,
                 apertures_theta500=(0.5, 1.0, 1.5, 2.0, 3.0),
                 npix_unlensed=512, n_maps_unlensed=60, n_clusters=400,
                 note=""):
        self.name = name
        self.beam = float(beam_arcsec)
        self.pix = float(pix_arcsec)
        self.sigma_n = float(sigma_n)
        self.sigma_nonp = float(sigma_nonp)
        self.s_cut = float(s_cut)
        self.u_grid = np.asarray(u_grid, float)
        self.core_measured = core_measured
        self.apertures_theta500 = tuple(apertures_theta500)
        self.npix_unlensed = int(npix_unlensed)
        self.n_maps_unlensed = int(n_maps_unlensed)
        self.n_clusters = int(n_clusters)
        self.note = note
        if self.beam / self.pix < 2.5:
            raise ValueError(f"{name}: beam under-sampled, "
                             f"FWHM/pix = {self.beam / self.pix:.2f} < 2.5")

    # -- geometry ------------------------------------------------------------
    @property
    def beam_fwhm_pix(self):
        return self.beam / self.pix

    @property
    def om_beam_arcmin2(self):
        """Gaussian beam solid angle, 1.133 FWHM^2 [arcmin^2]."""
        return 1.133 * (self.beam / 60.0) ** 2

    def map_area_deg2(self, npix):
        return (npix * self.pix / 3600.0) ** 2

    def npix_for_aperture(self, r_arcmin, pad=1.4, minimum=64):
        """
        Smallest cutout comfortably containing a disc of radius `r_arcmin`.

        Rounded up to a multiple of 64 rather than to a power of two: at CCAT's
        3" pixel the aperture needs 384 px and the power-of-two rule would give
        512, a 78% pixel-count penalty for no benefit.  numpy's FFT handles
        multiples of 64 efficiently (2^6 times a small factor).

        The margin is safe against the periodic boundaries of the FFT beam
        convolution, and the whole-map mean subtraction is unbiased between the
        lensed and control arms because magnification conserves the mean
        intensity exactly: n_mu(S) = mu^-2 n_0(S/mu) gives
        int S n_mu dS = int S n_0 dS.  Only the S_cut truncation breaks that,
        at the sub-percent level.
        """
        need = pad * 2.0 * r_arcmin * 60.0 / self.pix
        return int(64 * int(np.ceil(max(need, minimum) / 64.0)))

    # -- Gaussian budget -----------------------------------------------------
    @property
    def sigma_gauss_extra(self):
        """Everything Gaussian that is NOT the Poisson source population:
        instrument noise plus the non-Poisson sky (audit R2, R3)."""
        return float(np.hypot(self.sigma_n, self.sigma_nonp))

    def __repr__(self):
        return (f"<{self.name}: {self.beam:.2f}\" beam, {self.pix:.1f}\" pix, "
                f"sigma_N={self.sigma_n:.3f}"
                + (f", sigma_nonP={self.sigma_nonp:.1f}" if self.sigma_nonp
                   else "")
                + f", S_cut={self.s_cut:.0f} mJy>")


# %% Section 4 — the benchmark lens
def m500_to_mvir(m500, z_l, cosmo=Planck18):
    """
    Convert M_500c to M_vir by matching the NFW enclosed mass at R_500.

    Uses the Bryan & Norman (1998) virial overdensity relative to the CRITICAL
    density and the Duffy et al. (2008) 'full' virial concentration, i.e. the
    identical convention `Revised_Planck_analysis.md` Sec. 4.2 applies to the
    PSZ2 catalogue.  Solving is a 1-D root find because c_vir depends on M_vir.

    Returns (m_vir, r500_Mpc, theta500_arcmin, c_vir).
    """
    rho_c = cosmo.critical_density(z_l).to(as_u.Msun / as_u.Mpc ** 3).value
    r500 = (m500 / (4.0 / 3.0 * np.pi * 500.0 * rho_c)) ** (1.0 / 3.0)
    d_a = cosmo.angular_diameter_distance(z_l).to(as_u.Mpc).value
    theta500 = np.degrees(r500 / d_a) * 60.0

    def _m_nfw(x):
        return np.log(1.0 + x) - x / (1.0 + x)

    def _enclosed_at_r500(m_vir):
        dc = bryan_norman_delta_c(z_l)
        c = duffy_cvir(m_vir, z_l)
        r_vir = (m_vir / (4.0 / 3.0 * np.pi * dc * rho_c)) ** (1.0 / 3.0)
        return m_vir * _m_nfw(r500 / (r_vir / c)) / _m_nfw(c)

    m_vir = brentq(lambda m: _enclosed_at_r500(m) - m500, 1e13, 1e17,
                   xtol=1e9)
    return float(m_vir), float(r500), float(theta500), \
        float(duffy_cvir(m_vir, z_l))


class Benchmark:
    """The single fixed lens shared by all three experiments (audit R8)."""

    def __init__(self, m500=M500_BENCHMARK, z_l=Z_LENS, z_s=Z_SOURCE):
        self.m500, self.z_l, self.z_s = float(m500), float(z_l), float(z_s)
        (self.m_vir, self.r500_mpc,
         self.theta500, self.c_vir) = m500_to_mvir(self.m500, self.z_l)
        self.lens = NFWLens(m_vir=self.m_vir, z_l=self.z_l, z_s=self.z_s)

    def mu(self, theta_arcmin):
        return self.lens.mu(theta_arcmin, mu_max=MU_MAX)

    def disc_mixture(self, r_theta500, n_rings=200, r_inner=1e-3):
        """
        Area-weighted magnification mixture P_A(mu) of a FILLED disc of radius
        `r_theta500` * theta_500.

        A filled disc, not the annulus the old modules used: the data analyses
        (`Revised_Planck_analysis.md` Sec. 8.2, H2 Sec. 7.3) both measure inside
        filled discs r < R theta_500, so this is what the forecast must match.

        Returns (mus, weights, r_arcmin).
        """
        r_out = r_theta500 * self.theta500
        e = np.linspace(r_inner, r_out, n_rings + 1)
        mid = 0.5 * (e[1:] + e[:-1])
        w = e[1:] ** 2 - e[:-1] ** 2
        return self.mu(mid), w / w.sum(), r_out

    def disc_stats(self, r_theta500, n_rings=2000):
        mus, w, r_out = self.disc_mixture(r_theta500, n_rings=n_rings)
        return dict(r_arcmin=r_out,
                    mean_mu=float(np.sum(w * mus)),
                    mean_ln_mu=float(np.sum(w * np.log(mus))),
                    max_mu=float(mus.max()))

    def summary(self):
        return (f"NFW benchmark: M_500 = {self.m500:.3e} Msun at z_l = "
                f"{self.z_l}, z_s = {self.z_s}\n"
                f"  -> R_500 = {self.r500_mpc:.3f} Mpc, theta_500 = "
                f"{self.theta500:.3f}', M_vir = {self.m_vir:.4e} Msun, "
                f"c_vir = {self.c_vir:.3f}")


def binned_mixture(mus, w, n_bins=64):
    """Compress a fine mu-mixture to `n_bins` components (speed, no bias:
    weights and the weighted mean mu are preserved bin by bin)."""
    edges = np.geomspace(mus.min(), mus.max() * (1 + 1e-12), n_bins + 1)
    idx = np.clip(np.digitize(mus, edges) - 1, 0, n_bins - 1)
    wb = np.bincount(idx, weights=w, minlength=n_bins)
    mb = np.bincount(idx, weights=w * mus, minlength=n_bins)
    k = wb > 0
    return mb[k] / wb[k], wb[k] / wb.sum()


# %% Section 5 — noise bookkeeping (audit R1, R2, R3)
def budget(model, inst, s_min=S_MIN):
    """
    Full noise bookkeeping for one (count model, instrument) pair.

    Unlike the old `counts_350um.noise_budget`, sigma_N is NOT derived from
    sigma_c: it is the instrument's measured value, so changing the count model
    no longer silently changes the detector (audit R2).

    Returns a dict with
      sigma_c      : Condon rms of the FULL source population [mJy/beam]
      sigma_faint  : rms of the sub-S_split population, absorbed into a Gaussian
      s_split      : flux below which sources are not injected individually
      sigma_n      : instrument noise (from the instrument record)
      sigma_nonp   : non-Poisson sky Gaussian (Planck only)
      sigma_gauss  : the Gaussian actually handed to CIBMapSimulator
      sigma_tot    : quadrature total = the simulated core width
      n_beam       : sources per beam above s_min
      f_n_implied  : sigma_n / sigma_c, for comparison with the old f_N
    """
    sp = faint_split(model, inst.beam, inst.s_cut, s_min=s_min)
    full = confusion_sigma(model, inst.beam, s_lo=s_min, s_cut=inst.s_cut)
    sig_c = float(sp["sigma_c_total"])
    sig_gauss = float(np.sqrt(sp["sigma_faint"] ** 2
                              + inst.sigma_n ** 2 + inst.sigma_nonp ** 2))
    return dict(sigma_c=sig_c,
                sigma_faint=float(sp["sigma_faint"]),
                s_split=float(sp["s_split"]),
                sigma_n=inst.sigma_n,
                sigma_nonp=inst.sigma_nonp,
                sigma_gauss=sig_gauss,
                sigma_tot=float(np.hypot(sig_c, inst.sigma_gauss_extra)),
                n_beam=float(full["n_beam"]),
                gamma1=float(full["gamma1"]),
                f_n_implied=inst.sigma_n / sig_c)


def bright_only(model, s_split, s_cut):
    """A shallow copy of `model` restricted to [s_split, s_cut] — the part the
    simulator injects explicitly.  Same trick as `main_sims_lensed.py`."""
    c = type(model).__new__(type(model))
    c.__dict__.update(model.__dict__)
    c.s_min, c.s_max = float(s_split), float(s_cut)
    return c


def simulator(model, inst, bud, npix):
    """A `CIBMapSimulator` wired with this instrument's full Gaussian budget."""
    return CIBMapSimulator(bright_only(model, bud["s_split"], inst.s_cut),
                           beam_fwhm_arcsec=inst.beam,
                           pix_arcsec=inst.pix,
                           npix=npix,
                           sigma_noise=bud["sigma_gauss"],
                           s_cut=inst.s_cut,
                           noise_mode="beam")


# %% Section 6 — estimators
def declustered_peaks(m, beam_fwhm_pix):
    """Local maxima over a one-FWHM square window.  NaNs excluded."""
    a = np.where(np.isfinite(m), m, -np.inf)
    size = max(1, int(round(beam_fwhm_pix)))
    return m[(a == maximum_filter(a, size=size)) & np.isfinite(m)]


def disc_peaks(m, beam_fwhm_pix, pix_arcsec, r_arcmin, r_inner_arcmin=0.0):
    """
    Declustered peaks inside a filled disc centred on the cutout.

    Declustering runs on the WHOLE cutout before the radius cut, so a retained
    peak still had to beat neighbours outside the disc — the selection is then
    an unbiased spatial subsample (`GPD_beam_and_declustering_v2.md` Sec. 5.4).
    """
    a = np.where(np.isfinite(m), m, -np.inf)
    size = max(1, int(round(beam_fwhm_pix)))
    cy = (m.shape[0] - 1) / 2.0
    yy, xx = np.indices(m.shape)
    th = np.hypot(yy - cy, xx - cy) * pix_arcsec / 60.0
    sel = ((a == maximum_filter(a, size=size)) & np.isfinite(m)
           & (th < r_arcmin) & (th >= r_inner_arcmin))
    return m[sel]



def stable_seed(name, modulus=9973):
    """
    Deterministic integer seed derived from a string (e.g. a counts-model name).

    Release note (2026-09-24): the runs behind the paper used
    ``hash(name) % 9973``.  Python randomizes ``str.__hash__`` per interpreter
    session (PYTHONHASHSEED), so that seed -- which drives the bootstrap and the
    rate-thinning of over-large peak pools in `blocked_scan` -- differed from
    run to run, and a fresh run was a statistically equivalent but not
    bit-identical realization of the published one.  CRC32 is stable across
    sessions and platforms, so re-runs are now reproducible.  They still do not
    reproduce the paper's realization bit for bit (its seeds are unrecoverable);
    the published curves are shipped in `results/`.
    """
    return zlib.crc32(str(name).encode("utf-8")) % modulus

def blocked_scan(units, thresholds, n_boot=N_BOOT_DEFAULT,
                 min_exceed=MIN_EXCEED_DEFAULT, seed=0,
                 max_pooled=MAX_POOLED_PEAKS):
    """
    GPD xi-hat versus threshold with a BLOCK bootstrap over `units`.

    `units` is a list of per-realisation peak arrays (one array per map, or per
    cluster).  The resampling unit is the realisation, never the individual
    peak, because peaks inside one map share its large-scale modes.  The full
    replicate matrix is returned so that paired differences (Delta_xi) can be
    formed replicate by replicate rather than by adding half-widths.

    Returns dict(xi, lo, hi, sigma, n_exc, reps, n_units).
    """
    rng = np.random.default_rng(seed)
    units = [np.asarray(x, float) for x in units if np.size(x)]
    if not units:
        n = len(thresholds)
        return dict(xi=np.full(n, np.nan), lo=np.full(n, np.nan),
                    hi=np.full(n, np.nan), sigma=np.full(n, np.nan),
                    n_exc=np.zeros(n, int), reps=np.full((n_boot, n), np.nan),
                    n_units=0)
    tot = sum(x.size for x in units)
    if tot > max_pooled:                       # thin uniformly, keeps units
        f = max_pooled / tot
        units = [x[rng.random(x.size) < f] for x in units]
    pooled = np.concatenate(units)
    nu = len(units)

    reps = np.full((n_boot, len(thresholds)), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, nu, nu)
        pb = np.concatenate([units[k] for k in pick])
        for j, thr in enumerate(thresholds):
            y = pb[pb > thr] - thr
            if y.size >= min_exceed:
                reps[b, j] = fit_gpd(y)["xi"]

    xi, ne = [], []
    for thr in thresholds:
        y = pooled[pooled > thr] - thr
        ne.append(int(y.size))
        xi.append(fit_gpd(y)["xi"] if y.size >= min_exceed else np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(reps, [16, 84], axis=0)
        sig = np.nanstd(reps, axis=0)

    #  Guard against a degenerate bootstrap.  Near the exceedance floor most
    #  replicates fall below `min_exceed` and return NaN; with only one or two
    #  survivors np.nanstd returns 0 (or a meaningless number), which then
    #  propagates into `clusters_for_detection` as N = 0.  Require at least half
    #  the replicates -- and never fewer than 10 -- to have converged, and
    #  invalidate the point estimate too, because a threshold whose bootstrap
    #  cannot be formed is not one we can quote.
    valid = np.isfinite(reps).sum(axis=0)
    bad = valid < max(10, int(0.5 * n_boot))
    xi = np.asarray(xi)
    xi[bad] = np.nan
    lo[bad] = hi[bad] = sig[bad] = np.nan
    return dict(xi=xi, lo=lo, hi=hi, sigma=sig, n_valid=valid,
                n_exc=np.asarray(ne), reps=reps, n_units=nu)


def paired_delta(scan_lens, scan_ctrl):
    """
    Delta_xi and its interval from the PAIRED replicate difference.

    Requires the two scans to have been run with the same `n_boot`; replicate b
    of the lensed arm is differenced against replicate b of the control arm, so
    any common-mode fluctuation of the estimator cancels inside the replicate.
    """
    d = scan_lens["reps"] - scan_ctrl["reps"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(d, [16, 84], axis=0)
        sig = np.nanstd(d, axis=0)
    #  Same degenerate-bootstrap guard as `blocked_scan`: a paired difference
    #  needs BOTH arms to have converged in the same replicate.
    valid = np.isfinite(d).sum(axis=0)
    bad = valid < max(10, int(0.5 * d.shape[0]))
    delta = scan_lens["xi"] - scan_ctrl["xi"]
    delta = np.where(bad, np.nan, delta)
    lo, hi, sig = (np.where(bad, np.nan, v) for v in (lo, hi, sig))
    return dict(delta=delta, lo=lo, hi=hi, sigma=sig, n_valid=valid, reps=d)


def xi_theory(model, inst, u_grid, mu_spec=1.0, sigma_extra=None,
              n_fft=2 ** 17):
    """
    Analytic xi(u): the KL projection of the exact P(D) exceedance density onto
    the GPD family.  This is the value the MLE converges to with infinite data,
    so it is the right thing to compare a Monte Carlo against.

    `mu_spec` is 1.0 (unlensed) or a (mus, weights) mixture for a lensed
    aperture.  `sigma_extra` defaults to the instrument's Gaussian budget
    (instrument noise + non-Poisson sky); the faint source population is NOT
    added here because PofD uses the full unsplit counts.
    """
    sig = inst.sigma_gauss_extra if sigma_extra is None else float(sigma_extra)
    #  d_span must cover the true support of P(D) but NOT much more: the FFT
    #  output is clipped with np.maximum(p, 0), so a grid extending far past the
    #  support carries a positive floor of numerical noise that the KL
    #  projection reads as an extremely heavy tail (see the caveat in
    #  `gpd_tail.xi_of_threshold_analytic`).  The support is set by the source
    #  contribution (a few x S_cut) and by the Gaussian (~10 sigma), whichever
    #  is larger -- which at Planck is the Gaussian, because sigma_nonP = 159
    #  mJy/beam dwarfs the 100 mJy truncation.
    pp = PofD(model, inst.beam, mu=mu_spec, sigma_noise=sig,
              s_cut=inst.s_cut, n_fft=n_fft,
              d_span=max(6.0 * inst.s_cut, 10.0 * sig))
    return np.asarray(xi_of_threshold_analytic(pp.d, pp.p, u_grid)[0])


# %% Section 7 — the cluster-count forecast (audit R8)
def clusters_for_detection(delta_theory, sigma_n_clusters, n_clusters,
                           n_sigma=3.0, inflation=None):
    """
    How many benchmark clusters are needed for an `n_sigma` detection.

    Parameters
    ----------
    delta_theory : array
        Analytic Delta_xi(u) for ONE cluster's aperture — this is a per-cluster
        signal amplitude and does NOT scale with the sample size.
    sigma_n_clusters : array
        The Monte Carlo bootstrap sigma on Delta_xi measured from an ensemble of
        `n_clusters` identical copies.
    n_clusters : int
        Size of that ensemble.

    Returns
    -------
    dict with
      sigma_1   : per-cluster error, sigma_N * sqrt(N).  Independent clusters
                  average as 1/sqrt(N), which the cluster-level bootstrap
                  already assumes, so this inversion is exact within the MC.
      n_needed  : (n_sigma * sigma_1 / Delta_xi)^2, Poisson-sim value.
      n_needed_infl : the same with the variance inflated by `inflation`^2,
                  because `full_simulation_summary_v2.md` Sec. 10.6 measures the
                  exceedance-count scatter of a clustered sky to be ~2x the
                  Poisson rms.  QUOTE THIS ONE as the realistic figure.
      u_best, n_best, n_best_infl : the threshold minimising n_needed.

    Note the strong threshold dependence: Delta_xi peaks well above the core, but
    so does the error, so the optimum is usually not at the maximum of
    |Delta_xi|.
    """
    inflation = CLUSTERING_INFLATION if inflation is None else float(inflation)
    dt = np.abs(np.asarray(delta_theory, float))
    sg = np.asarray(sigma_n_clusters, float).copy()
    #  A zero or negative bootstrap width means the replicate set was degenerate,
    #  not that the measurement is infinitely precise.  Drop those thresholds.
    sg[~(sg > 0)] = np.nan
    dt = np.where(dt > 0, dt, np.nan)
    sigma_1 = sg * np.sqrt(float(n_clusters))
    with np.errstate(divide="ignore", invalid="ignore"):
        n_needed = (n_sigma * sigma_1 / dt) ** 2
    n_needed_infl = n_needed * inflation ** 2
    ok = np.isfinite(n_needed)
    if ok.any():
        j = int(np.nanargmin(np.where(ok, n_needed, np.inf)))
        best = (j, float(n_needed[j]), float(n_needed_infl[j]))
    else:
        best = (-1, np.nan, np.nan)
    return dict(sigma_1=sigma_1, n_needed=n_needed,
                n_needed_infl=n_needed_infl,
                j_best=best[0], n_best=best[1], n_best_infl=best[2])


def peak_budget(inst, r_arcmin):
    """Expected declustered peaks in a filled disc of radius r [arcmin]."""
    n_beams = np.pi * r_arcmin ** 2 / inst.om_beam_arcmin2
    return n_beams, n_beams * PEAKS_PER_BEAM


# %% Section 8 — caching, figures, logging
class Run:
    """Per-script bookkeeping: output dirs, disk memoisation, a tee'd log."""

    def __init__(self, tag, config_key):
        self.tag = tag
        self.fig_dir = os.environ.get(
            "CIB_FIGURE_DIR", os.path.join(_HERE, "figures", tag))
        self.res_dir = os.environ.get(
            "CIB_RESULT_DIR", os.path.join(_HERE, "results"))
        self.cache_dir = os.environ.get(
            "CIB_CACHE_DIR", os.path.join(_HERE, "tmp", f"cache_{tag}"))
        for d in (self.fig_dir, self.res_dir):
            os.makedirs(d, exist_ok=True)
        self.key = hashlib.md5(repr(config_key).encode()).hexdigest()[:10]
        self.t0 = time.time()
        self.lines = []

    def say(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def rule(self, title=None, ch="="):
        self.say(ch * 82)
        if title:
            self.say(title)
            self.say(ch * 82)

    def cached(self, name, fn, *a, quiet=False, **kw):
        if not USE_CACHE:
            return fn(*a, **kw)
        os.makedirs(self.cache_dir, exist_ok=True)
        p = os.path.join(self.cache_dir, f"{name}_{self.key}.pkl")
        if os.path.exists(p):
            if not quiet:
                self.say(f"  [cache] {os.path.basename(p)}")
            with open(p, "rb") as fh:
                return pickle.load(fh)
        out = fn(*a, **kw)
        with open(p, "wb") as fh:
            pickle.dump(out, fh, protocol=4)
        return out

    def finish(self, fig, name):
        if SHOW:
            plt.show()
        else:
            p = os.path.join(self.fig_dir, f"{name}.png")
            fig.savefig(p, dpi=160, bbox_inches="tight")
            self.say(f"  wrote {p}")
            plt.close(fig)

    def save(self, **arrays):
        p = os.path.join(self.res_dir, f"{self.tag}.npz")
        np.savez_compressed(p, **arrays)
        self.say(f"\n  wrote {p}")
        q = os.path.join(self.res_dir, f"{self.tag}.log")
        with open(q, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")
        self.say(f"  wrote {q}")
        self.say(f"  total runtime {time.time() - self.t0:.1f} s")


def build_ensemble(run, key, one, n_total, chunk=50):
    """Build a list of per-realisation peak arrays, cached in chunks so an
    interrupted run resumes (with CIB_CACHE=1)."""
    out = []
    for a in range(0, n_total, chunk):
        b = min(a + chunk, n_total)
        out.extend(run.cached(f"{key}_{a:05d}_{b:05d}",
                              lambda a=a, b=b: [one(i) for i in range(a, b)],
                              quiet=True))
    return out


# %% Section 9 — shared reporting blocks
def report_models(run, inst, models, buds):
    """Print the noise budget table and the audit cross-checks."""
    run.rule(f"{inst.name} — noise budget at S_min = {S_MIN} mJy "
             f"(audit R1), S_cut = {inst.s_cut:.0f} mJy")
    run.say(f"  {inst!r}")
    run.say(f"  provenance: {inst.note}")
    run.say("")
    run.say(f"{'model':>10s} {'sigma_c':>9s} {'sigma_N':>9s} "
            f"{'sigma_nonP':>11s} {'Sigma_tot':>10s} {'N_beam':>9s} "
            f"{'skew':>7s} {'s_split':>8s} {'f_N impl':>9s}")
    for nm in MODEL_ORDER:
        b = buds[nm]
        run.say(f"{nm:>10s} {b['sigma_c']:9.2f} {b['sigma_n']:9.2f} "
                f"{b['sigma_nonp']:11.1f} {b['sigma_tot']:10.2f} "
                f"{b['n_beam']:9.1f} {b['gamma1']:7.2f} "
                f"{b['s_split']:8.2f} {b['f_n_implied']:9.3f}")
    if inst.core_measured:
        ref = buds["Schechter"]["sigma_tot"]
        run.say(f"\n  measured core width = {inst.core_measured:.2f} "
                f"mJy/beam;  simulated (Schechter) = {ref:.2f}  "
                f"-> ratio {ref / inst.core_measured:.3f}")
    run.say(f"  counts fitted over S = {FIT_RANGE_MJY[0]}-{FIT_RANGE_MJY[1]} "
            f"mJy; NO renormalisation applied anywhere (audit R4).")


def report_apertures(run, inst, bench):
    """Print the aperture / peak-budget table for the lensed arm."""
    run.rule(f"{inst.name} — benchmark lens apertures", ch="-")
    run.say(bench.summary())
    run.say(f"  beam = {inst.beam:.2f}\" = {inst.beam / 60:.3f}', "
            f"theta_500 = {bench.theta500:.3f}'  "
            f"-> theta_500 / FWHM = {bench.theta500 / (inst.beam / 60):.2f}")
    run.say("")
    run.say(f"{'r/theta500':>11s} {'r [arcmin]':>11s} {'<mu>':>8s} "
            f"{'<ln mu>':>9s} {'beams':>9s} {'peaks/cl':>9s}")
    rows = {}
    for R in inst.apertures_theta500:
        st = bench.disc_stats(R)
        nb, npk = peak_budget(inst, st["r_arcmin"])
        rows[R] = dict(st, n_beams=nb, peaks=npk)
        run.say(f"{R:11.2f} {st['r_arcmin']:11.3f} {st['mean_mu']:8.4f} "
                f"{st['mean_ln_mu']:9.4f} {nb:9.1f} {npk:9.1f}")
    run.say("\n  peaks/cluster = pi r^2 / (4 Omega_beam): declustered maxima "
            "occur at ~1 per 4 beam solid angles, not 1 per beam.")
    return rows


# %% Section 10 — run stage: the unlensed xi(u)
def stage_unlensed(run, inst, models, buds, n_maps=None, npix=None,
                   n_boot=None, min_exceed=None, seed=1000, label=""):
    """
    Monte Carlo xi-hat(u) for all three count models, against the analytic
    curves, on the instrument's ABSOLUTE threshold ladder (audit R6).

    Also validates the faint-end Gaussian substitution by comparing the MC pixel
    rms with the analytic Sigma_tot: the MC injects sources only above S_split
    and carries the rest as a Gaussian, while the analytic budget uses the full
    unsplit counts, so agreement is a non-trivial check.

    Returns dict keyed by model name plus 'u', 'area_deg2'.
    """
    n_maps = inst.n_maps_unlensed if n_maps is None else int(n_maps)
    npix = inst.npix_unlensed if npix is None else int(npix)
    n_boot = N_BOOT_DEFAULT if n_boot is None else int(n_boot)
    min_exceed = MIN_EXCEED_DEFAULT if min_exceed is None else int(min_exceed)
    area = inst.map_area_deg2(npix) * n_maps
    u = inst.u_grid
    tag = f" [{label}]" if label else ""

    run.rule(f"{inst.name}{tag} — UNLENSED xi(u):  {n_maps} maps of "
             f"{npix}^2 at {inst.pix:.1f}\" = {area:.2f} deg^2 total")
    run.say(f"  thresholds [mJy/beam]: "
            + ", ".join(f"{x:.1f}" for x in u))

    out = {"u": u, "area_deg2": area, "n_maps": n_maps, "npix": npix}
    for nm in MODEL_ORDER:
        b = buds[nm]
        sim = simulator(models[nm], inst, b, npix)
        bfp = inst.beam_fwhm_pix

        def one(i, sim=sim, bfp=bfp):
            m = sim.make_map(seed=seed + 7919 * i)
            return declustered_peaks(m, bfp), float(m.std())

        t0 = time.time()
        pairs = build_ensemble(run, f"unl_{inst.name}_{nm}{label}", one, n_maps)
        peaks = [p for p, _ in pairs]
        rms = np.array([r for _, r in pairs])
        scan = run.cached(f"unlscan_{inst.name}_{nm}{label}", blocked_scan,
                          peaks, u, n_boot=n_boot, min_exceed=min_exceed,
                          seed=stable_seed(nm))
        theo = run.cached(f"unlth_{inst.name}_{nm}{label}", xi_theory,
                          models[nm], inst, u)

        npk = sum(p.size for p in peaks)
        n_beams = area / (inst.om_beam_arcmin2 / 3600.0)
        out[nm] = dict(scan=scan, theory=theo, mc_rms=float(rms.mean()),
                       n_peaks=npk, peaks_per_beam=npk / n_beams)
        run.say(f"  {nm:>10s}  {npk:7d} peaks ({npk / n_beams:.3f}/beam)  "
                f"MC rms {rms.mean():7.2f} vs analytic Sigma_tot "
                f"{b['sigma_tot']:7.2f}  ratio {rms.mean() / b['sigma_tot']:.3f}"
                f"   [{time.time() - t0:.0f}s]")
        run.say(f"{'':14s}n_exc = "
                + ", ".join(f"{int(v)}" for v in scan["n_exc"]))

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for nm in MODEL_ORDER:
        d = out[nm]
        c = MODEL_COLORS[nm]
        ax.plot(u, d["theory"], c, lw=2.0, alpha=0.85,
                label=f"{nm} (analytic)")
        good = np.isfinite(d["scan"]["xi"])
        ax.errorbar(u[good] * (1 + 0.012 * MODEL_ORDER.index(nm)),
                    d["scan"]["xi"][good],
                    yerr=[d["scan"]["xi"][good] - d["scan"]["lo"][good],
                          d["scan"]["hi"][good] - d["scan"]["xi"][good]],
                    fmt="o", ms=4.2, color=c, capsize=2.5, lw=1.2,
                    label=f"{nm} (MC)")
    if inst.core_measured:
        ax.axvline(inst.core_measured, color="0.55", ls=":", lw=1.2)
        ax.text(inst.core_measured, ax.get_ylim()[1], r" $\sigma_{\rm core}$",
                fontsize=8, color="0.4", va="top")
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]",
           ylabel=r"$\xi(u)$",
           title=f"{inst.name}{tag} — unlensed $\\xi(u)$, "
                 f"{inst.beam:.2f}$''$ beam, "
                 fr"$\sigma_N$ = {inst.sigma_n:.2f} mJy/beam, "
                 f"{area:.0f} deg$^2$")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    run.finish(fig, f"{run.tag}_unlensed_xi{'_' + label if label else ''}")
    return out


# %% Section 11 — run stage: analytic Delta_xi(u) of the benchmark lens
def stage_lensed_theory(run, inst, models, buds, bench, n_mu_bins=64):
    """
    Exact analytic Delta_xi(u) = xi_lensed - xi_control for the benchmark lens,
    every count model, every aperture.  Deterministic — this is the genuine
    single-cluster statement (audit R8, deliverable 1).

    The lensed arm uses the area-weighted mu-mixture of the FILLED disc, matching
    the aperture geometry both data analyses actually use.
    """
    run.rule(f"{inst.name} — analytic $\\Delta\\xi(u)$ of the benchmark lens",
             ch="-")
    u = inst.u_grid
    out = {"u": u}
    run.say(f"{'r/th500':>8s} {'model':>10s} {'<mu>':>7s} "
            + " ".join(f"{x:>9.1f}" for x in u))
    for R in inst.apertures_theta500:
        mus, w, r_out = bench.disc_mixture(R)
        mm, ww = binned_mixture(mus, w, n_bins=n_mu_bins)
        st = bench.disc_stats(R)
        for nm in MODEL_ORDER:
            xc = run.cached(f"thc_{inst.name}_{nm}", xi_theory,
                            models[nm], inst, u, 1.0)
            xl = run.cached(f"thl_{inst.name}_{nm}_{R}", xi_theory,
                            models[nm], inst, u, (mm, ww))
            d = xl - xc
            out[(R, nm)] = dict(ctrl=xc, lens=xl, delta=d,
                                mean_mu=st["mean_mu"],
                                mean_ln_mu=st["mean_ln_mu"],
                                r_arcmin=r_out)
            run.say(f"{R:8.2f} {nm:>10s} {st['mean_mu']:7.3f} "
                    + " ".join(f"{v:9.5f}" for v in d))

    # ---- figure: one panel per model, curves by aperture --------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, nm in zip(axes, MODEL_ORDER):
        for k, R in enumerate(inst.apertures_theta500):
            d = out[(R, nm)]
            ax.plot(u, d["delta"],
                    color=cmap(k / max(1, len(inst.apertures_theta500) - 1)),
                    lw=2.0,
                    label=fr"$r<{R}\theta_{{500}}$ "
                          fr"($\langle\mu\rangle$={d['mean_mu']:.2f})")
        ax.axhline(0.0, color="0.4", lw=1.0)
        ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]", title=nm)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel(r"$\Delta\xi(u)$")
    fig.suptitle(f"{inst.name} — analytic $\\Delta\\xi(u)$ for one benchmark "
                 fr"cluster ($M_{{500}}=10^{{15}}M_\odot$, $z_l=0.5$, "
                 fr"$\theta_{{500}}={bench.theta500:.2f}'$)", fontsize=11)
    fig.tight_layout()
    run.finish(fig, f"{run.tag}_lensed_theory")
    return out


# %% Section 12 — run stage: the lensed Monte Carlo
def stage_lensed_mc(run, inst, model, bud, bench, mc_model_name="Schechter",
                    n_clusters=None, n_boot=None, min_exceed=None,
                    seed=5000):
    """
    Three-arm Monte Carlo with N identical copies of the benchmark lens.

      LENS  — cutouts containing the lens; peaks read inside each aperture
      CTRL  — statistically identical UNLENSED cutouts, same apertures
      NULL  — a second, independent unlensed set differenced against CTRL

    The NULL must return Delta_xi = 0 within its own error; it measures the
    pipeline's systematic floor and is what makes a small LENS-CTRL difference
    interpretable.  All three arms share the declustering, the radius cut, the
    threshold ladder and the bootstrap.

    Every aperture is read from the SAME declustering pass on each cutout, so
    the full aperture scan costs one map per realisation, not one per aperture.
    Nested apertures are therefore correlated, exactly as in the real analyses.
    """
    n_clusters = inst.n_clusters if n_clusters is None else int(n_clusters)
    n_boot = N_BOOT_DEFAULT if n_boot is None else int(n_boot)
    min_exceed = MIN_EXCEED_DEFAULT if min_exceed is None else int(min_exceed)
    r_max = max(inst.apertures_theta500) * bench.theta500
    npix = inst.npix_for_aperture(r_max)
    u = inst.u_grid
    bfp = inst.beam_fwhm_pix
    radii = {R: R * bench.theta500 for R in inst.apertures_theta500}

    run.rule(f"{inst.name} — LENSED Monte Carlo: {n_clusters} copies of the "
             f"benchmark, {npix}^2 cutouts at {inst.pix:.1f}\" "
             f"({npix * inst.pix / 60:.1f}' across), model = {mc_model_name}")

    sim = simulator(model, inst, bud, npix)

    def one(i, lens=None, off=0):
        m = sim.make_map(lens=lens, seed=seed + off + 104729 * i)
        a = np.where(np.isfinite(m), m, -np.inf)
        size = max(1, int(round(bfp)))
        ismax = (a == maximum_filter(a, size=size)) & np.isfinite(m)
        cy = (m.shape[0] - 1) / 2.0
        yy, xx = np.indices(m.shape)
        th = np.hypot(yy - cy, xx - cy) * inst.pix / 60.0
        return {R: m[ismax & (th < r)] for R, r in radii.items()}

    t0 = time.time()
    arms = {}
    for arm, lens, off in (("lens", bench.lens, 0),
                           ("ctrl", None, 300_000),
                           ("null", None, 600_000)):
        arms[arm] = build_ensemble(
            run, f"lmc_{inst.name}_{arm}",
            lambda i, lens=lens, off=off: one(i, lens, off), n_clusters)
    run.say(f"  {3 * n_clusters} cutouts built [{time.time() - t0:.0f}s]")

    out = {"u": u, "n_clusters": n_clusters, "npix": npix,
           "model": mc_model_name}
    run.say("")
    run.say(f"{'r/th500':>8s} {'pk/cl lens':>11s} {'pk/cl ctrl':>11s} "
            f"{'yield est':>10s}")
    for R in inst.apertures_theta500:
        _, est = peak_budget(inst, radii[R])
        yl = np.mean([a[R].size for a in arms["lens"]])
        yc = np.mean([a[R].size for a in arms["ctrl"]])
        run.say(f"{R:8.2f} {yl:11.2f} {yc:11.2f} {est:10.2f}")

    for R in inst.apertures_theta500:
        sc = {}
        for arm in ("lens", "ctrl", "null"):
            sc[arm] = run.cached(
                f"lscan_{inst.name}_{arm}_{R}", blocked_scan,
                [a[R] for a in arms[arm]], u, n_boot=n_boot,
                min_exceed=min_exceed, seed=17 + 3 * int(10 * R))
        dsig = paired_delta(sc["lens"], sc["ctrl"])
        dnull = paired_delta(sc["null"], sc["ctrl"])
        out[R] = dict(scan=sc, delta=dsig, null=dnull)

        run.say("")
        run.say(f"  aperture r < {R} theta_500 = {radii[R]:.3f}'")
        run.say(f"    {'u':>9s} {'n_exc':>7s} {'xi_ctrl':>9s} {'xi_lens':>9s} "
                f"{'Dxi_MC':>10s} {'+-':>8s} {'Dxi_NULL':>10s} {'+-':>8s}")
        for j, thr in enumerate(u):
            if not np.isfinite(dsig["delta"][j]):
                run.say(f"    {thr:9.1f} {sc['lens']['n_exc'][j]:7d} "
                        f"{'--':>9s} {'--':>9s} {'(below N_exc floor)':>29s}")
                continue
            run.say(f"    {thr:9.1f} {sc['lens']['n_exc'][j]:7d} "
                    f"{sc['ctrl']['xi'][j]:9.4f} {sc['lens']['xi'][j]:9.4f} "
                    f"{dsig['delta'][j]:+10.5f} {dsig['sigma'][j]:8.5f} "
                    f"{dnull['delta'][j]:+10.5f} {dnull['sigma'][j]:8.5f}")
        with np.errstate(invalid="ignore", divide="ignore"):
            pull = np.abs(dnull["delta"]) / dnull["sigma"]
        ok = np.isfinite(pull)
        if ok.any():
            run.say(f"    NULL check |Dxi|/sigma: max {np.nanmax(pull):.2f}  "
                    + ("PASS" if np.nanmax(pull) < 3.0 else "**FAIL**"))
    return out


# %% Section 13 — run stage: the cluster-count forecast
def stage_forecast(run, inst, theory, mc, bench, mc_model_name="Schechter",
                   n_sigma=3.0):
    """
    Combine the analytic per-cluster signal with the Monte Carlo per-cluster
    noise into the number of benchmark clusters needed for an n_sigma detection
    (audit R8, deliverable 3).  This is the headline comparison across the three
    experiments.
    """
    run.rule(f"{inst.name} — FORECAST: benchmark clusters needed for "
             f"{n_sigma:.0f}-sigma", ch="=")
    run.say(f"  signal  = analytic Delta_xi(u) for ONE cluster "
            f"({mc_model_name} counts)")
    run.say(f"  noise   = MC bootstrap sigma on Delta_xi from "
            f"{mc['n_clusters']} copies, rescaled to one cluster by sqrt(N)")
    run.say(f"  N_infl  = the same with variance inflated by "
            f"{CLUSTERING_INFLATION:.0f}^2, because Poisson simulations "
            f"under-scatter\n            the exceedance counts of a clustered "
            f"sky by ~{CLUSTERING_INFLATION:.0f}x in rms "
            f"(v2 memo Sec. 10.6).  QUOTE N_infl.")
    u = inst.u_grid
    out = {}
    run.say("")
    run.say(f"{'r/th500':>8s} {'u':>9s} {'Dxi_theory':>11s} "
            f"{'sigma_1':>10s} {'N(3sig)':>12s} {'N_infl':>12s}")
    for R in inst.apertures_theta500:
        dth = theory[(R, mc_model_name)]["delta"]
        sg = mc[R]["delta"]["sigma"]
        f = clusters_for_detection(dth, sg, mc["n_clusters"], n_sigma=n_sigma)
        out[R] = f
        for j, thr in enumerate(u):
            if not np.isfinite(f["n_needed"][j]):
                continue
            mark = "  <-- best" if j == f["j_best"] else ""
            run.say(f"{R if j == 0 else '':>8} {thr:9.1f} "
                    f"{dth[j]:+11.5f} {f['sigma_1'][j]:10.4f} "
                    f"{f['n_needed'][j]:12.3e} {f['n_needed_infl'][j]:12.3e}"
                    f"{mark}")
        run.say("")

    # ---- the headline: best aperture ---------------------------------------
    best_R, best_N = None, np.inf
    for R, f in out.items():
        if np.isfinite(f["n_best_infl"]) and f["n_best_infl"] < best_N:
            best_R, best_N = R, f["n_best_infl"]
    if best_R is not None:
        j = out[best_R]["j_best"]
        run.say(f"  BEST CONFIGURATION: r < {best_R} theta_500 "
                f"(= {best_R * bench.theta500:.2f}'), u = {u[j]:.1f} mJy/beam")
        run.say(f"    Delta_xi = {theory[(best_R, mc_model_name)]['delta'][j]:+.5f}"
                f"   sigma_1 = {out[best_R]['sigma_1'][j]:.4f}")
        run.say(f"    N clusters for {n_sigma:.0f}-sigma = "
                f"{out[best_R]['n_best']:.3e} (Poisson MC), "
                f"{best_N:.3e} (clustering-inflated)")
    else:
        run.say("  No threshold cleared the exceedance floor on any aperture "
                "— the benchmark signal is unmeasurable at this resolution.")
    out["best_R"], out["best_N"] = best_R, best_N
    return out


# %% Section 14 — figures for the lensed arm and the forecast
def figure_lensed_mc(run, inst, theory, mc, bench, mc_model_name="Schechter"):
    """
    Monte Carlo Delta_xi(u) against the analytic curve, with the null arm.

    One panel per aperture.  The NULL points (two independent unlensed sets
    differenced) must scatter about zero: they are the pipeline's own systematic
    floor, and any lensed signal has to clear them, not just clear the
    statistical error.
    """
    u = inst.u_grid
    aps = inst.apertures_theta500
    fig, axes = plt.subplots(1, len(aps), figsize=(4.0 * len(aps), 4.3),
                             sharey=True, squeeze=False)
    for ax, R in zip(axes[0], aps):
        d, n = mc[R]["delta"], mc[R]["null"]
        th = theory[(R, mc_model_name)]
        g = np.isfinite(d["delta"])
        ax.errorbar(u[g], d["delta"][g], yerr=d["sigma"][g], fmt="o", ms=4.5,
                    color="C3", capsize=2.5, lw=1.3, label=r"MC $\Delta\xi$")
        gn = np.isfinite(n["delta"])
        ax.errorbar(u[gn] * 1.02, n["delta"][gn], yerr=n["sigma"][gn],
                    fmt="s", ms=3.6, color="0.55", capsize=2.0, lw=1.0,
                    alpha=0.85, label="null (unlensed$-$unlensed)")
        ax.plot(u, th["delta"], "C0", lw=2.0, label="analytic")
        ax.axhline(0.0, color="0.4", lw=1.0)
        ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]",
               title=fr"$r<{R}\,\theta_{{500}}$ = {R * bench.theta500:.2f}$'$, "
                     fr"$\langle\mu\rangle$={th['mean_mu']:.3f}")
        ax.legend(fontsize=7.5)
    axes[0][0].set_ylabel(r"$\Delta\xi(u)$")
    fig.suptitle(f"{inst.name} — lensed Monte Carlo, {mc['n_clusters']} copies "
                 f"of the benchmark cluster ({mc_model_name} counts).  "
                 "The analytic curve is per-cluster and does not scale with N.",
                 fontsize=10.5)
    fig.tight_layout()
    run.finish(fig, f"{run.tag}_lensed_mc")


def figure_forecast(run, inst, fc, bench, n_sigma=3.0, u_floor=None):
    """
    Clusters needed for an n_sigma detection, versus threshold, by aperture.

    The headline figure of the lensed arm: the minimum over this surface is the
    number that answers whether the measurement is worth attempting.
    """
    u = inst.u_grid
    aps = [R for R in inst.apertures_theta500 if R in fc]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    cmap = plt.get_cmap("viridis")
    #  The validity floor, when supplied, is drawn AND the starred optimum is
    #  the FLOORED one.  Starring the unconstrained minimum would advertise a
    #  threshold the analysis explicitly declines to quote (see
    #  simulate_ccat.py Sec. 8b), which is exactly the figure/text
    #  contradiction this argument exists to avoid.
    if u_floor is not None:
        ax.axvspan(u.min() * 0.9, u_floor, color="0.88", zorder=0)
        ax.axvline(u_floor, color="crimson", ls="--", lw=1.4, zorder=1)
    for k, R in enumerate(aps):
        y = fc[R]["n_needed_infl"]
        g = np.isfinite(y)
        if not g.any():
            continue
        c = cmap(k / max(1, len(aps) - 1))
        ax.plot(u[g], y[g], "o-", color=c, lw=1.8, ms=4.5,
                label=fr"$r<{R}\,\theta_{{500}}$ "
                      f"({R * bench.theta500:.2f}')")
        if u_floor is None:
            j = fc[R]["j_best"]
        else:
            masked = np.where((u >= u_floor) & np.isfinite(y), y, np.inf)
            j = int(np.argmin(masked)) if np.isfinite(masked).any() \
                and np.min(masked) < np.inf else -1
        if j >= 0 and np.isfinite(y[j]):
            ax.plot(u[j], y[j], "*", color=c, ms=16, mec="k", mew=0.6)
    ax.set(xscale="log", yscale="log",
           xlabel=r"threshold $u$ [mJy/beam]",
           ylabel=fr"benchmark clusters for {n_sigma:.0f}$\sigma$",
           title=f"{inst.name} — forecast (clustering-inflated).  "
                 fr"$M_{{500}}=10^{{15}}M_\odot$, $z_l=0.5$, "
                 fr"$\theta_{{500}}$={bench.theta500:.2f}$'$")
    ax.grid(alpha=0.3, which="both")
    if u_floor is not None:
        ax.text(u_floor, ax.get_ylim()[1], r" $u=3\sigma_c$ ",
                fontsize=8.5, color="crimson", va="top", ha="left")
        #  placed in the UPPER part of the shaded band, where the curves
        #  never go (N falls toward low u), so it cannot overlap data
        ax.text(u.min() * 0.98, ax.get_ylim()[1] * 0.10,
                "below validity floor:\nno interior optimum,\n"
                "$N$ falls without bound",
                fontsize=8.0, color="0.30", va="top", ha="left")
    ax.legend(fontsize=8.5,
              title=("stars mark the optimum above the floor"
                     if u_floor is not None else "stars mark the optimum"),
              title_fontsize=8)
    fig.tight_layout()
    run.finish(fig, f"{run.tag}_forecast")


if __name__ == "__main__":
    b = Benchmark()
    print(b.summary())
    print("\n  theta[']      mu     kappa")
    for t in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        print(f"  {t:8.2f} {b.mu(t):9.4f} {b.lens.kappa(t):9.4f}")
    print("\n  filled-disc aperture statistics:")
    for R in (0.5, 1.0, 1.5, 2.0, 3.0):
        s = b.disc_stats(R)
        print(f"   r < {R:.1f} theta500 = {s['r_arcmin']:6.3f}'  "
              f"<mu> = {s['mean_mu']:.4f}  <ln mu> = {s['mean_ln_mu']:.4f}")
