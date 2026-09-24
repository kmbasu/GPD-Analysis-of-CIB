# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module M4a-v2 — Unlensed GPD analysis of the Planck 857 GHz CIB P(D)
#
# `planck_unlensed_pd_v2.py` · CIB-lensing / GPD project · Planck analysis
#
# **What this module is.**  The ξ(u) characterization of the *unlensed*
# CIB P(D), measured from random (cluster-avoiding) cutouts of the Lenz,
# Doré & Lagache (2019) 857 GHz CIB map.  It is the control-side
# measurement that stands alone as a paper deliverable and that fixes the
# threshold window inherited by the differential module
# (`planck_lensed_pd_v2.py`).
#
# ---
#
# ## Why there is a v2
#
# v1 (`planck_unlensed_pd.py`, July 2026) reported ξ̂(u) climbing out of
# the Gaussian floor and turning **positive above both Poisson models** at
# k ≈ 3.5–4, read as a detection of tail structure heavier than Poisson
# confusion.  The Herschel analysis
# (`Herschel_analysis/Full_Analysis_Summary_Herschel.md`, §5.8 and §11)
# subsequently showed that the analogous claim there was inflated by
# treating threshold-to-threshold values as independent, and listed six
# items to port back.  This module implements that port **and** adds one
# Planck-specific check that the Herschel list does not contain.
#
# The three ranked suspects for an inflated v1 significance, all now
# tested rather than assumed:
#
# 1. **Threshold covariance.**  Exceedance sets are nested (peaks above
#    u₂ are a subset of those above u₁ < u₂) and all thresholds come from
#    the same bootstrap-resampled cutouts, so ξ̂(uᵢ) is strongly
#    correlated across the ladder.  At Herschel the effective number of
#    independent thresholds was 1.56–2.23 of 21.  Reading "positive at
#    k = 3.5 *and* k = 4 *and* k = 4.5" as corroborating evidence, or
#    combining thresholds in quadrature, then inflates the significance by
#    roughly √n_eff.  §11 below measures ρ(uᵢ,uⱼ), n_eff, and replaces
#    every combined statistic with S² = dᵀC⁻¹d carrying the Hartlap
#    (2007) correction.
#
#    *v1 saved only the bootstrap percentiles, not the replicate matrix,
#    which is why this could not be retro-fitted and the run had to be
#    repeated.  v2 saves the full replicate array.*
#
# 2. **Small-sample GPD MLE bias — the Planck-specific item.**  At k = 4
#    the v1 run had **78 exceedances**; above k ≈ 4.5 it falls below the
#    fitting floor.  The Herschel measurement had 15,264.  In this regime
#    the maximum-likelihood estimator of ξ carries a positive bias of
#    order 1/n and a strongly right-skewed sampling distribution (Hosking
#    & Wallis 1987, *Technometrics* **29**, 339; Smith 1985, *Biometrika*
#    **72**, 67), so a percentile bootstrap band centred on a biased point
#    estimate can manufacture exactly the ξ > 0 excursion v1 reported.
#    §10 calibrates this with a null Monte Carlo run through the identical
#    pipeline, and quotes ξ̂ against the *simulated null distribution*
#    rather than against an analytic curve.
#
# 3. **Peak independence at a 5′ beam on a 1′ grid.**  Declustering uses a
#    `maximum_filter` window of exactly one FWHM = 5 px, but the map
#    autocorrelation of a 5′ Gaussian beam has FWHM √2 × 5′ = 7.07′, so
#    retained peaks separated by 5′ still carry ρ ≈ 0.25 in amplitude.
#    §7b scans the declustering window over 5/7/10 px and reports the
#    induced shift in ξ̂ as a systematic.
#
# **What we already know is *not* the problem:** the v1 cutout-level
# bootstrap band is only 1.0–1.4× the iid-peak band across the ladder, so
# intra-cutout correlation was being handled and the *single-threshold*
# error bars were roughly honest.  The inflation, if any, is in the
# combination across thresholds and in the model-comparison statistics.
#
# ---
#
# ## Other changes from v1
#
# * **Mask variant `4.0e+20_gp40`** (was `2.5e+20_gp20`).  f_sky = 0.338 =
#   13,955 deg², nearly double v1's 0.184 = 7,599 deg².  The looser N_HI
#   cut buys sky at the cost of cirrus; the non-Poisson Gaussian budget is
#   measured, not assumed, so this is self-correcting in the model
#   comparison but must be quoted.
# * **Three count models** — Schechter, SPL, DPL — replacing v1's single
#   Guerrero fiducial, using the final adopted 350 µm fits of
#   `num_count_fits/fitting_results_number_counts.md` §7 (GOODS-N points
#   excluded, N = 9 points over 6–94.6 mJy).  857 GHz *is* 350 µm, so
#   these transfer to Planck without recolouring.  This matches the
#   Herschel treatment exactly, so the two chapters are like-for-like.
# * **CIB monopole check** (§4b) — a constraint Planck can impose and
#   Herschel structurally cannot.  The counts models are extrapolated
#   below the fitted range down to s_min; requiring the predicted
#   background not to exceed the FIRAS value (Fixsen et al. 1998) is an
#   independent test of that extrapolation.
# * **Simulated, not analytic, null curves** (Herschel port item 3):
#   `xi_of_threshold_analytic` computes a *pixel-level* ξ, while we
#   measure a *declustered-peak* ξ.  H1c measured |Δξ| ≈ 0.047 between the
#   two.  §10 produces peak-level null curves from simulation.
# * **Bright-source census** (port item 4) — observed N(>S) against the
#   model prediction near the point-source-mask limit.
# * `noise_mode="beam"` everywhere in simulation (port item 6).
#
# ---
#
# ## The physical headline
#
# Independently of any statistics, one number states the case.  With the
# adopted Schechter counts truncated at S_cut = 100 mJy, the Planck 5′
# beam gives σ_c = 95 mJy/beam, so
#
#     S_cut / σ_c = 1.05   (Planck, 300")      vs   12.6   (Herschel, 25")
#
# The brightest source the intrinsic counts allow cannot deposit even one
# σ of confusion in a Planck beam.  A Poisson power-law shoulder can only
# rise above the Gaussian core if single sources deposit several σ_c, so
# at Planck resolution ξ_pop(u) < 0 everywhere accessible and there is no
# positive-ξ regime to detect.  Everything below is the quantitative
# demonstration of that statement; a null result here is a *measurement*
# of beam dilution, not a failure of the method.
#
# ---
#
# ## Usage
#
# Run top-to-bottom, or cell-by-cell in Spyder (`#%%` separators).
# Expensive stages are memoised under an md5 of the configuration in
# `results/v2_cache/`, so a run resumes where it stopped and any parameter
# change invalidates the affected stage automatically.
#
# | env switch | default | effect |
# |---|---|---|
# | `CIB_DATA_DIR` | `<here>/data_857` | data location |
# | `CIB_MASK_VARIANT` | `4.0e+20_gp40` | mask variant |
# | `CIB_N_CUTOUTS` | `1600` | control-sample size |
# | `CIB_N_BOOT` | `400` | cutout-level bootstrap replicates |
# | `CIB_N_NULL_MAPS` | `24` | null-MC simulated fields |
# | `CIB_N_NULL_DRAW` | `2000` | null-MC resampling draws |
# | `CIB_SMIN_MJY` | `1.0` | counts faint-end truncation |
# | `CIB_SCUT_MJY` | `100.0` | counts bright-end truncation |
# | `CIB_SSPLIT_MJY` | `10.0` | sim: explicit-source / Gaussian split |
# | `CIB_USE_CACHE` | `1` | set 0 to force recomputation |
# | `CIB_SAVE_FIGURES` | `0` | 1 writes PNGs to `CIB_FIGURE_DIR` |
# | `CIB_QUICK_TEST` | `0` | fast smoke run |
#
# Dependencies: numpy, scipy, matplotlib, astropy, **healpy**, plus the
# project's `analysis_modules` (`counts`, `pofd_analytic`, `gpd_tail`,
# `simulate_maps`).
#
# **N_BOOT must exceed the number of usable thresholds by a comfortable
# margin** — the Hartlap factor is (N − p − 2)/(N − 1) and the covariance
# inverse is singular for N ≤ p + 2.  With 15 thresholds and N = 400 the
# factor is 0.955.

# %% [markdown]
# ## 1. Imports and paths

# %%
import os
import sys
import time
import json
import pickle
import hashlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u_ap

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()
_MODULE_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..",
                                           "analysis_modules"))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from counts import BaseCounts, Schechter, DoublePowerLaw       # noqa: E402
from pofd_analytic import PofD                                 # noqa: E402
from gpd_tail import fit_gpd, xi_of_threshold_analytic         # noqa: E402
from simulate_maps import CIBMapSimulator                      # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# %% [markdown]
# ## 2. Configuration
#
# Note the two flux truncations, which do different jobs.  `S_CUT_MJY` is
# a statement about the *counts model*: above ~100 mJy the observed 350 µm
# counts are lensing-dominated (Lima et al. 2010) and no longer trace the
# intrinsic dN/dS, and the pipeline supplies its own lensing, so the
# models are truncated there.  `S_MIN_MJY` is a *modelling choice* forced
# on us by the SPL: with β = −3.279 the second moment ∫S²dN/dS diverges at
# the faint end, so σ_c and the P(D) core width depend on where the
# integral is cut.  We adopt 1 mJy for all three models, matching the
# Herschel convention (`CIB_SMIN_MJY`), and quantify the consequence in
# §4b rather than hiding it: only the *tail* slope, which is s_min
# independent and is what ξ(u) reads at high threshold, should be treated
# as a model statement.

# %%
# ---- data location ----------------------------------------------------------
DATA_DIR = os.environ.get("CIB_DATA_DIR",
                          os.path.join(_BASE_DIR, "data_857"))
MASK_VARIANT = os.environ.get("CIB_MASK_VARIANT", "4.0e+20_gp40")
CATALOG_FILE = os.path.join(_BASE_DIR, "Planck_772_cluster_sample.txt")

# ---- beam / units -----------------------------------------------------------
BEAM_FWHM_ARCSEC = 300.0
OM_BEAM_ARCMIN2 = 1.133 * (BEAM_FWHM_ARCSEC / 60.0) ** 2
OM_BEAM_SR = OM_BEAM_ARCMIN2 * (np.pi / (180.0 * 60.0)) ** 2
MJY_SR_TO_MJY_BEAM = OM_BEAM_SR * 1e9          # 1 MJy/sr -> mJy/beam (~2397)

# ---- cutout geometry --------------------------------------------------------
NPIX_CUT = 64                    # cutout side [pixels]
PIX_ARCMIN = 1.0                 # native 1.72'; mild oversampling so the
                                 # beam FWHM spans exactly 5 cutout pixels
CUT_SIZE_ARCMIN = NPIX_CUT * PIX_ARCMIN
BEAM_FWHM_PIX = BEAM_FWHM_ARCSEC / 60.0 / PIX_ARCMIN     # = 5 pixels

# ---- control-sample construction --------------------------------------------
N_CUTOUTS = int(os.environ.get("CIB_N_CUTOUTS", "1600"))
MIN_VALID_FRAC = 0.97
CLUSTER_AVOID_DEG = 2.0
MIN_SEP_ARCMIN = CUT_SIZE_ARCMIN
RNG_SEED = 20260802
MAX_DRAWS = 500_000

# ---- GPD analysis -----------------------------------------------------------
K_GRID = np.arange(1.0, 8.01, 0.5)   # thresholds u = mu_core + k sigma_core
N_BOOT = int(os.environ.get("CIB_N_BOOT", "400"))
MIN_EXCEED = 40

# ---- declustering-window systematic (v2, suspect 3) -------------------------
DECLUSTER_WINDOWS_PIX = [5, 7, 10]   # 1.0, 1.41 (autocorr FWHM), 2.0 FWHM
DECLUSTER_PRIMARY = 5

# ---- count models: final adopted 350 um fits --------------------------------
#  num_count_fits/fitting_results_number_counts.md, Sec. 7
#  ("Final adopted fits", GOODS-N excluded, N = 9 points over 6-94.6 mJy).
#  These SUPERSEDE the v1 fiducial (alpha = -1.3, S* = 14.5, n* = 8902),
#  which came from the un-revised Guerrero memo.
SCH_PARS = dict(alpha=-1.890, sstar=19.01, nstar=7014.0,
                d_alpha=0.186, d_sstar=2.76, d_nstar=2156.0)
SPL_PARS = dict(N0=1.053e5, S0=2.2, beta=-3.279)
DPL_PARS = dict(alpha=1.082, beta=3.928, sstar=11.70, phistar=1.33e4)

S_MIN_MJY = float(os.environ.get("CIB_SMIN_MJY", "1.0"))
S_CUT_MJY = float(os.environ.get("CIB_SCUT_MJY", "100.0"))
D_SPAN_MJY, N_FFT, SF_FLOOR = 6000.0, 2 ** 17, 1e-9

# Planck point-source mask limit at 857 GHz (S/N > 5).  Used only for the
# bright-source census and the truncation argument, never as a model cut.
S_PSMASK_MJY = 650.0

# ---- null Monte Carlo -------------------------------------------------------
N_NULL_MAPS = int(os.environ.get("CIB_N_NULL_MAPS", "24"))
NULL_MAP_NPIX = 512              # 512' = 8.53 deg on a side
N_NULL_DRAW = int(os.environ.get("CIB_N_NULL_DRAW", "2000"))
#  Sources fainter than S_SPLIT are absorbed into the Gaussian component
#  rather than injected one by one.  At the Planck beam the counts give
#  ~680 sources/beam above 1 mJy but only ~4 above 10 mJy, so the faint
#  population is Gaussian to excellent accuracy by the CLT and injecting
#  it explicitly would cost ~100x the runtime for no change in the tail.
#  Section 10 verifies that the split reproduces the total sigma_c.
S_SPLIT_MJY = float(os.environ.get("CIB_SSPLIT_MJY", "10.0"))

# ---- quick-test / output ----------------------------------------------------
QUICK_TEST = bool(int(os.environ.get("CIB_QUICK_TEST", "0")))
if QUICK_TEST:
    N_CUTOUTS, N_BOOT = 120, 80
    N_NULL_MAPS, N_NULL_DRAW, NULL_MAP_NPIX = 4, 300, 256
    print("QUICK_TEST on: reduced cutout / bootstrap / null-MC counts.")

SAVE_FIGURES = bool(int(os.environ.get("CIB_SAVE_FIGURES", "0")))
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", RESULTS_DIR)
CACHE_DIR = os.path.join(RESULTS_DIR, "v2_cache")
USE_CACHE = os.environ.get("CIB_USE_CACHE", "1") == "1"
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG_HASH = hashlib.md5(json.dumps(
    dict(var=MASK_VARIANT, nc=N_CUTOUTS, npix=NPIX_CUT, pix=PIX_ARCMIN,
         seed=RNG_SEED, smin=S_MIN_MJY, scut=S_CUT_MJY,
         ssplit=S_SPLIT_MJY, nb=N_BOOT, quick=QUICK_TEST,
         # anything else a cached stage depends on, or the cache lies
         kmin=float(K_GRID[0]), kmax=float(K_GRID[-1]),
         nk=int(K_GRID.size), me=MIN_EXCEED,
         dw=list(DECLUSTER_WINDOWS_PIX), dp=DECLUSTER_PRIMARY,
         nnm=N_NULL_MAPS, nnd=N_NULL_DRAW, nnp=NULL_MAP_NPIX),
    sort_keys=True).encode()).hexdigest()[:10]

T0 = time.time()


def cached(tag, fn):
    """Memoise an expensive stage to results/v2_cache/, keyed on CFG_HASH.

    The Cowork sandbox caps a single shell call at 45 s, so long stages must
    be resumable; on a normal machine this simply makes re-runs instant.
    Any configuration change alters CFG_HASH and invalidates the entry.
    """
    key = "%s_%s.pkl" % (tag, CFG_HASH)
    path = os.path.join(CACHE_DIR, key)
    if USE_CACHE and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                val = pickle.load(fh)
            print("    [cache hit] %s" % key)
            return val
        except Exception as exc:                       # noqa: BLE001
            #  an entry left half-written by an interrupted run is a miss,
            #  not a fatal error -- otherwise one killed process poisons
            #  the cache permanently
            print("    [cache CORRUPT, recomputing] %s (%s)" % (key, exc))
    val = fn()
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        pickle.dump(val, fh)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.replace(tmp, path)
    except OSError:                                    # rename not allowed
        with open(path, "wb") as fh:
            pickle.dump(val, fh)
    print("    [cached] %s  (t = %.0f s)" % (key, time.time() - T0))
    return val


CHUNK = int(os.environ.get("CIB_CHUNK", "50"))


def cached_chunks(tag, n_total, fn, chunk=None):
    """Memoise a long replicate loop in BLOCKS, so it can resume.

    `fn(i0, i1)` must return an (i1 - i0, m) array; each block is cached
    separately and the blocks are stacked.  A run interrupted part-way
    (or a shell that caps call duration, as the Cowork sandbox does at
    45 s) resumes at the first missing block instead of restarting the
    whole loop.  Each block seeds its own RNG from i0, so the result is
    reproducible and independent of how the work was divided.
    """
    chunk = chunk or CHUNK
    out = []
    for i0 in range(0, n_total, chunk):
        i1 = min(i0 + chunk, n_total)
        out.append(cached("%s_%06d" % (tag, i0),
                          lambda i0=i0, i1=i1: fn(i0, i1)))
    return np.vstack(out) if out else np.empty((0, 0))


def _finish_figure(fig, name):
    """Save <FIGURE_DIR>/<name>.png if CIB_SAVE_FIGURES=1, else show."""
    if SAVE_FIGURES:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        path = os.path.join(FIGURE_DIR, name + ".png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"Wrote {path}")
    else:
        plt.show()


class SinglePowerLaw(BaseCounts):
    """Single power law, dN/dS = N0 (S/S0)^beta, in mJy^-1 deg^-2.

    Patanchon et al. (2009) form with S0 fixed at their 2.2 mJy, refit to
    Bethermin+2012 in the project's counts memo (Sec. 7).

    NOTE on s_min.  With beta = -3.279 the second moment int S^2 dN/dS dS
    diverges at the faint end (|beta| > 3), so both the confusion noise and
    the P(D) core width depend on where the integral is cut off.  The s_min
    passed here is a modelling CHOICE, not a property of the model.  Only the
    tail slope -- s_min-independent, and what xi(u) reads at high threshold --
    should be treated as a model statement.  This matters far more at Planck
    than at Herschel: the beam solid angle is 144x larger, so the faint
    population that the integral is sensitive to is 144x more numerous.
    """

    def __init__(self, N0, S0, beta, s_min=1.0, s_max=1e4):
        self.N0, self.S0, self.beta = float(N0), float(S0), float(beta)
        self.s_min, self.s_max = float(s_min), float(s_max)

    def dnds(self, S):
        S = np.asarray(S, float)
        out = self.N0 * (S / self.S0) ** self.beta
        return np.where(S > 0, out, 0.0)


def build_models(s_min=S_MIN_MJY, s_max=S_CUT_MJY):
    """The three adopted 350 um count models on a common flux range."""
    return {
        "Schechter": Schechter(alpha=SCH_PARS["alpha"],
                               log_sstar=np.log10(SCH_PARS["sstar"]),
                               log_phistar=np.log10(SCH_PARS["nstar"]),
                               s_min=s_min, s_max=s_max),
        "SPL": SinglePowerLaw(SPL_PARS["N0"], SPL_PARS["S0"],
                              SPL_PARS["beta"], s_min=s_min, s_max=s_max),
        "DPL": DoublePowerLaw(alpha=DPL_PARS["alpha"], beta=DPL_PARS["beta"],
                              log_sstar=np.log10(DPL_PARS["sstar"]),
                              log_phistar=np.log10(DPL_PARS["phistar"]),
                              s_min=s_min, s_max=s_max),
    }


MODELS = build_models()
MODEL_NAMES = list(MODELS)
PRIMARY_MODEL = "Schechter"          # for the lensed module and the null MC


def robust_core(x):
    """(median, MAD-sigma) of the finite entries of x."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


# %% [markdown]
# ## 3. Load maps; instrument noise from the half-ring difference
#
# The Lenz products store masked pixels as NaN; the boolean mask is 0/1.
# The half-difference (odd−even)/2 cancels the sky exactly and leaves a
# pure realization of the instrument noise.  All statistics are quoted in
# mJy/beam; the conversion touches only labels, not ξ.

# %%
D = os.path.join(DATA_DIR, MASK_VARIANT)
t0 = time.time()
cib = hp.read_map(os.path.join(D, "cib_fullmission.hpx.fits"),
                  field=0, dtype=np.float32)
mask = hp.read_map(os.path.join(D, "mask_bool.hpx.fits"),
                   dtype=np.float32) > 0.5
odd = hp.read_map(os.path.join(D, "cib_oddring.hpx.fits"),
                  field=0, dtype=np.float32)
even = hp.read_map(os.path.join(D, "cib_evenring.hpx.fits"),
                   field=0, dtype=np.float32)
NSIDE = hp.get_nside(cib)
noise_map = 0.5 * (odd - even)
del odd, even

cib_mjyb = cib * MJY_SR_TO_MJY_BEAM
noise_mjyb = noise_map * MJY_SR_TO_MJY_BEAM

v_map = cib_mjyb[mask]
v_noise = noise_mjyb[mask]
sky_frac = float(mask.mean())

mu_n, sigma_n = robust_core(v_noise)

print(f"Loaded 857 GHz / {MASK_VARIANT} in {time.time() - t0:.1f}s "
      f"(Nside {NSIDE}, f_sky {sky_frac:.4f} = {sky_frac * 41253:.0f} deg^2)")
print(f"  map    (masked): mean = {v_map.mean():8.1f}, "
      f"rms = {v_map.std():7.1f} mJy/beam")
print(f"  noise (odd-even)/2: sigma_N = {sigma_n:.1f} mJy/beam (MAD; "
      f"plain std {v_noise.std():.1f}) = "
      f"{sigma_n / MJY_SR_TO_MJY_BEAM:.4f} MJy/sr")

# %% [markdown]
# ## 4. Confusion noise and the beam-dilution statement
#
# σ_c for each of the three adopted models at the Planck beam, and the
# single number that frames the whole chapter:
#
#     S_cut / σ_c  ~ 1  at Planck (300")     vs   12.6 at Herschel (25")
#
# The brightest source the intrinsic counts allow cannot deposit even one
# σ of confusion in a Planck beam.  A positive-ξ Poisson shoulder requires
# S_bright/σ_c of at least a few, so it is *structurally* inaccessible
# here — this is the beam-dilution result, and it is independent of every
# statistical subtlety that follows.

# %%
print("\n" + "=" * 79)
print("  CONFUSION NOISE AT THE PLANCK BEAM  "
      f"(s_min = {S_MIN_MJY} mJy, s_cut = {S_CUT_MJY} mJy)")
print("=" * 79)
print("  %-11s %12s %14s %14s %12s"
      % ("model", "sigma_c", "sources/beam", "mean [mJy/bm]", "S_cut/sig_c"))
CONF = {}
for nm, c in MODELS.items():
    st = c.confusion_stats(BEAM_FWHM_ARCSEC)
    CONF[nm] = st
    print("  %-11s %12.2f %14.1f %14.1f %12.2f"
          % (nm, st["sigma_conf_mJy_per_beam"], st["sources_per_beam"],
             st["mean_mJy_per_beam"],
             S_CUT_MJY / st["sigma_conf_mJy_per_beam"]))
SIGMA_C = {nm: CONF[nm]["sigma_conf_mJy_per_beam"] for nm in MODELS}
SIGMA_C_PRIMARY = SIGMA_C[PRIMARY_MODEL]

_st25 = build_models()[PRIMARY_MODEL].confusion_stats(25.0)
print("\n  For contrast, the same Schechter model at the Herschel/SPIRE "
      "350 um beam (25\"):")
print("    sigma_c = %.2f mJy/beam, sources/beam = %.2f, S_cut/sigma_c = %.1f"
      % (_st25["sigma_conf_mJy_per_beam"], _st25["sources_per_beam"],
         S_CUT_MJY / _st25["sigma_conf_mJy_per_beam"]))
print("""
  THE BEAM-DILUTION STATEMENT.  Planck's 5' beam contains ~10^2-10^3
  sources above 1 mJy, so the central limit theorem has all but
  Gaussianised the confusion P(D); the brightest allowed intrinsic source
  raises a single beam by ~1 sigma_c.  Herschel's 25' beam sees the same
  counts with 12.6 sigma of dynamic range.  No amount of sky area or
  statistical care recovers a Poisson shoulder that the beam has averaged
  away -- which is why the honest Planck deliverable is a null with a
  measured dilution factor, not a detection.""")

# %% [markdown]
# ## 4b. The CIB monopole check — a constraint only Planck can impose
#
# All three models are fitted over 6–94.6 mJy and then extrapolated down
# to s_min = 1 mJy.  Nothing in the Herschel analysis tests that
# extrapolation, because a differential measurement is insensitive to the
# mean.  Planck measures an absolute background, so the predicted CIB
# monopole
#
#     I_nu = Omega_eff^-1 * D_bar  =  int S dN/dS dS
#
# can be compared with the FIRAS determination (Fixsen et al. 1998, *ApJ*
# **508**, 123): I_ν = 1.3e-5 (ν/ν₀)^0.64 B_ν(18.5 K) with ν₀ = 3000 GHz,
# evaluated at 857 GHz below.  The ~30% uncertainty on the FIRAS amplitude
# is the tolerance.

# %%
_H, _KB, _C = 6.62607015e-34, 1.380649e-23, 2.99792458e8


def firas_cib_mjy_sr(nu_ghz=857.0, amp=1.3e-5, index=0.64, t_dust=18.5):
    """Fixsen et al. (1998) FIRAS CIB spectrum, MJy/sr."""
    nu = nu_ghz * 1e9
    b_nu = (2 * _H * nu ** 3 / _C ** 2) / np.expm1(_H * nu / (_KB * t_dust))
    return amp * (nu_ghz / 3000.0) ** index * b_nu * 1e20


FIRAS_857 = firas_cib_mjy_sr()
print("\n" + "=" * 79)
print("  CIB MONOPOLE CHECK  (extrapolation of the counts below the fit "
      "range)")
print("=" * 79)
print(f"  FIRAS (Fixsen+1998) at 857 GHz: {FIRAS_857:.3f} MJy/sr "
      f"(+-30% on the amplitude)")
print("  %-11s %14s %14s %10s" % ("model", "I_nu [MJy/sr]", "vs FIRAS",
                                  "verdict"))
MONO = {}
for nm, c in MODELS.items():
    i_nu = CONF[nm]["mean_mJy_per_beam"] / MJY_SR_TO_MJY_BEAM
    MONO[nm] = i_nu
    ratio = i_nu / FIRAS_857
    verdict = ("OK" if 0.5 < ratio < 1.6 else
               "HIGH" if ratio >= 1.6 else "LOW")
    print("  %-11s %14.3f %13.2fx %10s" % (nm, i_nu, ratio, verdict))
print("""
  READING.  The SPL over-produces the 857 GHz background by a large factor
  once extrapolated to 1 mJy -- it is excluded as a GLOBAL count model by
  the background alone, independently of anything xi(u) says.  This does
  not invalidate it as a description of the 6-95 mJy counts, and it is
  retained in the comparison below for exactly that reason: the xi(u)
  comparison probes the model near the threshold flux, not at 1 mJy.  But
  it means the SPL's sigma_c and P(D) core width should not be quoted as
  model predictions -- they are s_min-dependent (see the SinglePowerLaw
  docstring), and s_min is now independently constrained.  This is a
  Planck-only result: Herschel's differential measurement cannot see it.""")

# %% [markdown]
# ## 5. The P(D) + noise histogram (Nguyen et al. 2010, Fig. 3 style)
#
# Black: pixel histogram of the (mean-subtracted) masked CIB map.
# Dashed: the same for the half-ring noise map.  Red: Gaussian fit to the
# noise.  Everything between the red curve and the black histogram is sky
# — confusion (Poisson + clustered) plus residual cirrus.  The grey dotted
# curve marks the Poisson-confusion Gaussian core (σ_c ⊕ σ_N) of the
# primary model; the bright-side excess over *that* is what the GPD tail
# analysis feeds on.

# %%
v0 = v_map - v_map.mean()
lo, hi = -1200.0, 1800.0
bins = np.linspace(lo, hi, 121)
bw = bins[1] - bins[0]
cen = 0.5 * (bins[1:] + bins[:-1])

h_map, _ = np.histogram(v0, bins=bins)
h_noise, _ = np.histogram(v_noise - mu_n, bins=bins)
gauss_n = (v_noise.size * bw / (np.sqrt(2 * np.pi) * sigma_n)
           * np.exp(-0.5 * (cen / sigma_n) ** 2))
sig_core_ref = np.hypot(SIGMA_C_PRIMARY, sigma_n)
gauss_c = (v0.size * bw / (np.sqrt(2 * np.pi) * sig_core_ref)
           * np.exp(-0.5 * (cen / sig_core_ref) ** 2))

fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.step(cen, h_map, where="mid", color="k", lw=1.4,
        label="857 GHz CIB map (masked, mean-subtracted)")
ax.step(cen, h_noise, where="mid", color="k", lw=1.0, ls="--",
        label="instrument noise: (odd$-$even)/2")
ax.plot(cen, gauss_n, "r-", lw=1.4,
        label=fr"Gaussian fit, $\sigma_N$ = {sigma_n:.0f} mJy/beam")
ax.plot(cen, gauss_c, color="0.6", ls=":", lw=1.6,
        label=fr"Poisson confusion $\oplus$ noise "
              fr"({sig_core_ref:.0f} mJy/beam)")
ax.set_yscale("log")
ax.set(xlim=(lo, hi), ylim=(0.8, 2.5 * h_map.max()),
       xlabel="pixel intensity  [mJy/beam]", ylabel=r"$N_{\rm pix}$",
       title=f"Planck 857 GHz CIB P(D) and noise  ({MASK_VARIANT}, "
             f"{sky_frac * 41253:.0f} deg$^2$)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
_finish_figure(fig, "planck_v2_unlensed_histogram")

# %% [markdown]
# ## 6. Random control cutouts
#
# Gnomonic tangent-plane projection without `healpy.gnomview` (adapted
# from `example_codes/extract_planck_cutouts.py`; bilinear interpolation
# via `hp.get_interp_val`, NaN-masked pixels propagate).  Acceptance:
# centre inside the mask; > CLUSTER_AVOID_DEG from all 772 catalog
# clusters; ≥ MIN_SEP_ARCMIN from previously accepted centres; valid-pixel
# fraction ≥ MIN_VALID_FRAC.  Accepted cutouts are DC+gradient baselined
# — its cirrus-suppression job, quantified below by the drop from the
# full-map rms to σ_core.

# %%
def gnomonic_cutout(skymap, l0_deg, b0_deg, n_pix, pix_arcmin):
    """Matplotlib-free gnomonic cutout of a RING HEALPix map centred on
    Galactic (l0, b0).  Inverse gnomonic deprojection with bilinear
    interpolation; xi West-positive, eta North-positive (healpy.gnomview
    orientation).  NaNs propagate, so masked regions stay masked."""
    reso = np.radians(pix_arcmin / 60.0)
    half = (n_pix - 1) / 2.0
    off = (np.arange(n_pix) - half) * reso
    XI, ETA = np.meshgrid(-off, off[::-1])
    RHO = np.hypot(XI, ETA)
    RHO_s = np.where(RHO == 0.0, 1e-30, RHO)
    C = np.arctan(RHO)
    sin_c, cos_c = np.sin(C), np.cos(C)
    b0, l0 = np.radians(b0_deg), np.radians(l0_deg)
    b_out = np.arcsin(np.clip(cos_c * np.sin(b0)
                              + ETA * sin_c * np.cos(b0) / RHO_s, -1, 1))
    l_out = l0 + np.arctan2(XI * sin_c,
                            RHO_s * np.cos(b0) * cos_c
                            - ETA * np.sin(b0) * sin_c)
    vals = hp.get_interp_val(skymap, np.pi / 2 - b_out.ravel(),
                             l_out.ravel() % (2 * np.pi), nest=False)
    return vals.reshape(n_pix, n_pix)


def fit_remove_plane(img):
    """DC + linear plane fit on finite pixels; NaNs preserved."""
    n = img.shape[0]
    xn = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    XN, YN = np.meshgrid(xn, xn)
    valid = np.isfinite(img)
    if valid.sum() < 10:
        return img
    A = np.column_stack([np.ones(valid.sum()), XN[valid], YN[valid]])
    coef, *_ = np.linalg.lstsq(A, img[valid], rcond=None)
    return img - (coef[0] + coef[1] * XN + coef[2] * YN)


def declustered_peaks(img, window_pix):
    """Local maxima over `window_pix`, NaN-aware (masked pixels are -inf
    for the filter and excluded from the output)."""
    filled = np.where(np.isfinite(img), img, -np.inf)
    size = max(1, int(round(window_pix)))
    is_max = (filled == maximum_filter(filled, size=size))
    is_max &= np.isfinite(img)
    return img[is_max]


cat = np.genfromtxt(CATALOG_FILE, comments="#",
                    dtype=[("name", "U24"), ("ra", "f8"), ("dec", "f8"),
                           ("z", "f8"), ("r500", "f8"), ("tx", "f8")])
gal = SkyCoord(ra=cat["ra"] * u_ap.deg, dec=cat["dec"] * u_ap.deg,
               frame="icrs").galactic
clus_vec = hp.ang2vec(gal.l.deg, gal.b.deg, lonlat=True)     # (772, 3)
COS_AVOID = np.cos(np.radians(CLUSTER_AVOID_DEG))
COS_MINSEP = np.cos(np.radians(MIN_SEP_ARCMIN / 60.0))


def _extract_cutouts():
    rng = np.random.default_rng(RNG_SEED)
    stack, cen_l, cen_b, acc_vecs = [], [], [], []
    n_drawn = n_rej_mask = n_rej_clus = n_rej_sep = n_rej_frac = 0
    while len(stack) < N_CUTOUTS and n_drawn < MAX_DRAWS:
        n_drawn += 1
        vec = hp.ang2vec(*hp.pix2ang(
            NSIDE, rng.integers(hp.nside2npix(NSIDE))))
        if not mask[hp.vec2pix(NSIDE, *vec)]:
            n_rej_mask += 1
            continue
        if np.max(clus_vec @ vec) > COS_AVOID:
            n_rej_clus += 1
            continue
        if acc_vecs and np.max(np.array(acc_vecs) @ vec) > COS_MINSEP:
            n_rej_sep += 1
            continue
        lon, lat = (a.item() for a in hp.vec2ang(vec, lonlat=True))
        img = gnomonic_cutout(cib_mjyb, lon, lat, NPIX_CUT, PIX_ARCMIN)
        if np.isfinite(img).mean() < MIN_VALID_FRAC:
            n_rej_frac += 1
            continue
        stack.append(fit_remove_plane(img).astype(np.float32))
        cen_l.append(lon)
        cen_b.append(lat)
        acc_vecs.append(vec)
    print(f"    {len(stack)} cutouts from {n_drawn} draws "
          f"({n_rej_mask} off-mask, {n_rej_clus} near-cluster, "
          f"{n_rej_sep} too-close, {n_rej_frac} low-valid-frac)")
    return (np.array(stack), np.array(cen_l), np.array(cen_b))


stack, cen_l, cen_b = cached("cutouts", _extract_cutouts)

n_cut = stack.shape[0]
area_total_deg2 = n_cut * (CUT_SIZE_ARCMIN / 60.0) ** 2
n_beams_total = area_total_deg2 * 3600.0 / OM_BEAM_ARCMIN2
print(f"Control sample: {n_cut} cutouts x ({CUT_SIZE_ARCMIN:.0f}')^2 = "
      f"{area_total_deg2:.0f} deg^2 = {n_beams_total:.0f} beams")

# %% [markdown]
# ## 7. Pooled pixel statistics, core fit, declustered peaks
#
# Core parameters are estimated robustly (median / MAD — the heavy tail
# biases plain moments) from the pooled baselined pixels.  σ_core is
# *smaller* than the full-map rms of §3 because the per-cutout DC+gradient
# removal has taken out large-scale cirrus and clustering modes.
# Thresholds are anchored to this post-baseline core (manual §7b).

# %%
pix_pool = stack[np.isfinite(stack)]
mu_core, sigma_core = robust_core(pix_pool)
neg = pix_pool[pix_pool < mu_core]
sigma_neg = float(np.sqrt(np.mean((neg - mu_core) ** 2)))
print(f"pooled pixels: {pix_pool.size} | core: mu = {mu_core:.1f}, "
      f"sigma_MAD = {sigma_core:.1f}, sigma_negside = {sigma_neg:.1f} "
      f"mJy/beam (plain std {pix_pool.std():.1f})")
print(f"  vs full-map rms {v_map.std():.1f} -> baselining removed "
      f"{np.sqrt(max(v_map.std()**2 - pix_pool.std()**2, 0)):.1f} mJy/beam "
      f"of large-scale power")

#  The non-Poisson Gaussian budget: clustered CIB + residual cirrus, i.e.
#  everything in the core width that Poisson confusion does not explain.
SIGMA_EXTRA = {nm: float(np.sqrt(max(sigma_core ** 2 - SIGMA_C[nm] ** 2, 0.0)))
               for nm in MODELS}
print("  non-Poisson Gaussian budget sqrt(sigma_core^2 - sigma_c^2):")
for nm in MODEL_NAMES:
    flag = ""
    if SIGMA_C[nm] >= sigma_core:
        flag = ("   <-- WARNING: this model's Poisson confusion ALONE "
                "exceeds the\n                    measured core width, so "
                "it over-predicts the P(D) core.\n                    Its "
                "Gaussian budget is clipped to zero and its xi(u) curve\n"
                "                    should be read as an upper envelope, "
                "not a fit.")
    print("    %-11s %7.1f mJy/beam%s" % (nm, SIGMA_EXTRA[nm], flag))

peaks_by_cut = [declustered_peaks(img, DECLUSTER_PRIMARY) for img in stack]
peaks_pool = np.concatenate(peaks_by_cut)
print(f"declustered peaks (window {DECLUSTER_PRIMARY} px): "
      f"{peaks_pool.size} total ({peaks_pool.size / n_cut:.1f}/cutout; "
      f"{peaks_pool.size / n_beams_total:.2f}/beam)")

U_GRID = mu_core + K_GRID * sigma_core

# %% [markdown]
# ## 7b. Declustering-window systematic (suspect 3)
#
# The primary declustering window is one beam FWHM = 5 px.  But the map's
# autocorrelation is the beam convolved with itself, a Gaussian of FWHM
# √2 × 5′ = 7.07′, so two retained peaks 5′ apart still share ρ ≈ 0.25 in
# amplitude — they are not independent draws, which is the literal "pixel
# covariance at the 5′ beam" concern.  Under-declustering inflates the
# apparent number of exceedances and can bias ξ̂; over-declustering throws
# away real sources.  The scan below reports the induced shift in ξ̂(u) as
# a systematic band rather than adopting one window on faith.

# %%
def _decluster_scan():
    out = {}
    for w in DECLUSTER_WINDOWS_PIX:
        pbc = [declustered_peaks(img, w) for img in stack]
        pool = np.concatenate(pbc)
        xi_w = np.full(U_GRID.size, np.nan)
        n_w = np.zeros(U_GRID.size, int)
        for j, u in enumerate(U_GRID):
            y = pool[pool > u] - u
            n_w[j] = y.size
            if y.size >= MIN_EXCEED:
                xi_w[j] = fit_gpd(y)["xi"]
        out[w] = dict(n_peaks=pool.size, xi=xi_w, n_exc=n_w,
                      per_beam=pool.size / n_beams_total)
    return out


DECL = cached("decluster_scan", _decluster_scan)

print("\n  DECLUSTERING-WINDOW SCAN")
print("  window  peaks/beam  " + "  ".join("k=%.1f" % k for k in K_GRID[:8]))
for w in DECLUSTER_WINDOWS_PIX:
    row = "  ".join(("%+.3f" % v) if np.isfinite(v) else "  --  "
                    for v in DECL[w]["xi"][:8])
    print("  %4d px %10.3f  %s" % (w, DECL[w]["per_beam"], row))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns
    _dsys = np.nanmax([np.abs(DECL[w]["xi"] - DECL[DECLUSTER_PRIMARY]["xi"])
                       for w in DECLUSTER_WINDOWS_PIX], axis=0)
    _dwin = _dsys[(K_GRID >= 3.0) & (K_GRID <= 4.5)]
    _dmax = np.nanmax(_dwin) if np.isfinite(_dwin).any() else np.nan
print("  max |shift| vs the %d px primary, over k in [3, 4.5]: %.3f"
      % (DECLUSTER_PRIMARY, _dmax))
print("""  This is a SYSTEMATIC to be added to the error budget, not a
  correction: no window is "right", because the beam makes nearby maxima
  genuinely correlated.  Quote the primary value with this spread.""")

# %% [markdown]
# ## 8. ξ(u) scan with the cutout-level bootstrap
#
# The cutout is the resampling unit — peaks within one cutout share
# large-scale modes and are not independent.  **The full replicate matrix
# is retained**, because every covariance-aware statistic in §11–12 is
# built from it; v1 stored only percentiles, which is why its combined
# significances could not be repaired after the fact.

# %%
def _xi_point():
    xi_hat = np.full(U_GRID.size, np.nan)
    n_exc = np.zeros(U_GRID.size, int)
    for j, u in enumerate(U_GRID):
        y = peaks_pool[peaks_pool > u] - u
        n_exc[j] = y.size
        if y.size >= MIN_EXCEED:
            xi_hat[j] = fit_gpd(y)["xi"]
    return np.vstack([xi_hat, n_exc.astype(float)])


def _xi_boot_chunk(i0, i1):
    rng = np.random.default_rng(7 + i0)
    out = np.full((i1 - i0, U_GRID.size), np.nan)
    for b in range(i1 - i0):
        pick = rng.integers(0, n_cut, n_cut)
        pool_b = np.concatenate([peaks_by_cut[i] for i in pick])
        for j, u in enumerate(U_GRID):
            y = pool_b[pool_b > u] - u
            if y.size >= MIN_EXCEED:
                out[b, j] = fit_gpd(y)["xi"]
    return out


_pt = cached("xi_point", _xi_point)
XI_HAT, N_EXC = _pt[0], _pt[1].astype(int)
BOOT = cached_chunks("xi_boot", N_BOOT, _xi_boot_chunk)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    XI_LO, XI_HI = np.nanpercentile(BOOT, [16, 84], axis=0)
    XIERR = 0.5 * (XI_HI - XI_LO)

print("\n     k      u [mJy/bm]     n_exc     xi_hat +- (boot 16-84)/2")
for j, k in enumerate(K_GRID):
    if np.isfinite(XI_HAT[j]):
        print("  %5.1f %13.1f %9d     %+.4f +- %.4f"
              % (k, U_GRID[j], N_EXC[j], XI_HAT[j], XIERR[j]))
    else:
        print("  %5.1f %13.1f %9d     -- (below MIN_EXCEED = %d)"
              % (k, U_GRID[j], N_EXC[j], MIN_EXCEED))

# ---- mean-excess diagnostic -------------------------------------------------
me_grid = mu_core + np.linspace(0.5, 8.0, 40) * sigma_core
mean_excess = np.array([
    (peaks_pool[peaks_pool > u] - u).mean()
    if np.sum(peaks_pool > u) > 20 else np.nan for u in me_grid])

# %% [markdown]
# ## 9. Analytic (pixel-level) model curves — reference only
#
# These are the v1 comparison curves: `xi_of_threshold_analytic` applied
# to the FFT P(D) of each model sitting on its own measured Gaussian
# budget.  They are retained for continuity, but they are **pixel-level**,
# whereas the measurement is **declustered-peak-level**.  Herschel's H1c
# measured |Δξ| ≈ 0.047 between an analytic pixel curve and a simulated
# peak measurement of identical counts, and §11 of the Herschel memo lists
# replacing them as port item 3.  §10 below produces the peak-level
# curves; those are what the model comparison uses.

# %%
def _analytic_curves():
    out = {}
    for nm, c in MODELS.items():
        pp = PofD(c, BEAM_FWHM_ARCSEC, mu=1.0,
                  sigma_noise=SIGMA_EXTRA[nm], s_cut=S_CUT_MJY,
                  d_span=D_SPAN_MJY, n_fft=N_FFT)
        xi_m, _ = xi_of_threshold_analytic(pp.d, pp.p, U_GRID - mu_core)
        sf_m = pp.sf(U_GRID - mu_core)
        out[nm] = dict(xi=np.where(sf_m > SF_FLOOR, xi_m, np.nan), sf=sf_m)
    return out


XI_ANALYTIC = cached("analytic_curves", _analytic_curves)
print("\n  analytic (pixel-level) xi at k = 3.5 / 4.0:")
for nm in MODEL_NAMES:
    _j35, _j40 = int(np.argmin(abs(K_GRID - 3.5))), int(np.argmin(abs(K_GRID - 4.0)))
    print("    %-11s %+.4f  %+.4f"
          % (nm, XI_ANALYTIC[nm]["xi"][_j35], XI_ANALYTIC[nm]["xi"][_j40]))

# %% [markdown]
# ## 10. The null Monte Carlo — peak-level model curves and the null
# ## sampling distribution of ξ̂(u)
#
# This is the core of the v2 audit.  Rather than comparing the measured
# ξ̂(u) with an analytic curve and an error bar, we push each count model
# through the **identical pipeline** — simulate, cut into cutouts,
# baseline, decluster with the same window, apply the same threshold
# ladder, fit the same GPD by the same MLE — and read off (a) what ξ̂ the
# model *would produce in our measurement*, and (b) the full sampling
# distribution of that quantity at our sample size.  Nothing about the
# small-sample bias of the MLE, the peak-vs-pixel offset, the baselining,
# or the threshold correlation has to be modelled: it is all inherited.
#
# **Two-tier design.**
#
# *Tier 1 (map level).* `N_NULL_MAPS` fields of `NULL_MAP_NPIX`² at 1′
# pixels, each cut into 64′ cutouts, giving a null cutout library of the
# same size as the data sample.  Sources fainter than `S_SPLIT_MJY` are
# absorbed into the Gaussian component instead of being injected
# individually — at the Planck beam the counts give ~10² sources per beam
# below 10 mJy, so the CLT applies to them to high accuracy, and the split
# is verified below by checking that σ_c(<split) ⊕ σ_c(>split) reproduces
# σ_c(full).  `noise_mode="beam"` throughout (Herschel port item 6:
# `"white"` retains ~4× more peaks, nearly all noise spikes).
#
# *Tier 2 (sampling level).* The null cutout library is resampled
# `N_NULL_DRAW` times at the data's cutout count, and the ladder refit
# each time.  This yields the null distribution of ξ̂(u) — its median (the
# bias), its spread, and its **threshold covariance** — from which the
# measured ξ̂ gets an honest p-value with no Gaussianity assumption
# anywhere.
#
# **Caveat, stated rather than buried.** Both the analytic P(D) and
# `CIBMapSimulator` are Poisson; neither contains source clustering.  The
# Gaussian budget we inject reproduces the measured *variance* of the
# clustered CIB and cirrus but not their non-Gaussian structure.  The null
# is therefore "Poisson sources on a Gaussian background of the observed
# width", which is precisely the hypothesis v1's positive-ξ claim was
# asserted against — so it is the right null for this test, but a rejection
# of it is not by itself evidence for any particular alternative.

# %%
def _sigma_c_range(cnts_cls_pars, nm, s_lo, s_hi):
    """sigma_c of one model restricted to [s_lo, s_hi]."""
    m = build_models(s_min=s_lo, s_max=s_hi)[nm]
    return m.confusion_stats(BEAM_FWHM_ARCSEC)["sigma_conf_mJy_per_beam"]


print("\n" + "=" * 79)
print("  NULL MONTE CARLO")
print("=" * 79)
print("  split check: sigma_c(full) vs sqrt(sigma_c(<split)^2 + "
      "sigma_c(>split)^2)")
SIG_FAINT, SIG_BRIGHT = {}, {}
for nm in MODEL_NAMES:
    SIG_FAINT[nm] = _sigma_c_range(None, nm, S_MIN_MJY, S_SPLIT_MJY)
    SIG_BRIGHT[nm] = _sigma_c_range(None, nm, S_SPLIT_MJY, S_CUT_MJY)
    quad = np.hypot(SIG_FAINT[nm], SIG_BRIGHT[nm])
    print("    %-11s %8.2f   vs %8.2f   (rel. diff %.2e)"
          % (nm, SIGMA_C[nm], quad, abs(quad - SIGMA_C[nm]) / SIGMA_C[nm]))
print("    [exact by construction: the second moment is additive over "
      "disjoint flux ranges]")

#  Gaussian component injected in the simulation: the faint-source
#  confusion we chose not to inject explicitly, PLUS the measured
#  non-Poisson budget (clustering + cirrus + instrument noise, which
#  sigma_extra already contains since it was defined against sigma_core).
SIGMA_GAUSS_SIM = {nm: float(np.hypot(SIG_FAINT[nm], SIGMA_EXTRA[nm]))
                   for nm in MODEL_NAMES}
print("  injected Gaussian sigma per model (should return sigma_core = "
      "%.1f):" % sigma_core)
for nm in MODEL_NAMES:
    print("    %-11s %7.1f  ->  predicted core %7.1f"
          % (nm, SIGMA_GAUSS_SIM[nm],
             np.hypot(SIGMA_GAUSS_SIM[nm], SIG_BRIGHT[nm])))

N_TILE = NULL_MAP_NPIX // NPIX_CUT


def _simulate_maps_block(nm, seed0, i0, i1):
    """Simulate maps [i0, i1) for one model and reduce them to peaks.

    Cached per block so a long simulation resumes rather than restarting.
    Returns (list of per-cutout peak arrays, thinned pixel sample).
    """
    cnts_bright = build_models(s_min=S_SPLIT_MJY, s_max=S_CUT_MJY)[nm]
    sim = CIBMapSimulator(cnts_bright, BEAM_FWHM_ARCSEC,
                          PIX_ARCMIN * 60.0, npix=NULL_MAP_NPIX,
                          sigma_noise=SIGMA_GAUSS_SIM[nm],
                          s_cut=S_CUT_MJY, noise_mode="beam")
    lib, pix_all = [], []
    for i in range(i0, i1):
        m = sim.make_map(seed=seed0 + i)
        for a in range(N_TILE):
            for b in range(N_TILE):
                tile = m[a * NPIX_CUT:(a + 1) * NPIX_CUT,
                         b * NPIX_CUT:(b + 1) * NPIX_CUT]
                tb = fit_remove_plane(np.asarray(tile, float))
                lib.append(declustered_peaks(tb, DECLUSTER_PRIMARY)
                           .astype(np.float32))
                pix_all.append(tb.ravel()[::7].astype(np.float32))
    return lib, np.concatenate(pix_all)


def _simulate_peak_library(nm, seed0, block=4):
    """Run one count model through the whole pipeline, block by block."""
    lib, pix = [], []
    for i0 in range(0, N_NULL_MAPS, block):
        i1 = min(i0 + block, N_NULL_MAPS)
        l_b, p_b = cached("simblk_%s_%03d" % (nm, i0),
                          lambda nm=nm, i0=i0, i1=i1:
                          _simulate_maps_block(nm, seed0, i0, i1))
        lib.extend(l_b)
        pix.append(p_b)
    mu_s, sig_s = robust_core(np.concatenate(pix))
    return dict(lib=lib, mu_core=float(mu_s), sigma_core=float(sig_s),
                n_peaks=int(sum(p.size for p in lib)))


def _null_ladder_chunk(lib, u_offsets, n_target, i0, i1):
    """Resample the null cutout library and refit the ladder each time.

    u_offsets are thresholds measured FROM THE SIMULATION'S OWN core mean
    (the simulated maps are mean-subtracted), so the comparison with the
    data is in absolute flux above the core, as it must be.
    """
    rng = np.random.default_rng(4242 + i0)
    n_lib = len(lib)
    out = np.full((i1 - i0, u_offsets.size), np.nan)
    for b in range(i1 - i0):
        pick = rng.integers(0, n_lib, n_target)
        pool = np.concatenate([lib[i] for i in pick])
        for j, u in enumerate(u_offsets):
            y = pool[pool > u] - u
            if y.size >= MIN_EXCEED:
                out[b, j] = fit_gpd(y)["xi"]
    return out


U_OFFSET = U_GRID - mu_core          # absolute flux above the core

SIM = {}
for _i, nm in enumerate(MODEL_NAMES):
    SIM[nm] = cached("sim_%s" % nm,
                     lambda nm=nm, _i=_i: _simulate_peak_library(
                         nm, 900_000 + 10_000 * _i))
    s = SIM[nm]
    print("  %-11s simulated core sigma = %7.1f (data %7.1f), "
          "%d cutouts, %d peaks (%.3f/beam)"
          % (nm, s["sigma_core"], sigma_core, len(s["lib"]), s["n_peaks"],
             s["n_peaks"] / (len(s["lib"]) * NPIX_CUT ** 2
                             / OM_BEAM_ARCMIN2)))

#  Peak-level model curves: the ladder applied to the FULL simulated
#  library (all peaks), i.e. the model's own xi(u) as our pipeline sees it.
XI_PEAK = {}
for nm in MODEL_NAMES:
    pool = np.concatenate(SIM[nm]["lib"])
    xi_m = np.full(U_OFFSET.size, np.nan)
    for j, u in enumerate(U_OFFSET):
        y = pool[pool > u] - u
        if y.size >= MIN_EXCEED:
            xi_m[j] = fit_gpd(y)["xi"]
    XI_PEAK[nm] = xi_m

print("\n  PEAK-LEVEL vs ANALYTIC PIXEL-LEVEL model curves "
      "(Herschel port item 3):")
print("    %-11s %9s %9s %9s %9s"
      % ("model", "k=3 pix", "k=3 peak", "k=4 pix", "k=4 peak"))
_j3 = int(np.argmin(abs(K_GRID - 3.0)))
_j4 = int(np.argmin(abs(K_GRID - 4.0)))
for nm in MODEL_NAMES:
    print("    %-11s %+9.4f %+9.4f %+9.4f %+9.4f"
          % (nm, XI_ANALYTIC[nm]["xi"][_j3], XI_PEAK[nm][_j3],
             XI_ANALYTIC[nm]["xi"][_j4], XI_PEAK[nm][_j4]))
print("""    The two differ because declustering selects local maxima, whose
    marginal distribution is displaced above the pixel P(D) (manual Sec.
    13.7).  Only the peak-level column may be compared with the data.""")

#  Tier 2: the null sampling distribution at the data's sample size.
NULL_BOOT = {nm: cached_chunks(
    "nullboot_%s" % nm, N_NULL_DRAW,
    lambda i0, i1, nm=nm: _null_ladder_chunk(SIM[nm]["lib"], U_OFFSET,
                                             n_cut, i0, i1))
    for nm in MODEL_NAMES}

NB = NULL_BOOT[PRIMARY_MODEL]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    NULL_MED = np.nanmedian(NB, axis=0)
    NULL_LO, NULL_HI = np.nanpercentile(NB, [2.5, 97.5], axis=0)
    NULL_SD = np.nanstd(NB, axis=0)

#  One-sided p-value: how often does the null produce a xi at least as
#  large as the measured one?
P_ONE_SIDED = np.full(U_GRID.size, np.nan)
for j in range(U_GRID.size):
    col = NB[:, j]
    col = col[np.isfinite(col)]
    if col.size > 50 and np.isfinite(XI_HAT[j]):
        P_ONE_SIDED[j] = float(np.mean(col >= XI_HAT[j]))

print("\n  MEASURED xi-hat AGAINST THE SIMULATED NULL "
      "(%s model, %d draws)" % (PRIMARY_MODEL, N_NULL_DRAW))
print("    %5s %9s %10s %10s %12s %9s"
      % ("k", "xi_hat", "null med", "null sd", "null 95% hi", "p(>=)"))
for j, k in enumerate(K_GRID):
    if np.isfinite(XI_HAT[j]) and np.isfinite(NULL_MED[j]):
        print("    %5.1f %+9.4f %+10.4f %10.4f %+12.4f %9.3f"
              % (k, XI_HAT[j], NULL_MED[j], NULL_SD[j], NULL_HI[j],
                 P_ONE_SIDED[j]))
_bias_band = (K_GRID >= 3.0) & (K_GRID <= 4.5) & np.isfinite(NULL_MED)
print("""
  THE SMALL-SAMPLE BIAS, MEASURED.  The 'null med' column is what our
  pipeline returns when the truth is a pure Poisson-on-Gaussian null.  Any
  offset of that column from the analytic pixel-level curve of Sec. 9 is
  the combined peak-vs-pixel shift and MLE small-sample bias -- the thing
  v1 had no way to see.  At k >= 4 the exceedance count falls to O(10^2),
  where the GPD MLE is both biased upward and strongly right-skewed
  (Hosking & Wallis 1987; Smith 1985), so the null 97.5th percentile sits
  well above zero even though the truth has xi < 0.  A measured xi > 0
  there is NOT evidence of a heavy tail unless it clears that percentile.""")

# %% [markdown]
# ## 10b. The bright-source bracket — Herschel port item 4, adapted
#
# At Herschel the bright-source audit was done by *masking* the map above
# the counts truncation S_cut = 100 mJy, which there is 12.6 σ_c, and
# comparing with models truncated identically.  **That is impossible at
# Planck**, and the reason is itself a result: S_cut = 100 mJy is only
# 0.54 σ_core here, so masking the map at the model's truncation flux
# would remove most of the sky.  There is no flux at which map and model
# can be made to cover the same range by masking, because the model's
# bright cut lies *inside* the noise.
#
# The bracket therefore has to run the other way — extend the model
# instead of censoring the map.  The census of §13 shows the real map
# contains ~1300 declustered peaks above 650 mJy and ~120 above 1000
# mJy, i.e. sources far above the 100 mJy truncation, and these are
# precisely the objects that populate the k ≳ 3.5 exceedances.  The null
# below repeats the simulation with the Schechter counts extended to
# `S_BRIGHT_EXT_MJY`, so the simulated map contains a bright population
# too.  If the measured ξ̂ excursion collapses towards this extended null,
# the excursion is the bright population the fiducial models truncate
# away — the same conclusion H1c reached at Herschel — and *not* evidence
# about the sub-100 mJy counts the paper is trying to constrain.
#
# The extension is an extrapolation of the fitted Schechter above the
# range it was fitted over, and the real counts there are
# lensing-dominated with a shallower slope (Lima et al. 2010), so this
# null should be read as a *lower* bound on the bright population's
# effect.

# %%
S_BRIGHT_EXT_MJY = float(os.environ.get("CIB_SEXT_MJY", "1500.0"))
_ext_name = "Schechter_ext"


def _sim_ext():
    """Simulate the primary model with the counts extended to S_ext."""
    cnts = build_models(s_min=S_SPLIT_MJY, s_max=S_BRIGHT_EXT_MJY)[PRIMARY_MODEL]
    sim = CIBMapSimulator(cnts, BEAM_FWHM_ARCSEC, PIX_ARCMIN * 60.0,
                          npix=NULL_MAP_NPIX,
                          sigma_noise=SIGMA_GAUSS_SIM[PRIMARY_MODEL],
                          s_cut=S_BRIGHT_EXT_MJY, noise_mode="beam")
    lib, pix = [], []
    for i in range(N_NULL_MAPS):
        m = sim.make_map(seed=770_000 + i)
        for a in range(N_TILE):
            for b in range(N_TILE):
                tb = fit_remove_plane(np.asarray(
                    m[a * NPIX_CUT:(a + 1) * NPIX_CUT,
                      b * NPIX_CUT:(b + 1) * NPIX_CUT], float))
                lib.append(declustered_peaks(tb, DECLUSTER_PRIMARY)
                           .astype(np.float32))
                pix.append(tb.ravel()[::7].astype(np.float32))
    mu_s, sig_s = robust_core(np.concatenate(pix))
    return dict(lib=lib, mu_core=float(mu_s), sigma_core=float(sig_s),
                n_peaks=int(sum(p.size for p in lib)))


SIM_EXT = cached("sim_ext", _sim_ext)
pool_ext = np.concatenate(SIM_EXT["lib"])
XI_PEAK_EXT = np.full(U_OFFSET.size, np.nan)
for j, u in enumerate(U_OFFSET):
    y = pool_ext[pool_ext > u] - u
    if y.size >= MIN_EXCEED:
        XI_PEAK_EXT[j] = fit_gpd(y)["xi"]

NULL_EXT = cached_chunks(
    "nullboot_ext", N_NULL_DRAW,
    lambda i0, i1: _null_ladder_chunk(SIM_EXT["lib"], U_OFFSET, n_cut,
                                      i0 + 500_000, i1 + 500_000))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    EXT_MED = np.nanmedian(NULL_EXT, axis=0)
    EXT_LO, EXT_HI = np.nanpercentile(NULL_EXT, [2.5, 97.5], axis=0)
P_EXT = np.full(U_GRID.size, np.nan)
for j in range(U_GRID.size):
    col = NULL_EXT[:, j]
    col = col[np.isfinite(col)]
    if col.size > 50 and np.isfinite(XI_HAT[j]):
        P_EXT[j] = float(np.mean(col >= XI_HAT[j]))

print("\n  BRIGHT-EXTENDED NULL (%s counts to %.0f mJy), simulated core "
      "%.1f vs data %.1f" % (PRIMARY_MODEL, S_BRIGHT_EXT_MJY,
                             SIM_EXT["sigma_core"], sigma_core))
print("    %5s %9s %12s %12s %12s %9s"
      % ("k", "xi_hat", "trunc null", "ext null", "ext 95% hi", "p_ext"))
for j, k in enumerate(K_GRID):
    if np.isfinite(XI_HAT[j]) and np.isfinite(EXT_MED[j]):
        print("    %5.1f %+9.4f %+12.4f %+12.4f %+12.4f %9.3f"
              % (k, XI_HAT[j], NULL_MED[j], EXT_MED[j], EXT_HI[j], P_EXT[j]))
print("""    If 'ext null' tracks xi_hat where 'trunc null' does not, the
    measured excursion is the >100 mJy population, not the faint counts.""")

# %% [markdown]
# ## 11. Threshold covariance — the v2 correction
#
# ξ̂ at neighbouring thresholds is strongly correlated: the peaks above
# u₂ are a *subset* of those above u₁ < u₂, and every threshold is
# computed from the same bootstrap-resampled cutouts.  Two consequences.
#
# 1. Statements of the form "positive at k = 3.5 *and* 4 *and* 4.5" count
#    one measurement several times.
# 2. Any quadrature combination across thresholds — the v1 reading, and
#    the pre-v3 Herschel statistic — inflates the result by roughly
#    √n_eff.
#
# The correct statistic is S² = dᵀC⁻¹d with C the bootstrap covariance and
# the Hartlap (2007, *A&A* **464**, 399) factor (N−p−2)/(N−1) correcting
# the bias of an inverted sample covariance.  Because the deep,
# low-variance modes of C are exactly where a sample covariance is least
# reliable, we also report PC-truncated versions and a half-sample
# stability test, following the Herschel v3 treatment.

# %%
GOOD = np.isfinite(XI_HAT) & np.isfinite(XIERR) & (XIERR > 0)
_bok = GOOD & np.all(np.isfinite(BOOT), axis=0)
BMAT = BOOT[:, _bok]
NB_EFF, NP_EFF = BMAT.shape
if NB_EFF - NP_EFF - 2 <= 0:
    raise SystemExit("ERROR: %d bootstrap replicates cannot support a "
                     "%d x %d covariance inverse.  Raise CIB_N_BOOT."
                     % (NB_EFF, NP_EFF, NP_EFF))
COV = np.cov(BMAT, rowvar=False)
DIAG = np.sqrt(np.diag(COV))
CORR = COV / np.outer(DIAG, DIAG)
HARTLAP = (NB_EFF - NP_EFF - 2.0) / (NB_EFF - 1.0)
COVI = np.linalg.inv(COV) * HARTLAP

print("\n" + "=" * 79)
print("  THRESHOLD COVARIANCE  (why naive quadrature is wrong)")
print("=" * 79)
print("  %d usable thresholds, %d bootstrap replicates" % (NP_EFF, NB_EFF))
print("  correlation of xi_hat vs threshold separation:")
_dk = K_GRID[1] - K_GRID[0]
for _lag in (1, 2, 3, 4, 6):
    if _lag < NP_EFF:
        print("     lag %2d (dk = %.1f): mean rho = %.3f"
              % (_lag, _lag * _dk, float(np.mean(np.diag(CORR, _lag)))))
_ev = np.linalg.eigvalsh(CORR)[::-1]
N_EFF_SUM = float(NP_EFF ** 2 / CORR.sum())
N_EFF_PCA = float(_ev.sum() ** 2 / np.sum(_ev ** 2))
print("  effective independent thresholds: %.2f (sum of correlations), "
      "%.2f (eigenvalue spread), of %d" % (N_EFF_SUM, N_EFF_PCA, NP_EFF))
print("  leading principal component carries %.0f%% of the variance"
      % (100 * _ev[0] / _ev.sum()))
print("  Hartlap factor (N-p-2)/(N-1) = %.3f" % HARTLAP)


def _sub_cov(sel):
    """Covariance sub-block, its Hartlap-corrected inverse, and the factor.

    A model curve can be NaN at a threshold where the simulated null had
    too few exceedances, so the usable set differs from one comparison to
    the next.  Rather than returning NaN for the whole statistic (which is
    what a naive implementation does), restrict to the thresholds that ARE
    finite and invert the corresponding sub-block; the Hartlap factor is
    recomputed for the reduced dimension.
    """
    p = int(sel.sum())
    if p == 0 or NB_EFF - p - 2 <= 0:
        return None, np.nan, 0
    C = COV[np.ix_(sel, sel)]
    h = (NB_EFF - p - 2.0) / (NB_EFF - 1.0)
    return np.linalg.inv(C) * h, h, p


def sep_full(d_vec, n_pc=None, restrict=None):
    """Covariance-correct separation sqrt(d^T C^-1 d).

    restrict : optional boolean mask over the FULL threshold grid, ANDed
    with the covariance's own usable set (e.g. the science window).
    n_pc : if given, invert only within the leading n_pc principal
    components.  A full-covariance number that collapses under mild
    truncation is being carried by poorly-determined low-variance modes
    and should not be quoted.
    """
    d_all = np.asarray(d_vec, float)
    sel = np.isfinite(d_all[_bok])
    if restrict is not None:
        sel &= restrict[_bok]
    d = d_all[_bok][sel]
    Ci, h, p = _sub_cov(sel)
    if Ci is None:
        return np.nan
    if n_pc is None:
        return float(np.sqrt(max(d @ Ci @ d, 0.0)))
    C = COV[np.ix_(sel, sel)]
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:min(n_pc, p)]
    proj = V[:, idx].T @ d
    return float(np.sqrt(max(np.sum(proj ** 2 / w[idx]) * h, 0.0)))


def sep_half_sample(d_vec, n_rep=20, restrict=None):
    """Recompute sep_full from random halves of the bootstrap ensemble."""
    d_all = np.asarray(d_vec, float)
    sel = np.isfinite(d_all[_bok])
    if restrict is not None:
        sel &= restrict[_bok]
    d = d_all[_bok][sel]
    p = int(sel.sum())
    nh = NB_EFF // 2
    if p == 0 or nh - p - 2 <= 0:
        return np.nan, np.nan
    vals = []
    for rep in range(n_rep):
        sub = np.random.default_rng(1000 + rep).permutation(NB_EFF)[:nh]
        #  atleast_2d: a one-threshold window gives a 0-d np.cov (release fix,
        #  reached only by the synthetic smoke test; no effect for p >= 2)
        Ch = np.atleast_2d(np.cov(BMAT[np.ix_(sub, np.where(sel)[0])],
                                  rowvar=False))
        hh = (nh - p - 2.0) / (nh - 1.0)
        vals.append(np.sqrt(max(d @ (np.linalg.inv(Ch) * hh) @ d, 0.0)))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan,) * 2


# %% [markdown]
# ## 12. Is there a detection?  Three statistics, honestly labelled
#
# **(a) Best single threshold** — the conservative headline; no
# combination, no covariance, no Gaussianity assumption beyond the
# bootstrap percentile.
#
# **(b) Naive quadrature across the window** — the *incorrect* v1-style
# statistic, printed only so the size of the error is visible.
#
# **(c) Full covariance** — S² = dᵀC⁻¹d, the optimal combination if C is
# taken at face value, with PC-truncated and half-sample columns as the
# robustness check.
#
# **(d) Simulated-null p-value** — the statistic that assumes least: how
# often the null pipeline of §10 produces a ξ̂ at least as extreme.  When
# (a)–(c) and (d) disagree, (d) wins, because it is the only one that
# knows about the MLE's small-sample behaviour.
#
# The vector d is the measured ξ̂ minus the **peak-level** model curve.

# %%
WIN = (K_GRID >= 3.0) & (K_GRID <= 4.5)
print("\n" + "=" * 79)
print("  IS THERE A DETECTION OF SUPER-POISSON TAIL STRUCTURE?")
print("=" * 79)
print("""  RESTRICTED TO THE SCIENCE WINDOW k in [3, 4.5].  Below k ~ 3 the
  measurement is floor-dominated: the models carry the non-Poisson budget
  as a GAUSSIAN of the measured width, whereas the real floor is clustered
  CIB plus cirrus, which is neither Gaussian nor beam-correlated.  A large
  data-minus-model residual there is a statement about the foreground
  model, not about the counts, so including those thresholds would inflate
  every number below without adding information about dN/dS.""")
print("  %-11s %11s %11s %13s %9s %9s"
      % ("vs model", "best single", "naive quad", "FULL COV", "PC=2", "PC=3"))
DETECT = {}
for nm in MODEL_NAMES:
    d = XI_HAT - XI_PEAK[nm]
    r = d / XIERR
    ok = GOOD & WIN & np.isfinite(r)
    if not ok.any():
        continue
    ib = int(np.nanargmax(np.where(ok, np.abs(r), np.nan)))
    #  the naive statistic must use the SAME threshold set as the
    #  covariance one, or the comparison is not like-for-like
    naive = float(np.sqrt(np.nansum(np.where(ok, r, np.nan) ** 2)))
    full = sep_full(d, restrict=WIN)
    hm, hs = sep_half_sample(d, restrict=WIN)
    DETECT[nm] = dict(best=float(r[ib]), best_k=float(K_GRID[ib]),
                      naive_window=naive, full_cov=full,
                      half_mean=hm, half_sd=hs, n_thr=int(ok.sum()))
    print("  %-11s %11.2f %11.2f %13.2f %9.2f %9.2f"
          % (nm, r[ib], naive, full, sep_full(d, 2, restrict=WIN),
             sep_full(d, 3, restrict=WIN)))
    print("  %-11s   (best at k = %.1f, %d thresholds; half-sample "
          "full-cov %.2f +- %.2f)" % ("", K_GRID[ib], int(ok.sum()), hm, hs))

#  The same question asked of the simulated null, which is the reading we
#  actually trust.
_pmin = np.nanmin(P_ONE_SIDED[WIN]) if np.isfinite(P_ONE_SIDED[WIN]).any() \
    else np.nan
_ntrial = int(np.sum(WIN & np.isfinite(P_ONE_SIDED)))
print("\n  SIMULATED-NULL READING (%s null, %d draws):" % (PRIMARY_MODEL,
                                                           N_NULL_DRAW))
print("    smallest one-sided p over the window k in [3, 4.5]: %.4f "
      "(%d thresholds scanned)" % (_pmin, _ntrial))
#  Look-elsewhere over a correlated ladder: the effective number of
#  independent trials is n_eff, not the number of thresholds.
_neff_win = max(1.0, N_EFF_SUM * _ntrial / max(NP_EFF, 1))
_p_global = 1.0 - (1.0 - _pmin) ** _neff_win if np.isfinite(_pmin) else np.nan
print("    effective independent trials in the window: %.2f  ->  "
      "look-elsewhere-corrected p = %.4f" % (_neff_win, _p_global))
print("""    The trials correction uses n_eff, NOT the number of thresholds:
    correcting by 4 when the ladder contains only ~1-2 independent numbers
    would over-penalise as badly as not correcting at all under-penalises.""")

# %% [markdown]
# ## 12b. Model ranking, diagonal vs full covariance
#
# χ² of each model against the measurement, and the pairwise model
# separations — which express *discriminating power*, i.e. how far apart
# the models are relative to the data errors, and are **not** detection
# significances.

# %%
CHI2_DIAG, CHI2_COV = {}, {}
for nm in MODEL_NAMES:
    d = XI_HAT - XI_PEAK[nm]
    r = d / XIERR
    okw = GOOD & WIN
    CHI2_DIAG[nm] = float(np.nansum(np.where(okw, r, np.nan) ** 2))
    _s = sep_full(d, restrict=WIN)
    CHI2_COV[nm] = float(_s ** 2) if np.isfinite(_s) else np.nan
print("\n  chi2 of each model against the measured xi-hat "
      "(peak-level curves, window only):")
print("    %-11s %18s %18s"
      % ("model", "diagonal (%d thr)" % int((GOOD & WIN).sum()),
         "full covariance"))
for nm in MODEL_NAMES:
    print("    %-11s %18.1f %18.1f" % (nm, CHI2_DIAG[nm], CHI2_COV[nm]))
_ord_d = sorted(CHI2_DIAG, key=CHI2_DIAG.get)
_ord_c = sorted((m for m in CHI2_COV if np.isfinite(CHI2_COV[m])),
                key=CHI2_COV.get)
print("    ranking, diagonal:        %s" % " < ".join(_ord_d))
print("    ranking, full covariance: %s" % " < ".join(_ord_c))

PAIRS = [("Schechter", "SPL"), ("Schechter", "DPL"), ("SPL", "DPL")]
print("\n  pairwise MODEL SEPARATIONS (discriminating power, not "
      "significance):")
print("    %-20s %11s %11s %13s"
      % ("pair", "best single", "naive quad", "FULL COV"))
SEP = {}
for a, b in PAIRS:
    g = XI_PEAK[a] - XI_PEAK[b]
    r = np.abs(g) / XIERR
    ok = GOOD & np.isfinite(r)
    if not ok.any():
        continue
    ib = int(np.nanargmax(np.where(ok, r, np.nan)))
    naive = float(np.sqrt(np.nansum(np.where(ok, r, np.nan) ** 2)))
    full = sep_full(g, restrict=WIN)
    SEP["%s_vs_%s" % (a, b)] = dict(best=float(r[ib]), naive=naive,
                                    full_cov=full)
    print("    %-20s %11.2f %11.2f %13.2f"
          % ("%s vs %s" % (a, b), r[ib], naive, full))
print("""    The effect of the covariance is NOT a uniform deflation.  Pairs
    whose model difference is mostly a vertical OFFSET are penalised,
    because a coherent shift of the whole curve is also the dominant
    bootstrap noise mode; a pair whose curves CROSS is rewarded, its
    difference being nearly orthogonal to that mode.  Diagonal errors
    cannot see this (Herschel memo Sec. 5.8).""")

# %% [markdown]
# ## 13. Bright-source census and the point-source-mask ceiling
#
# Herschel port item 4.  The Planck 857 GHz PS mask removes S/N > 5
# sources, i.e. roughly S ≳ 650 mJy ≈ 4 σ_core — right where v1's ξ̂(u)
# turned over.  A right-truncated tail reads as ξ < 0 (manual §13.5), so
# the decline at k ≳ 4.5 may be an artefact of the mask rather than a
# property of the counts.  The census below compares the observed
# declustered-peak counts above a flux with the model prediction, so the
# truncation is *measured* rather than asserted.

# %%
print("\n" + "=" * 79)
print("  BRIGHT-SOURCE CENSUS")
print("=" * 79)
print(f"  PS-mask limit {S_PSMASK_MJY:.0f} mJy/beam = "
      f"{(S_PSMASK_MJY - mu_core) / sigma_core:.2f} sigma_core above the core")
print("  %10s %12s %14s %14s %14s"
      % ("S [mJy/bm]", "observed", "Schechter", "SPL", "DPL"))
CENSUS = {}
for s in [200.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0]:
    n_obs = int(np.sum(peaks_pool > s))
    row = [n_obs]
    for nm in MODEL_NAMES:
        #  model N(>S) per deg^2 over the surveyed area; valid only for
        #  S >> sigma_core, where a peak is one source rather than a
        #  confusion fluctuation
        m_un = build_models(s_min=S_MIN_MJY, s_max=1e5)[nm]
        row.append(float(m_un.n_gtr(s) * area_total_deg2))
    CENSUS[s] = row
    print("  %10.0f %12d %14.1f %14.1f %14.1f" % (s, *row))
print("""  The model column is the number of INTRINSIC sources brighter than
  S expected in the surveyed area (counts extrapolated above S_cut, hence
  no lensing boost).  Where observed << model, the map has been censored:
  that is the PS mask, and it sets the hard ceiling of the usable
  threshold window.  Where observed >> model, we are counting confusion
  fluctuations, not sources -- which is the regime below ~3 sigma_core.""")

# %% [markdown]
# ## 14. Figures
#
# **A — the stability scan.**  Measured ξ̂(u) with the cutout bootstrap
# band, the three peak-level model curves, and — the panel that carries
# the argument — the shaded 95% band of the *simulated null*.  A point
# inside that band is consistent with Poisson-on-Gaussian no matter what
# its own error bar says.
#
# **B — diagnostics.**  Mean excess; the declustering-window systematic;
# the threshold correlation matrix; and the null-vs-measured comparison of
# the combined statistic.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

ax = axes[0]
ax.axvspan(K_GRID.min(), 3.0, color="0.93", zorder=0)
ax.fill_between(K_GRID, NULL_LO, NULL_HI, color="0.55", alpha=0.30,
                zorder=1, label="simulated null, 95% band")
ax.plot(K_GRID, NULL_MED, color="0.35", ls="-.", lw=1.3, zorder=2,
        label="simulated null, median")
for nm, c, ls in zip(MODEL_NAMES, ["C2", "C3", "C4"], ["--", "-", ":"]):
    ax.plot(K_GRID, XI_PEAK[nm], c + ls, lw=1.6, zorder=3,
            label=f"{nm} (peak-level)")
ax.errorbar(K_GRID, XI_HAT, yerr=XIERR, fmt="C0o", ms=5, capsize=2, lw=1.2,
            zorder=5, label=r"measured $\hat\xi(u)$ (cutout bootstrap)")
ax.axhline(0, color="0.7", lw=0.8)
ax.set(xlabel=r"threshold  $k = (u - \mu_{\rm core})/\sigma_{\rm core}$",
       ylabel=r"$\xi(u)$",
       title=f"GPD stability scan, 857 GHz ({MASK_VARIANT}, {n_cut} cutouts)")
ax.legend(fontsize=7.5, loc="upper left")

ax = axes[1]
ax.plot((me_grid - mu_core) / sigma_core, mean_excess / sigma_core,
        "C0.-", ms=4)
ax.set(xlabel=r"$k = (v - \mu_{\rm core})/\sigma_{\rm core}$",
       ylabel=r"mean excess  $e(v)/\sigma_{\rm core}$",
       title="mean-excess (declustered peaks): straight = GPD;\n"
             "negative slope = bounded tail, positive = heavy")
fig.tight_layout()
_finish_figure(fig, "planck_v2_unlensed_stability")

# ---- diagnostics panel ------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

ax = axes[0]
for w, c in zip(DECLUSTER_WINDOWS_PIX, ["C0", "C1", "C5"]):
    ax.plot(K_GRID, DECL[w]["xi"], c + "o-", ms=4, lw=1.2,
            label=f"window {w} px ({w / BEAM_FWHM_PIX:.1f} FWHM)")
ax.axhline(0, color="0.7", lw=0.8)
ax.axvspan(3.0, 4.5, color="C2", alpha=0.08)
ax.set(xlabel=r"$k$", ylabel=r"$\hat\xi(u)$",
       title="declustering-window systematic")
ax.legend(fontsize=8)

ax = axes[1]
_kk = K_GRID[_bok]
im = ax.imshow(CORR, origin="lower", vmin=0, vmax=1, cmap="viridis",
               extent=[_kk.min(), _kk.max(), _kk.min(), _kk.max()])
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set(xlabel=r"$k$", ylabel=r"$k$",
       title=r"threshold correlation of $\hat\xi$"
             "\n" + r"$n_{\rm eff}$ = %.2f of %d" % (N_EFF_SUM, NP_EFF))

ax = axes[2]
_jbest = int(np.argmin(np.abs(K_GRID - 4.0)))
_col = NB[:, _jbest]
_col = _col[np.isfinite(_col)]
if _col.size <= 20:
    #  fall back to the deepest threshold that the null actually reaches
    _valid = [j for j in range(U_GRID.size)
              if np.isfinite(NB[:, j]).sum() > 20]
    if _valid:
        _jbest = _valid[-1]
        _col = NB[:, _jbest]
        _col = _col[np.isfinite(_col)]
if _col.size > 20:
    ax.hist(_col, bins=40, color="0.6", label="simulated null")
    if np.isfinite(XI_HAT[_jbest]):
        ax.axvline(XI_HAT[_jbest], color="C3", lw=2,
                   label=r"measured $\hat\xi$ = %+.3f" % XI_HAT[_jbest])
    ax.axvline(0.0, color="0.3", ls=":", lw=1)
    ax.legend(fontsize=8)
ax.set(xlabel=r"$\hat\xi$ at $k = %.1f$" % K_GRID[_jbest], ylabel="draws",
       title="null sampling distribution\n(small-sample MLE skew visible)")
fig.tight_layout()
_finish_figure(fig, "planck_v2_unlensed_diagnostics")

# %% [markdown]
# ## 15. Save results and closing summary
#
# The npz carries everything the paper section and the differential module
# need — including, unlike v1, **the full bootstrap replicate matrix**, so
# any future covariance treatment can be applied without re-running.

# %%
OUT = os.path.join(RESULTS_DIR, f"planck_v2_unlensed_857_{MASK_VARIANT}.npz")
np.savez_compressed(
    OUT,
    # configuration
    mask_variant=MASK_VARIANT, k_grid=K_GRID, u_grid=U_GRID,
    n_cutouts=n_cut, npix_cut=NPIX_CUT, pix_arcmin=PIX_ARCMIN,
    beam_fwhm_arcsec=BEAM_FWHM_ARCSEC, s_min_mjy=S_MIN_MJY,
    s_cut_mjy=S_CUT_MJY, s_split_mjy=S_SPLIT_MJY,
    decluster_primary=DECLUSTER_PRIMARY, sky_frac=sky_frac,
    area_total_deg2=area_total_deg2, n_beams_total=n_beams_total,
    # noise / core budget
    mu_core=mu_core, sigma_core=sigma_core, sigma_neg=sigma_neg,
    sigma_n=sigma_n, map_rms_fullsky=v_map.std(),
    sigma_c_json=json.dumps(SIGMA_C),
    sigma_extra_json=json.dumps(SIGMA_EXTRA),
    monopole_json=json.dumps(MONO), firas_857=FIRAS_857,
    # the measurement
    peaks_pool=peaks_pool, xi=XI_HAT, xi_err=XIERR,
    xi_lo=XI_LO, xi_hi=XI_HI, n_exc=N_EXC,
    boot=BOOT,                       # <-- the replicate matrix (new in v2)
    me_grid=me_grid, mean_excess=mean_excess,
    # models
    xi_analytic_json=json.dumps({nm: list(map(float, XI_ANALYTIC[nm]["xi"]))
                                 for nm in MODEL_NAMES}),
    xi_peak_json=json.dumps({nm: list(map(float, XI_PEAK[nm]))
                             for nm in MODEL_NAMES}),
    # null MC
    null_med=NULL_MED, null_lo=NULL_LO, null_hi=NULL_HI, null_sd=NULL_SD,
    null_boot=NB, p_one_sided=P_ONE_SIDED,
    # bright-extended null (Sec. 10b)
    s_bright_ext=S_BRIGHT_EXT_MJY, xi_peak_ext=XI_PEAK_EXT,
    ext_med=EXT_MED, ext_lo=EXT_LO, ext_hi=EXT_HI, p_ext=P_EXT,
    # covariance
    thr_cov=COV, thr_corr=CORR, hartlap=HARTLAP,
    n_eff_sum=N_EFF_SUM, n_eff_pca=N_EFF_PCA, bok=_bok,
    # statistics
    detect_json=json.dumps(DETECT), sep_json=json.dumps(SEP),
    chi2_diag_json=json.dumps(CHI2_DIAG), chi2_cov_json=json.dumps(CHI2_COV),
    decluster_sys=_dsys,
    census_json=json.dumps({str(k): list(map(float, v))
                            for k, v in CENSUS.items()}),
)
print(f"\nWrote {OUT}")

print("\n" + "=" * 79)
print("  M4a-v2 SUMMARY")
print("=" * 79)
print(f"  mask variant {MASK_VARIANT}: f_sky = {sky_frac:.4f} "
      f"({sky_frac * 41253:.0f} deg^2); {n_cut} cutouts = "
      f"{area_total_deg2:.0f} deg^2")
print(f"  sigma_N = {sigma_n:.1f}, sigma_core (post-baseline) = "
      f"{sigma_core:.1f} mJy/beam")
print(f"  Poisson confusion: " + ", ".join(
    "%s %.1f" % (nm, SIGMA_C[nm]) for nm in MODEL_NAMES) + " mJy/beam")
print(f"  beam dilution: S_cut/sigma_c = "
      f"{S_CUT_MJY / SIGMA_C_PRIMARY:.2f} (Planck) vs 12.6 (Herschel 25\")")
print(f"  declustered peaks: {peaks_pool.size} "
      f"({peaks_pool.size / n_beams_total:.2f}/beam)")
print(f"  effective independent thresholds: {N_EFF_SUM:.2f} "
      f"(sum-rho) / {N_EFF_PCA:.2f} (PCA) of {NP_EFF}")

_win_ok = WIN & np.isfinite(XI_HAT) & np.isfinite(NULL_HI)
if _win_ok.any():
    _above = _win_ok & (XI_HAT > NULL_HI)
    print("\n  VERDICT over the window k in [3, 4.5]:")
    print("    xi-hat range: %+.3f .. %+.3f  (mean bootstrap error %.3f)"
          % (np.nanmin(XI_HAT[_win_ok]), np.nanmax(XI_HAT[_win_ok]),
             np.nanmean(XIERR[_win_ok])))
    print("    null 95%% upper edge over the same window: %+.3f .. %+.3f"
          % (np.nanmin(NULL_HI[_win_ok]), np.nanmax(NULL_HI[_win_ok])))
    if _above.any():
        print("    ABOVE the TRUNCATED-model null at k = "
              + ", ".join("%.1f" % k for k in K_GRID[_above])
              + "  -> look-elsewhere-corrected p = %.4f" % _p_global)
        _above_ext = _win_ok & np.isfinite(EXT_HI) & (XI_HAT > EXT_HI)
        if _above_ext.any():
            print("    ALSO above the BRIGHT-EXTENDED null at k = "
                  + ", ".join("%.1f" % k for k in K_GRID[_above_ext]))
            print("    -> not attributable to the >100 mJy population "
                  "alone; the remaining\n       candidates are clustered "
                  "CIB and cirrus non-Gaussianity, neither of\n       which "
                  "the null contains.")
        else:
            print("    but NOT above the bright-extended null of Sec. 10b.")
            print("""    -> THE EXCURSION IS THE BRIGHT POPULATION THE MODELS TRUNCATE AWAY.
       It is a real feature of the map, but it is not evidence about the
       sub-100 mJy counts this analysis is meant to constrain: the same
       conclusion H1c reached at Herschel, reached here by a different
       route because Planck cannot mask at the truncation flux.""")
        print("    Declustering-window systematic over the window: %.3f"
              % np.nanmax(_dsys[WIN]))
    else:
        print("    NO threshold exceeds the simulated null 95% band.")
        print("    The tail is consistent with Poisson confusion on the "
              "measured\n    Gaussian budget.  Combined with S_cut/sigma_c "
              "~ 1, this is the\n    beam-dilution null: at a 5' beam the "
              "positive-xi regime is not\n    merely undetected, it is "
              "structurally absent.")
print("\n  Runtime %.0f s.  Next: planck_lensed_pd_v2.py (module M5-v2)."
      % (time.time() - T0))

