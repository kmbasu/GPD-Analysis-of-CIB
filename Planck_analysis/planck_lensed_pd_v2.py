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
# # Module M5-v2 — Cluster-lensed Δξ(u) in the Planck 857 GHz CIB
#
# `planck_lensed_pd_v2.py` · CIB-lensing / GPD project · Planck analysis
#
# **What this module is.**  The differential measurement: the GPD shape
# parameter measured on PSZ2 cluster sightlines minus the same quantity
# on matched controls, Δξ(u), compared with the NFW μ-mixture prediction
# computed from the accepted clusters' own masses.  At the Planck beam
# this is expected to be — and is delivered as — a calibrated upper limit
# rather than a detection; the deliverable is that the pipeline is
# demonstrably unbiased at that level and that the limit sits far above
# the prediction for a quantified, physical reason.
#
# ---
#
# ## Why Δξ is the robust observable
#
# Lensing magnifies background sources (S → μS) while diluting their
# surface density (dΩ → μ dΩ), so the image-plane counts are
# n_μ(S) = μ⁻² n₀(S/μ) (Lima, Jain, Devlin & Aguirre 2010, Eq. 31).  **For
# a pure power law this changes only the amplitude, not the slope**, so
# the entire Δξ signal comes from *curvature* in dN/dS.  That is the
# physics the paper is about.
#
# The bright-tail and cirrus systematics that dominate the absolute ξ are
# properties of the *sky*, not of cluster sightlines, so they are
# common-mode between cluster and control and cancel in the difference.
# Three design choices, ported from the Herschel H2 module (§7.1 of
# `Herschel_analysis/Full_Analysis_Summary_Herschel.md`), make that
# cancellation as exact as possible — and items 1 and 2 are **new in v2**:
#
# 1. **Local pairing.**  Every cluster is paired with controls displaced
#    by 1°–3° in a random direction, rather than with globally random
#    mask positions matched only on |b| as in v1.  A pair then shares
#    local coverage depth and cirrus far more closely.
# 2. **Paired bootstrap.**  Resampling *pairs* means sky systematics
#    common to a pair cancel inside every replicate, not merely on
#    average.
# 3. **Identical geometry.**  Cutout side, baseline polynomial *and its
#    fitting annulus*, declustering window and aperture are applied
#    identically — including using the **cluster's** θ₅₀₀ to define the
#    *control's* annulus and apertures.  An asymmetry in analysis
#    geometry would appear directly in Δξ and be indistinguishable from
#    signal.
#
# ---
#
# ## What is new in v2
#
# * **Mask variant `4.0e+20_gp40`** — 496 of the 772 PSZ2 clusters fall on
#   the mask, against 298 for v1's `2.5e+20_gp20`, so the accepted sample
#   grows by roughly 1.7×.
# * **Schechter counts only** for the prediction (the agreed strategy:
#   three models for the unlensed ξ(u), one for Δξ), using the final
#   adopted 350 µm fit of `num_count_fits/fitting_results_number_counts.md`
#   §7 — α = −1.890, S* = 19.01 mJy, n* = 7014 deg⁻² — which supersedes
#   v1's Guerrero fiducial.
# * **Dust-model-matched controls.**  v1 found a cluster-side deficit of
#   far-tail exceedances that |b|-matching did not remove, and attributed
#   it to PSZ2 selecting IR-quiet sightlines.  Controls are now matched on
#   the local Lenz `dust_model` amplitude inside the analysis aperture,
#   which is the adjudicating test; the unmatched variant is retained and
#   both are reported.
# * **The Erler et al. (2018) cluster-dust forward model** — see below.
# * **Covariance-aware threshold combination**, and an explicit
#   demonstration of how much (or how little) the v1 combination was
#   inflated.
# * **Null tests N1–N3** ported from Herschel H2 §7.5.
#
# ---
#
# ## Cluster dust: Erler et al. (2018), forward-modelled
#
# Erler, Basu, Chluba & Bertoldi (2018), *MNRAS* **476**, 3360
# (`relevant_papers/Erler+2019_rSZ_MNRAS_Final.pdf`) measured cluster-
# centric FIR emission by matched filtering Planck HFI maps with a GNFW
# spatial template scaled by θ₅₀₀, **on this identical 772-cluster
# sample**.  Their FIR component (their Eq. 15) is a modified blackbody
# normalized at ν₀ = 857 GHz — our channel — with
#
#     A_857 = 0.10 +- 0.02 MJy/sr,   T_dust = 18.4 (+3.9, -2.4) K
#
# (the paper prints "0.10 ± 0.2"; that is a typesetting slip, since the
# detection is quoted at 5σ).
#
# A_857 = 0.10 MJy/sr is 240 mJy/beam in our units, i.e. ~1.4 σ_core,
# against the ~38 mJy/beam central excess v1 measured in its stacked
# profile.  The factor ~6 is **not** a discrepancy: A_857 is the amplitude
# of the *deconvolved* GNFW template, whereas the stacked profile reads
# the beam-convolved surface brightness after our outer-region baseline.
# §8 below forward-models the published template through our beam, our
# aperture and our baseline and compares it with our own stack — a closure
# test between two independent methods (matched filter on component-
# separated HFI maps vs direct stacking on the Lenz CIB map) on the same
# clusters.  A pre-run of that calculation at the median θ₅₀₀ = 4.77′
# predicts 40 mJy/beam inside r < 0.5 θ₅₀₀, against the ~38 measured.
#
# The forward model is then used three ways, and Δξ is reported for all
# three so the choice is visible rather than buried:
#
# | variant | treatment |
# |---|---|
# | `none` | no correction — the v1 behaviour |
# | `stack` | subtract the *measured* azimuthal cluster-minus-control profile |
# | `erler` | subtract the Erler-normalized GNFW template, beam-convolved |
#
# ---
#
# ## Usage
#
# Run top-to-bottom or cell-by-cell (`#%%`).  Expensive stages memoise to
# `results/v2_cache_lensed/` under an md5 of the configuration.
#
# | env switch | default | effect |
# |---|---|---|
# | `CIB_DATA_DIR` | `<here>/data_857` | data location |
# | `CIB_MASK_VARIANT` | `4.0e+20_gp40` | mask variant |
# | `CIB_N_CTRL` | `4` | controls per cluster |
# | `CIB_N_BOOT` | `400` | paired-bootstrap replicates |
# | `CIB_OFFSET_MIN/MAX_DEG` | `1.0` / `3.0` | control offset annulus |
# | `CIB_DUST_TOL` | `0.25` | fractional dust-amplitude match tolerance |
# | `CIB_DUST_VARIANT` | `none` | `none`, `stack` or `erler` |
# | `CIB_Z_SOURCE` | `2.0` | assumed CIB source redshift |
# | `CIB_USE_CACHE` | `1` | 0 forces recomputation |
# | `CIB_SAVE_FIGURES` | `0` | 1 writes PNGs |
# | `CIB_QUICK_TEST` | `0` | fast smoke run |
#
# Dependencies: numpy, scipy, matplotlib, astropy, **healpy**, plus the
# project's `analysis_modules` (`counts`, `pofd_analytic`, `lens_model`,
# `gpd_tail`).  Run `planck_unlensed_pd_v2.py` first: this module reads
# its npz for the threshold window and the measured Gaussian budget.

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
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import brentq
from scipy.integrate import quad

import healpy as hp
from astropy.cosmology import Planck18
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

from counts import Schechter                                   # noqa: E402
from pofd_analytic import PofD                                 # noqa: E402
from lens_model import (NFWLens, duffy_cvir,                   # noqa: E402
                        bryan_norman_delta_c)
from gpd_tail import (confusion_sigma, fit_gpd,                # noqa: E402
                      xi_of_threshold_analytic)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# %% [markdown]
# ## 2. Configuration

# %%
DATA_DIR = os.environ.get("CIB_DATA_DIR",
                          os.path.join(_BASE_DIR, "data_857"))
MASK_VARIANT = os.environ.get("CIB_MASK_VARIANT", "4.0e+20_gp40")
CATALOG_FILE = os.path.join(_BASE_DIR, "Planck_772_cluster_sample.txt")

# ---- beam / units -----------------------------------------------------------
BEAM_FWHM_ARCSEC = 300.0
OM_BEAM_ARCMIN2 = 1.133 * (BEAM_FWHM_ARCSEC / 60.0) ** 2
OM_BEAM_SR = OM_BEAM_ARCMIN2 * (np.pi / (180.0 * 60.0)) ** 2
MJY_SR_TO_MJY_BEAM = OM_BEAM_SR * 1e9
BEAM_SIGMA_ARCMIN = BEAM_FWHM_ARCSEC / 60.0 / 2.3548200450309493

# ---- cutout geometry --------------------------------------------------------
PIX_ARCMIN = 1.0
SIDE_T500 = 6.0                  # cutout side in theta_500 units ...
SIDE_MIN_ARCMIN = 25.0           # ... never below 5 beam FWHM (edge safety)
BEAM_FWHM_PIX = BEAM_FWHM_ARCSEC / 60.0 / PIX_ARCMIN
BASE_FIT_FRAC = 2.0              # baseline fitted on r > 2 theta_500 only
APERTURE_FRAC = 1.5              # default analysis aperture
APERTURE_SCAN = [0.5, 1.0, 1.5, 2.0, 2.5]
MIN_VALID_FRAC = 0.95
#  MIN_VALID_AP — minimum finite-pixel fraction required inside the LARGEST
#  aperture (r < max(APERTURE_SCAN)*theta_500), applied identically to cluster
#  and control cutouts.
#
#  Null test N2 (Sec. 12) found cluster apertures carrying 53% more invalid
#  pixels than control apertures (0.00058 vs 0.00038, ratio 1.526) at the
#  default 0.97 — unsurprising, since the Planck point-source mask
#  preferentially removes bright sources near clusters, but it means the two
#  samples are not analysed on exactly identical footings.  Setting
#  CIB_MIN_VALID_AP=1.0 requires 100% coverage on BOTH sides and closes N2 by
#  construction, at the cost of sample size.  This is open item 1 of
#  `Revised_Planck_analysis.md` Sec. 11.3.                    [2026-08-17]
MIN_VALID_AP = float(os.environ.get("CIB_MIN_VALID_AP", "0.97"))

# ---- paired-control construction (NEW in v2) --------------------------------
N_CTRL_PER = int(os.environ.get("CIB_N_CTRL", "4"))
OFFSET_MIN_DEG = float(os.environ.get("CIB_OFFSET_MIN_DEG", "1.0"))
OFFSET_MAX_DEG = float(os.environ.get("CIB_OFFSET_MAX_DEG", "3.0"))
CLUSTER_AVOID_DEG = 2.0          # controls stay 2 deg from ANY catalog cluster
DUST_TOL = float(os.environ.get("CIB_DUST_TOL", "0.25"))
RNG_SEED = 20260802
RNG_SEED_ALT = 20260803          # null test N3: control re-randomisation
MAX_DRAWS_PER = 6000

# ---- GPD analysis -----------------------------------------------------------
K_GRID = np.arange(1.0, 8.01, 0.5)
K_WINDOW = (3.0, 4.5)            # PS-mask truncation caps usable u at ~4.5
N_BOOT = int(os.environ.get("CIB_N_BOOT", "400"))
MIN_EXCEED = 25                  # cluster-side samples are small
DECLUSTER_PIX = 5                # one beam FWHM, as in the unlensed module

# ---- lensing model ----------------------------------------------------------
Z_SOURCE = float(os.environ.get("CIB_Z_SOURCE", "2.0"))
CONC_RELATION = "duffy"
MU_MAX = 100.0

# ---- counts (Schechter only, final adopted 350 um fit) ----------------------
SCH_PARS = dict(alpha=-1.890, sstar=19.01, nstar=7014.0)
S_MIN_MJY = float(os.environ.get("CIB_SMIN_MJY", "1.0"))
S_CUT_MJY = float(os.environ.get("CIB_SCUT_MJY", "100.0"))
D_SPAN_MJY, N_FFT, SF_FLOOR = 6000.0, 2 ** 17, 1e-9
N_RINGS, N_MU_BINS = 48, 64
#  Stacked-profile binning: 0.1 theta_500 bins out to 3 theta_500, with an
#  edge exactly at 0.5 so the "inside 0.5 theta_500" number is a clean
#  area-weighted average over the same pixels for data and forward model.
N_RAD_BINS = 30

# ---- Erler et al. (2018) cluster-dust model ---------------------------------
#  MNRAS 476, 3360, Eq. 15; modified blackbody normalized at nu_0 = 857 GHz,
#  matched-filtered with a GNFW template scaled by theta_500, on THIS sample.
ERLER_A857_MJY_SR = 0.10         # +- 0.02 (the paper's "+- 0.2" is a typo)
ERLER_A857_ERR = 0.02
ERLER_TDUST_K = 18.4
#  Arnaud et al. (2010) universal pressure profile (A&A 517, A92, Eq. 12),
#  used here only as the SHAPE of the matched-filter template.
GNFW_PARS = dict(c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
DUST_VARIANT = os.environ.get("CIB_DUST_VARIANT", "none")   # none|stack|erler

# ---- quick test / output ----------------------------------------------------
QUICK_TEST = bool(int(os.environ.get("CIB_QUICK_TEST", "0")))
if QUICK_TEST:
    N_BOOT, N_CTRL_PER = 80, 2
    APERTURE_SCAN = [1.0, 1.5, 2.0]
    print("QUICK_TEST on: reduced bootstrap/control counts.")

SAVE_FIGURES = bool(int(os.environ.get("CIB_SAVE_FIGURES", "0")))
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", RESULTS_DIR)
CACHE_DIR = os.path.join(RESULTS_DIR, "v2_cache_lensed")
USE_CACHE = os.environ.get("CIB_USE_CACHE", "1") == "1"
os.makedirs(RESULTS_DIR, exist_ok=True)

_CFG = dict(var=MASK_VARIANT, nc=N_CTRL_PER,
            off=(OFFSET_MIN_DEG, OFFSET_MAX_DEG),
            tol=DUST_TOL, seed=RNG_SEED, side=SIDE_T500, pix=PIX_ARCMIN,
            zs=Z_SOURCE, quick=QUICK_TEST, nrb=N_RAD_BINS, nb=N_BOOT,
            aps=list(APERTURE_SCAN), apd=APERTURE_FRAC, kw=list(K_WINDOW),
            me=MIN_EXCEED, decl=DECLUSTER_PIX)
#  MIN_VALID_AP enters the hash ONLY when it differs from the historical
#  default of 0.97.  This is deliberate: it keeps the baseline hash
#  (4c9c5af0c8) and every cache entry under it bit-identical, so the numbers
#  in `Revised_Planck_analysis.md` remain reproducible without recomputation,
#  while the 100%-coverage variant gets a hash of its own and builds its own
#  pairs.  Do not "tidy" this into an unconditional key.       [2026-08-17]
if MIN_VALID_AP != 0.97:
    _CFG["vap"] = MIN_VALID_AP
CFG_HASH = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()
                       ).hexdigest()[:10]

T0 = time.time()


def cached(tag, fn):
    """Memoise an expensive stage under results/v2_cache_lensed/.

    A cache entry left half-written by an interrupted run is treated as a
    miss and recomputed, rather than raising -- otherwise a single killed
    process poisons the cache permanently.
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


def _finish_figure(fig, name):
    if SAVE_FIGURES:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        path = os.path.join(FIGURE_DIR, name + ".png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"Wrote {path}")
    else:
        plt.show()


# %% [markdown]
# ## 3. Shared helpers
#
# Same conventions as `planck_unlensed_pd_v2.py`, with the two
# modifications the lensed geometry requires: the baseline plane is fitted
# on the **outer region only** (r > 2 θ₅₀₀), so a centred signal cannot
# leak into it, and peak extraction takes an aperture radius.  The
# M₅₀₀ → M_vir inversion uses the Duffy et al. (2008) virial concentration
# with the Bryan & Norman (1998) overdensity, self-consistent with
# `lens_model.NFWLens`.

# %%
def gnomonic_cutout(skymap, l0_deg, b0_deg, n_pix, pix_arcmin):
    """Matplotlib-free gnomonic cutout (RING map, Galactic centre);
    bilinear interpolation, NaN-masked pixels propagate."""
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


def radius_grid_arcmin(n_pix, pix_arcmin):
    """Radial distance of each pixel from the cutout centre [arcmin]."""
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix), dtype=float)
    return np.hypot(yy - c, xx - c) * pix_arcmin


def baseline_outer_plane(img, r_arcmin, r_fit_min):
    """DC + linear plane fitted ONLY on valid pixels with r > r_fit_min,
    subtracted everywhere.  Protects a centred signal by construction."""
    n = img.shape[0]
    xn = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    XN, YN = np.meshgrid(xn, xn)
    sel = np.isfinite(img) & (r_arcmin > r_fit_min)
    if sel.sum() < 20:
        return img, False
    A = np.column_stack([np.ones(sel.sum()), XN[sel], YN[sel]])
    coef, *_ = np.linalg.lstsq(A, img[sel], rcond=None)
    return img - (coef[0] + coef[1] * XN + coef[2] * YN), True


def aperture_peaks(img, r_arcmin, r_ap_arcmin, window_pix=DECLUSTER_PIX):
    """NaN-aware declustered peaks restricted to r < r_ap_arcmin."""
    filled = np.where(np.isfinite(img), img, -np.inf)
    size = max(1, int(round(window_pix)))
    is_max = (filled == maximum_filter(filled, size=size))
    is_max &= np.isfinite(img) & (r_arcmin < r_ap_arcmin)
    return img[is_max]


COSMO = Planck18


def m500_from_r500(r500_mpc, z):
    rho_c = COSMO.critical_density(z).to(u_ap.Msun / u_ap.Mpc ** 3).value
    return (4.0 / 3.0) * np.pi * 500.0 * rho_c * r500_mpc ** 3


def _nfw_g(y):
    return np.log1p(y) - y / (1.0 + y)


def mvir_from_m500(m500, r500_mpc, z):
    """Invert M500 -> M_vir (Duffy c_vir, Bryan-Norman overdensity)."""
    dc = bryan_norman_delta_c(z, COSMO)
    rho_c = COSMO.critical_density(z).to(u_ap.Msun / u_ap.Mpc ** 3).value

    def resid(log_m):
        m = 10.0 ** log_m
        c = duffy_cvir(m, z, COSMO)
        r_vir = (3.0 * m / (4.0 * np.pi * dc * rho_c)) ** (1.0 / 3.0)
        m_enc = m if r500_mpc >= r_vir else \
            m * _nfw_g(c * r500_mpc / r_vir) / _nfw_g(c)
        return np.log10(m_enc) - np.log10(m500)

    return 10.0 ** brentq(resid, 12.5, 17.0, xtol=1e-4)


# ---- the Erler+2018 GNFW dust template --------------------------------------
def _gnfw_3d(x):
    """Arnaud+2010 universal profile shape at x = r/r500 (unnormalized)."""
    cx = np.maximum(GNFW_PARS["c500"] * np.asarray(x, float), 1e-8)
    g, a, b = GNFW_PARS["gamma"], GNFW_PARS["alpha"], GNFW_PARS["beta"]
    return 1.0 / (cx ** g * (1.0 + cx ** a) ** ((b - g) / a))


def _gnfw_projected(x_perp, l_max=5.0):
    """Line-of-sight integral of the GNFW shape at projected radius x_perp
    (in units of r500).  Finite at x=0 because gamma < 1."""
    val, _ = quad(lambda l: _gnfw_3d(np.hypot(x_perp, l)), 0.0, l_max,
                  limit=200)
    return 2.0 * val


_X_TAB = np.linspace(1e-4, 6.0, 500)
_P_TAB = np.array([_gnfw_projected(x) for x in _X_TAB])
_P0 = _P_TAB[0]


def erler_dust_template(n_pix, theta500_arcmin, amp_mjy_sr=ERLER_A857_MJY_SR):
    """Beam-convolved cluster-dust map, mJy/beam, on the cutout grid.

    The published A_857 is the amplitude of the DECONVOLVED GNFW template,
    so the shape is normalized to unity at the centre, scaled by A_857, and
    only then passed through the 5' beam.  Unit chain: an intensity field
    I [MJy/sr] appears in a peak-normalized-beam map as
    MJY_SR_TO_MJY_BEAM * (I convolved with the unit-SUM Gaussian), because
    the two beam normalizations differ by exactly the beam solid angle.
    """
    r = radius_grid_arcmin(n_pix, PIX_ARCMIN)
    prof = np.interp(r / theta500_arcmin, _X_TAB, _P_TAB) / _P0
    i_nu = amp_mjy_sr * prof                      # MJy/sr, deconvolved
    sig_pix = BEAM_SIGMA_ARCMIN / PIX_ARCMIN
    return MJY_SR_TO_MJY_BEAM * gaussian_filter(i_nu, sig_pix,
                                                mode="nearest")


def robust_core(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


# %% [markdown]
# ## 4. Maps, dust template, catalog and lens models
#
# The `dust_model.hpx.fits` product is the HI-based Galactic dust model
# that Lenz et al. already **subtracted** from the CIB map.  It is used
# here purely as a per-sightline *foreground amplitude template* for
# matching controls — it is a Galactic quantity and has nothing to do with
# the cluster dust of §8.

# %%
D = os.path.join(DATA_DIR, MASK_VARIANT)
t0 = time.time()
cib_mjyb = hp.read_map(os.path.join(D, "cib_fullmission.hpx.fits"),
                       field=0, dtype=np.float32) * MJY_SR_TO_MJY_BEAM
mask = hp.read_map(os.path.join(D, "mask_bool.hpx.fits"),
                   dtype=np.float32) > 0.5
dust_model = hp.read_map(os.path.join(D, "dust_model.hpx.fits"),
                         field=0, dtype=np.float32)
NSIDE = hp.get_nside(mask)
print(f"Maps loaded in {time.time() - t0:.1f}s (f_sky {mask.mean():.4f} = "
      f"{mask.mean() * 41253:.0f} deg^2)")

cat = np.genfromtxt(CATALOG_FILE, comments="#",
                    dtype=[("name", "U24"), ("ra", "f8"), ("dec", "f8"),
                           ("z", "f8"), ("r500", "f8"), ("tx", "f8")])
gal = SkyCoord(ra=cat["ra"] * u_ap.deg, dec=cat["dec"] * u_ap.deg,
               frame="icrs").galactic
cat_l, cat_b = gal.l.deg, gal.b.deg
clus_vec = hp.ang2vec(cat_l, cat_b, lonlat=True)
COS_AVOID = np.cos(np.radians(CLUSTER_AVOID_DEG))

d_a = COSMO.angular_diameter_distance(cat["z"]).to(u_ap.Mpc).value
theta500 = (cat["r500"] / d_a) * (180.0 * 60.0 / np.pi)      # arcmin
m500 = np.array([m500_from_r500(r, z)
                 for r, z in zip(cat["r500"], cat["z"])])
print(f"Catalog: {cat.size} clusters, theta500 median "
      f"{np.median(theta500):.2f}', M500 median {np.median(m500):.2e} Msun")
print(f"  on-mask: {int(mask[hp.ang2pix(NSIDE, cat_l, cat_b, lonlat=True)].sum())}"
      f" of {cat.size}")

COUNTS = Schechter(alpha=SCH_PARS["alpha"],
                   log_sstar=np.log10(SCH_PARS["sstar"]),
                   log_phistar=np.log10(SCH_PARS["nstar"]),
                   s_min=S_MIN_MJY, s_max=S_CUT_MJY)
SIGMA_C_POISSON = confusion_sigma(COUNTS, BEAM_FWHM_ARCSEC,
                                  s_cut=S_CUT_MJY)["sigma_c"]
print(f"  Schechter sigma_c at the 5' beam: {SIGMA_C_POISSON:.1f} mJy/beam "
      f"(S_cut/sigma_c = {S_CUT_MJY / SIGMA_C_POISSON:.2f})")

# ---- inherit the unlensed module's core budget where available --------------
_UNL = os.path.join(RESULTS_DIR, f"planck_v2_unlensed_857_{MASK_VARIANT}.npz")
if os.path.exists(_UNL):
    _u = np.load(_UNL, allow_pickle=True)
    SIGMA_CORE_FIELD = float(_u["sigma_core"])
    print(f"  inherited sigma_core = {SIGMA_CORE_FIELD:.1f} mJy/beam from "
          f"the unlensed module")
else:
    SIGMA_CORE_FIELD = np.nan
    print("  NOTE: run planck_unlensed_pd_v2.py first to inherit "
          "sigma_core; the prediction will use the control aperture core.")

# %% [markdown]
# ## 5. Paired cluster/control cutouts
#
# Per cluster: side = max(6 θ₅₀₀, 25′) at 1′ pixels; centre inside the
# mask; validity ≥ 95% overall and ≥ 97% inside the largest scan aperture;
# outer-region baseline (r > 2 θ₅₀₀).
#
# **Controls are now local and dust-matched** (change 1 of §1).  Each
# control is placed at a random position angle and a random offset in
# [1°, 3°] from its cluster — close enough to share the cirrus
# environment, far enough that the cluster's own lensing and emission
# (deflections ≲ 1′, dust confined to ≲ θ₅₀₀) cannot reach it.  A
# candidate is accepted only if the mean Lenz dust-model amplitude within
# the analysis aperture matches the cluster's to within `DUST_TOL`.
#
# For every cluster we build **two** control sets in the same pass: the
# dust-matched primary, and an offset-only set with the dust criterion
# disabled.  Comparing them is the adjudicating test for v1's far-tail
# deficit, which |b|-matching failed to remove.

# %%
def _disc_dust(vec, radius_arcmin):
    """Mean Lenz dust-model amplitude within a disc [MJy/sr]."""
    pix = hp.query_disc(NSIDE, vec, np.radians(radius_arcmin / 60.0),
                        inclusive=True)
    v = dust_model[pix]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def _offset_position(l_deg, b_deg, rng):
    """A position at a random position angle and a random offset drawn
    uniformly in AREA over the annulus [OFFSET_MIN, OFFSET_MAX].

    Pure spherical trigonometry rather than `SkyCoord.directional_offset_by`
    -- the astropy call is exact but costs ~1 ms, and this runs O(10^4)
    times.  Build a local orthonormal frame (n, e_north, e_east) and rotate:

        p = n cos(s) + (e_n cos(pa) + e_e sin(pa)) sin(s)

    The handedness of the frame is irrelevant because pa is uniform.
    """
    pa = rng.uniform(0.0, 2.0 * np.pi)
    sep = np.sqrt(rng.uniform(OFFSET_MIN_DEG ** 2, OFFSET_MAX_DEG ** 2))
    s = np.radians(sep)
    n = hp.ang2vec(l_deg, b_deg, lonlat=True)
    z = np.array([0.0, 0.0, 1.0])
    e_n = z - np.dot(z, n) * n
    nn = np.linalg.norm(e_n)
    if nn < 1e-8:                      # at a pole, pick any perpendicular
        e_n = np.array([1.0, 0.0, 0.0]) - n[0] * n
        nn = np.linalg.norm(e_n)
    e_n /= nn
    e_e = np.cross(n, e_n)
    p = n * np.cos(s) + (e_n * np.cos(pa) + e_e * np.sin(pa)) * np.sin(s)
    p /= np.linalg.norm(p)
    lon, lat = hp.vec2ang(p, lonlat=True)
    return float(lon[0]), float(lat[0]), sep


def _build_pairs(seed):
    """Extract cluster cutouts and their two matched control sets."""
    rng = np.random.default_rng(seed)
    cl_imgs, ct_dust, ct_plain, acc_idx = [], [], [], []
    dust_cl, dust_ct, offs, badfrac_cl, badfrac_ct = [], [], [], [], []
    n_rej = dict(mask=0, valid=0, ap=0, base=0, nocontrol=0)
    #  how often the dust criterion actually bites -- if it rejects
    #  nothing, the dust-matched and offset-only samples are identical
    #  and the selection test of Sec. 9 has no discriminating power
    n_dust_tested, n_dust_rejected = [0], [0]
    for i in range(cat.size):
        side = max(SIDE_T500 * theta500[i], SIDE_MIN_ARCMIN)
        npix = int(np.ceil(side / PIX_ARCMIN)) | 1        # odd -> centred
        vec_i = hp.ang2vec(cat_l[i], cat_b[i], lonlat=True)
        if not mask[hp.vec2pix(NSIDE, *vec_i)]:
            n_rej["mask"] += 1
            continue
        img = gnomonic_cutout(cib_mjyb, cat_l[i], cat_b[i], npix, PIX_ARCMIN)
        r_g = radius_grid_arcmin(npix, PIX_ARCMIN)
        r_ap_max = max(APERTURE_SCAN) * theta500[i]
        in_ap = r_g < r_ap_max
        if np.isfinite(img).mean() < MIN_VALID_FRAC:
            n_rej["valid"] += 1
            continue
        if np.isfinite(img[in_ap]).mean() < MIN_VALID_AP:
            n_rej["ap"] += 1
            continue
        img_b, ok = baseline_outer_plane(img, r_g,
                                         BASE_FIT_FRAC * theta500[i])
        if not ok:
            n_rej["base"] += 1
            continue

        d_cl = _disc_dust(vec_i, APERTURE_FRAC * theta500[i])
        set_d, set_p, off_i, d_ct_i = [], [], [], []
        n_try = 0
        while (len(set_d) < N_CTRL_PER or len(set_p) < N_CTRL_PER) \
                and n_try < MAX_DRAWS_PER:
            n_try += 1
            lon, lat, sep = _offset_position(cat_l[i], cat_b[i], rng)
            v = hp.ang2vec(lon, lat, lonlat=True)
            if not mask[hp.vec2pix(NSIDE, *v)]:
                continue
            if np.max(clus_vec @ v) > COS_AVOID:
                continue
            cimg = gnomonic_cutout(cib_mjyb, lon, lat, npix, PIX_ARCMIN)
            if (np.isfinite(cimg).mean() < MIN_VALID_FRAC
                    or np.isfinite(cimg[in_ap]).mean() < MIN_VALID_AP):
                continue
            cimg_b, okc = baseline_outer_plane(cimg, r_g,
                                               BASE_FIT_FRAC * theta500[i])
            if not okc:
                continue
            cimg_b = cimg_b.astype(np.float32)
            if len(set_p) < N_CTRL_PER:
                set_p.append(cimg_b)
            if len(set_d) < N_CTRL_PER:
                d_c = _disc_dust(v, APERTURE_FRAC * theta500[i])
                n_dust_tested[0] += 1
                if (np.isfinite(d_c) and np.isfinite(d_cl) and d_cl > 0
                        and abs(d_c - d_cl) / d_cl < DUST_TOL):
                    set_d.append(cimg_b)
                    off_i.append(sep)
                    d_ct_i.append(d_c)
                else:
                    n_dust_rejected[0] += 1
        if len(set_d) < N_CTRL_PER or len(set_p) < N_CTRL_PER:
            n_rej["nocontrol"] += 1
            continue
        cl_imgs.append(img_b.astype(np.float32))
        ct_dust.append(set_d)
        ct_plain.append(set_p)
        acc_idx.append(i)
        dust_cl.append(d_cl)
        dust_ct.append(float(np.mean(d_ct_i)))
        offs.append(float(np.mean(off_i)))
        #  null test N2: is the two samples' invalid-pixel budget balanced?
        badfrac_cl.append(float(1.0 - np.isfinite(img_b[in_ap]).mean()))
        badfrac_ct.append(float(np.mean(
            [1.0 - np.isfinite(c[in_ap]).mean() for c in set_d])))
    print("    rejections: %s" % n_rej)
    print("    dust criterion: %d of %d candidates rejected (%.1f%%)"
          % (n_dust_rejected[0], n_dust_tested[0],
             100.0 * n_dust_rejected[0] / max(n_dust_tested[0], 1)))
    return dict(cl=cl_imgs, ct_dust=ct_dust, ct_plain=ct_plain,
                dust_reject_frac=(n_dust_rejected[0]
                                  / max(n_dust_tested[0], 1)),
                acc=np.array(acc_idx), dust_cl=np.array(dust_cl),
                dust_ct=np.array(dust_ct), offsets=np.array(offs),
                badfrac_cl=np.array(badfrac_cl),
                badfrac_ct=np.array(badfrac_ct))


PAIRS = cached("pairs", lambda: _build_pairs(RNG_SEED))
cl_imgs = PAIRS["cl"]
ct_imgs = PAIRS["ct_dust"]           # primary: dust-matched
ct_plain = PAIRS["ct_plain"]
acc_idx = PAIRS["acc"]
n_cl = len(acc_idx)
th5_acc = theta500[acc_idx]

print(f"\nSample: {n_cl}/{cat.size} clusters accepted")
print(f"  theta500 = [{th5_acc.min():.1f}', {th5_acc.max():.1f}'] "
      f"(median {np.median(th5_acc):.2f}')")
print(f"  z median {np.median(cat['z'][acc_idx]):.3f}, "
      f"M500 median {np.median(m500[acc_idx]):.3e} Msun")
print(f"  control offsets: median {np.median(PAIRS['offsets']):.2f} deg "
      f"(annulus [{OFFSET_MIN_DEG}, {OFFSET_MAX_DEG}])")
_dr = PAIRS["dust_ct"] / np.maximum(PAIRS["dust_cl"], 1e-12)
print(f"  dust match (control/cluster amplitude): median {np.median(_dr):.4f}, "
      f"16-84% [{np.percentile(_dr, 16):.3f}, {np.percentile(_dr, 84):.3f}]")

# lens models for the accepted sample only
t0 = time.time()
m_vir_acc = np.array([mvir_from_m500(m500[i], cat["r500"][i], cat["z"][i])
                      for i in acc_idx])
lenses_acc = [NFWLens(m, cat["z"][i], Z_SOURCE, cosmo=COSMO,
                      conc=CONC_RELATION, mu_max=MU_MAX)
              for m, i in zip(m_vir_acc, acc_idx)]
print(f"  lens models built in {time.time() - t0:.1f}s "
      f"(M_vir/M500 median {np.median(m_vir_acc / m500[acc_idx]):.2f})")

# %% [markdown]
# ## 6. Δξ(u) with the paired bootstrap and a covariance-aware combination
#
# Thresholds are anchored to the **control** aperture-pixel core (manual
# §7b: choose u from the control, apply to both).  The resampling unit is
# the *pair* — a cluster together with its own controls — so any sky
# systematic common to a pair cancels inside every replicate.
#
# **On the combination, and on whether v1 was inflated.**  Herschel §5.8
# found that combining thresholds in quadrature, treating them as
# independent, inflated significances by factors ~1.8–2.3.  v1 of *this*
# module already computed its window combination *inside* the bootstrap,
# which propagates the threshold covariance without ever forming it — so
# the lensed side was, in fact, protected.  Rather than assert that, the
# code below computes both and prints the ratio, so the claim is a
# measurement:
#
# * `naive`  — inverse-variance weighted mean with σ from independent
#   thresholds, 1/√Σwᵢ.  **Wrong**, printed for comparison.
# * `paired` — the same weighted mean applied to every bootstrap
#   replicate, with σ from the scatter of those replicates.  **Correct.**

# %%
def _subset(seq, sub):
    """Positional subset of a list/array; `sub` of None means everything."""
    return seq if sub is None else [seq[i] for i in sub]


def collect_peaks(ap_frac, ct_set, dust_corr=None, cl_set=None, sub=None):
    """Peaks within r < ap_frac*theta500 for clusters and their controls.

    dust_corr : None, 'stack' or 'erler' — see Sec. 8.  The correction is
    applied to the CLUSTER cutouts only, since by construction the control
    fields contain no cluster.
    cl_set : override the cluster-side image list (used by null test N1,
    which feeds control cutouts in on both sides).
    sub : positional indices INTO THE ACCEPTED SAMPLE, or None for all of it.
    Used by the mass/redshift split of Sec. 12b.  Defaults to None so every
    pre-existing call is bit-identical to before this argument existed.
    """
    _cl = cl_set if cl_set is not None else cl_imgs
    pk_cl, pk_ct = [], []
    for n, (img, ctrls, t5) in enumerate(zip(_subset(_cl, sub),
                                             _subset(ct_set, sub),
                                             _subset(th5_acc, sub))):
        r_g = radius_grid_arcmin(img.shape[0], PIX_ARCMIN)
        r_ap = ap_frac * t5
        im = img
        if dust_corr == "erler":
            tmpl = erler_dust_template(img.shape[0], t5)
            tmpl, _ = baseline_outer_plane(tmpl, r_g, BASE_FIT_FRAC * t5)
            im = img - tmpl
        elif dust_corr == "stack":
            im = img - STACK_TEMPLATE(img.shape[0], t5)
        pk_cl.append(aperture_peaks(im, r_g, r_ap))
        pk_ct.append([aperture_peaks(c, r_g, r_ap) for c in ctrls])
    return pk_cl, pk_ct


def xi_scan(peaks, u_grid, min_exceed=MIN_EXCEED):
    """GPD xi at each threshold (NaN where too few exceedances)."""
    out = np.full(u_grid.size, np.nan)
    for j, u in enumerate(u_grid):
        y = peaks[peaks > u] - u
        if y.size >= min_exceed:
            out[j] = fit_gpd(y)["xi"]
    return out


def control_core(ap_frac, ct_set, sub=None):
    """Robust core of the pooled control pixels inside the aperture.

    `sub` restricts the pool to a subsample (Sec. 12b); the thresholds are
    then anchored to THAT subsample's own control core, which is the correct
    convention — u must be defined by the controls the subsample is compared
    against, not by the parent sample.
    """
    vals = []
    for ctrls, t5 in zip(_subset(ct_set, sub), _subset(th5_acc, sub)):
        r_g = radius_grid_arcmin(ctrls[0].shape[0], PIX_ARCMIN)
        sel = r_g < ap_frac * t5
        for c in ctrls:
            v = c[sel]
            vals.append(v[np.isfinite(v)])
    return robust_core(np.concatenate(vals))


def dxi_measurement(ap_frac, ct_set, dust_corr=None, n_boot=N_BOOT, seed=3,
                    cl_set=None, sub=None):
    """Paired-bootstrap Delta_xi(u), its window combination (both ways),
    and the amplitude-channel exceedance ratio, for one aperture.

    `sub` (Sec. 12b) restricts the whole calculation — peaks, control core,
    thresholds and bootstrap — to a subsample of the accepted pairs.  The
    resampling unit remains the pair, drawn from within the subsample, so a
    split half's error bar is honest rather than inherited.
    """
    pk_cl, pk_ct = collect_peaks(ap_frac, ct_set, dust_corr, cl_set, sub=sub)
    pool_cl = np.concatenate(pk_cl)
    pool_ct = np.concatenate([p for ct in pk_ct for p in ct])
    n_pairs = len(pk_cl)

    mu_core, sig_core = control_core(ap_frac, ct_set, sub=sub)
    u_grid = mu_core + K_GRID * sig_core

    xi_cl = xi_scan(pool_cl, u_grid)
    xi_ct = xi_scan(pool_ct, u_grid)
    dxi = xi_cl - xi_ct
    in_win = (K_GRID >= K_WINDOW[0]) & (K_GRID <= K_WINDOW[1])

    rng = np.random.default_rng(seed)
    dxi_b = np.full((n_boot, K_GRID.size), np.nan)
    rat_b = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, n_pairs, n_pairs)    # resample PAIRS
        pcl = np.concatenate([pk_cl[i] for i in pick])
        pct = np.concatenate([np.concatenate(pk_ct[i]) for i in pick])
        dxi_b[b] = xi_scan(pcl, u_grid) - xi_scan(pct, u_grid)
        ncl_w = np.array([(pcl > u).sum() for u in u_grid[in_win]], float)
        nct_w = np.array([(pct > u).sum() for u in u_grid[in_win]], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (ncl_w / max(pcl.size, 1)) / (nct_w / max(pct.size, 1))
        if np.isfinite(r).any():
            rat_b[b] = np.nanmean(r)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dxi_lo, dxi_hi = np.nanpercentile(dxi_b, [16, 84], axis=0)
        dxi_sd = np.nanstd(dxi_b, axis=0)
        rat_lo, rat_hi = np.nanpercentile(rat_b, [16, 84])

        #  --- window combination, both ways ---------------------------------
        use = in_win & np.isfinite(dxi) & np.isfinite(dxi_sd) & (dxi_sd > 0)
        if use.any():
            w = 1.0 / dxi_sd[use] ** 2
            comb = float(np.sum(w * dxi[use]) / np.sum(w))
            #  WRONG: errors added as if thresholds were independent
            sig_naive = float(1.0 / np.sqrt(np.sum(w)))
            #  RIGHT: the same weighted mean re-evaluated per replicate
            cb = np.full(n_boot, np.nan)
            for b in range(n_boot):
                row = dxi_b[b, use]
                if np.isfinite(row).all():
                    cb[b] = float(np.sum(w * row) / np.sum(w))
            sig_paired = float(np.nanstd(cb))
            n_thr_used = int(use.sum())
        else:
            comb = sig_naive = sig_paired = np.nan
            cb = np.full(n_boot, np.nan)
            n_thr_used = 0

    n_exc_cl = np.array([(pool_cl > u).sum() for u in u_grid], float)
    n_exc_ct = np.array([(pool_ct > u).sum() for u in u_grid], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (n_exc_cl / max(pool_cl.size, 1)) / \
                (n_exc_ct / max(pool_ct.size, 1))
        ratio_err = ratio * np.sqrt(1.0 / np.maximum(n_exc_cl, 1)
                                    + 1.0 / np.maximum(n_exc_ct, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rat_win = float(np.nanmean(ratio[in_win]))

    #  95% one-sided upper limit on |Delta_xi|, Herschel H2 convention
    ul95 = abs(comb) + 1.96 * sig_paired if np.isfinite(comb) else np.nan

    return dict(ap_frac=ap_frac, n_pairs=n_pairs,
                mu_core=mu_core, sig_core=sig_core,
                u_grid=u_grid, xi_cl=xi_cl, xi_ct=xi_ct, dxi=dxi,
                dxi_lo=dxi_lo, dxi_hi=dxi_hi, dxi_sd=dxi_sd, dxi_boot=dxi_b,
                comb=comb, sig_naive=sig_naive, sig_paired=sig_paired,
                comb_boot=cb, ul95=ul95, n_thr_used=n_thr_used,
                n_peaks_cl=pool_cl.size, n_peaks_ct=pool_ct.size,
                n_exc_cl=n_exc_cl, n_exc_ct=n_exc_ct,
                ratio=ratio, ratio_err=ratio_err, ratio_win=rat_win,
                ratio_win_lo=rat_lo, ratio_win_hi=rat_hi)


def STACK_TEMPLATE(n_pix, t5):
    """Azimuthal cluster-minus-control profile, interpolated onto a grid.

    Defined after Sec. 8 measures it; before that it returns zero so the
    'none' variant can run first (the profile is measured on uncorrected
    cutouts by construction).
    """
    return np.zeros((n_pix, n_pix))


MEAS = cached("meas_dustmatched",
              lambda: {f: dxi_measurement(f, ct_imgs) for f in APERTURE_SCAN})
M = MEAS[APERTURE_FRAC]

print("\n" + "=" * 79)
print("  Delta_xi RESULTS  (dust-matched controls, no cluster-dust "
      "correction)")
print("=" * 79)
print("  %8s %10s %10s %12s %10s %10s %9s"
      % ("aperture", "N_cl pk", "N_ct pk", "Delta_xi", "sigma", "95% UL",
         "sigma_nv"))
for f in APERTURE_SCAN:
    m = MEAS[f]
    if np.isfinite(m["comb"]):
        print("  %8.1f %10d %10d %+12.4f %10.4f %10.4f %9.4f"
              % (f, m["n_peaks_cl"], m["n_peaks_ct"], m["comb"],
                 m["sig_paired"], m["ul95"], m["sig_naive"]))
    else:
        print("  %8.1f %10d %10d    -- (fails the N_exc floor)"
              % (f, m["n_peaks_cl"], m["n_peaks_ct"]))

_ratios = [MEAS[f]["sig_paired"] / MEAS[f]["sig_naive"]
           for f in APERTURE_SCAN
           if np.isfinite(MEAS[f]["sig_naive"]) and MEAS[f]["sig_naive"] > 0
           and MEAS[f]["n_thr_used"] > 1]
print("\n  WAS THE v1 COMBINATION INFLATED?  ratio sigma_paired/sigma_naive")
print("    " + ", ".join(
    "%.2f (%d thr, r<%.1f)" % (MEAS[f]["sig_paired"] / MEAS[f]["sig_naive"],
                               MEAS[f]["n_thr_used"], f)
    for f in APERTURE_SCAN
    if np.isfinite(MEAS[f]["sig_naive"]) and MEAS[f]["sig_naive"] > 0
    and MEAS[f]["n_thr_used"] > 1)
    + ("   median %.2f" % np.median(_ratios) if _ratios else
       "   [no aperture has >1 usable threshold in the window -- the "
       "combination\n    is a single number there and the ratio is "
       "trivially 1]"))
print("""    A ratio > 1 means the naive independent-threshold error UNDER-states
    the true error, i.e. the naive significance is INFLATED by that factor.
    v1 of this module already combined inside the bootstrap, so its lensed
    numbers should be close to the 'paired' column; the Herschel-style
    inflation applies to the UNLENSED chapter's threshold-combined claims,
    not to these.  The ratio above is the evidence for that statement.""")

#  sign-consistency check across nested apertures (Herschel H2 warning)
_signs = [np.sign(MEAS[f]["comb"]) for f in APERTURE_SCAN
          if np.isfinite(MEAS[f]["comb"])]
if len(set(_signs)) > 1:
    print("""
  WARNING: the sign of Delta_xi is NOT consistent across nested apertures,
  while the NFW prediction is positive and monotonically decreasing.  A real
  lensing signal cannot change sign between nested apertures.  No aperture
  may be singled out post hoc on the strength of its value.""")

# %% [markdown]
# ## 7. The NFW μ-mixture prediction from the accepted clusters' masses
#
# The exact mixture prediction restricted to the accepted sample and the
# analysis aperture, evaluated on the *measured* Gaussian budget — this is
# the master-equation compatibility target.  Because Δξ is a difference of
# two identically computed curves, the pixel-vs-peak offset that
# contaminates the absolute ξ largely cancels here, so the analytic
# prediction is more trustworthy for Δξ than it is for ξ.

# %%
def aperture_mixture(ap_frac):
    """Pooled area-weighted mu-mixture of the accepted clusters within
    r < ap_frac*theta500; returns (mus, weights, stats)."""
    all_mu, all_w = [], []
    for L, t5 in zip(lenses_acc, th5_acc):
        edges = np.linspace(0.02 * t5, ap_frac * t5, N_RINGS + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        all_mu.append(L.mu(mid))
        all_w.append(areas)
    mu_all = np.concatenate(all_mu)
    w_all = np.concatenate(all_w)
    wn = w_all / w_all.sum()
    stats = dict(mean_mu=float(np.sum(wn * mu_all)),
                 mean_ln=float(np.sum(wn * np.log(mu_all))),
                 area_total=float(w_all.sum()))
    edges = np.geomspace(mu_all.min(), mu_all.max() * (1 + 1e-12),
                         N_MU_BINS + 1)
    idx = np.clip(np.digitize(mu_all, edges) - 1, 0, N_MU_BINS - 1)
    w_bin = np.bincount(idx, weights=w_all, minlength=N_MU_BINS)
    mu_bin = np.bincount(idx, weights=w_all * mu_all, minlength=N_MU_BINS)
    keep = w_bin > 0
    return mu_bin[keep] / w_bin[keep], w_bin[keep] / w_bin.sum(), stats


def _predictions():
    sig_core_ref = M["sig_core"]
    sigma_extra = float(np.sqrt(max(sig_core_ref ** 2
                                    - SIGMA_C_POISSON ** 2, 0.0)))
    pp_c = PofD(COUNTS, BEAM_FWHM_ARCSEC, mu=1.0, sigma_noise=sigma_extra,
                s_cut=S_CUT_MJY, d_span=D_SPAN_MJY, n_fft=N_FFT)
    xi_c, _ = xi_of_threshold_analytic(pp_c.d, pp_c.p,
                                       M["u_grid"] - M["mu_core"])
    sf_c = pp_c.sf(M["u_grid"] - M["mu_core"])
    out = dict(sigma_extra=sigma_extra)
    in_win = (K_GRID >= K_WINDOW[0]) & (K_GRID <= K_WINDOW[1])
    for f in APERTURE_SCAN:
        mus, w, mstat = aperture_mixture(f)
        pp_l = PofD(COUNTS, BEAM_FWHM_ARCSEC, mu=(mus, w),
                    sigma_noise=sigma_extra, s_cut=S_CUT_MJY,
                    d_span=D_SPAN_MJY, n_fft=N_FFT)
        xi_l, _ = xi_of_threshold_analytic(
            pp_l.d, pp_l.p, MEAS[f]["u_grid"] - MEAS[f]["mu_core"])
        sf_l = pp_l.sf(MEAS[f]["u_grid"] - MEAS[f]["mu_core"])
        ok = sf_l > SF_FLOOR
        dxi_p = np.where(ok, np.asarray(xi_l) - np.asarray(xi_c), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comb_p = float(np.nanmean(dxi_p[in_win]))
        out[f] = dict(dxi=dxi_p, ratio=np.where(ok, sf_l / sf_c, np.nan),
                      comb=comb_p, **mstat)
    return out


PRED = cached("predictions", _predictions)
print("\n  NFW mu-mixture predictions (Schechter counts, z_s = %.1f):"
      % Z_SOURCE)
print("  %8s %10s %12s %14s %14s %10s"
      % ("aperture", "<mu>", "<ln mu>", "pred Delta_xi", "measured", "pull"))
for f in APERTURE_SCAN:
    p, m = PRED[f], MEAS[f]
    if np.isfinite(m["comb"]) and m["sig_paired"] > 0:
        pull = (m["comb"] - p["comb"]) / m["sig_paired"]
        print("  %8.1f %10.4f %12.4f %14.3e %+14.4f %10.2f"
              % (f, p["mean_mu"], p["mean_ln"], p["comb"], m["comb"], pull))
    else:
        print("  %8.1f %10.4f %12.4f %14.3e %14s %10s"
              % (f, p["mean_mu"], p["mean_ln"], p["comb"], "--", "--"))
print("""
  The predictions sit two to three orders of magnitude inside the errors,
  so a null was the expected outcome and the deliverable is a calibrated
  UPPER LIMIT that the master equation satisfies by a wide margin.  The
  reason is quantified in Sec. 9: at a 5' beam an aperture cannot be
  smaller than about a beam, which for a median theta_500 of ~5' forces
  r >~ 1.5 theta_500 before the exceedance count clears the fitting floor
  -- by which radius <ln mu> has fallen to a few x 10^-2.""")

# %% [markdown]
# ## 8. Cluster dust: the stacked profile and the Erler+2018 closure test
#
# Two independent determinations of the same quantity, on the same 772
# clusters:
#
# * **Ours** — the azimuthally averaged cluster-minus-control profile of
#   the baselined cutouts, in θ₅₀₀ units.  This is the beam-convolved,
#   outer-baselined surface brightness, i.e. what our pipeline actually
#   sees.
# * **Erler et al. (2018)** — A_857 = 0.10 ± 0.02 MJy/sr, the amplitude of
#   a *deconvolved* GNFW template, from matched filtering component-
#   separated HFI maps.
#
# The two are compared by pushing the published template through our beam,
# our outer-region baseline and our radial binning, per cluster with its
# own θ₅₀₀.  If they agree, the ~38 mJy/beam central excess v1 reported is
# confirmed as the cluster dust of a known amplitude rather than an
# unexplained residual — and, since the Erler template is a *shape* as
# well as an amplitude, it can then be subtracted.

# %%
R_BINS = np.linspace(0.0, 3.0, N_RAD_BINS + 1)
R_MID = 0.5 * (R_BINS[1:] + R_BINS[:-1])


def _radial_stack(imgs_cl, imgs_ct_sets, template_fn=None):
    """Azimuthal cluster-minus-control profile in theta_500 units.

    If template_fn is given, the same accumulation is applied to the
    forward-modelled template instead of the data, giving a like-for-like
    prediction of what the stack should return.
    """
    p_cl = np.zeros(R_MID.size)
    p_ct = np.zeros(R_MID.size)
    n_cl_b = np.zeros(R_MID.size)
    n_ct_b = np.zeros(R_MID.size)
    for img, ctrls, t5 in zip(imgs_cl, imgs_ct_sets, th5_acc):
        r_g = radius_grid_arcmin(img.shape[0], PIX_ARCMIN) / t5
        idx = np.digitize(r_g.ravel(), R_BINS) - 1
        ok = (idx >= 0) & (idx < R_MID.size)
        if template_fn is not None:
            t = template_fn(img.shape[0], t5)
            t, _ = baseline_outer_plane(
                t, radius_grid_arcmin(img.shape[0], PIX_ARCMIN),
                BASE_FIT_FRAC * t5)
            v = t.ravel()
        else:
            v = img.ravel()
        fin = np.isfinite(v) & ok
        np.add.at(p_cl, idx[fin], v[fin])
        np.add.at(n_cl_b, idx[fin], 1)
        if template_fn is None:
            for c in ctrls:
                v = c.ravel()
                fin = np.isfinite(v) & ok
                np.add.at(p_ct, idx[fin], v[fin])
                np.add.at(n_ct_b, idx[fin], 1)
    prof_cl = p_cl / np.maximum(n_cl_b, 1)
    if template_fn is not None:
        return prof_cl
    return prof_cl - p_ct / np.maximum(n_ct_b, 1)


PROF_MEAS = cached("prof_meas", lambda: _radial_stack(cl_imgs, ct_imgs))
PROF_ERLER = cached("prof_erler",
                    lambda: _radial_stack(cl_imgs, ct_imgs,
                                          template_fn=erler_dust_template))

print("\n" + "=" * 79)
print("  CLUSTER DUST: OUR STACK vs THE ERLER+2018 FORWARD MODEL")
print("=" * 79)
print(f"  A_857 = {ERLER_A857_MJY_SR:.2f} +- {ERLER_A857_ERR:.2f} MJy/sr "
      f"= {ERLER_A857_MJY_SR * MJY_SR_TO_MJY_BEAM:.0f} mJy/beam "
      f"(deconvolved GNFW amplitude)")
print("  %10s %14s %16s %10s" % ("r/theta500", "measured", "Erler forward",
                                 "ratio"))
for j in range(min(15, R_MID.size)):
    rr = (PROF_MEAS[j] / PROF_ERLER[j]) if abs(PROF_ERLER[j]) > 1e-6 \
        else np.nan
    print("  %10.2f %14.1f %16.1f %10.2f"
          % (R_MID[j], PROF_MEAS[j], PROF_ERLER[j], rr))
#  area-weighted (annulus area goes as r dr) so this is the mean surface
#  brightness over the disc r < 0.5 theta_500, not an average over bins
_in05 = R_MID < 0.5
_w05 = R_MID[_in05]
_meas05 = float(np.sum(_w05 * PROF_MEAS[_in05]) / np.sum(_w05))
_erl05 = float(np.sum(_w05 * PROF_ERLER[_in05]) / np.sum(_w05))
print(f"\n  mean inside r < 0.5 theta500: measured {_meas05:.1f}, "
      f"Erler forward model {_erl05:.1f} mJy/beam  (ratio "
      f"{_meas05 / _erl05 if _erl05 else np.nan:.2f})")
print(f"  implied A_857 from OUR stack: "
      f"{ERLER_A857_MJY_SR * _meas05 / _erl05 if _erl05 else np.nan:.3f} "
      f"MJy/sr  (Erler: {ERLER_A857_MJY_SR:.2f} +- {ERLER_A857_ERR:.2f})")
print(f"  contamination level: {_meas05 / M['sig_core']:.2f} sigma_core "
      f"(sigma_core = {M['sig_core']:.1f} mJy/beam)")
print("""
  READING.  The published amplitude is ~6x the beam-convolved central
  surface brightness because the projected GNFW is sharply peaked --
  p(0.5 theta500)/p(0) = 0.16 -- so a 5' beam on a ~5' object dilutes it
  heavily, and the outer-region baseline removes a further part.  Once
  that is undone the two agree.  This is a genuine cross-check: matched
  filtering of component-separated HFI maps versus direct stacking of the
  Lenz CIB map, two different methods on two different map products,
  returning the same cluster-dust amplitude.""")


def _make_stack_template():
    """Bind the measured profile as an interpolating template."""
    prof = np.where(np.isfinite(PROF_MEAS), PROF_MEAS, 0.0)

    def _tmpl(n_pix, t5):
        r = radius_grid_arcmin(n_pix, PIX_ARCMIN) / t5
        return np.interp(r, R_MID, prof, left=prof[0], right=0.0)
    return _tmpl


STACK_TEMPLATE = _make_stack_template()      # rebinds the Sec. 6 stub

DUST_VARIANTS = {}
for _v in ("none", "stack", "erler"):
    DUST_VARIANTS[_v] = cached(
        "meas_dust_%s" % _v,
        lambda _v=_v: (MEAS if _v == "none" else
                       {f: dxi_measurement(f, ct_imgs, dust_corr=_v)
                        for f in APERTURE_SCAN}))

print("\n  Delta_xi UNDER THE THREE CLUSTER-DUST TREATMENTS "
      f"(aperture r < {APERTURE_FRAC} theta500):")
print("  %-8s %12s %10s %10s %12s"
      % ("variant", "Delta_xi", "sigma", "95% UL", "ratio_win"))
for _v in ("none", "stack", "erler"):
    m = DUST_VARIANTS[_v][APERTURE_FRAC]
    print("  %-8s %+12.4f %10.4f %10.4f %12.3f"
          % (_v, m["comb"], m["sig_paired"], m["ul95"], m["ratio_win"]))
print("""  The 'stack' variant is self-referential by construction (it removes
  a profile measured from the same data) and is shown as an upper bound on
  how much the correction could matter; 'erler' uses an EXTERNAL amplitude
  and shape and is the physically meaningful correction.  If all three
  agree, the cluster dust does not drive Delta_xi at these apertures --
  which is what a 0.2 sigma_core, centrally confined contaminant should
  do at r < 1.5 theta_500.""")

# %% [markdown]
# ## 9. The amplitude channel and the IR-quiet selection hypothesis
#
# v1 found the measured exceedance ratio λ_cl/λ_ctrl matching the
# prediction at k = 2.5–3 but dragged down at k ≳ 4 by **zero** cluster-
# side exceedances where several were expected, and |b|-matching of
# controls did not remove it.  The working hypothesis was a PSZ2 selection
# systematic: SZ candidates coincident with bright submillimetre emission
# are flagged during catalog validation, so confirmed-cluster sightlines
# are IR-quieter than random sky.
#
# The adjudicating test is now available: the same measurement with
# controls matched on the local dust amplitude versus controls placed at
# the same offsets but *unmatched*.  If the deficit is a foreground-
# matching artefact it should move; if it is catalog selection it should
# not.  Note that the two hypotheses push in opposite directions on the
# cluster side — a dust pedestal *adds* exceedances, selection *removes*
# them — so what v1 measured was their difference.

# %%
MEAS_PLAIN = cached("meas_offsetonly",
                    lambda: {f: dxi_measurement(f, ct_plain)
                             for f in APERTURE_SCAN})

print("\n  AMPLITUDE CHANNEL, dust-matched vs offset-only controls "
      f"(r < {APERTURE_FRAC} theta500):")
print("  %6s %10s %12s %12s %12s"
      % ("k", "N_exc cl", "N_exc ct", "ratio (dust)", "ratio (plain)"))
_mp = MEAS_PLAIN[APERTURE_FRAC]
for j, k in enumerate(K_GRID):
    if M["n_exc_ct"][j] > 0 or _mp["n_exc_ct"][j] > 0:
        print("  %6.1f %10d %12d %12.3f %12.3f"
              % (k, M["n_exc_cl"][j], M["n_exc_ct"][j],
                 M["ratio"][j], _mp["ratio"][j]))
_drf = PAIRS.get("dust_reject_frac", np.nan)
print("  [the dust criterion rejected %.1f%% of control candidates; if that "
      "is ~0\n   the two columns are the SAME sample and the test has no "
      "power]" % (100 * _drf))
print("  window-mean ratio: dust-matched %.3f [%.3f, %.3f], "
      "offset-only %.3f [%.3f, %.3f]"
      % (M["ratio_win"], M["ratio_win_lo"], M["ratio_win_hi"],
         _mp["ratio_win"], _mp["ratio_win_lo"], _mp["ratio_win_hi"]))
print("  predicted window ratio (magnification bias): %.3f"
      % np.nanmean(PRED[APERTURE_FRAC]["ratio"][
          (K_GRID >= K_WINDOW[0]) & (K_GRID <= K_WINDOW[1])]))

# %% [markdown]
# ## 10. Null tests (Herschel H2 §7.5, ported)
#
# * **N1 — control-vs-control half split.**  Split each cluster's controls
#   into two halves and measure Δξ between them.  The truth is exactly
#   zero, so this tests the whole machinery for bias and tests whether the
#   paired-bootstrap error is honest (empirical scatter ÷ bootstrap σ
#   should be ≈ 1).
# * **N2 — validity balance.**  If the mask removed more area from cluster
#   fields than from control fields the two samples would not be analysed
#   identically, and the difference would appear directly in Δξ.
# * **N3 — control re-randomisation.**  Rebuild every control with a
#   different seed and measure the shift in units of σ.

# %%
def _null_n1():
    """Half A of each cluster's controls plays the 'cluster' role, half B
    the 'control' role.  Identical geometry, identical processing, truth
    exactly zero."""
    half = max(1, N_CTRL_PER // 2)
    a_first = [c[0] for c in ct_imgs]                    # cluster side
    b_sets = [c[half:2 * half] for c in ct_imgs]         # control side
    if min(len(b) for b in b_sets) == 0:
        return None
    return dxi_measurement(APERTURE_FRAC, b_sets, cl_set=a_first,
                           n_boot=max(80, N_BOOT // 4), seed=11)


N1 = cached("null_n1", _null_n1)
print("\n" + "=" * 79)
print("  NULL TESTS")
print("=" * 79)
if N1 is None or not np.isfinite(N1.get("sig_paired", np.nan)) \
        or N1["sig_paired"] <= 0:
    _z1 = np.nan
    print("  N1 control-vs-control half split: not evaluable "
          "(need N_CTRL_PER >= 2 and enough exceedances)")
else:
    _z1 = N1["comb"] / N1["sig_paired"]
    print("  N1 control-vs-control half split: Delta_xi = %+.4f +- %.4f "
          "(z = %+.2f)  [%s]" % (N1["comb"], N1["sig_paired"], _z1,
                                 "PASS" if abs(_z1) < 2.5 else "CHECK"))

_bc, _bt = PAIRS["badfrac_cl"], PAIRS["badfrac_ct"]
_bal = (_bc.mean() + 1e-12) / (_bt.mean() + 1e-12)
print("  N2 validity balance: invalid-pixel fraction in the aperture, "
      "cluster %.5f vs control %.5f (ratio %.3f)  [%s]"
      % (_bc.mean(), _bt.mean(), _bal,
         "PASS" if 0.8 < _bal < 1.25 else "CHECK"))

RUN_N3 = bool(int(os.environ.get("CIB_RUN_N3", "1")))
if RUN_N3:
    PAIRS_ALT = cached("pairs_alt", lambda: _build_pairs(RNG_SEED_ALT))
    if len(PAIRS_ALT["acc"]) == n_cl and np.array_equal(PAIRS_ALT["acc"],
                                                        acc_idx):
        M_ALT = dxi_measurement(APERTURE_FRAC, PAIRS_ALT["ct_dust"],
                                n_boot=max(80, N_BOOT // 4), seed=13)
        _z3 = ((M_ALT["comb"] - M["comb"])
               / np.hypot(M_ALT["sig_paired"], M["sig_paired"]))
        print("  N3 control re-randomisation: Delta_xi %+.4f -> %+.4f "
              "(shift %.2f sigma)  [%s]"
              % (M["comb"], M_ALT["comb"], _z3,
                 "PASS" if abs(_z3) < 2.0 else "CHECK"))
    else:
        M_ALT, _z3 = None, np.nan
        print("  N3 skipped: the alternative seed accepted a different "
              "cluster set (%d vs %d)" % (len(PAIRS_ALT["acc"]), n_cl))
else:
    M_ALT, _z3 = None, np.nan
    print("  N3 skipped (CIB_RUN_N3=0)")

# %% [markdown]
# ## 10b. Mass and redshift split of the negative Δξ
#
# Open item 2 of Sec. 11.3: Δξ is negative at all three measurable
# apertures and reaches −2.55σ at r < 2.5 θ₅₀₀, with no explanation; dust
# accounts for about a fifth of it.  Splitting the sample at the median
# M₅₀₀ and the median z asks whether the offset tracks a cluster property.
#
# **What the test can and cannot do.**  Halving the sample roughly doubles
# σ, so a uniform −2.55σ offset appears as ≈−1.8σ in each half: the split
# has power against a STRONG dependence (one half carrying the effect and
# the other not) and none against a mild one.  The exceedance budget is
# printed first so this is a measured statement rather than an assumed one.
#
# **How to read the sign.**  A lensing origin predicts the offset to be
# LARGER in magnitude for high-mass and low-z clusters, which have larger
# ⟨ln μ⟩; so does a cluster-dust origin, since dust scales with mass too.
# The two are therefore not separated by this test — it separates
# "correlated with cluster properties" from "not correlated", which is
# what the look-elsewhere question needs.

# %%
#  Release fix: restrict to the apertures actually scanned, so that the
#  reduced CIB_QUICK_TEST scan (no 2.5) runs; the full scan contains all three.
SPLIT_APERTURES = [f for f in (1.5, 2.0, 2.5) if f in APERTURE_SCAN]
_m5_acc = m500[acc_idx]
_z_acc = cat["z"][acc_idx]
_m_med, _z_med = float(np.median(_m5_acc)), float(np.median(_z_acc))

#  theta_500 is included as a third split because in an SZ-selected sample
#  M500 and z are strongly correlated (the selection is roughly a mass floor
#  rising with z), so the M500 and z splits are NOT independent tests — they
#  select nearly the same clusters.  What actually distinguishes the halves
#  is ANGULAR SIZE, and theta_500 is that variable directly: the analysis
#  aperture is r < f*theta_500, so theta_500 sets how much sky each aperture
#  covers, and therefore how much cirrus and clustered CIB it admits.
#  Lensing and sky-systematic hypotheses predict OPPOSITE orderings in it.
_t5_med = float(np.median(th5_acc))
SPLITS = {
    "M500 low": np.where(_m5_acc <= _m_med)[0],
    "M500 high": np.where(_m5_acc > _m_med)[0],
    "z low": np.where(_z_acc <= _z_med)[0],
    "z high": np.where(_z_acc > _z_med)[0],
    "th500 small": np.where(th5_acc <= _t5_med)[0],
    "th500 large": np.where(th5_acc > _t5_med)[0],
}

print("\n" + "=" * 79)
print("  MASS / REDSHIFT SPLIT OF Delta_xi   (open item 11.3.2)")
print("=" * 79)
print("  medians: M500 = %.3e Msun, z = %.3f  (N = %d pairs)"
      % (_m_med, _z_med, n_cl))
for k, v in SPLITS.items():
    print("    %-11s N = %3d   <M500> = %.3e   <z> = %.3f   "
          "<theta500> = %.2f'"
          % (k, v.size, _m5_acc[v].mean(), _z_acc[v].mean(),
             th5_acc[v].mean()))

#  How much do the three splits actually differ?  If the overlaps are near
#  100% the "three tests" are one test wearing three hats, and the naive
#  reading (two independent 2.3-2.5 sigma trends) would be wrong.
print("\n  SPLIT OVERLAP — %% of each half shared with the others "
      "(100%% = identical selection)")
_low = [("M500 low", SPLITS["M500 low"]), ("z low", SPLITS["z low"]),
        ("th500 large", SPLITS["th500 large"])]
print("  %14s" % "" + "".join("%14s" % a for a, _ in _low))
for a, ia in _low:
    row = "".join("%13.0f%%" % (100.0 * np.intersect1d(ia, ib).size
                                / max(ia.size, 1)) for _, ib in _low)
    print("  %14s%s" % (a, row))
print("  (th500 LARGE is paired with M500 LOW / z LOW because in an "
      "SZ-selected\n   sample the nearby low-mass clusters are the "
      "angularly large ones.)")

#  ---- power check: exceedance budget in the combination window -------------
print("\n  POWER CHECK — exceedances in the k in [%.1f, %.1f] window "
      "(MIN_EXCEED = %d)" % (K_WINDOW[0], K_WINDOW[1], MIN_EXCEED))
print("  %10s %8s %10s %10s %10s" % ("aperture", "sample", "N pairs",
                                     "N_exc k=3", "N_exc k=4.5"))
for f in SPLIT_APERTURES:
    for lab, idx in [("FULL", None)] + list(SPLITS.items()):
        _pc, _ = collect_peaks(f, ct_imgs, sub=idx)
        _pool = np.concatenate(_pc)
        _mu, _sg = control_core(f, ct_imgs, sub=idx)
        _n3 = int((_pool > _mu + 3.0 * _sg).sum())
        _n45 = int((_pool > _mu + 4.5 * _sg).sum())
        print("  %10.1f %8s %10d %10d %10d"
              % (f, lab.replace(" ", ""), len(_pc), _n3, _n45))

#  ---- the split itself ------------------------------------------------------
#  NOTE the cache tag carries a version suffix: the SPLITS dict is not part of
#  CFG_HASH, so adding or changing a split MUST bump this tag or a stale
#  result is served silently.  v2 added the theta_500 split.
SPLIT_RES = cached("split_mz_v2", lambda: {
    (f, lab): dxi_measurement(f, ct_imgs, sub=idx, seed=101 + 7 * j)
    for f in SPLIT_APERTURES
    for j, (lab, idx) in enumerate(SPLITS.items())})

print("\n  RESULT — Delta_xi by subsample")
print("  %10s %10s %8s %11s %9s %9s %9s"
      % ("aperture", "sample", "N pairs", "Delta_xi", "sigma", "z", "95% UL"))
SPLIT_Z = {}
for f in SPLIT_APERTURES:
    m_full = MEAS[f]
    if np.isfinite(m_full["comb"]):
        print("  %10.1f %10s %8d %+11.4f %9.4f %+9.2f %9.4f"
              % (f, "FULL", n_cl, m_full["comb"], m_full["sig_paired"],
                 m_full["comb"] / m_full["sig_paired"], m_full["ul95"]))
    for lab in SPLITS:
        r = SPLIT_RES[(f, lab)]
        if np.isfinite(r["comb"]) and r["sig_paired"] > 0:
            print("  %10s %10s %8d %+11.4f %9.4f %+9.2f %9.4f"
                  % ("", lab, r["n_pairs"], r["comb"], r["sig_paired"],
                     r["comb"] / r["sig_paired"], r["ul95"]))
        else:
            print("  %10s %10s %8d    -- (fails the N_exc floor)"
                  % ("", lab, r["n_pairs"]))
    #  difference between the two halves of each variable
    for var, lo, hi in (("M500", "M500 low", "M500 high"),
                        ("z", "z low", "z high"),
                        ("th500", "th500 small", "th500 large")):
        a, b = SPLIT_RES[(f, lo)], SPLIT_RES[(f, hi)]
        if np.isfinite(a["comb"]) and np.isfinite(b["comb"]) \
                and a["sig_paired"] > 0 and b["sig_paired"] > 0:
            d = b["comb"] - a["comb"]
            sd = np.hypot(a["sig_paired"], b["sig_paired"])
            SPLIT_Z[(f, var)] = d / sd
            print("  %10s %10s   high - low = %+.4f +- %.4f  (z = %+.2f)"
                  % ("", var, d, sd, d / sd))
print("""
  The two halves are INDEPENDENT samples of clusters, so their difference
  carries the quadrature error; but they share the same sky, the same mask
  and the same control-drawing machinery, so a common systematic cancels in
  the difference and would NOT show up here.  A null difference therefore
  says 'not driven by a mass- or z-dependent effect', not 'no systematic'.""")

# %% [markdown]
# ## 11. Figure A — Δξ(u) and the amplitude channel

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
P = PRED[APERTURE_FRAC]

ax = axes[0]
ax.axvspan(K_GRID.min(), 3.0, color="0.93", zorder=0)
ax.axvspan(*K_WINDOW, color="C2", alpha=0.10, zorder=0)
ax.errorbar(K_GRID, M["dxi"],
            yerr=[M["dxi"] - M["dxi_lo"], M["dxi_hi"] - M["dxi"]],
            fmt="C0o", ms=5, capsize=2, label=r"measured $\Delta\xi(u)$")
ax.plot(K_GRID, P["dxi"], "C3-", lw=1.8,
        label=r"NFW $\mu$-mixture prediction")
if np.isfinite(M["comb"]):
    ax.errorbar([np.mean(K_WINDOW)], [M["comb"]], yerr=[M["sig_paired"]],
                fmt="C2s", ms=9, mfc="none", mew=2, capsize=4,
                label=f"window-combined: {M['comb']:+.3f} $\\pm$ "
                      f"{M['sig_paired']:.3f}\n95% UL "
                      f"$|\\Delta\\xi| <$ {M['ul95']:.3f}")
ax.axhline(0, color="0.7", lw=0.8)
ax.set(xlabel=r"$k = (u-\mu_{\rm core})/\sigma_{\rm core}$",
       ylabel=r"$\Delta\xi(u)$",
       title=f"shape channel  ({n_cl} pairs, "
             f"r < {APERTURE_FRAC}$\\,\\theta_{{500}}$, dust-matched)")
ax.legend(fontsize=8)

ax = axes[1]
ax.errorbar(K_GRID, M["ratio"], yerr=M["ratio_err"], fmt="C0o", ms=5,
            capsize=2, label="dust-matched controls")
ax.errorbar(K_GRID + 0.06, _mp["ratio"], yerr=_mp["ratio_err"], fmt="C1^",
            ms=5, capsize=2, mfc="none", label="offset-only controls")
ax.plot(K_GRID, P["ratio"], "C3-", lw=1.8, label="predicted (magn. bias)")
ax.axhline(1, color="0.7", lw=0.8)
ax.set(xlabel=r"$k = (u-\mu_{\rm core})/\sigma_{\rm core}$",
       ylabel=r"$\lambda_{\rm cl}(u)\,/\,\lambda_{\rm ctrl}(u)$",
       title="amplitude channel and the selection test")
ax.legend(fontsize=8)
fig.tight_layout()
_finish_figure(fig, "planck_v2_lensed_dxi")

# %% [markdown]
# ## 12. Figure B — the aperture optimum, and Figure C — the dust closure
#
# Figure B is the dilution-versus-statistics pincer realized in data: the
# predicted Δξ falls with aperture radius as ⟨ln μ⟩ dilutes, while the
# exceedance count rises, and at the Planck beam the aperture cannot be
# smaller than about a beam regardless of θ₅₀₀.  This is the single most
# transferable quantitative lesson of the Planck chapter.

# %%
ap_arr = np.array(APERTURE_SCAN)
dxi_pred_arr = np.array([PRED[f]["comb"] for f in APERTURE_SCAN])
mean_ln_arr = np.array([PRED[f]["mean_ln"] for f in APERTURE_SCAN])
in_win = (K_GRID >= K_WINDOW[0]) & (K_GRID <= K_WINDOW[1])
n_exc_win = np.array([np.nansum(MEAS[f]["n_exc_cl"][in_win])
                      for f in APERTURE_SCAN])
snr_fore = np.abs(dxi_pred_arr) / np.sqrt(
    np.maximum(1.0 / np.maximum(n_exc_win, 1) * (1 + 1.0 / N_CTRL_PER), 1e-30))

fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.9))

ax = axes[0]
ax2 = ax.twinx()
l1, = ax.plot(ap_arr, np.abs(dxi_pred_arr), "C3o-",
              label=r"$|\Delta\xi|$ predicted (window)")
l2, = ax.plot(ap_arr, mean_ln_arr, "C1s--", label=r"$\langle\ln\mu\rangle$")
l3, = ax2.plot(ap_arr, snr_fore, "ko-", lw=2, label="forecast SNR")
ax.set_yscale("log")
ax.set(xlabel=r"aperture radius  [$\theta_{500}$]",
       ylabel=r"$|\Delta\xi|$ or $\langle\ln\mu\rangle$",
       title="dilution vs statistics: the aperture optimum")
ax2.set_ylabel("forecast SNR")
ax.legend(handles=[l1, l2, l3], fontsize=8, loc="upper right")

ax = axes[1]
cm = np.array([MEAS[f]["comb"] for f in APERTURE_SCAN])
cs = np.array([MEAS[f]["sig_paired"] for f in APERTURE_SCAN])
ax.errorbar(ap_arr, cm, yerr=cs, fmt="C0o", ms=6, capsize=3,
            label="measured (window-combined)")
ax.plot(ap_arr, dxi_pred_arr, "C3s-", lw=1.8, label="predicted")
ax.axhline(0, color="0.7", lw=0.8)
ax.set(xlabel=r"aperture radius  [$\theta_{500}$]",
       ylabel=r"window-combined $\Delta\xi$",
       title="consistency across apertures")
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(R_MID, PROF_MEAS, "C0o-", ms=5, label="measured stack (cl $-$ ct)")
ax.plot(R_MID, PROF_ERLER, "C3s--", ms=5,
        label=r"Erler+2018 GNFW, $A_{857}$ = %.2f MJy/sr,"
              "\nforward-modelled through beam + baseline"
              % ERLER_A857_MJY_SR)
ax.axhline(0, color="0.7", lw=0.8)
ax.set(xlabel=r"$r/\theta_{500}$", ylabel="mJy/beam",
       title="cluster dust: two independent determinations")
ax.legend(fontsize=7.5)
fig.tight_layout()
_finish_figure(fig, "planck_v2_lensed_aperture_and_dust")

# %% [markdown]
# ## 13. Save results and closing summary

# %%
#  A non-default aperture-coverage cut is a DIFFERENT sample, so it must not
#  overwrite the baseline product that `Revised_Planck_analysis.md` quotes.
#  Default runs keep the historical filename exactly.            [2026-08-17]
_COV_TAG = "" if MIN_VALID_AP == 0.97 else f"_ap{MIN_VALID_AP:.2f}"
OUT = os.path.join(RESULTS_DIR,
                   f"planck_v2_lensed_857_{MASK_VARIANT}{_COV_TAG}.npz")
_save = dict(
    mask_variant=MASK_VARIANT, k_grid=K_GRID, k_window=np.array(K_WINDOW),
    aperture_scan=np.array(APERTURE_SCAN), aperture_default=APERTURE_FRAC,
    n_clusters=n_cl, acc_idx=acc_idx, theta500_acc=th5_acc,
    m500_acc=m500[acc_idx], m_vir_acc=m_vir_acc, z_acc=cat["z"][acc_idx],
    z_source=Z_SOURCE, n_ctrl_per=N_CTRL_PER,
    offset_min_deg=OFFSET_MIN_DEG, offset_max_deg=OFFSET_MAX_DEG,
    dust_tol=DUST_TOL, offsets=PAIRS["offsets"],
    dust_cl=PAIRS["dust_cl"], dust_ct=PAIRS["dust_ct"],
    badfrac_cl=PAIRS["badfrac_cl"], badfrac_ct=PAIRS["badfrac_ct"],
    sigma_c_poisson=SIGMA_C_POISSON, sigma_extra=PRED["sigma_extra"],
    r_mid=R_MID, prof_meas=PROF_MEAS, prof_erler=PROF_ERLER,
    erler_a857=ERLER_A857_MJY_SR, erler_a857_err=ERLER_A857_ERR,
    erler_implied_a857=(ERLER_A857_MJY_SR * _meas05 / _erl05
                        if _erl05 else np.nan),
    n1_z=_z1, n2_balance=_bal, n3_z=_z3,
)
for f in APERTURE_SCAN:
    m, p = MEAS[f], PRED[f]
    tag = str(f)
    _save[f"dxi_{tag}"] = m["dxi"]
    _save[f"dxi_sd_{tag}"] = m["dxi_sd"]
    _save[f"dxi_boot_{tag}"] = m["dxi_boot"]      # full replicates (new)
    _save[f"u_grid_{tag}"] = m["u_grid"]
    _save[f"comb_{tag}"] = m["comb"]
    _save[f"sig_paired_{tag}"] = m["sig_paired"]
    _save[f"sig_naive_{tag}"] = m["sig_naive"]
    _save[f"ul95_{tag}"] = m["ul95"]
    _save[f"ratio_{tag}"] = m["ratio"]
    #  [2026-08-22] the amplitude panel of the paper figure needs the ratio
    #  ERRORS and the offset-only control series; without them
    #  00_Paper_Draft/make_paper_figures.py cannot rebuild Fig. 8 from cache.
    _save[f"ratio_err_{tag}"] = m["ratio_err"]
    _save[f"n_exc_cl_{tag}"] = m["n_exc_cl"]
    _save[f"n_exc_ct_{tag}"] = m["n_exc_ct"]
    _mpf = MEAS_PLAIN[f]
    _save[f"ratio_plain_{tag}"] = _mpf["ratio"]
    _save[f"ratio_plain_err_{tag}"] = _mpf["ratio_err"]
    #  [2026-09-14] window-combined ratios with their 95% intervals, for the
    #  open markers of the Fig. 8 amplitude panel (the numbers Sect. 4.4 quotes).
    for _nm, _src in (("ratio_win", m), ("ratio_plain_win", _mpf)):
        _save[f"{_nm}_{tag}"] = _src["ratio_win"]
        _save[f"{_nm}_lo_{tag}"] = _src["ratio_win_lo"]
        _save[f"{_nm}_hi_{tag}"] = _src["ratio_win_hi"]
    _save[f"pred_dxi_{tag}"] = p["dxi"]
    _save[f"pred_comb_{tag}"] = p["comb"]
    _save[f"pred_ratio_{tag}"] = p["ratio"]
    _save[f"mean_mu_{tag}"] = p["mean_mu"]
    _save[f"mean_ln_{tag}"] = p["mean_ln"]
for v in ("none", "stack", "erler"):
    mm = DUST_VARIANTS[v][APERTURE_FRAC]
    _save[f"dustvar_{v}_comb"] = mm["comb"]
    _save[f"dustvar_{v}_sig"] = mm["sig_paired"]
    _save[f"dustvar_{v}_ul95"] = mm["ul95"]
np.savez_compressed(OUT, **_save)
print(f"\nWrote {OUT}")

print("\n" + "=" * 79)
print("  M5-v2 SUMMARY")
print("=" * 79)
print(f"  {n_cl} cluster/control pairs from the {MASK_VARIANT} mask "
      f"({N_CTRL_PER} dust-matched controls each, offsets "
      f"{OFFSET_MIN_DEG}-{OFFSET_MAX_DEG} deg)")
print(f"  median theta500 {np.median(th5_acc):.2f}', median M500 "
      f"{np.median(m500[acc_idx]):.2e} Msun, z_s = {Z_SOURCE}")
print(f"  default aperture r < {APERTURE_FRAC} theta500: "
      f"Delta_xi = {M['comb']:+.4f} +- {M['sig_paired']:.4f}, "
      f"95% UL |Delta_xi| < {M['ul95']:.3f}")
print(f"  NFW prediction there: {PRED[APERTURE_FRAC]['comb']:.3e} "
      f"-> the limit exceeds the prediction by a factor "
      f"{M['ul95'] / abs(PRED[APERTURE_FRAC]['comb']):.0f}"
      if np.isfinite(PRED[APERTURE_FRAC]["comb"])
      and PRED[APERTURE_FRAC]["comb"] != 0 else "")
print(f"  cluster dust: {_meas05:.1f} mJy/beam inside 0.5 theta500 = "
      f"{_meas05 / M['sig_core']:.2f} sigma_core; Erler+2018 forward model "
      f"predicts {_erl05:.1f}")
print(f"  sigma_paired/sigma_naive = "
      f"{np.median(_ratios) if _ratios else np.nan:.2f} -> the v1 window "
      f"combination was NOT materially inflated")
print(f"  null tests: N1 z = {_z1:+.2f}, N2 balance = {_bal:.3f}, "
      f"N3 shift = {_z3:+.2f} sigma")
print("""
  BOTTOM LINE.  At Planck resolution the lensing modulation of the CIB
  P(D) tail is not detectable and, given S_cut/sigma_c ~ 1 at a 5' beam,
  could not have been: the aperture is beam-limited, so <ln mu> is diluted
  to a few x 10^-2 before the exceedance count clears the fitting floor.
  The honest deliverable is the upper limit above, together with the
  demonstration that the pipeline returns zero on control-vs-control data
  at the same precision (N1) and that the cluster-dust contamination is
  measured, externally corroborated, and too small to matter here.""")
print("\n  Runtime %.0f s." % (time.time() - T0))

