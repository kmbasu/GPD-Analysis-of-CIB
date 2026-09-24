#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
herschel_lensed.py -- Module H2: the lensed Delta-xi(u) measurement behind
                      eFEDS clusters in the Herschel/SPIRE 350 um GAMA-09 field.
===============================================================================

PURPOSE
-------
The last measurement module of the Herschel arm.  H1v2 measured the unlensed
xi_hat(u) on blank fields; this module measures

    Delta-xi(u)  =  xi_hat_cluster(u)  -  xi_hat_control(u)

for a matched sample of eFEDS clusters and control fields, and compares it with
the NFW-lensing prediction built from each cluster's weak-lensing-calibrated
M500.  The physical content: gravitational lensing magnifies the background CIB
sources (S -> mu S) while diluting their surface density (dOmega -> mu dOmega),
so the image-plane counts become n_mu(S) = mu^-2 n_0(S/mu) (Lima, Jain & Devlin
2010, Eq. 31).  For a pure power law this changes only the amplitude, NOT the
slope -- so the entire Delta-xi signal comes from CURVATURE in dN/dS and from
finite-threshold effects.  That is the point of the paper.

WHY Delta-xi IS THE ROBUST OBSERVABLE HERE  (and why H2 matters more than H1v2)
-------------------------------------------------------------------------------
H1v2 + H1c established that the ABSOLUTE xi_hat(u) is dominated at high
threshold by the bright, strongly lensed source population, which none of the
fitted count models describes: 12 peaks above 200 mJy in 25.75 deg^2 where the
adopted Schechter predicts 0.05.  That systematic is a property of the SKY, not
of the cluster sightlines, so it is common-mode between cluster and control
fields and cancels in the difference.  Three design choices below make that
cancellation as exact as we can make it:

  1. Every cluster is PAIRED with exactly one control, obtained by displacing
     its centre by 1 degree in a random direction.  The pair therefore shares
     the local coverage depth, the local cirrus level and the same region of
     the map, far more closely than fully random control positions would.
  2. The bootstrap resamples PAIRS, not cluster and control fields
     independently, so any sky systematic common to a pair cancels inside every
     bootstrap replicate rather than merely on average.
  3. The bright-source mask, the baseline polynomial and its fitting annulus,
     the declustering window and the edge exclusion are applied IDENTICALLY to
     cluster and control -- including using the CLUSTER's theta500 to define the
     control's baseline annulus and aperture, so the two have identical
     geometry and identical baseline degrees of freedom.  A difference in
     analysis geometry between the two samples would appear directly in
     Delta-xi and is indistinguishable from signal.

The same argument applies to the pixel-model systematic H1c exposed (the
analytic KL-projected curve differs from a simulated peak measurement by up to
|dxi| ~ 0.047): since the prediction below is a DIFFERENCE of two curves
computed identically, that offset largely cancels, and the predicted Delta-xi is
considerably more trustworthy than the predicted absolute xi.  This is stated
explicitly because it is the reverse of the situation in H1v2.

SAMPLE  (user-approved 2026-07-29)
----------------------------------
Clusters: eFEDS V3.2 catalogue cross-matched by name to Table C1 of Chiu et al.
(2022), A&A 661, A11 -- the HSC weak-lensing mass calibration -- taking
logM500R0.5B (the no-core, R > 0.5 h^-1 Mpc mass, bytes 46-51, in h^-1 Msun).
Accepted if the FULL padded 43.3' extraction region is 100% unmasked and a mass
and redshift exist: **110 clusters**.

The 100%-clean requirement is deliberately strict and is the reason this sample
is smaller than the 161 quoted in earlier sessions (which used the unpadded
30' cutout and a >=95% coverage criterion).  The justification is specific to
the self-filtering scheme: the matched filter is INVERSE-VARIANCE WEIGHTED, so
masked pixels anywhere inside the 13.5' kernel footprint locally distort the
filter -- and that distortion would in general differ between a cluster field
and its control, which is exactly the kind of difference Delta-xi is sensitive
to.  Relaxing to >=99% or >=95% coverage would give 132 or 140 clusters
(~10-13% smaller errors) but would require demonstrating the distortion is
common-mode.  Env switch CIB_COVER_FRAC exists to test that later.

Controls: one per cluster, displaced 1.0 degree in a uniformly random direction,
required to satisfy the same 100%-clean padded condition and to lie at least 5'
from ANY eFEDS cluster (not just its own parent).  If a direction fails, further
random directions are tried; in practice 110/110 succeed with a median of one
try, so the control sample is not selection-biased towards particular regions.

PROCESSING -- INHERITED UNCHANGED FROM H1v2, DO NOT DRIFT
---------------------------------------------------------
Map        data/GAMA-09_SPIRE350_v1.0.fits, HDU 1 (IMAGE)
Filter     our own matched filter, MFILT = [(d/sigma^2)*K]/[(1/sigma^2)*K^2],
           applied to a 325-pix padded extraction and trimmed to 225 pix (30').
           Padding by the kernel half-width (50 pix) makes this identical to
           full-map filtering to machine precision (H1v2 cell 1a/1b).
Beam       25.15" FWHM (the filtered profile's Condon-equivalent FWHM)
Mask       map-level bright-source mask at S_MASK = 100 mJy grown by one FWHM,
           matching the counts truncation S_cut used in the models
Baseline   DC + linear, fitted on the OUTER region r > BASE_RINNER*theta500 of
           the cutout (clusters AND their controls, same theta500)
Peaks      local maxima over one FWHM (3 pix), edge-excluded by one FWHM, then
           restricted to r < AP_FRAC*theta500 of the cutout centre
Thresholds absolute flux (mJy/beam), the same grid as H1v2

APERTURES
---------
Delta-xi is measured in several apertures r < AP_FRAC*theta500.  Small apertures
have higher magnification but few peaks; large apertures have many peaks but
dilute the signal.  With the median theta500 = 2.45' and the measured peak
density 0.374/beam, an aperture of 1.0 theta500 yields ~36 peaks per cluster
(~3,900 over the sample) and 2.0 theta500 about four times that.  All apertures
are reported; none is designated "the" result a priori, and the reader is given
the full table so the aperture choice cannot be tuned post hoc.

NULL TESTS (cell 6) -- these decide whether any detection is believable
-----------------------------------------------------------------------
  N1  CONTROL-vs-CONTROL.  Split the 110 controls at random into two halves and
      form Delta-xi between them, many times.  This must be consistent with
      zero; its scatter is an empirical check on the bootstrap errors, which is
      more honest than trusting the bootstrap alone.
  N2  MASK BALANCE.  The masked area fraction and the fraction of peaks removed
      must agree between cluster and control samples.  If clusters lose more
      peaks to the bright mask (they might: cluster members and lensed sources
      are bright), the two samples are no longer analysed identically and
      Delta-xi is biased.  This is REPORTED, not corrected.
  N3  ROTATION.  Re-run with the control offset direction re-randomised under a
      different seed; the result must not move by more than the errors.

WHAT THIS MODULE DOES NOT DO
----------------------------
The predicted Delta-xi below uses the analytic pixel-level P(D) machinery
(MixtureCounts + xi_of_threshold_analytic), as the Planck chapter did.  H1c
showed that peak-level and pixel-level xi differ; the difference largely
cancels in Delta-xi (see above) but "largely" is not "exactly".  A
simulation-derived peak-level Delta-xi prediction, using the H1c machinery with
a lens injected, is the natural next step (H2b) and is required before the
prediction curve is quoted as a precise number rather than an expectation.

OUTPUTS
-------
results/herschel_lensed_350.npz
results/herschel_h2_dxi.png              the main result, all apertures
results/herschel_h2_sample.png           sample properties, sky positions
results/herschel_h2_nulls.png            the three null tests
results/herschel_h2_prediction.png       NFW mu-mixture and predicted Delta-xi

ENVIRONMENT SWITCHES
--------------------
CIB_HERSCHEL_DIR, CIB_FIGURE_DIR, CIB_SAVE_FIGURES, CIB_QUICK_TEST
CIB_COVER_FRAC     required unmasked fraction of the padded region (default 1.0)
CIB_CTRL_SHIFT_DEG control offset in degrees (default 1.0)
CIB_RANDOM_SEED    default 1234
CIB_N_BOOT         paired bootstrap replicates (default 400)
CIB_AP_FRACS       comma-separated aperture list (default 0.5,1.0,1.5,2.0)
CIB_BASE_RINNER    baseline annulus inner radius in theta500 (default 2.0)
CIB_SMASK_MJY      bright mask (default 100)
CIB_SCUT_MJY       counts truncation (default 100)
CIB_Z_SOURCE       CIB source redshift for the lensing (default 2.0)
CIB_MODELS         "schechter", "dpl", or "both" (default both)

Author: CIB-lensing / GPD project, Herschel task, July 2026.
"""

#%% ---------------------------------------------------------------- imports --
import os
import sys
import json
import time
import warnings

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck18
import astropy.units as u_ap
from scipy.ndimage import maximum_filter
from scipy.signal import oaconvolve
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis_modules"))

from counts import Schechter, DoublePowerLaw, MixtureCounts   # noqa: E402
from pofd_analytic import PofD                                # noqa: E402
from lens_model import (NFWLens, duffy_cvir,                  # noqa: E402
                        bryan_norman_delta_c)
from gpd_tail import fit_gpd, xi_of_threshold_analytic        # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ------------------------------------------------------------------ config --
DATA_DIR = os.environ.get("CIB_HERSCHEL_DIR",
                          os.path.join(HERE, "data"))
MAP_FILE = os.path.join(DATA_DIR, "GAMA-09_SPIRE350_v1.0.fits")
CLUSTER_FILE = os.path.join(DATA_DIR, "eFEDS_clusters_V3.2.fits")
TABLEC1 = os.path.join(DATA_DIR, "tablec1.dat")
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", os.path.join(HERE, "results"))
SAVE_FIGURES = os.environ.get("CIB_SAVE_FIGURES", "1") == "1"
QUICK = os.environ.get("CIB_QUICK_TEST", "0") == "1"
SEED = int(os.environ.get("CIB_RANDOM_SEED", "1234"))

COVER_FRAC = float(os.environ.get("CIB_COVER_FRAC", "1.0"))
CTRL_SHIFT_DEG = float(os.environ.get("CIB_CTRL_SHIFT_DEG", "1.0"))
N_BOOT = int(os.environ.get("CIB_N_BOOT", "60" if QUICK else "400"))
AP_FRACS = [float(s) for s in
            os.environ.get("CIB_AP_FRACS", "0.5,1.0,1.5,2.0").split(",")]
BASE_RINNER = float(os.environ.get("CIB_BASE_RINNER", "2.0"))
S_MASK_MJY = float(os.environ.get("CIB_SMASK_MJY", "100.0"))
S_CUT_MJY = float(os.environ.get("CIB_SCUT_MJY", "100.0"))
S_MIN_MJY = float(os.environ.get("CIB_SMIN_MJY", "1.0"))
Z_SOURCE = float(os.environ.get("CIB_Z_SOURCE", "2.0"))
WHICH_MODELS = os.environ.get("CIB_MODELS", "both").lower()

CUTOUT_ARCMIN = 30.0
MASK_GROW = 3
CLUSTER_AVOID_ARCMIN = 5.0
N_RINGS, N_MU_BINS = 48, 64
MIN_EXC = 50
COSMO = Planck18
H_LITTLE = COSMO.h

os.makedirs(FIGURE_DIR, exist_ok=True)
JY2MJY, MAD2SIG = 1.0e3, 1.4826
T0 = time.time()
RES = {}

# ---------------------------------------------------------------- caching --
# The full run takes a few minutes, dominated by the paired bootstrap and by
# xi_of_threshold_analytic (~2.3 s per call).  Each expensive stage is cached to
# disk under a key that hashes the configuration, so:
#   * the module can be run in stages on a machine with a short job limit;
#   * figures and summaries can be regenerated instantly without redoing the
#     cutout extraction;
#   * changing any parameter that affects a stage invalidates its cache
#     automatically, so a stale result cannot silently survive a config change.
# Set CIB_USE_CACHE=0 to force a full recomputation.
import hashlib                                                   # noqa: E402
import pickle                                                    # noqa: E402

CACHE_DIR = os.path.join(FIGURE_DIR, "h2_cache")
#  Release fix: the comparison was inverted (== "0"), so the cache was written
#  but never read.  Results are unaffected; only re-runs are now faster.
USE_CACHE = os.environ.get("CIB_USE_CACHE", "1") == "1"
_CFG = json.dumps(dict(cover=COVER_FRAC, shift=CTRL_SHIFT_DEG, seed=SEED,
                       aps=AP_FRACS, rin=BASE_RINNER, smask=S_MASK_MJY,
                       scut=S_CUT_MJY, smin=S_MIN_MJY, zs=Z_SOURCE,
                       cut=CUTOUT_ARCMIN, quick=QUICK), sort_keys=True)
CFG_HASH = hashlib.md5(_CFG.encode()).hexdigest()[:10]


def cached(tag, fn, extra=""):
    """Return fn(), memoised on disk under (tag, config hash, extra)."""
    key = "%s_%s%s.pkl" % (tag, CFG_HASH, extra)
    path = os.path.join(CACHE_DIR, key)
    if USE_CACHE and os.path.exists(path):
        with open(path, "rb") as fh:
            print("    [cache hit] %s" % key)
            return pickle.load(fh)
    val = fn()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(val, fh)
    print("    [cached] %s (%.0f s elapsed)" % (key, time.time() - T0))
    return val

SCH_PARS = dict(alpha=-1.890, sstar=19.01, nstar=7014.0)
DPL_PARS = dict(alpha=1.082, beta=3.928, sstar=11.70, phistar=1.33e4)


def robust_sigma(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return MAD2SIG * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def apply_matched_filter(d, sigma, valid, K, K2):
    """Chapin+2011 inverse-variance matched filter; see H1v2 for verification.
    MUST be called on a cutout padded by >= half the kernel width."""
    w = np.where(valid, 1.0 / np.maximum(sigma, 1e-30) ** 2, 0.0)
    num = oaconvolve(np.where(valid, d, 0.0) * w, K, mode="same")
    den = oaconvolve(w, K2, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def poly_baseline_region(cut, order, fit_mask, valid):
    """Fit the polynomial on `fit_mask` only, subtract it everywhere on `valid`.

    Cluster cutouts must not have their baseline fitted on the cluster itself,
    so the fit uses the OUTER region r > BASE_RINNER*theta500.  The control
    cutout uses the SAME annulus (its parent cluster's theta500) so that the
    two have identical baseline degrees of freedom -- otherwise a difference in
    the fit would propagate straight into Delta-xi.
    """
    ny, nx = cut.shape
    y, x = np.mgrid[:ny, :nx]
    xs = (x - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
    ys = (y - (ny - 1) / 2.0) / ((ny - 1) / 2.0)
    terms = [(xs ** i) * (ys ** (t - i))
             for t in range(order + 1) for i in range(t + 1)]
    sel = fit_mask & valid
    if sel.sum() < 3 * len(terms):
        sel = valid
    A = np.stack([t[sel] for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, cut[sel], rcond=None)
    surf = sum(c * t for c, t in zip(coef, terms))
    return np.where(valid, cut - surf, np.nan)


def mvir_from_m500(m500, r500_mpc, z):
    """Invert M500 -> M_vir (Duffy c_vir, Bryan-Norman overdensity).
    Copied from Planck_analysis/planck_lensed_pd.py so the two chapters use the
    identical conversion."""
    dc = bryan_norman_delta_c(z, COSMO)
    rho_c = COSMO.critical_density(z).to(u_ap.Msun / u_ap.Mpc ** 3).value

    def _g(c):
        return np.log(1.0 + c) - c / (1.0 + c)

    def resid(log_m):
        m = 10.0 ** log_m
        c = duffy_cvir(m, z, COSMO)
        r_vir = (3.0 * m / (4.0 * np.pi * dc * rho_c)) ** (1.0 / 3.0)
        m_enc = m if r500_mpc >= r_vir else m * _g(c * r500_mpc / r_vir) / _g(c)
        return np.log10(m_enc) - np.log10(m500)

    return 10.0 ** brentq(resid, 12.5, 17.0, xtol=1e-4)


#%% ------------- cell 1: map, catalogue, and the paired sample ---------------
print("=" * 79)
print("H2  Lensed Delta-xi(u) behind eFEDS clusters, GAMA-09 350 um")
print("=" * 79)
if not os.path.exists(MAP_FILE):
    raise SystemExit("ERROR: %s not found; set CIB_HERSCHEL_DIR." % MAP_FILE)

hdul = fits.open(MAP_FILE, memmap=True)
hm = hdul[8].header
FWHM, PIX = float(hm["FWHM"]), float(hm["PIXSIZE"])
K = hdul[8].data.astype(np.float64)
K2 = K ** 2
KHW = K.shape[0] // 2
BEAM_PIX = 1.133 * (FWHM / PIX) ** 2
wcs = WCS(hdul[1].header)
MASK = hdul[5].data
NY, NX = MASK.shape

SIDE = int(round(CUTOUT_ARCMIN * 60.0 / PIX)) | 1
PAD = KHW
BIG = SIDE + 2 * PAD
HALF, HALFB = SIDE // 2, BIG // 2
EDGE = int(np.ceil(FWHM / PIX))
DECL = int(round(FWHM / PIX)) | 1
TRIM = (slice(PAD, PAD + SIDE), slice(PAD, PAD + SIDE))

print("beam %.2f\" | pixel %.1f\" | extract %d pix (%.1f') -> trim %d pix (%.0f')"
      % (FWHM, PIX, BIG, BIG * PIX / 60.0, SIDE, CUTOUT_ARCMIN))

# --- catalogue ---------------------------------------------------------------
cat = fits.getdata(CLUSTER_FILE, 1)
names = np.array([str(s).strip().replace("eFEDS ", "") for s in cat["ID"]])
mass_tab = {}
for line in open(TABLEC1):
    try:
        mass_tab[line[0:16].strip()] = float(line[45:51])   # logM500R0.5B
    except ValueError:
        pass
print("\neFEDS catalogue %d entries | Chiu+2022 Table C1 masses %d"
      % (len(cat), len(mass_tab)))

cx_all, cy_all = wcs.all_world2pix(cat["RA"], cat["DEC"], 0)


def padded_clean(xc, yc, frac=COVER_FRAC):
    """Is the FULL padded extraction usable?  See docstring for why this is
    strict: masked pixels inside the kernel footprint distort the filter."""
    xc, yc = int(round(xc)), int(round(yc))
    if not (HALFB <= xc < NX - HALFB and HALFB <= yc < NY - HALFB):
        return False
    sub = MASK[yc - HALFB:yc + HALFB + 1, xc - HALFB:xc + HALFB + 1]
    if sub.shape != (BIG, BIG):
        return False
    return (sub == 0).mean() >= frac


has_m = np.array([n in mass_tab for n in names])
z_ok = np.isfinite(cat["z"]) & (cat["z"] > 0)
keep = np.zeros(len(cat), bool)
for i in range(len(cat)):
    if has_m[i] and z_ok[i] and np.isfinite(cx_all[i]):
        keep[i] = padded_clean(cx_all[i], cy_all[i])
IDX = np.where(keep)[0]
NCL = len(IDX)
print("clusters with mass + z + %.0f%%-clean padded cutout: %d"
      % (100 * COVER_FRAC, NCL))
if QUICK:
    IDX = IDX[:30]
    NCL = len(IDX)
    print("  QUICK mode: using %d" % NCL)

# --- masses, radii, lenses ---------------------------------------------------
z_cl = cat["z"][IDX].astype(float)
logM_h = np.array([mass_tab[n] for n in names[IDX]])
M500 = 10.0 ** logM_h / H_LITTLE                 # h^-1 Msun -> Msun
rho_c = COSMO.critical_density(z_cl).to(u_ap.Msun / u_ap.Mpc ** 3).value
R500 = (3.0 * M500 / (4.0 * np.pi * 500.0 * rho_c)) ** (1.0 / 3.0)     # Mpc
DA = COSMO.angular_diameter_distance(z_cl).to(u_ap.Mpc).value
TH500 = np.degrees(R500 / DA) * 60.0                                   # arcmin
MVIR = np.array([mvir_from_m500(M500[i], R500[i], z_cl[i]) for i in range(NCL)])
LENSES = [NFWLens(MVIR[i], z_cl[i], Z_SOURCE, cosmo=COSMO) for i in range(NCL)]

print("\nsample: z %.2f-%.2f (median %.2f)" % (z_cl.min(), z_cl.max(), np.median(z_cl)))
print("        log10 M500/Msun %.2f-%.2f (median %.2f)"
      % (np.log10(M500).min(), np.log10(M500).max(), np.median(np.log10(M500))))
print("        log10 Mvir/Msun median %.2f" % np.median(np.log10(MVIR)))
print("        theta500 %.2f-%.2f' (median %.2f')"
      % (TH500.min(), TH500.max(), np.median(TH500)))
print("        source redshift assumed z_s = %.1f" % Z_SOURCE)
_mu_c = np.array([L.mu(np.array([0.5 * t]))[0] for L, t in zip(LENSES, TH500)])
print("        magnification at 0.5*theta500: %.4f-%.4f (median %.4f)"
      % (_mu_c.min(), _mu_c.max(), np.median(_mu_c)))
_bad_ap = [(TH500 * f > 0.5 * CUTOUT_ARCMIN).sum() for f in AP_FRACS]
print("        apertures exceeding the cutout half-width: %s (of %d)"
      % (dict(zip(AP_FRACS, _bad_ap)), NCL))

# --- controls: one per cluster, random direction, same cleanliness ----------
rng = np.random.default_rng(SEED)
shift_pix = CTRL_SHIFT_DEG * 3600.0 / PIX
avoid_pix = CLUSTER_AVOID_ARCMIN * 60.0 / PIX
ctrl_x, ctrl_y, ctrl_try = [], [], []
for i in IDX:
    placed = False
    for t in range(200):
        th = rng.uniform(0, 2 * np.pi)
        xc = cx_all[i] + shift_pix * np.cos(th)
        yc = cy_all[i] + shift_pix * np.sin(th)
        if not padded_clean(xc, yc):
            continue
        if np.nanmin((cx_all - xc) ** 2 + (cy_all - yc) ** 2) < avoid_pix ** 2:
            continue
        ctrl_x.append(xc)
        ctrl_y.append(yc)
        ctrl_try.append(t + 1)
        placed = True
        break
    if not placed:
        ctrl_x.append(np.nan)
        ctrl_y.append(np.nan)
        ctrl_try.append(-1)
ctrl_x, ctrl_y = np.array(ctrl_x), np.array(ctrl_y)
PAIR_OK = np.isfinite(ctrl_x)
print("\ncontrols placed at %.2f deg offset: %d/%d (median %d direction tries)"
      % (CTRL_SHIFT_DEG, PAIR_OK.sum(), NCL, int(np.median(
          [t for t in ctrl_try if t > 0]))))
if PAIR_OK.sum() < NCL:
    print("  %d clusters dropped for want of a control (pairs must be complete)"
          % (NCL - PAIR_OK.sum()))
IDX, TH500, z_cl, M500, MVIR = (IDX[PAIR_OK], TH500[PAIR_OK], z_cl[PAIR_OK],
                                M500[PAIR_OK], MVIR[PAIR_OK])
LENSES = [L for L, ok in zip(LENSES, PAIR_OK) if ok]
ctrl_x, ctrl_y = ctrl_x[PAIR_OK], ctrl_y[PAIR_OK]
NPAIR = len(IDX)
print("FINAL SAMPLE: %d cluster/control pairs" % NPAIR)
RES.update(n_pairs=NPAIR, z=z_cl, m500=M500, mvir=MVIR, th500=TH500,
           z_source=Z_SOURCE, ap_fracs=np.array(AP_FRACS),
           ra=cat["RA"][IDX], dec=cat["DEC"][IDX],
           ctrl_shift_deg=CTRL_SHIFT_DEG)


#%% -------- cell 2: process every cutout, identically for both samples -------
print("\n" + "-" * 79)
print("cell 2: filtering, masking, baselining and peak extraction")
print("-" * 79)
print("""  Every operation below is applied identically to the cluster and to its
  paired control, INCLUDING the use of the cluster's theta500 to define the
  control's baseline annulus and apertures.  Any asymmetry in the analysis
  geometry would appear directly in Delta-xi and be indistinguishable from
  signal.""")

_yy, _xx = np.mgrid[:SIDE, :SIDE]
R_PIX = np.hypot(_yy - HALF, _xx - HALF) * PIX / 60.0        # arcmin from centre
INTERIOR = np.zeros((SIDE, SIDE), bool)
INTERIOR[EDGE:SIDE - EDGE, EDGE:SIDE - EDGE] = True


def process_cutout(xc, yc, th500):
    """Return (peak values, peak radii [arcmin], diagnostics) for one cutout."""
    xc, yc = int(round(xc)), int(round(yc))
    B = (slice(yc - HALFB, yc + HALFB + 1), slice(xc - HALFB, xc + HALFB + 1))
    img = hdul[1].data[B].astype(np.float64) * JY2MJY
    err = hdul[3].data[B].astype(np.float64) * JY2MJY
    val = np.isfinite(img) & np.isfinite(err) & (err > 0) & (MASK[B] == 0)
    cut = apply_matched_filter(img, err, val, K, K2)[TRIM]
    valid = np.isfinite(cut)

    # bright-source mask, identical rule for cluster and control
    hot = valid & (cut > S_MASK_MJY)
    n_hot = int(hot.sum())
    if MASK_GROW > 0 and n_hot:
        hot = maximum_filter(hot, size=2 * MASK_GROW + 1, mode="constant",
                             cval=False)
    valid_m = valid & ~hot

    # baseline fitted on the outer region only
    fit_mask = R_PIX > BASE_RINNER * th500
    res = poly_baseline_region(cut, 1, fit_mask, valid_m)

    filled = np.where(np.isfinite(res), res, -np.inf)
    pk = np.isfinite(res) & INTERIOR & (filled == maximum_filter(
        filled, size=DECL, mode="constant", cval=-np.inf))
    return (res[pk].astype(np.float32), R_PIX[pk].astype(np.float32),
            dict(mask_area=float(hot.sum()) / max(valid.sum(), 1),
                 n_bright=n_hot,
                 base_pix=int((fit_mask & valid_m).sum())))


def _extract_all():
    cl_pk, cl_r, ct_pk, ct_r = [], [], [], []
    dg = {"cl": [], "ct": []}
    for j in range(NPAIR):
        p, r, d = process_cutout(cx_all[IDX[j]], cy_all[IDX[j]], TH500[j])
        cl_pk.append(p)
        cl_r.append(r)
        dg["cl"].append(d)
        p, r, d = process_cutout(ctrl_x[j], ctrl_y[j], TH500[j])
        ct_pk.append(p)
        ct_r.append(r)
        dg["ct"].append(d)
        if (j + 1) % 25 == 0:
            print("    ... %d/%d pairs (%.0f s)" % (j + 1, NPAIR,
                                                    time.time() - T0))
    return cl_pk, cl_r, ct_pk, ct_r, dg


CL_PK, CL_R, CT_PK, CT_R, diag = cached("peaks", _extract_all)

n_cl = sum(p.size for p in CL_PK)
n_ct = sum(p.size for p in CT_PK)
print("\n  peaks: %d cluster, %d control (full 30' cutouts)" % (n_cl, n_ct))
print("  peak density: cluster %.4f/beam, control %.4f/beam"
      % (n_cl / (NPAIR * INTERIOR.sum() / BEAM_PIX),
         n_ct / (NPAIR * INTERIOR.sum() / BEAM_PIX)))

# --- NULL TEST N2: is the bright mask balanced between the two samples? ------
ma_cl = np.array([d["mask_area"] for d in diag["cl"]])
ma_ct = np.array([d["mask_area"] for d in diag["ct"]])
nb_cl = np.array([d["n_bright"] for d in diag["cl"]])
nb_ct = np.array([d["n_bright"] for d in diag["ct"]])
print("\n  NULL TEST N2 -- bright-mask balance (must be similar, else the two")
print("  samples are not analysed identically):")
print("    masked area fraction : cluster %.5f  control %.5f  ratio %.3f"
      % (ma_cl.mean(), ma_ct.mean(),
         ma_cl.mean() / max(ma_ct.mean(), 1e-12)))
print("    cutouts with any bright pixel: cluster %d, control %d of %d"
      % (int((nb_cl > 0).sum()), int((nb_ct > 0).sum()), NPAIR))
print("    total bright pixels  : cluster %d  control %d" % (nb_cl.sum(), nb_ct.sum()))
if ma_ct.mean() > 0 and ma_cl.mean() / ma_ct.mean() > 2.0:
    print("    *** WARNING: clusters lose substantially more area to the mask.")
    print("        Delta-xi may be biased; this is REPORTED, not corrected.")
RES.update(mask_area_cl=ma_cl, mask_area_ct=ma_ct,
           n_bright_cl=nb_cl, n_bright_ct=nb_ct)

#%% ---------------- cell 3: Delta-xi(u) with a PAIRED bootstrap --------------
print("\n" + "-" * 79)
print("cell 3: Delta-xi(u) by aperture, paired bootstrap")
print("-" * 79)
U_GRID = np.arange(25.0, 76.0, 2.5)


def xi_of(pk_list, u_grid):
    y_all = np.concatenate(pk_list) if pk_list else np.array([])
    xi = np.full(len(u_grid), np.nan)
    nex = np.zeros(len(u_grid), int)
    for i, uu in enumerate(u_grid):
        y = y_all[y_all > uu] - uu
        nex[i] = y.size
        if y.size >= MIN_EXC:
            xi[i] = fit_gpd(y)["xi"]
    return xi, nex


def in_aperture(pk_list, r_list, ap):
    """Peaks within r < ap*theta500, per cutout."""
    return [p[r < ap * t] for p, r, t in zip(pk_list, r_list, TH500)]


def _measure(ap):
    cl_a = in_aperture(CL_PK, CL_R, ap)
    ct_a = in_aperture(CT_PK, CT_R, ap)
    xi_cl, nx_cl = xi_of(cl_a, U_GRID)
    xi_ct, nx_ct = xi_of(ct_a, U_GRID)
    # PAIRED bootstrap: resample pair indices, so sky systematics common to a
    # pair cancel inside every replicate, not merely on average.
    r_ = np.random.default_rng(SEED + int(1000 * ap))
    bt = np.full((N_BOOT, len(U_GRID)), np.nan)
    for b in range(N_BOOT):
        s = r_.integers(0, NPAIR, NPAIR)
        a1, _ = xi_of([cl_a[i] for i in s], U_GRID)
        a2, _ = xi_of([ct_a[i] for i in s], U_GRID)
        bt[b] = a1 - a2
    lo, hi = np.nanpercentile(bt, [16, 84], axis=0)
    return dict(xi_cl=xi_cl, xi_ct=xi_ct, dxi=xi_cl - xi_ct,
                err=0.5 * (hi - lo), boot=bt, n_cl=nx_cl, n_ct=nx_ct,
                npk_cl=sum(p.size for p in cl_a),
                npk_ct=sum(p.size for p in ct_a))


APR = {}
for ap in AP_FRACS:
    APR[ap] = cached("dxi", lambda ap=ap: _measure(ap),
                     extra="_ap%.1f_nb%d" % (ap, N_BOOT))
    print("  aperture %.1f theta500: %6d cluster / %6d control peaks  (%.0f s)"
          % (ap, APR[ap]["npk_cl"], APR[ap]["npk_ct"], time.time() - T0))

for ap in AP_FRACS:
    A = APR[ap]
    g = np.isfinite(A["dxi"]) & np.isfinite(A["err"]) & (A["err"] > 0)
    if not g.any():
        print("\n  aperture %.1f theta500: no usable thresholds" % ap)
        continue
    w = 1.0 / A["err"][g] ** 2
    dbar = float(np.sum(w * A["dxi"][g]) / np.sum(w))
    # Naive error, treating thresholds as independent.  They are NOT: they share
    # exceedances, so this understates the error badly.  Kept only to show how
    # badly, against the bootstrap-propagated version below.
    dbar_e_naive = float(1.0 / np.sqrt(np.sum(w)))
    # Correct version: apply the SAME weighted mean to every bootstrap replicate
    # and take the scatter of the result.  This propagates the full
    # threshold-threshold covariance without ever forming the covariance matrix.
    bm = np.array([np.sum(w * r[g]) / np.sum(w) for r in A["boot"]
                   if np.all(np.isfinite(r[g]))])
    dbar_e = float(np.std(bm)) if bm.size > 20 else np.nan
    A["dxi_mean"] = dbar
    A["dxi_mean_err"] = dbar_e
    A["dxi_mean_err_naive"] = dbar_e_naive
    A["n_boot_used"] = int(bm.size)
    print("\n  aperture %.1f theta500  (%d usable thresholds)" % (ap, g.sum()))
    print("     u[mJy]   xi_cluster   xi_control     Delta-xi +- err")
    for i, uu in enumerate(U_GRID):
        if g[i] and i % 2 == 0:
            print("   %8.1f   %+9.4f    %+9.4f    %+8.4f +- %.4f"
                  % (uu, A["xi_cl"][i], A["xi_ct"][i], A["dxi"][i], A["err"][i]))
    print("     threshold-combined Delta-xi: %+.4f +- %.4f  (%.2f sigma)"
          % (dbar, dbar_e, abs(dbar) / dbar_e))
    print("     [error propagated through the %d bootstrap replicates, so the"
          % A["n_boot_used"])
    print("      threshold-threshold correlation is included.  Treating the %d"
          % g.sum())
    print("      thresholds as independent would give +-%.4f (%.2f sigma) --"
          % (dbar_e_naive, abs(dbar) / dbar_e_naive))
    print("      an over-statement by a factor %.1f.]" % (dbar_e / dbar_e_naive))

RES.update(u_grid=U_GRID, **{("ap%.1f_%s" % (ap, k)): APR[ap][k]
                             for ap in AP_FRACS
                             for k in ("dxi", "err", "xi_cl", "xi_ct",
                                       "boot", "n_cl", "n_ct")})


#%% ---------------- cell 4: the NFW lensing prediction ----------------------
print("\n" + "-" * 79)
print("cell 4: predicted Delta-xi from the NFW lens models")
print("-" * 79)
print("""  Each cluster's M500 is inverted to M_vir (Duffy concentration,
  Bryan-Norman overdensity) and used to build a truncated-NFW lens.  Within an
  aperture the magnification varies with radius, so the image-plane counts are
  the AREA-WEIGHTED mixture n_eff(S) = sum_k w_k mu_k^-2 n_0(S/mu_k), pooled
  over all clusters (`counts.MixtureCounts`).  Delta-xi_pred is then the
  difference of the xi(u) curves computed from that mixture and from the
  unlensed counts -- and because it IS a difference of two identically computed
  curves, the pixel-vs-peak offset H1c exposed largely cancels.  That is why the
  predicted Delta-xi is more trustworthy than the predicted absolute xi.""")

MODELS = {}
if WHICH_MODELS in ("both", "schechter"):
    MODELS["Schechter"] = Schechter(alpha=SCH_PARS["alpha"],
                                    log_sstar=np.log10(SCH_PARS["sstar"]),
                                    log_phistar=np.log10(SCH_PARS["nstar"]),
                                    s_min=S_MIN_MJY)
if WHICH_MODELS in ("both", "dpl"):
    MODELS["DPL"] = DoublePowerLaw(alpha=DPL_PARS["alpha"],
                                   beta=DPL_PARS["beta"],
                                   log_sstar=np.log10(DPL_PARS["sstar"]),
                                   log_phistar=np.log10(DPL_PARS["phistar"]),
                                   s_min=S_MIN_MJY)
MCOL = {"Schechter": "crimson", "DPL": "darkgreen"}


def aperture_mixture(ap_frac):
    """Pooled area-weighted mu-mixture within r < ap_frac*theta500.
    Same construction as Planck_analysis/planck_lensed_pd.py."""
    all_mu, all_w = [], []
    for L, t5 in zip(LENSES, TH500):
        edges = np.linspace(0.02 * t5, ap_frac * t5, N_RINGS + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        all_mu.append(L.mu(mid))
        all_w.append(areas)
    mu_all = np.concatenate(all_mu)
    w_all = np.concatenate(all_w)
    wn = w_all / w_all.sum()
    edges = np.geomspace(mu_all.min(), mu_all.max() * (1 + 1e-12), N_MU_BINS + 1)
    idx = np.clip(np.digitize(mu_all, edges) - 1, 0, N_MU_BINS - 1)
    w_bin = np.bincount(idx, weights=w_all, minlength=N_MU_BINS)
    mu_bin = np.bincount(idx, weights=w_all * mu_all, minlength=N_MU_BINS)
    ok = w_bin > 0
    return (mu_bin[ok] / w_bin[ok], w_bin[ok] / w_bin.sum(),
            dict(mean_mu=float(np.sum(wn * mu_all)),
                 max_mu=float(mu_all.max())))


SIGMA_INST = 5.402      # measured matched-filter noise, H1v2
PRED = {}
print("\n  %-10s %8s %10s %10s %14s" % ("model", "aperture", "<mu>", "mu_max",
                                        "max |Dxi_pred|"))
MU0 = float(np.median(np.concatenate(CT_PK)))
for nm, base in MODELS.items():
    def _unlensed(base=base):
        pp0 = PofD(base, FWHM, mu=1.0, sigma_noise=SIGMA_INST,
                   s_cut=S_CUT_MJY, d_span=300.0, n_fft=2 ** 17)
        return np.asarray(xi_of_threshold_analytic(pp0.d, pp0.p,
                                                   U_GRID - MU0)[0], float)

    xi0 = cached("xi0", _unlensed, extra="_%s" % nm)
    for ap in AP_FRACS:
        def _lensed(base=base, ap=ap):
            mus, ws, st = aperture_mixture(ap)
            mix = MixtureCounts(base, mus, ws)
            ppL = PofD(mix, FWHM, mu=1.0, sigma_noise=SIGMA_INST,
                       s_cut=S_CUT_MJY, d_span=300.0, n_fft=2 ** 17)
            return (np.asarray(xi_of_threshold_analytic(
                ppL.d, ppL.p, U_GRID - MU0)[0], float), st)

        xiL, st = cached("xiL", _lensed, extra="_%s_ap%.1f" % (nm, ap))
        PRED[(nm, ap)] = xiL - xi0
        print("  %-10s %8.1f %10.5f %10.4f %14.2e"
              % (nm, ap, st["mean_mu"], st["max_mu"],
                 np.nanmax(np.abs(xiL - xi0))))

print("""
  The predicted signal is small because eFEDS clusters are low mass (median
  log10 M500/Msun = %.2f) and the CIB counts at 350 um are steep: for a pure
  power law the lensing transformation changes only the AMPLITUDE of the counts,
  not the slope, so Delta-xi comes entirely from curvature in dN/dS.  A null
  result is therefore the expected outcome, and the deliverable of this module
  is a calibrated UPPER LIMIT plus the demonstration that the machinery is
  unbiased (cell 5).""" % np.median(np.log10(M500)))
RES.update(pred_json=json.dumps({("%s_ap%.1f" % k): list(map(float, v))
                                 for k, v in PRED.items()}))

#%% ------------------------ cell 5: null tests N1 and N3 --------------------
print("\n" + "-" * 79)
print("cell 5: null tests")
print("-" * 79)
AP_REF = AP_FRACS[min(1, len(AP_FRACS) - 1)]
cl_ref = in_aperture(CL_PK, CL_R, AP_REF)
ct_ref = in_aperture(CT_PK, CT_R, AP_REF)

print("  N1  CONTROL-vs-CONTROL: split the controls at random in half and form")
print("      Delta-xi between the halves.  Must be consistent with zero, and its")
print("      scatter is an empirical check on the paired-bootstrap errors.")
n_null = 30 if QUICK else 120


def _null1():
    r_ = np.random.default_rng(SEED + 77)
    out = np.full((n_null, len(U_GRID)), np.nan)
    for b in range(n_null):
        perm = r_.permutation(NPAIR)
        h1, h2 = perm[:NPAIR // 2], perm[NPAIR // 2:]
        a1, _ = xi_of([ct_ref[i] for i in h1], U_GRID)
        a2, _ = xi_of([ct_ref[i] for i in h2], U_GRID)
        out[b] = a1 - a2
    return out


null = cached("null1", _null1, extra="_ap%.1f_n%d" % (AP_REF, n_null))
null_mean = np.nanmean(null, axis=0)
null_sd = np.nanstd(null, axis=0)
g = np.isfinite(null_mean) & np.isfinite(null_sd) & (null_sd > 0)
z_null = null_mean[g] / (null_sd[g] / np.sqrt(n_null))
print("     mean null Delta-xi over thresholds: %+.5f (max |z| = %.2f)"
      % (np.nanmean(null_mean[g]), np.nanmax(np.abs(z_null))))
# half-sample split has ~2x the variance of the full paired comparison
emp = null_sd[g] / np.sqrt(2.0)
boot_e = APR[AP_REF]["err"][g]
rat = np.nanmedian(emp / boot_e)
print("     empirical error (half-split, /sqrt2) vs paired bootstrap: ratio %.2f"
      % rat)
print("     [%s] bootstrap errors are %s"
      % ("PASS" if 0.5 < rat < 2.0 else "CHECK",
         "consistent with the empirical scatter" if 0.5 < rat < 2.0
         else "NOT consistent -- investigate before quoting a detection"))

print("\n  N3  ROTATION: re-place the controls with a different random seed;")
print("      the result must not move by more than the errors.")
def _null3():
    rng2 = np.random.default_rng(SEED + 991)
    cx2, cy2 = [], []
    for i in IDX:
        for t in range(200):
            th = rng2.uniform(0, 2 * np.pi)
            xc = cx_all[i] + shift_pix * np.cos(th)
            yc = cy_all[i] + shift_pix * np.sin(th)
            if padded_clean(xc, yc) and np.nanmin(
                    (cx_all - xc) ** 2 + (cy_all - yc) ** 2) >= avoid_pix ** 2:
                cx2.append(xc)
                cy2.append(yc)
                break
        else:
            cx2.append(np.nan)
            cy2.append(np.nan)
    pk2, r2 = [], []
    for j in range(NPAIR):
        if np.isfinite(cx2[j]):
            p, r, _ = process_cutout(cx2[j], cy2[j], TH500[j])
        else:
            p, r = CT_PK[j], CT_R[j]
        pk2.append(p)
        r2.append(r)
    return pk2, r2


ct2_pk, ct2_r = cached("null3", _null3)
ct2_a = in_aperture(ct2_pk, ct2_r, AP_REF)
xi_ct2, _ = xi_of(ct2_a, U_GRID)
dxi2 = APR[AP_REF]["xi_cl"] - xi_ct2
d13 = dxi2 - APR[AP_REF]["dxi"]
gg = np.isfinite(d13) & (APR[AP_REF]["err"] > 0)
print("     max |Delta-xi(seed2) - Delta-xi(seed1)| / error = %.2f"
      % np.nanmax(np.abs(d13[gg] / APR[AP_REF]["err"][gg])))
print("     [%s] control placement is %s"
      % ("PASS" if np.nanmax(np.abs(d13[gg] / APR[AP_REF]["err"][gg])) < 3
         else "CHECK",
         "not driving the result" if np.nanmax(
             np.abs(d13[gg] / APR[AP_REF]["err"][gg])) < 3 else "suspicious"))
RES.update(null_mean=null_mean, null_sd=null_sd, null_ratio=float(rat),
           dxi_seed2=dxi2, ap_ref=AP_REF)


#%% -------------------------------- cell 6: figures --------------------------
print("\n" + "-" * 79)
print("cell 6: figures")
print("-" * 79)


def _save(fig, name):
    if SAVE_FIGURES:
        p = os.path.join(FIGURE_DIR, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("  wrote", p)


ncol = len(AP_FRACS)
fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.4), sharey=True)
axes = np.atleast_1d(axes)
for a, ap in zip(axes, AP_FRACS):
    A = APR[ap]
    a.errorbar(U_GRID, A["dxi"], yerr=A["err"], fmt="o", color="k", ms=4,
               capsize=2, label=r"measured $\Delta\xi$")
    for nm in MODELS:
        a.plot(U_GRID, PRED[(nm, ap)], "-", color=MCOL[nm], lw=2,
               label="%s NFW prediction" % nm)
    a.axhline(0, color="0.6", lw=0.9)
    a.set_xlabel("threshold $u$ [mJy/beam]")
    a.set_title(r"$r<%.1f\,\theta_{500}$  (%d/%d peaks)"
                % (ap, A["npk_cl"], A["npk_ct"]), fontsize=10)
    a.grid(alpha=0.25)
axes[0].set_ylabel(r"$\Delta\xi(u)=\xi_{\rm cl}-\xi_{\rm ctrl}$")
axes[0].legend(fontsize=7)
fig.suptitle("H2: lensed $\\Delta\\xi(u)$, %d eFEDS cluster/control pairs"
             % NPAIR, fontsize=12)
fig.tight_layout()
_save(fig, "herschel_h2_dxi.png")

fig2, ax2 = plt.subplots(1, 3, figsize=(14.0, 4.2))
ax2[0].scatter(z_cl, np.log10(M500), c=TH500, cmap="viridis", s=26)
cb = fig2.colorbar(ax2[0].collections[0], ax=ax2[0])
cb.set_label(r"$\theta_{500}$ [arcmin]", fontsize=8)
ax2[0].set_xlabel("redshift $z$")
ax2[0].set_ylabel(r"$\log_{10} M_{500}/M_\odot$")
ax2[0].set_title("Cluster sample", fontsize=11)
ax2[0].grid(alpha=0.25)
ax2[1].scatter(cat["RA"][IDX], cat["DEC"][IDX], s=18, c="crimson",
               label="clusters")
_rr, _dd = wcs.all_pix2world(ctrl_x, ctrl_y, 0)
ax2[1].scatter(_rr, _dd, s=14, c="0.5", marker="x", label="controls")
ax2[1].set_xlabel("RA [deg]")
ax2[1].set_ylabel("Dec [deg]")
ax2[1].invert_xaxis()
ax2[1].set_title("Sky positions", fontsize=11)
ax2[1].legend(fontsize=8)
ax2[1].grid(alpha=0.25)
th = np.geomspace(0.05, 10.0, 200)
for L, t5 in list(zip(LENSES, TH500))[:40]:
    ax2[2].plot(th / t5, L.mu(th), color="0.7", lw=0.7)
ax2[2].plot(th / np.median(TH500),
            np.median([L.mu(th) for L in LENSES], axis=0), color="crimson",
            lw=2, label="median")
for ap in AP_FRACS:
    ax2[2].axvline(ap, color="k", ls=":", lw=0.8)
ax2[2].set_xscale("log")
ax2[2].set_xlim(0.05, 5)
ax2[2].set_xlabel(r"$\theta/\theta_{500}$")
ax2[2].set_ylabel(r"magnification $\mu$")
ax2[2].set_title("NFW magnification profiles", fontsize=11)
ax2[2].legend(fontsize=8)
ax2[2].grid(alpha=0.25)
fig2.tight_layout()
_save(fig2, "herschel_h2_sample.png")

fig3, ax3 = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax3[0].fill_between(U_GRID, -null_sd, null_sd, color="0.88",
                    label=r"$\pm1\sigma$ half-split scatter")
ax3[0].plot(U_GRID, null_mean, "o-", color="k", ms=4, label="N1 control-control")
ax3[0].axhline(0, color="0.5", lw=0.9)
ax3[0].set_xlabel("threshold $u$ [mJy/beam]")
ax3[0].set_ylabel(r"$\Delta\xi$ (null)")
ax3[0].set_title("N1: control-vs-control null", fontsize=11)
ax3[0].legend(fontsize=8)
ax3[0].grid(alpha=0.25)
ax3[1].errorbar(U_GRID, APR[AP_REF]["dxi"], yerr=APR[AP_REF]["err"], fmt="o",
                color="k", ms=4, capsize=2, label="seed 1")
ax3[1].plot(U_GRID, dxi2, "s--", color="tab:orange", ms=4, label="seed 2")
ax3[1].axhline(0, color="0.5", lw=0.9)
ax3[1].set_xlabel("threshold $u$ [mJy/beam]")
ax3[1].set_ylabel(r"$\Delta\xi$")
ax3[1].set_title(r"N3: control re-randomisation ($r<%.1f\theta_{500}$)" % AP_REF,
                 fontsize=11)
ax3[1].legend(fontsize=8)
ax3[1].grid(alpha=0.25)
fig3.tight_layout()
_save(fig3, "herschel_h2_nulls.png")

#%% -------------------------------- cell 7: summary --------------------------
print("\n" + "=" * 79)
print("SUMMARY  --  H2 lensed Delta-xi, GAMA-09 350 um")
print("=" * 79)
print("  pairs                   : %d eFEDS clusters + %d matched controls"
      % (NPAIR, NPAIR))
print("  median z / log10 M500   : %.2f / %.2f"
      % (np.median(z_cl), np.median(np.log10(M500))))
print("  median theta500         : %.2f arcmin  (z_s = %.1f)"
      % (np.median(TH500), Z_SOURCE))
print("  control offset          : %.2f deg, random direction" % CTRL_SHIFT_DEG)
print("  bright mask             : %.0f mJy, grown %d pix; area cl/ct %.5f/%.5f"
      % (S_MASK_MJY, MASK_GROW, ma_cl.mean(), ma_ct.mean()))
print("\n  %-9s %9s %9s %13s %9s %9s %11s"
      % ("aperture", "N_cl", "N_ct", "Delta-xi", "error", "sigma",
         "|pred| Sch"))
_signs = []
for ap in AP_FRACS:
    A = APR[ap]
    if "dxi_mean" in A and np.isfinite(A["dxi_mean_err"]):
        sg = abs(A["dxi_mean"]) / A["dxi_mean_err"]
        _signs.append(np.sign(A["dxi_mean"]))
        pv = np.nanmax(np.abs(PRED[("Schechter", ap)])) \
            if ("Schechter", ap) in PRED else np.nan
        print("  %-9.1f %9d %9d %+13.4f %9.4f %9.2f %11.2e"
              % (ap, A["npk_cl"], A["npk_ct"], A["dxi_mean"],
                 A["dxi_mean_err"], sg, pv))
print("  95%% upper limits |Delta-xi| < %s"
      % ", ".join("%.3f (r<%.1f)" % (abs(APR[ap]["dxi_mean"])
                                     + 1.96 * APR[ap]["dxi_mean_err"], ap)
                  for ap in AP_FRACS
                  if np.isfinite(APR[ap].get("dxi_mean_err", np.nan))))
if len(set(_signs)) > 1:
    print("""
  *** THE SIGN OF Delta-xi IS NOT CONSISTENT ACROSS APERTURES. ***
  The NFW prediction is positive and DECREASES monotonically with aperture, so a
  real lensing signal cannot change sign between nested apertures.  Combined
  with the fact that the largest |Delta-xi| is only comparable to the scatter
  once threshold correlations are propagated, this pattern is what noise looks
  like, and no aperture should be singled out as a detection.""")
print("\n  predicted |Delta-xi| (NFW, largest over thresholds):")
for (nm, ap), v in PRED.items():
    print("    %-10s r<%.1f theta500 : %.2e" % (nm, ap, np.nanmax(np.abs(v))))
print("\n  null tests: N1 mean %+.5f, error ratio %.2f | N2 mask ratio %.2f | "
      "N3 max shift %.2f sigma"
      % (np.nanmean(null_mean[g]), rat,
         ma_cl.mean() / max(ma_ct.mean(), 1e-12),
         np.nanmax(np.abs(d13[gg] / APR[AP_REF]["err"][gg]))))
print("  runtime                 : %.0f s" % (time.time() - T0))
print("=" * 79)
print("""
  HOW TO READ THIS RESULT.  The predicted Delta-xi is of order 1e-3 or smaller,
  far below the achievable errors: eFEDS clusters are low mass and the lensing
  transformation of a power law changes only the counts amplitude, not the
  slope, so Delta-xi is sourced purely by curvature in dN/dS.  A null detection
  is the EXPECTED outcome and is not a failure of the method -- the deliverable
  is a calibrated upper limit together with the null tests showing the pipeline
  is unbiased at that level.  If instead a significant Delta-xi appears, check
  N2 first: clusters host bright cluster-member and lensed sources, and if the
  bright mask removes more area from cluster fields than from controls, the two
  samples are no longer analysed identically.

  STILL TO DO (H2b), inherited from H1c: the prediction above is pixel-level,
  while the measurement is peak-level.  The offset largely cancels in a
  difference, but "largely" is not "exactly".  A simulation-derived peak-level
  Delta-xi -- CIBMapSimulator with a lens injected, through this identical
  pipeline -- is needed before the prediction curve is quoted as a number
  rather than an expectation.""")

npz = os.path.join(FIGURE_DIR, "herschel_lensed_350.npz")
np.savez_compressed(npz, **RES)
print("\nwrote", npz)
if SAVE_FIGURES and (os.environ.get("DISPLAY") or sys.platform == "darwin"):
    plt.show()
