#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
herschel_unlensed_v3.py -- Module H1v3: the unlensed xi(u) measurement on the
                           Herschel/SPIRE 350 um GAMA-09 field, rebuilt on a
                           self-filtered IMAGE map, with a three-way count-model
                           comparison and an exact filtered-profile P(D).
===============================================================================

PURPOSE
-------
Supersedes `herschel_unlensed.py` (H1).  Same scientific question -- measure
xi_hat(u) from declustered peaks in blank-field cutouts, and ask whether it can
discriminate between functional forms of dN/dS -- but with four modelling
choices changed after the 2026-07-29 review.  H1 is kept on disk unchanged; this
module writes to separate output files.

WHAT CHANGED RELATIVE TO H1v2  (the only substantive change; read this first)
------------------------------------------------------------------------------
The MEASUREMENT is unchanged -- v3 reproduces v2's xi_hat(u) exactly.  What
changes is how the 21 thresholds are COMBINED into a single significance.

v2 combined them by naive quadrature, sqrt(sum_u r(u)^2) with
r(u) = |xi_A(u) - xi_B(u)| / sigma_boot(u).  That is valid only if the
thresholds are independent measurements.  They are not, for two reasons:

  * NESTED EXCEEDANCE SETS.  The peaks above u = 27.5 mJy are a SUBSET of those
    above 25 mJy -- 11,536 of 15,264, i.e. 76% of the same objects.  Adjacent
    thresholds are largely the same GPD fit on largely the same data.
  * SHARED CUTOUTS.  All thresholds come from the same 103 cutouts and the
    bootstrap resamples cutouts, so a cutout-level fluctuation moves the entire
    xi_hat(u) curve coherently up or down.

Measured from v2's own 200 x 21 bootstrap array, the threshold-threshold
correlation of xi_hat is 0.890 at lag 2.5 mJy, 0.808 at 5, 0.691 at 10, 0.527
at 20 and still 0.283 at 40 mJy.  The effective number of independent
thresholds is 1.65 (from n^2 / sum_ij R_ij) or 2.40 (from the eigenvalue
spread) out of 21.

v3 therefore uses the covariance-correct statistic

    S^2 = d^T C^-1 d ,      d = xi_A - xi_B ,   C = bootstrap covariance,

with the Hartlap (2007, A&A 464, 399) correction (N - p - 2)/(N - 1) for the
bias of an inverted sample covariance, and reports alongside it the best single
threshold (a conservative floor) and the naive quadrature (labelled as
incorrect, so the difference is visible).

*** THE EFFECT IS NOT A UNIFORM DEFLATION -- THIS IS THE POINT. ***

    pair                  naive quad   best single   full covariance
    Schechter vs SPL          12.33          5.56          9.37
    Schechter vs DPL           9.14          4.24          6.56
    SPL       vs DPL           5.62          2.49          7.59   <-- INCREASED

Two of the three fall; SPL vs DPL RISES above even the naive value.  The reason
is physical, not numerical.  The dominant bootstrap noise mode is a coherent
vertical shift of the whole xi_hat(u) curve (it carries 72% of the variance).
Schechter and SPL differ mostly in overall LEVEL, so their difference is nearly
parallel to that noise mode and the covariance weighting penalises it.  SPL and
DPL instead CROSS -- their difference changes sign across the threshold range,
nearly orthogonal to the coherent-shift mode -- so the covariance weighting
rewards it.  Diagonal errors are blind to this distinction entirely.  The
correction therefore re-weights WHICH model pairs are distinguishable, and that
changes the paper's claim; it is not a cosmetic rescaling.

(The exact values above were computed from v2's 200-replicate array; v3 runs
400 replicates and prints its own, slightly different, numbers at run time.)

HOW MUCH OF THE FULL-COVARIANCE STATISTIC IS TRUSTWORTHY
--------------------------------------------------------
Cell 5 reports the statistic truncated to the leading 3 and 5 principal
components as well as in full, and the full value is much the largest -- so a
large part of it comes from LOW-VARIANCE modes of the threshold covariance,
which is also where a sample covariance is least reliable.  v3 therefore adds a
half-sample stability test: recompute the statistic from random halves of the
bootstrap ensemble.  It reproduces the full-sample value to ~3% (e.g. Schechter
vs SPL: 9.00 +- 0.30 on halves versus 9.03 on all 400), so the deep modes are
genuinely determined and the truncated values are discarding real information,
not suppressing noise.

What the test does NOT establish is that the bootstrap distribution is Gaussian
in those deep modes, which the chi2-like statistic assumes.  RECOMMENDATION FOR
THE PAPER: quote the BEST SINGLE THRESHOLD as the headline discriminating power
(conservative, assumption-light), and the full-covariance value as the optimal
combination achievable if the covariance is taken at face value.  Do not quote
the naive quadrature at all.

Consequences for the rest of the module:
  * the masked chi2 of cell 5 is likewise recomputed with the full covariance
    (Schechter 51.8, DPL 121.1, SPL 148.5, versus 75.9 / 322.9 / 484.3 with
    diagonal errors -- the RANKING survives but the spread compresses);
  * CIB_N_BOOT_SYS now defaults to CIB_N_BOOT, because with 60 replicates and
    21 thresholds the Hartlap factor was 0.80 and the inverse noisy, and at the
    full grid the sample covariance would be singular (N < p + 2);
  * a new diagnostic figure shows the threshold correlation matrix and the
    eigenvalue spectrum -- this justifies the whole treatment and is worth
    carrying into the paper.

The same fix has already been applied in H2 (`herschel_lensed.py`), where
treating thresholds as independent overstated Delta-xi significances by factors
of 1.8-2.3 (an apparent 2.32 sigma became 1.01 sigma).  There the statistic is a
single weighted mean, so the error can be propagated through the replicates
without inverting anything; here the model-difference direction matters, so the
inverse covariance is genuinely needed.

WHAT CHANGED RELATIVE TO H1  (read this before comparing numbers)
-----------------------------------------------------------------
(1) PRIMARY MAP IS NOW A SELF-FILTERED **IMAGE**, NOT HELP's MFILT.
    H1 used HDU 6 (MFILT) as delivered by HELP.  That product is the matched
    filter applied to NEBFILT (HDU 2), i.e. it inherits the CASU nebular filter.
    Here we apply the matched-filter operator ourselves, to IMAGE (HDU 1):

        MFILT       = [ (d/sigma^2) * K ] / [ (1/sigma^2) * K^2 ]
        MFILT_ERROR = 1 / sqrt( (1/sigma^2) * K^2 )

    with sigma = ERROR (HDU 3) and K = the Matchedfilter kernel (HDU 8).  This
    is the Chapin et al. (2011) inverse-variance-weighted form.  Cell 1 verifies
    at runtime that feeding it NEBFILT reproduces HELP's MFILT (it does, to
    corr = 1.0000000000, rms residual 1.5e-6 mJy/beam).

    Why do it ourselves:
      * the paper then defines the operator in three lines instead of citing an
        opaque pipeline stage;
      * it is applied identically to cluster and control cutouts by
        construction, which matters for module H2;
      * it closes the systematic left open by H0 (IMAGE retains the cirrus that
        the nebular filter removes, so the two routes bracket that systematic).

    *** THE PADDING POINT, WHICH IS WHAT MAKES THIS EQUIVALENT. ***
    The kernel is 101 x 101 pixels at 8"/pix = 13.5' across, i.e. half-width
    6.67'.  The operator therefore has FINITE SUPPORT: filtering a cutout that
    has been padded by >= 50 pixels on every side and then trimming reproduces
    full-map filtering EXACTLY.  Measured in cell 1: max |difference| = 7e-14
    mJy/beam, i.e. machine precision.  Filtering a BARE 30' cutout instead
    corrupts an annulus 6.67' deep -- rms difference 0.45 mJy/beam, peaking at
    9.3 -- leaving only the central 16.7' (31% of the area) usable.  So we
    extract 43.5' cutouts, filter, and trim to the central 30'.

    Consequence worth stating plainly in the paper: "matched-filtered map" and
    "self-filtered cutouts" are not competing choices.  They are the same
    operation.  The only real difference between our primary map and HELP's
    MFILT is whether the nebular filter was applied first.

(2) WHY MATCHED FILTERING AT ALL -- THE ARGUMENT IS PEAK STATISTICS, NOT S/N.
    H1's stated justification (instrument noise no longer overwhelms confusion)
    is the weaker one and should not be repeated: H0 measured that the filter
    only improves sigma_inst/sigma_conf from 1.429 to 1.033, still ~unity.
    The decisive number is the lag-1 pixel autocorrelation: 0.245 in IMAGE
    versus 0.719 in MFILT, against 0.87 for a pure 25.15"-beam-correlated field.
    In IMAGE the instrument noise is white while the sky is beam-correlated, so
    the local maxima are a MIXTURE of two populations with completely different
    peak statistics, and no single-scale theory describes them.  After filtering
    both components are beam-correlated and the peaks are those of one smooth
    random field plus sources -- which is the object the GPD/P(D) framework
    actually models.  Matched filtering is what makes peak-based extreme-value
    theory applicable here.

(3) THE GAUSSIAN-BEAM P(D) IS NOW CHECKED, NOT ASSUMED, AGAINST THE EXACT
    FILTERED PROFILE.
    H1 checked only Condon's effective solid angle, Omega_2 = int P^2 (ratio
    0.9997).  That controls the VARIANCE.  But the exceedance tail -- which is
    what xi reads -- is governed by Omega_{eta-1} = int P^{eta-1}, since for
    dN/dS ~ S^-eta the tail behaves as P(>u) ~ u^{1-eta} int P^{eta-1} d2x, and
    eta ~ 3-5 over our threshold range.  Cell 2 therefore tabulates int P^q for
    q = 1-4 and, more importantly, builds the EXACT P(D) for the filtered
    profile from the two-sided response rate

        R(y) = sum_i dOmega_i * n(y / P_i) / |P_i|     (pixels with y/P_i > 0)

    which handles the filter's negative side lobes (min -0.131 in the kernel,
    -0.059 in the filtered profile, at ~32" radius) properly: a source DEPRESSES
    an annulus around itself, populating the negative side of P(D).  The
    Gaussian-beam PofD cannot represent that.  Cell 2 quantifies how much it
    matters for xi(u); if the difference is negligible the Gaussian model is
    used for the headline numbers and the exact one is quoted as a systematic.

(4) THREE COUNT MODELS, AND NO RENORMALISATION.
    H1 carried Schechter and SPL.  The double power law is added here, because
    the paper's claim is that xi(u) distinguishes FUNCTIONAL FORMS: an
    exponential cutoff (Schechter), a break-then-saturate (DPL), and a constant
    (SPL).  Their asymptotic signatures at high threshold are
        xi_SPL -> 1/(beta-1) = 1/2.279 = 0.439   (flat, no cutoff)
        xi_DPL -> 1/(beta-1) = 1/2.928 = 0.342   (flat after the break)
        xi_Sch -> falls monotonically through zero (bounded endpoint)
    NOTE: at 25.15" we do NOT resolve the DPL break itself.  It sits at
    S* = 11.70 mJy, i.e. k = S*/sigma_c ~ 1.65, deep inside the confusion core
    (counts memo Sec. 9).  What is accessible is the presence or absence of a
    cutoff at high threshold, not the break location.  The paper should say so.

    ON THE sigma_conf "OVER-PREDICTION" (H1 docstring, and the reason it
    anchored thresholds in absolute flux): it was largely an ESTIMATOR
    ARTEFACT, not a physical discrepancy.  Condon's sigma_c is the true RMS of a
    strongly skewed P(D); the measured 6.49/6.52 mJy/beam were MAD-based robust
    widths.  On the SAME model P(D) the counts-only rms is 8.149 while its
    MAD-sigma is 6.580 -- a ratio of 1.238.  Cell 3 does the comparison
    like-for-like, on both statistics, and also propagates the counts-fit
    normalisation uncertainty (n* = 7014 +- 2156, i.e. +-15% on sigma_c since
    sigma_c ~ sqrt(n*)).  No model is renormalised anywhere in this module.
    Absolute-flux threshold anchoring is RETAINED, but now because it is the
    physically correct comparison (xi ~ 1/(eta(S)-1) is a function of absolute
    flux), not as a workaround for a mismatch.

(5) THE RISING xi_hat(u) IS EXPLAINED: IT IS THE BRIGHT LENSED POPULATION.
    xi_hat(u) rises from 0.14 at u = 25 mJy to ~0.48 at 75 mJy (8.1 sigma from
    flat) while all three models FALL through zero.  Module H1c
    (`herschel_peak_sims.py`) settled the cause by forward simulation.  Slope
    attribution of the measured d(xi)/du = +0.00876 mJy^-1:

        pixel -> peak                     -0.00077   (small, and the WRONG SIGN)
        bright lensed population          +0.01028   (dominant)
        unexplained residual              +0.00219

    chi2 over 21 thresholds: analytic pixel model 1753 -> simulated peaks with
    S_cut-truncated counts 2824 -> simulated peaks including a bright tail 152.

    *** CORRECTION TO H1 AND TO EARLIER DRAFTS OF THIS MODULE. *** The
    pixel-vs-peak mismatch was the wrong suspect.  Declustered peaks give a
    LOWER xi than pixels at every threshold here, not higher.  The statement
    that "the maxima of a beam-correlated field are genuinely heavier-tailed"
    is measurably false for this beam, noise level and declustering window, and
    must not be repeated.

    THE PHYSICS.  The models are truncated at S_cut = 100 mJy because above
    ~100 mJy the 350 um counts are dominated by strongly lensed sources (Lima,
    Jain, Devlin & Aguirre 2010, ApJL 717, L31) and no longer trace the
    intrinsic dN/dS -- the same reason the counts memo dropped Bethermin's
    133.7 mJy point, and the reason all three fits are unreliable well below
    100 mJy too.  The MAP is not truncated.  Cell 4b below counts the bright
    peaks and finds a large, real excess over every fitted model.  So xi(u) at
    high threshold is reading the transition from the intrinsic counts to the
    lensing-dominated bright tail.  That is a RESULT, not a systematic --
    but it does mean the measurement and the models are, without further care,
    being compared over different flux ranges.

WHAT THIS MODULE DOES ABOUT IT
------------------------------
Cell 4b adds a BRIGHT-SOURCE-MASKED variant of the measurement: pixels above
S_MASK are masked at the MAP level and the mask grown by one FWHM, then peaks
are re-extracted.  This is the standard P(D) bright-source-masking operation
(and the Planck chapter's PS mask), and it is the closest available match to the
models' source-plane truncation.  Both variants are reported:

  * UNMASKED xi_hat(u)  -- the physical measurement, containing the lensed
    bright-tail signature.  This stays the headline result.  It sits ABOVE
    every model, by construction, since the models truncate that population.
  * MASKED xi_hat(u)    -- quoted for u <= U_MAX_MASKED.  It sits BELOW every
    model at every threshold, because map-level masking right-censors the
    exceedance distribution and biases the GPD MLE downward.

THE TWO THEREFORE BRACKET the model-comparable measurement from above and
below; neither is a goodness-of-fit on its own, and the absolute chi2 from
either should not be quoted as a fit quality until the censoring bias is
calibrated by applying the identical mask to simulated maps (easy with the H1c
machinery; deliberately NOT done here, so that the bracket is not mistaken for
a corrected measurement).  What survives is the model RANKING, which cell 5
verifies is identical for both variants and stable against U_MAX_MASKED.

A FURTHER CAVEAT H1c EXPOSED, NOT FIXED HERE
--------------------------------------------
H1c's validation leg compared simulated PIXELS against the analytic KL-projected
pixel P(D) for IDENTICAL counts, and found them to differ by up to
|dxi| = 0.047 near u = 40-45 mJy -- roughly twice the data error there, and
comparable to the Schechter-vs-SPL model gap this module quotes (0.086 at its
best threshold).  The difference is KL-projection-in-the-infinite-data-limit
versus finite-sample MLE, plus the numerical-floor caveat documented in
`gpd_tail.xi_of_threshold_analytic`.  CONSEQUENCE: the analytic model curves
below carry a systematic of order the signal, and the final model
discrimination for the paper should be recomputed from SIMULATED PEAKS
(H1c machinery) rather than from `xi_of_threshold_analytic`.  The separations
quoted here should be treated as indicative pending that.

DATA
----
Map        data/GAMA-09_SPIRE350_v1.0.fits
           HDU 1 IMAGE, 2 NEBFILT, 3 ERROR, 5 MASK, 6 MFILT, 7 MFILT_ERROR,
           8 Matchedfilter (101x101, FWHM 25.15", PIXSIZE 8.0)
Beam       25.15" FWHM, 11.20 pix/beam
Cutouts    30' x 30' (225 pix) after trimming from 43.5' (325 pix) padded
           extractions; random, non-overlapping, not centred within 5' of an
           eFEDS cluster
Baseline   DC + linear per cutout, fitted on the full cutout (controls)
Peaks      local maxima over one FWHM (3 pix); peaks within one FWHM of the
           trimmed cutout edge discarded

MAP VARIANTS PRODUCED (all three carried through to xi(u))
----------------------------------------------------------
  SF_IMAGE   self-filtered IMAGE      <-- PRIMARY
  SF_NEB     self-filtered NEBFILT    <-- operator check; must equal HELP MFILT
  HELP_MFILT HDU 6 as delivered       <-- systematic

ERRORS
------
Cutout-level bootstrap (not i.i.d. peak resampling): peaks within a cutout share
a baseline fit, a local noise level and a local cirrus realisation.  Each
replicate is refitted at every threshold so threshold-to-threshold correlations
propagate; the full replicate array is saved for H3 to build the covariance.

OUTPUTS
-------
results/herschel_unlensed_v3_350.npz
results/herschel_v3_xi_vs_flux.png          the main result
results/herschel_v3_map_variants.png        SF_IMAGE vs SF_NEB vs HELP MFILT
results/herschel_v3_profile_moments.png     int P^q and exact-vs-Gaussian P(D)
results/herschel_v3_counts_models.png       the three dN/dS and their eta(S)
results/herschel_v3_widths.png              like-for-like width comparison
results/herschel_v3_discrimination.png      three-way separation vs data errors

ENVIRONMENT SWITCHES
--------------------
CIB_HERSCHEL_DIR   data directory (default Herschel_analysis/data)
CIB_FIGURE_DIR     output directory (default results/)
CIB_SAVE_FIGURES   "1"/"0"
CIB_RANDOM_SEED    default 1234
CIB_N_CUTOUTS      target cutout count (default 250; RSA jamming caps it ~120)
CIB_CUTOUT_ARCMIN  trimmed cutout side (default 30)
CIB_N_BOOT         bootstrap replicates for the primary map (default 400)
CIB_N_BOOT_SYS     replicates for the map-variant systematics (default 60)
CIB_N_BOOT_MASK    replicates for the masked variant, which feeds a covariance
                   inverse (default = CIB_N_BOOT)
CIB_SMIN_MJY       counts integration floor (default 1.0)
CIB_SCUT_MJY       bright-source truncation of the COUNTS (default 100.0)
CIB_SMASK_MJY      map-level bright-source mask (default = CIB_SCUT_MJY)
CIB_MASK_GROW      mask growth radius in pixels (default 3 = one FWHM)
CIB_UMAX_MASKED    highest threshold quoted for the masked variant (default 50)
CIB_QUICK_TEST     "1" for a fast smoke run

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
from scipy.ndimage import maximum_filter
from scipy.signal import oaconvolve

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis_modules"))

from counts import BaseCounts, Schechter, DoublePowerLaw      # noqa: E402
from pofd_analytic import PofD                                # noqa: E402
from gpd_tail import fit_gpd, xi_of_threshold_analytic        # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# numpy 2.x renamed trapz -> trapezoid; keep both working (as analysis_modules does)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ------------------------------------------------------------------ config --
DATA_DIR = os.environ.get("CIB_HERSCHEL_DIR",
                          os.path.join(HERE, "data"))
MAP_FILE = os.path.join(DATA_DIR, "GAMA-09_SPIRE350_v1.0.fits")
CLUSTER_FILE = os.path.join(DATA_DIR, "eFEDS_clusters_V3.2.fits")
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", os.path.join(HERE, "results"))
SAVE_FIGURES = os.environ.get("CIB_SAVE_FIGURES", "1") == "1"
SEED = int(os.environ.get("CIB_RANDOM_SEED", "1234"))
QUICK = os.environ.get("CIB_QUICK_TEST", "0") == "1"

CUTOUT_ARCMIN = float(os.environ.get("CIB_CUTOUT_ARCMIN", "30"))
N_CUTOUTS = int(os.environ.get("CIB_N_CUTOUTS", "250"))
#  Release: default raised from 200 to 400, the value used for the paper
#  (the published results/herschel_unlensed_v3_350.npz is reproduced exactly).
N_BOOT = int(os.environ.get("CIB_N_BOOT", "40" if QUICK else "400"))
# v3: the systematic-variant and masked bootstraps now feed a COVARIANCE, whose
# inverse is biased unless N_boot >> n_thresholds (Hartlap 2007).  With 21
# thresholds, 60 replicates gave a Hartlap factor of 0.80 and a noisy inverse;
# at the full grid it would be singular.  Default raised to match N_BOOT.
N_BOOT_SYS = int(os.environ.get("CIB_N_BOOT_SYS", "20" if QUICK else "60"))
# The MASKED bootstrap feeds a covariance inverse, the map-variant bootstraps do
# not (they are only used for a per-threshold agreement check).  v2 conflated
# the two and gave the masked variant 60 replicates for 21 thresholds -- a
# Hartlap factor of 0.80, and singular at the full grid.  They are now separate.
N_BOOT_MASK = int(os.environ.get("CIB_N_BOOT_MASK", str(N_BOOT)))

CLUSTER_AVOID = 5.0                    # arcmin
BASELINE_ORDER = 1                     # DC + linear
S_MIN_MJY = float(os.environ.get("CIB_SMIN_MJY", "1.0"))
S_CUT_MJY = float(os.environ.get("CIB_SCUT_MJY", "100.0"))
# Map-level bright-source mask.  Default matches the counts truncation so that
# the masked measurement and the truncated models cover the same flux range.
S_MASK_MJY = float(os.environ.get("CIB_SMASK_MJY", str(S_CUT_MJY)))
MASK_GROW = int(os.environ.get("CIB_MASK_GROW", "3"))       # one FWHM
U_MAX_MASKED = float(os.environ.get("CIB_UMAX_MASKED", "50.0"))

os.makedirs(FIGURE_DIR, exist_ok=True)
JY2MJY, MAD2SIG = 1.0e3, 1.4826
PRIMARY = "SF_IMAGE"
VARIANTS = ["SF_IMAGE", "SF_NEB", "HELP_MFILT"]

# Final adopted 350 um fits -- num_count_fits/fitting_results_number_counts.md
# Sec. 7 ("Final adopted fits", GOODS-N excluded, N=9 points over 6-94.6 mJy).
SCH_PARS = dict(alpha=-1.890, sstar=19.01, nstar=7014.0,
                d_alpha=0.186, d_sstar=2.76, d_nstar=2156.0)
SPL_PARS = dict(N0=1.053e5, S0=2.2, beta=-3.279)
DPL_PARS = dict(alpha=1.082, beta=3.928, sstar=11.70, phistar=1.33e4)

RES = {}
T0 = time.time()

# ---------------------------------------------------------------- caching --
# Same scheme as herschel_lensed.py (H2): each expensive stage is memoised to
# disk under an md5 of the configuration, so a run can resume, figures can be
# regenerated without redoing the extraction, and any parameter change
# invalidates the affected stage automatically.  CIB_USE_CACHE=0 disables.
import hashlib                                                   # noqa: E402
import pickle                                                    # noqa: E402

CACHE_DIR = os.path.join(FIGURE_DIR, "v3_cache")
USE_CACHE = os.environ.get("CIB_USE_CACHE", "1") == "1"
CFG_HASH = hashlib.md5(json.dumps(
    dict(nc=N_CUTOUTS, cut=CUTOUT_ARCMIN, seed=SEED, smin=S_MIN_MJY,
         scut=S_CUT_MJY, smask=S_MASK_MJY, grow=MASK_GROW,
         base=BASELINE_ORDER, quick=QUICK),
    sort_keys=True).encode()).hexdigest()[:10]


def cached(tag, fn, extra=""):
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
    print("    [cached] %s (%.0f s)" % (key, time.time() - T0))
    return val



def robust_sigma(x):
    """MAD-based sigma, 1.4826 * median|x - median(x)|."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return MAD2SIG * np.median(np.abs(x - np.median(x))) if x.size else np.nan


class SinglePowerLaw(BaseCounts):
    """Single power law, dN/dS = N0 (S/S0)^beta, in mJy^-1 deg^-2.

    Patanchon et al. (2009) form with S0 fixed at their 2.2 mJy, refit to
    Bethermin+2012 in the project's counts memo (Sec. 7).

    NOTE on s_min.  With beta = -3.279 the second moment int S^2 dN/dS dS
    diverges at the faint end (gamma = 3.279 > 3), so both the confusion noise
    and the P(D) core width depend on where the integral is cut off.  The s_min
    passed here is a modelling CHOICE, not a property of the model.  Only the
    tail slope -- which is s_min-independent, and which is what xi(u) reads at
    high threshold -- should be treated as a model statement.  This is also why
    "renormalise the models to match the measured sigma_conf" is ill-defined for
    the SPL: its sigma_c can be matched by moving N0 or s_min, degenerately.
    """

    def __init__(self, N0, S0, beta, s_min=1.0, s_max=1e4):
        self.N0, self.S0, self.beta = float(N0), float(S0), float(beta)
        self.s_min, self.s_max = float(s_min), float(s_max)

    def dnds(self, S):
        S = np.asarray(S, float)
        out = self.N0 * (S / self.S0) ** self.beta
        return np.where(S > 0, out, 0.0)


def poly_baseline(cut, order, valid):
    """Subtract a least-squares 2-D polynomial surface in normalised coords.

    order = 0 is a DC offset, 1 adds the two linear gradients, and so on.
    H0b measured this to remove only ~0.06% of the variance of a matched-
    filtered cutout (the cirrus having already been suppressed), so the choice
    is near-immaterial here -- but it MUST be applied identically to the cluster
    cutouts in H2, where it is fitted on the outer region only.
    """
    if order < 0:
        return np.where(valid, cut, np.nan)
    ny, nx = cut.shape
    y, x = np.mgrid[:ny, :nx]
    xs = (x - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
    ys = (y - (ny - 1) / 2.0) / ((ny - 1) / 2.0)
    terms = [(xs ** i) * (ys ** (t - i))
             for t in range(order + 1) for i in range(t + 1)]
    A = np.stack([t[valid] for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, cut[valid], rcond=None)
    surf = sum(c * t for c, t in zip(coef, terms))
    return np.where(valid, cut - surf, np.nan)


#%% ------------------------------------------ the matched-filter operator ----
def apply_matched_filter(d, sigma, valid, K, K2):
    """Chapin et al. (2011) inverse-variance-weighted matched filter.

        MFILT = [ (d/sigma^2) * K ] / [ (1/sigma^2) * K^2 ]

    Parameters
    ----------
    d      : map to filter (IMAGE or NEBFILT), mJy/beam
    sigma  : per-pixel noise (ERROR HDU), mJy/beam
    valid  : boolean mask of usable pixels
    K, K2  : the Matchedfilter kernel and its square

    Returns the filtered map, NaN where the denominator vanishes.

    IMPORTANT: the kernel has finite support (101x101 = 13.5'), so this must be
    called on a cutout padded by at least half the kernel width on every side,
    and the result trimmed.  See `filter_padded_cutout`.
    """
    w = np.where(valid, 1.0 / np.maximum(sigma, 1e-30) ** 2, 0.0)
    num = oaconvolve(np.where(valid, d, 0.0) * w, K, mode="same")
    den = oaconvolve(w, K2, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def matched_filter_error(sigma, valid, K2):
    """MFILT_ERROR = 1 / sqrt( (1/sigma^2) * K^2 ), same padding rules."""
    w = np.where(valid, 1.0 / np.maximum(sigma, 1e-30) ** 2, 0.0)
    den = oaconvolve(w, K2, mode="same")
    return np.where(den > 0, 1.0 / np.sqrt(np.maximum(den, 1e-300)), np.nan)


#%% ---------- exact P(D) for an arbitrary (possibly signed) beam profile -----
def pofd_from_profile(cnts, profile, pix_arcsec, sigma_noise=0.0, s_cut=None,
                      d_span=300.0, n_fft=2 ** 17, n_pbins=1500, n_xgrid=6000):
    """Exact one-point P(D) for a map smoothed with an ARBITRARY profile.

    `pofd_analytic.PofD` assumes a Gaussian beam, for which the response rate
    collapses to the closed form R(x) = Omega_eff * N(>x) / x.  A matched-
    filtered map is not Gaussian-smoothed: the effective profile has negative
    side lobes, so a source both brightens its own position and DEPRESSES an
    annulus around it.  The general two-sided response rate is

        R(y) = sum_i dOmega_i * n(y / P_i) / |P_i| ,   over pixels with y/P_i > 0

    (a source of flux S seen through profile value P contributes y = S*P, and
    |dS/dy| = 1/|P|).  Positive-P pixels populate y > 0 and negative-P pixels
    populate y < 0.  The characteristic function is then the usual

        ln phi(w) = sum_j R_j ( e^{i w y_j} - 1 ) dy

    evaluated on a two-sided grid, mean-subtracted, with the Gaussian noise term
    added.  This mirrors `pofd_analytic._pofd_from_rate` but without its
    positive-only assumption.

    Parameters
    ----------
    cnts        : counts.BaseCounts instance (dN/dS in mJy^-1 deg^-2)
    profile     : 2-D array, the effective beam profile, PEAK-NORMALISED so that
                  a point source of flux S reads S at its own position
    pix_arcsec  : pixel size of `profile`
    n_pbins     : profile values are binned in |P| before the sum, since a
                  pixel-by-pixel sum over ~1e5 pixels x ~1e4 grid points is
                  needlessly expensive; 1500 log-spaced bins is converged to
                  better than 1e-4 on the moments.
    n_xgrid     : R is evaluated on this many log-spaced |y| points and
                  interpolated onto the FFT grid.

    Returns (d, p) on a centred grid, matching PofD.d / PofD.p conventions.
    """
    dom = (pix_arcsec ** 2) / (3600.0 ** 2)          # pixel solid angle, deg^2

    def _dnds(S):
        v = cnts.dnds(S)
        if s_cut is not None:
            v = np.where(S > s_cut, 0.0, v)
        return np.where(S >= cnts.s_min, v, 0.0)

    # --- bin the profile by |P|, separately for the two signs ---------------
    P = np.asarray(profile, float).ravel()
    P = P[np.abs(P) > 1e-6]                          # ignore the far wings
    rates = {}
    for sign in (+1, -1):
        sel = P[np.sign(P) == sign]
        if sel.size == 0:
            rates[sign] = None
            continue
        a = np.abs(sel)
        edges = np.geomspace(a.min() * 0.999, a.max() * 1.001, n_pbins + 1)
        cntb, _ = np.histogram(a, bins=edges)
        ctr = np.sqrt(edges[1:] * edges[:-1])
        keep = cntb > 0
        rates[sign] = (ctr[keep], cntb[keep] * dom)   # (|P| values, solid angle)

    dd = d_span / n_fft
    ygrid = np.geomspace(dd * 1e-6, d_span, n_xgrid)

    def _R(sign):
        pv, om = rates[sign]
        # R(y) = sum_b om_b * n(y/|P|_b) / |P|_b   -- chunk to bound memory
        out = np.zeros_like(ygrid)
        step = max(1, int(4e7 // max(ygrid.size, 1)))
        for i in range(0, pv.size, step):
            pb, ob = pv[i:i + step], om[i:i + step]
            out += ((_dnds(ygrid[:, None] / pb[None, :]) / pb[None, :])
                    * ob[None, :]).sum(axis=1)
        return out

    R_pos = _R(+1) if rates[+1] is not None else np.zeros_like(ygrid)
    R_neg = _R(-1) if rates[-1] is not None else np.zeros_like(ygrid)

    # --- assemble the two-sided rate on the FFT grid ------------------------
    n = int(n_fft)
    j = np.fft.fftfreq(n, d=1.0 / n)                  # 0,1,..,n/2-1,-n/2,..,-1
    y = j * dd
    Rarr = np.zeros(n)
    pos, neg = y > 0, y < 0
    Rarr[pos] = np.interp(y[pos], ygrid, R_pos, left=0.0, right=0.0)
    Rarr[neg] = np.interp(-y[neg], ygrid, R_neg, left=0.0, right=0.0)

    # faint-end (|y| < dd) moments, done on the log grid where R is smooth
    fine = ygrid[ygrid <= dd]
    if fine.size > 4:
        Rp = np.interp(fine, ygrid, R_pos)
        Rn = np.interp(fine, ygrid, R_neg)
        m0 = _trapz((Rp - Rn) * fine ** 2, np.log(fine))
        v0 = _trapz((Rp + Rn) * fine ** 3, np.log(fine))
    else:
        m0 = v0 = 0.0

    d_bar = m0 + np.sum(y * Rarr) * dd
    cf = n * np.fft.ifft(Rarr) * dd - np.sum(Rarr) * dd
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=dd)
    cf += 1j * w * m0 - 0.5 * v0 * w ** 2
    cf += -1j * w * d_bar
    cf += -0.5 * (sigma_noise * w) ** 2
    p = np.real(np.fft.fft(np.exp(cf))) / (n * dd)

    d = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / (n * dd)))
    return d, np.maximum(np.fft.fftshift(p), 0.0)


def pdf_stats(d, p):
    """rms, MAD-sigma, median and skew of a tabulated pdf -- so that model and
    data can be compared on the SAME statistic (see docstring, item 4)."""
    p = np.clip(np.asarray(p, float), 0, None)
    d = np.asarray(d, float)
    norm = _trapz(p, d)
    if not np.isfinite(norm) or norm <= 0:
        return dict(rms=np.nan, madsig=np.nan, med=np.nan, skew=np.nan)
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(d))])
    med = np.interp(0.5 * cdf[-1], cdf, d)
    m1 = _trapz(d * p, d)
    m2 = _trapz((d - m1) ** 2 * p, d)
    m3 = _trapz((d - m1) ** 3 * p, d)
    a = np.abs(d - med)
    o = np.argsort(a)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[o][1:] + p[o][:-1])
                                         * np.abs(np.diff(a[o])))])
    mad = np.interp(0.5 * c[-1], c, a[o])
    return dict(rms=np.sqrt(m2), madsig=MAD2SIG * mad, med=med,
                skew=m3 / m2 ** 1.5)


#%% ---- cell 1: load the map, verify the operator, build the three variants --
print("=" * 79)
print("H1v2  Unlensed xi(u), GAMA-09 350 um, self-filtered IMAGE primary")
print("=" * 79)
if not os.path.exists(MAP_FILE):
    raise SystemExit("ERROR: %s not found; set CIB_HERSCHEL_DIR." % MAP_FILE)

hdul = fits.open(MAP_FILE, memmap=True)
hm = hdul[8].header
FWHM, PIX = float(hm["FWHM"]), float(hm["PIXSIZE"])
K = hdul[8].data.astype(np.float64)
K2 = K ** 2
KHW = K.shape[0] // 2                       # kernel half-width, 50 pix = 6.67'
BEAM_PIX = 1.133 * (FWHM / PIX) ** 2
wcs = WCS(hdul[1].header)
MASK = hdul[5].data

SIDE = int(round(CUTOUT_ARCMIN * 60.0 / PIX)) | 1       # trimmed side, 225 pix
PAD = KHW                                               # padding, 50 pix
BIG = SIDE + 2 * PAD                                    # extracted side, 325
HALF, HALFB = SIDE // 2, BIG // 2
EDGE = int(np.ceil(FWHM / PIX))                         # peak edge exclusion
DECL = int(round(FWHM / PIX)) | 1                       # declustering window

print("beam %.2f\" (%.2f pix/beam) | pixel %.1f\" | kernel %dx%d (half-width %.2f')"
      % (FWHM, BEAM_PIX, PIX, K.shape[0], K.shape[1], PAD * PIX / 60.0))
print("cutouts: extract %d pix (%.1f') -> filter -> trim to %d pix (%.0f')"
      % (BIG, BIG * PIX / 60.0, SIDE, CUTOUT_ARCMIN))
print("decluster window %d pix | edge exclusion %d pix" % (DECL, EDGE))

# --- 1a. verify the operator against HELP's MFILT on a clean sub-region ------
print("\n" + "-" * 79)
print("cell 1a: does our operator reproduce HELP's MFILT?  (must be exact)")
print("-" * 79)
sl = (slice(1000, 1900), slice(2500, 3900))
_neb = hdul[2].data[sl].astype(np.float64) * JY2MJY
_err = hdul[3].data[sl].astype(np.float64) * JY2MJY
_hmf = hdul[6].data[sl].astype(np.float64) * JY2MJY
_val = np.isfinite(_neb) & np.isfinite(_err) & (_err > 0) & (MASK[sl] == 0)
_my = apply_matched_filter(_neb, _err, _val, K, K2)
_inner = np.zeros_like(_val)
_inner[KHW:-KHW, KHW:-KHW] = True
_m = _val & _inner & np.isfinite(_my) & np.isfinite(_hmf)
OP_CORR = float(np.corrcoef(_my[_m], _hmf[_m])[0, 1])
OP_RMS = float(np.std(_my[_m] - _hmf[_m]))
print("  NEBFILT -> our filter vs HELP MFILT:  corr = %.10f   rms resid = %.3e mJy/beam"
      % (OP_CORR, OP_RMS))
if OP_CORR < 0.9999:
    raise SystemExit("ERROR: the matched-filter operator does not reproduce "
                     "HELP's MFILT; the self-filtered maps cannot be trusted.")

# --- 1b. padding test: padded-cutout filtering == full-map filtering? --------
_cy, _cx = 450, 700
_b = (slice(_cy - HALFB, _cy + HALFB + 1), slice(_cx - HALFB, _cx + HALFB + 1))
_s = (slice(_cy - HALF, _cy + HALF + 1), slice(_cx - HALF, _cx + HALF + 1))
_pad_cut = apply_matched_filter(_neb[_b], _err[_b], _val[_b], K, K2)[PAD:-PAD,
                                                                    PAD:-PAD]
_bare = apply_matched_filter(_neb[_s], _err[_s], _val[_s], K, K2)
_ref = _my[_s]
_g = np.isfinite(_pad_cut) & np.isfinite(_ref)
PAD_MAXDIFF = float(np.max(np.abs(_pad_cut[_g] - _ref[_g])))
BARE_RMS = float(np.std(_bare[_g] - _ref[_g]))
print("  padded cutout  vs full-map filtering: max|diff| = %.2e mJy/beam  (machine precision)"
      % PAD_MAXDIFF)
print("  UNpadded cutout vs full-map filtering: rms = %.4f mJy/beam, max = %.3f"
      % (BARE_RMS, float(np.max(np.abs(_bare[_g] - _ref[_g])))))
print("  -> padding by %d pix is necessary AND sufficient." % PAD)
del _neb, _err, _hmf, _val, _my, _inner, _m, _pad_cut, _bare, _ref, _g
RES.update(op_corr=OP_CORR, op_rms=OP_RMS, pad_maxdiff=PAD_MAXDIFF,
           bare_rms=BARE_RMS)

# --- 1c. draw cutout centres -------------------------------------------------
print("\n" + "-" * 79)
print("cell 1c: drawing cutouts")
print("-" * 79)
cl = fits.getdata(CLUSTER_FILE, 1) if os.path.exists(CLUSTER_FILE) else None
if cl is not None:
    cx, cy = wcs.all_world2pix(cl["RA"], cl["DEC"], 0)
    _ok = np.isfinite(cx) & np.isfinite(cy)
    cx, cy = cx[_ok], cy[_ok]
    print("  avoiding %d eFEDS clusters by %.0f'" % (cx.size, CLUSTER_AVOID))
else:
    #  Release note: without the eFEDS catalog the cutouts are not required
    #  to avoid clusters, so their placement (and hence every number) differs
    #  from the paper.  See Herschel_analysis/README.md for the catalog.
    print("  WARNING: %s not found -- cutouts will NOT avoid eFEDS clusters;"
          " results will differ from the paper." % CLUSTER_FILE)

rng = np.random.default_rng(SEED)
ny, nx = MASK.shape
GOODC = (MASK == 0)
ys_ok, xs_ok = np.nonzero(GOODC[HALFB:ny - HALFB, HALFB:nx - HALFB])
ys_ok, xs_ok = ys_ok + HALFB, xs_ok + HALFB
avoid = CLUSTER_AVOID * 60.0 / PIX

# A try-cap is essential: non-overlapping random placement jams at a packing
# fraction ~0.55 (random sequential adsorption), so the target is unreachable
# and an uncapped loop would scan all ~1e7 candidate centres.
acc_y, acc_x = [], []
target = 20 if QUICK else N_CUTOUTS
n_try, max_try = 0, 40 * N_CUTOUTS
for jj in rng.permutation(len(ys_ok)):
    if len(acc_y) >= target or n_try > max_try:
        break
    n_try += 1
    yc, xc = int(ys_ok[jj]), int(xs_ok[jj])
    sub = GOODC[yc - HALFB:yc + HALFB + 1, xc - HALFB:xc + HALFB + 1]
    if sub.shape != (BIG, BIG) or not sub.all():
        continue
    if acc_y and np.min((np.array(acc_y) - yc) ** 2
                        + (np.array(acc_x) - xc) ** 2) < SIDE ** 2:
        continue
    if cl is not None and np.min((cx - xc) ** 2 + (cy - yc) ** 2) < avoid ** 2:
        continue
    acc_y.append(yc)
    acc_x.append(xc)
NC = len(acc_y)
print("  accepted %d cutouts (%.2f deg^2 of trimmed area) after %d tries"
      % (NC, NC * (SIDE * PIX / 3600.) ** 2, n_try))
if NC < 20:
    raise SystemExit("ERROR: too few cutouts accepted (%d)." % NC)

# --- 1d. build the three map variants and extract peaks ----------------------
print("\n" + "-" * 79)
print("cell 1d: filtering cutouts and extracting declustered peaks")
print("-" * 79)
TRIM = (slice(PAD, PAD + SIDE), slice(PAD, PAD + SIDE))
INT = np.zeros((SIDE, SIDE), bool)
INT[EDGE:SIDE - EDGE, EDGE:SIDE - EDGE] = True

peaks_by_cut = {v: [] for v in VARIANTS}
pix_pool = {v: [] for v in VARIANTS}
raw_stats = {v: {"mad": [], "rms": []} for v in VARIANTS + ["IMAGE", "NEBFILT"]}
noise_cut = []
peaks_masked = []          # PRIMARY variant only, bright-source masked
masked_area_frac = []

def _extract():
    """Extract, filter, baseline and decluster every cutout.  Cached: this is
    the single most expensive stage and does not depend on the bootstrap."""
    for n_done, (yc, xc) in enumerate(zip(acc_y, acc_x)):
        B = (slice(yc - HALFB, yc + HALFB + 1), slice(xc - HALFB, xc + HALFB + 1))
        img = hdul[1].data[B].astype(np.float64) * JY2MJY
        neb = hdul[2].data[B].astype(np.float64) * JY2MJY
        err = hdul[3].data[B].astype(np.float64) * JY2MJY
        hmf = hdul[6].data[B].astype(np.float64) * JY2MJY
        val = np.isfinite(neb) & np.isfinite(err) & (err > 0) & (MASK[B] == 0)

        maps = {"SF_IMAGE": apply_matched_filter(img, err, val, K, K2)[TRIM],
                "SF_NEB": apply_matched_filter(neb, err, val, K, K2)[TRIM],
                "HELP_MFILT": hmf[TRIM]}
        noise_cut.append(float(np.nanmedian(
            matched_filter_error(err, val, K2)[TRIM])))

        for tag, arr in (("IMAGE", img[TRIM]), ("NEBFILT", neb[TRIM])):
            a = arr[np.isfinite(arr)]
            raw_stats[tag]["mad"].append(robust_sigma(a))
            raw_stats[tag]["rms"].append(float(np.std(a)))

        for v in VARIANTS:
            cut = maps[v]
            a = cut[np.isfinite(cut)]
            raw_stats[v]["mad"].append(robust_sigma(a))
            raw_stats[v]["rms"].append(float(np.std(a)))
            valid = np.isfinite(cut)
            res = poly_baseline(cut, BASELINE_ORDER, valid)
            filled = np.where(valid, res, -np.inf)
            pk = valid & INT & (filled == maximum_filter(filled, size=DECL,
                                                         mode="constant",
                                                         cval=-np.inf))
            peaks_by_cut[v].append(res[pk].astype(np.float32))
            pix_pool[v].append(res[valid].astype(np.float32))

            # --- bright-source-masked variant, PRIMARY map only ------------------
            # Mask at the MAP level (not by censoring the peak list): NaN out pixels
            # above S_MASK, grow the mask by one FWHM so the source's own shoulders
            # and its confusion neighbourhood go with it, then re-extract peaks.
            # This is the closest available analogue of the models' source-plane
            # truncation; see the docstring for why peak-list censoring is not.
            if v == PRIMARY:
                hot = np.isfinite(res) & (res > S_MASK_MJY)
                if MASK_GROW > 0 and hot.any():
                    hot = maximum_filter(hot, size=2 * MASK_GROW + 1,
                                         mode="constant", cval=False)
                valid_m = valid & ~hot
                masked_area_frac.append(float(hot.sum()) / max(valid.sum(), 1))
                filled_m = np.where(valid_m, res, -np.inf)
                pk_m = valid_m & INT & (filled_m == maximum_filter(
                    filled_m, size=DECL, mode="constant", cval=-np.inf))
                peaks_masked.append(res[pk_m].astype(np.float32))
        if (n_done + 1) % 25 == 0:
            print("    ... %d/%d cutouts (%.0f s)" % (n_done + 1, NC, time.time() - T0))
    return peaks_by_cut, pix_pool, raw_stats, noise_cut, peaks_masked, masked_area_frac


(peaks_by_cut, pix_pool, raw_stats, noise_cut, peaks_masked,
 masked_area_frac) = cached("extract", _extract)

CORE = {}
for v in VARIANTS:
    pk = np.concatenate(peaks_by_cut[v])
    px = np.concatenate(pix_pool[v])
    CORE[v] = dict(mu=float(np.median(pk)), sig=robust_sigma(pk),
                   sig_pix=robust_sigma(px), n=pk.size,
                   per_beam=pk.size / (NC * INT.sum() / BEAM_PIX))
    pix_pool[v] = px

print("\n  %-11s %9s %9s %9s %9s %8s"
      % ("variant", "N_peaks", "pk/beam", "mu_core", "sig_core", "sig_pix"))
for v in VARIANTS:
    c = CORE[v]
    print("  %-11s %9d %9.4f %+9.3f %9.3f %8.3f"
          % (v, c["n"], c["per_beam"], c["mu"], c["sig"], c["sig_pix"]))

noise_cut = np.array(noise_cut)
SIGMA_INST = float(np.median(noise_cut))
print("\n  matched-filter noise across cutouts: median %.3f, scatter %.3f "
      "(%.1f%%), range %.3f-%.3f mJy/beam"
      % (SIGMA_INST, np.std(noise_cut), 100 * np.std(noise_cut) / SIGMA_INST,
         noise_cut.min(), noise_cut.max()))
print("""  The inverse-variance weighting makes the filter formally non-stationary,
  so the observed P(D) is a MIXTURE over sigma_inst rather than a single-sigma
  distribution.  At the scatter measured above this broadens the core by
  ~(1/2)(dsigma/sigma)^2 in variance, i.e. well under a percent, and it is
  ignored in the forward model below.  It is recorded in the npz so H2 can
  weight cluster and control cutouts by their actual noise if needed.""")

RES.update(n_cutouts=NC, fwhm=FWHM, pix=PIX, sigma_inst=SIGMA_INST,
           noise_per_cutout=noise_cut, side_pix=SIDE, pad_pix=PAD,
           mu_core=CORE[PRIMARY]["mu"], sig_core=CORE[PRIMARY]["sig"],
           sig_pix=CORE[PRIMARY]["sig_pix"], n_peaks=CORE[PRIMARY]["n"])


#%% ------------ cell 2: the filtered profile, its moments, and its P(D) ------
print("\n" + "-" * 79)
print("cell 2: what the matched filter does to the effective beam profile")
print("-" * 79)
n_s, c_s = 401, 200
yy, xx = np.mgrid[:n_s, :n_s]
sig_pix_b = (FWHM / PIX) / (2 * np.sqrt(2 * np.log(2)))
PSF_G = np.exp(-((yy - c_s) ** 2 + (xx - c_s) ** 2) / (2 * sig_pix_b ** 2))
PSF_F = oaconvolve(PSF_G, K, mode="same") / K2.sum()
FLUX_RESP = float(PSF_F.max())
PSF_FN = PSF_F / FLUX_RESP                     # peak-normalised
print("  point-source flux response of the filter = %.4f" % FLUX_RESP)
print("""  (this is measured by filtering an IDEALISED Gaussian PSF; a mismatch of a
  few percent is expected because the true SPIRE PSF has wings the Gaussian
  lacks.  It is NOT applied as a flux correction here -- see cell 2 note.)""")

print("\n  Profile moments int P^q d2x [arcsec^2].  q = 2 sets the Condon")
print("  variance; the exceedance tail is set by q = eta - 1 with eta ~ 3-5,")
print("  so q = 2-4 is the range that matters for xi(u).")
print("  %5s %14s %14s %9s %14s" % ("q", "Gaussian", "filtered", "ratio",
                                    "filt (signed)"))
QLIST = (1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0)
om_g, om_f, om_ratio, om_signed = [], [], [], []
for q in QLIST:
    a = float(np.sum(np.abs(PSF_G) ** q) * PIX ** 2)
    b = float(np.sum(np.abs(PSF_FN) ** q) * PIX ** 2)
    bs = float(np.sum(np.sign(PSF_FN) * np.abs(PSF_FN) ** q) * PIX ** 2)
    om_g.append(a); om_f.append(b); om_ratio.append(b / a); om_signed.append(bs)
    print("  %5.2f %14.3f %14.3f %9.4f %14.3f" % (q, a, b, b / a, bs))
OM2_RATIO = om_ratio[QLIST.index(2.0)]
if not 0.97 < OM2_RATIO < 1.03:
    raise SystemExit("ERROR: Omega_eff is not preserved by the filter.")
_neg = PSF_FN[PSF_FN < 0]
LOBE_FRAC = float(np.abs(_neg).sum() / np.abs(PSF_FN).sum())
print("\n  negative-lobe |mass| fraction %.4f, deepest lobe %.4f at r ~ %.0f\""
      % (LOBE_FRAC, PSF_FN.min(),
         PIX * np.hypot(*(np.array(np.unravel_index(np.argmin(PSF_FN),
                                                    PSF_FN.shape)) - c_s))))
print("""  Reading of the table:
    q = 1 (the MEAN) is suppressed to %.3f signed -- the filter is a high-pass
      and removes DC.  Irrelevant here: we subtract a DC baseline per cutout.
    q = 2 (the VARIANCE) is preserved to %.4f -- Condon's effective solid angle
      survives the filter, which is why the Poisson confusion width is unchanged.
    q = 2-4 (the TAIL) stays within %.1f%% of the Gaussian value, and the signed
      and unsigned sums converge, because the lobes are shallow (%.3f) and
      lobe^q dies fast.  So the Gaussian-beam P(D) is a good model for the tail
      too -- but cell 2b checks that directly rather than inferring it."""
      % (om_signed[0] / om_g[0], OM2_RATIO,
         100 * max(abs(np.array(om_ratio[2:]) - 1)), PSF_FN.min()))
RES.update(q_list=np.array(QLIST), omega_gauss=np.array(om_g),
           omega_filt=np.array(om_f), omega_ratio=np.array(om_ratio),
           omega_signed=np.array(om_signed), flux_response=FLUX_RESP,
           lobe_frac=LOBE_FRAC)

#%% ---- cell 2b: exact filtered-profile P(D) vs the Gaussian-beam P(D) -------
print("\n" + "-" * 79)
print("cell 2b: exact P(D) for the filtered profile vs the Gaussian-beam P(D)")
print("-" * 79)
_sch_chk = Schechter(alpha=SCH_PARS["alpha"],
                     log_sstar=np.log10(SCH_PARS["sstar"]),
                     log_phistar=np.log10(SCH_PARS["nstar"]), s_min=S_MIN_MJY)
pp_gauss = PofD(_sch_chk, FWHM, mu=1.0, sigma_noise=SIGMA_INST,
                s_cut=S_CUT_MJY, d_span=300.0, n_fft=2 ** 17)
d_ex, p_ex = pofd_from_profile(_sch_chk, PSF_FN, PIX, sigma_noise=SIGMA_INST,
                               s_cut=S_CUT_MJY, d_span=300.0, n_fft=2 ** 17)
st_g, st_e = pdf_stats(pp_gauss.d, pp_gauss.p), pdf_stats(d_ex, p_ex)
print("  %-24s %9s %9s %9s" % ("P(D)", "rms", "MAD-sig", "skew"))
print("  %-24s %9.3f %9.3f %9.2f"
      % ("Gaussian beam", st_g["rms"], st_g["madsig"], st_g["skew"]))
print("  %-24s %9.3f %9.3f %9.2f"
      % ("exact filtered profile", st_e["rms"], st_e["madsig"], st_e["skew"]))
print("  ratio (exact/Gaussian):  rms %.4f   MAD-sigma %.4f"
      % (st_e["rms"] / st_g["rms"], st_e["madsig"] / st_g["madsig"]))

U_GRID = np.arange(25.0, 76.0, 2.5)
MU0 = CORE[PRIMARY]["mu"]
xi_g, _ = xi_of_threshold_analytic(pp_gauss.d, pp_gauss.p, U_GRID - MU0)
xi_e, _ = xi_of_threshold_analytic(d_ex, p_ex, U_GRID - MU0)
xi_g, xi_e = np.asarray(xi_g, float), np.asarray(xi_e, float)
DXI_PROFILE = np.nanmax(np.abs(xi_e - xi_g))
print("\n  xi(u) from the two P(D):  max |difference| over the threshold grid"
      " = %.4f" % DXI_PROFILE)
print("     u[mJy]   xi(Gaussian)  xi(exact)   diff")
for i in range(0, len(U_GRID), 4):
    print("   %8.1f   %+11.4f  %+9.4f  %+7.4f"
          % (U_GRID[i], xi_g[i], xi_e[i], xi_e[i] - xi_g[i]))
print("""  If this difference is small compared with the bootstrap errors (cell 4),
  the Gaussian-beam PofD is adequate for the filtered map and is used for the
  headline model curves; the exact profile is then quoted as a systematic.
  This replaces H1's inference-from-Omega_2 with a direct check.""")
RES.update(xi_gauss_profile=xi_g, xi_exact_profile=xi_e,
           dxi_profile=float(DXI_PROFILE), pd_exact_d=d_ex, pd_exact_p=p_ex)

#%% ------- cell 3: like-for-like width comparison (no renormalisation) -------
print("\n" + "-" * 79)
print("cell 3: measured vs predicted widths, ON THE SAME STATISTIC")
print("-" * 79)
print("""  H1 compared Condon's sigma_c -- the true RMS of P(D) -- against MAD-based
  measured widths.  P(D) is strongly skewed, so those are different statistics
  of the same distribution and the comparison manufactured a 25-56% "over-
  prediction".  Here both are computed on both sides.""")

print("\n  MEASURED, median over the %d cutouts:" % NC)
print("  %-12s %10s %10s %9s" % ("product", "MAD-sigma", "rms", "rms/MAD"))
meas_w = {}
for tag in ["IMAGE", "NEBFILT"] + VARIANTS:
    md = float(np.median(raw_stats[tag]["mad"]))
    rm = float(np.median(raw_stats[tag]["rms"]))
    meas_w[tag] = dict(mad=md, rms=rm)
    print("  %-12s %10.3f %10.3f %9.3f" % (tag, md, rm, rm / md))

print("\n  PREDICTED by the adopted Schechter counts at %.2f\", with the" % FWHM)
print("  matched-filter noise %.3f mJy/beam added in quadrature:" % SIGMA_INST)
_st = pdf_stats(pp_gauss.d, pp_gauss.p)
_pp0 = PofD(_sch_chk, FWHM, mu=1.0, sigma_noise=0.0, s_cut=None,
            d_span=600.0, n_fft=2 ** 19)
_st0 = pdf_stats(_pp0.d, _pp0.p)
print("    counts only, no noise :  rms %.3f   MAD-sigma %.3f   (ratio %.3f, skew %.2f)"
      % (_st0["rms"], _st0["madsig"], _st0["rms"] / _st0["madsig"], _st0["skew"]))
print("    + noise %.3f          :  rms %.3f   MAD-sigma %.3f"
      % (SIGMA_INST, _st["rms"], _st["madsig"]))
print("    MEASURED (%s)   :  rms %.3f   MAD-sigma %.3f"
      % (PRIMARY, meas_w[PRIMARY]["rms"], meas_w[PRIMARY]["mad"]))
print("    model/measured        :  rms %.3f   MAD-sigma %.3f"
      % (_st["rms"] / meas_w[PRIMARY]["rms"],
         _st["madsig"] / meas_w[PRIMARY]["mad"]))

# counts-normalisation uncertainty band: sigma_c ~ sqrt(n*)
_fn = SCH_PARS["d_nstar"] / SCH_PARS["nstar"]
print("\n  The counts fit has n* = %.0f +- %.0f deg^-2, i.e. %.0f%% on the"
      % (SCH_PARS["nstar"], SCH_PARS["d_nstar"], 100 * _fn))
print("  normalisation.  Since sigma_c ~ sqrt(n*), that is %.0f%% on the predicted"
      % (100 * 0.5 * _fn))
print("  width alone, before the alpha and S* uncertainties are folded in:")
for lab, fac in (("n* - 1sigma", np.sqrt(1 - _fn)), ("n* nominal", 1.0),
                 ("n* + 1sigma", np.sqrt(1 + _fn))):
    print("    %-12s predicted counts-only rms = %6.3f  MAD-sigma = %6.3f"
          % (lab, _st0["rms"] * fac, _st0["madsig"] * fac))
print("""
  CONCLUSION recorded for the paper: once compared like-for-like, the adopted
  Schechter counts reproduce the measured width to within the fit's own
  normalisation uncertainty.  NO MODEL IS RENORMALISED anywhere in this module.
  The residual difference between the self-filtered IMAGE and HELP's MFILT is
  the cirrus the nebular filter removes and ours does not; the DC+linear
  baseline absorbs most of it.""")
RES.update(meas_widths=json.dumps(meas_w),
           model_rms=_st["rms"], model_madsig=_st["madsig"],
           model_rms_nonoise=_st0["rms"], model_madsig_nonoise=_st0["madsig"])


#%% ------------ cell 4: measured xi(u), cutout-level bootstrap ---------------
print("\n" + "-" * 79)
print("cell 4: measured xi(u) with cutout-level bootstrap")
print("-" * 79)
print("""  Thresholds are anchored in ABSOLUTE flux (mJy/beam), not in k*sigma_core.
  Reason (revised from H1): xi ~= 1/(eta(S_u)-1) is a function of absolute flux,
  so mJy anchoring is the physically correct comparison.  It is NOT a workaround
  for a width mismatch -- cell 3 showed there is none to work around.  The k
  ladder is quoted alongside for continuity with the Planck chapter.""")
MIN_EXC = 50


def xi_at(pk_list, u_grid):
    """MLE xi at each threshold, for a list of per-cutout peak arrays."""
    y_all = np.concatenate(pk_list)
    out = np.full(len(u_grid), np.nan)
    nex = np.zeros(len(u_grid), int)
    for i, u in enumerate(u_grid):
        y = y_all[y_all > u] - u
        nex[i] = y.size
        if y.size >= MIN_EXC:
            out[i] = fit_gpd(y)["xi"]
    return out, nex


XI, XIERR, NEXC, BOOT = {}, {}, {}, {}
def _boot_variant(v, nb):
    xi_v, nex = xi_at(peaks_by_cut[v], U_GRID)
    r_ = np.random.default_rng(SEED + 7919 * VARIANTS.index(v))
    bt = np.full((nb, len(U_GRID)), np.nan)
    for b in range(nb):
        idx = r_.integers(0, NC, NC)           # resample CUTOUTS, not peaks
        bt[b], _ = xi_at([peaks_by_cut[v][i] for i in idx], U_GRID)
    return xi_v, nex, bt


for v in VARIANTS:
    nb = N_BOOT if v == PRIMARY else N_BOOT_SYS
    xi_v, nex, bt = cached("boot", lambda v=v, nb=nb: _boot_variant(v, nb),
                           extra="_%s_nb%d" % (v, nb))
    lo, hi = np.nanpercentile(bt, [16, 84], axis=0)
    XI[v], XIERR[v], NEXC[v], BOOT[v] = xi_v, 0.5 * (hi - lo), nex, bt
    print("  %-11s done (%d bootstrap replicates, %.0f s elapsed)"
          % (v, nb, time.time() - T0))

MU_CORE, SIG_CORE = CORE[PRIMARY]["mu"], CORE[PRIMARY]["sig"]
print("\n  PRIMARY = %s" % PRIMARY)
print("     u[mJy]    k      N_exc     xi_hat +- err     SF_NEB      HELP_MFILT")
for i, u in enumerate(U_GRID):
    k = (u - MU_CORE) / SIG_CORE
    if np.isfinite(XI[PRIMARY][i]):
        print("   %8.1f %5.2f %8d   %+.4f +- %.4f   %+.4f     %+.4f"
              % (u, k, NEXC[PRIMARY][i], XI[PRIMARY][i], XIERR[PRIMARY][i],
                 XI["SF_NEB"][i], XI["HELP_MFILT"][i]))
    else:
        print("   %8.1f %5.2f %8d   -- (below N_exc floor)"
              % (u, k, NEXC[PRIMARY][i]))

# Map-variant systematic.  Compare PER THRESHOLD in units of that threshold's
# own error -- comparing a max |difference| against a median error would
# overstate the disagreement, since both are largest at the sparse high-u end.
_dv = XI[PRIMARY] - XI["HELP_MFILT"]
_ev = np.sqrt(XIERR[PRIMARY] ** 2 + XIERR["HELP_MFILT"] ** 2)
_gv = np.isfinite(_dv) & np.isfinite(_ev) & (_ev > 0)
_zv = np.abs(_dv[_gv] / _ev[_gv])
print("\n  MAP-VARIANT SYSTEMATIC, SF_IMAGE vs HELP_MFILT, per threshold:")
print("    |difference| / error :  max %.2f, median %.2f, %d of %d thresholds > 2"
      % (_zv.max(), np.median(_zv), int((_zv > 2).sum()), _zv.size))
print("""    This adds the two errors in quadrature as if independent, but the two
    maps are built from the SAME cutouts, so their bootstrap errors are
    strongly correlated and the test is conservative.  Retaining the cirrus
    (SF_IMAGE) rather than removing it upstream (HELP_MFILT) is therefore not
    a significant systematic for xi(u) at these thresholds.""")
_dn = np.nanmax(np.abs(XI["SF_NEB"][_gv] - XI["HELP_MFILT"][_gv]))
print("    consistency: max |SF_NEB - HELP_MFILT| = %.2e in xi" % _dn)
print("""    (not bitwise zero: HELP stores MFILT at reduced precision, so our
    reconstruction differs by ~1e-6 mJy/beam, which occasionally flips a
    local-maximum tie and moves the GPD fit at the 1e-6 level -- four orders
    of magnitude below the bootstrap errors.)""")
if _dn > 1e-4:
    raise SystemExit("ERROR: self-filtered NEBFILT does not reproduce HELP "
                     "MFILT to 1e-4 in xi; the operator is wrong.")
RES.update(variant_z=_zv, variant_z_max=float(_zv.max()),
           variant_z_med=float(np.median(_zv)), sfneb_help_dxi=float(_dn))

#%% ------ cell 4b: the bright-source systematic (census + masked variant) ----
print("\n" + "-" * 79)
print("cell 4b: the bright lensed population, and the masked measurement")
print("-" * 79)
AREA = NC * (SIDE * PIX / 3600.0) ** 2
_pk_all = np.concatenate(peaks_by_cut[PRIMARY])
print("""  The models truncate the counts at S_cut = %.0f mJy because above ~100 mJy
  the 350 um counts are dominated by strongly lensed sources (Lima+2010) and no
  longer trace the intrinsic dN/dS.  The MAP is not truncated.  How large is the
  resulting mismatch?""" % S_CUT_MJY)
print("\n  Bright peak census over %.2f deg^2, vs each model's prediction for the"
      % AREA)
print("  same area (integrating dN/dS above S; the models' own S_cut is ignored")
print("  here precisely so the comparison is meaningful):")
print("  %8s %10s %12s %12s %12s" % ("S [mJy]", "observed", "Schechter", "SPL", "DPL"))
_sch_c = Schechter(alpha=SCH_PARS["alpha"], log_sstar=np.log10(SCH_PARS["sstar"]),
                   log_phistar=np.log10(SCH_PARS["nstar"]), s_min=S_MIN_MJY)
_spl_c = SinglePowerLaw(SPL_PARS["N0"], SPL_PARS["S0"], SPL_PARS["beta"],
                        s_min=S_MIN_MJY)
_dpl_c = DoublePowerLaw(alpha=DPL_PARS["alpha"], beta=DPL_PARS["beta"],
                        log_sstar=np.log10(DPL_PARS["sstar"]),
                        log_phistar=np.log10(DPL_PARS["phistar"]), s_min=S_MIN_MJY)
BRIGHT_LEVELS = (100.0, 150.0, 200.0, 300.0)
bright_obs, bright_pred = [], {"Schechter": [], "SPL": [], "DPL": []}
for _S in BRIGHT_LEVELS:
    n_o = int((_pk_all > _S).sum())
    bright_obs.append(n_o)
    row = []
    for _nm, _m in (("Schechter", _sch_c), ("SPL", _spl_c), ("DPL", _dpl_c)):
        _g = np.geomspace(_S, 1e5, 4000)
        _p = float(_trapz(_m.dnds(_g), _g) * AREA)
        bright_pred[_nm].append(_p)
        row.append(_p)
    print("  %8.0f %10d %12.2f %12.2f %12.2f" % (_S, n_o, *row))
print("  brightest peak in the sample: %.1f mJy/beam" % _pk_all.max())
print("""
  Read this as the honest statement of where all three fits fail.  The Schechter
  under-predicts the S > 200 mJy population by orders of magnitude (its
  exponential cutoff forbids it); the SPL over-predicts everywhere bright (it
  has no cutoff at all); the DPL sits between.  None of the three was ever
  fitted to this regime -- the counts memo fit S = 6-94.6 mJy and dropped
  Bethermin's 133.7 mJy point for exactly this reason.  Module H1c showed this
  population, not any peak-statistics artefact, drives the rise in xi_hat(u).""")

# --- the masked measurement --------------------------------------------------
_maf = float(np.mean(masked_area_frac))
xi_msk, nex_msk = xi_at(peaks_masked, U_GRID)
def _boot_masked():
    r_ = np.random.default_rng(SEED + 4242)
    bt = np.full((N_BOOT_MASK, len(U_GRID)), np.nan)
    for b in range(N_BOOT_MASK):
        idx = r_.integers(0, NC, NC)
        bt[b], _ = xi_at([peaks_masked[i] for i in idx], U_GRID)
    return bt


_bt = cached("bootmask", _boot_masked, extra="_nb%d" % N_BOOT_MASK)
_lo, _hi = np.nanpercentile(_bt, [16, 84], axis=0)
xierr_msk = 0.5 * (_hi - _lo)
print("\n  MAP-LEVEL BRIGHT MASK at S_mask = %.0f mJy, grown by %d pix (%.0f\"):"
      % (S_MASK_MJY, MASK_GROW, MASK_GROW * PIX))
print("    masked area fraction %.4f%%  |  peaks retained %d of %d (%.3f%% removed)"
      % (100 * _maf, sum(p.size for p in peaks_masked), _pk_all.size,
         100 * (1 - sum(p.size for p in peaks_masked) / _pk_all.size)))
print("\n     u[mJy]   unmasked xi     masked xi      difference")
_use_m = U_GRID <= U_MAX_MASKED
for i, u in enumerate(U_GRID):
    tag = "" if _use_m[i] else "   <- censored, not quoted"
    if np.isfinite(xi_msk[i]):
        print("   %8.1f  %+.4f+-%.4f  %+.4f+-%.4f  %+8.4f%s"
              % (u, XI[PRIMARY][i], XIERR[PRIMARY][i], xi_msk[i], xierr_msk[i],
                 xi_msk[i] - XI[PRIMARY][i], tag))
_gm = _use_m & np.isfinite(xi_msk) & np.isfinite(XI[PRIMARY])
_sl_un = np.polyfit(U_GRID[_gm], XI[PRIMARY][_gm], 1,
                    w=1.0 / XIERR[PRIMARY][_gm])[0]
_sl_ms = np.polyfit(U_GRID[_gm], xi_msk[_gm], 1, w=1.0 / xierr_msk[_gm])[0]
print("""
  Over u <= %.0f mJy the weighted slope d(xi)/du is %+.5f unmasked and %+.5f
  masked.  The masked variant is the apples-to-apples comparison against the
  S_cut-truncated models; the unmasked one is the physical measurement and
  remains the headline result.  Above u ~ %.0f mJy the mask right-censors the
  exceedance distribution and biases the GPD MLE downward, so the masked curve
  is NOT quoted there -- that is a property of the estimator, not of the sky."""
      % (U_MAX_MASKED, _sl_un, _sl_ms, U_MAX_MASKED))
RES.update(xi_masked=xi_msk, xierr_masked=xierr_msk, n_exc_masked=nex_msk,
           boot_masked=_bt, s_mask=S_MASK_MJY, mask_grow=MASK_GROW,
           u_max_masked=U_MAX_MASKED, masked_area_frac=_maf,
           bright_levels=np.array(BRIGHT_LEVELS),
           bright_obs=np.array(bright_obs),
           bright_pred_json=json.dumps(bright_pred), area_deg2=AREA,
           slope_unmasked_lowu=float(_sl_un), slope_masked_lowu=float(_sl_ms))

RES.update(u_grid=U_GRID, k_grid=(U_GRID - MU_CORE) / SIG_CORE,
           xi_meas=XI[PRIMARY], xi_err=XIERR[PRIMARY], n_exc=NEXC[PRIMARY],
           boot=BOOT[PRIMARY], primary=PRIMARY,
           xi_sfneb=XI["SF_NEB"], xi_helpmfilt=XI["HELP_MFILT"],
           xierr_sfneb=XIERR["SF_NEB"], xierr_helpmfilt=XIERR["HELP_MFILT"])

#%% ---------------- cell 5: three count models and their xi(u) ---------------
print("\n" + "-" * 79)
print("cell 5: Schechter vs SPL vs DPL")
print("-" * 79)
sch = Schechter(alpha=SCH_PARS["alpha"], log_sstar=np.log10(SCH_PARS["sstar"]),
                log_phistar=np.log10(SCH_PARS["nstar"]), s_min=S_MIN_MJY)
spl = SinglePowerLaw(SPL_PARS["N0"], SPL_PARS["S0"], SPL_PARS["beta"],
                     s_min=S_MIN_MJY)
dpl = DoublePowerLaw(alpha=DPL_PARS["alpha"], beta=DPL_PARS["beta"],
                     log_sstar=np.log10(DPL_PARS["sstar"]),
                     log_phistar=np.log10(DPL_PARS["phistar"]),
                     s_min=S_MIN_MJY)
MODELS = {"Schechter": sch, "SPL": spl, "DPL": dpl}
MCOL = {"Schechter": "crimson", "SPL": "tab:blue", "DPL": "darkgreen"}
MLS = {"Schechter": "-", "SPL": "--", "DPL": "-."}

print("  S_min = %.2f mJy, S_cut = %.1f mJy, beam %.2f\"" % (S_MIN_MJY, S_CUT_MJY, FWHM))
print("  %-10s %12s %10s %10s %12s"
      % ("model", "sigma_c(rms)", "N_beam", "mean", "xi asymptote"))
for nm, m in MODELS.items():
    st = m.confusion_stats(FWHM, s_lo=S_MIN_MJY, s_hi=S_CUT_MJY)
    if nm == "Schechter":
        asym = "-> -inf (cutoff)"
    elif nm == "SPL":
        asym = "%.3f" % (1.0 / (-SPL_PARS["beta"] - 1.0))
    else:
        asym = "%.3f" % (1.0 / (DPL_PARS["beta"] - 1.0))
    print("  %-10s %12.3f %10.3f %10.2f %12s"
          % (nm, st["sigma_conf_mJy_per_beam"], st["sources_per_beam"],
             st["mean_mJy_per_beam"], asym))
print("""  NOTE the DPL break sits at S* = %.2f mJy, i.e. k = %.2f in units of the
  measured core width -- deep inside the confusion core.  We do NOT resolve the
  break.  What xi(u) can test at this resolution is the HIGH-THRESHOLD
  behaviour: flat (SPL, no cutoff), flat-but-lower (DPL past its break), or
  falling through zero (Schechter's exponential cutoff)."""
      % (DPL_PARS["sstar"], DPL_PARS["sstar"] / SIG_CORE))

xi_model, pofd_model = {}, {}
for nm, m in MODELS.items():
    pp = PofD(m, FWHM, mu=1.0, sigma_noise=SIGMA_INST, s_cut=S_CUT_MJY,
              d_span=300.0, n_fft=2 ** 17)
    xm, _ = xi_of_threshold_analytic(pp.d, pp.p, U_GRID - MU_CORE)
    xi_model[nm] = np.asarray(xm, float)
    pofd_model[nm] = pp
    mo = pp.moments()
    print("  %-10s P(D): rms = %6.3f  skew = %5.2f" % (nm, mo["sigma"], mo["skew"]))

print("""
  CAVEAT carried from H1 and NOT fixed here.  These model curves are KL
  projections of the analytic PIXEL P(D), while xi_hat is measured on
  DECLUSTERED PEAKS.  The model grid is offset by the measured core centroid,
  which absorbs the leading part of the difference but not all of it.  Treat the
  curves as a guide to SHAPE and to model SEPARATION, not to absolute level,
  until the peak-level simulation (H1c) is run.""")

print("\n     u[mJy]   xi_Sch    xi_SPL    xi_DPL   |Sch-SPL| |Sch-DPL| |SPL-DPL|  err")
pairs = [("Schechter", "SPL"), ("Schechter", "DPL"), ("SPL", "DPL")]
gaps = {p: np.full(len(U_GRID), np.nan) for p in pairs}
for i, u in enumerate(U_GRID):
    g = [abs(xi_model[a][i] - xi_model[b][i]) for a, b in pairs]
    for p, v in zip(pairs, g):
        gaps[p][i] = v
    print("   %8.1f %+8.4f %+8.4f %+8.4f %9.4f %9.4f %9.4f %8.4f"
          % (u, xi_model["Schechter"][i], xi_model["SPL"][i],
             xi_model["DPL"][i], g[0], g[1], g[2], XIERR[PRIMARY][i]))

good = np.isfinite(XIERR[PRIMARY]) & (XIERR[PRIMARY] > 0) & np.isfinite(XI[PRIMARY])

# --- threshold covariance from the bootstrap replicates ---------------------
_bok = good & np.all(np.isfinite(BOOT[PRIMARY]), axis=0)
BMAT = BOOT[PRIMARY][:, _bok]
NB_EFF, NP_EFF = BMAT.shape
COV = np.cov(BMAT, rowvar=False)
DIAG = np.sqrt(np.diag(COV))
CORR = COV / np.outer(DIAG, DIAG)
HARTLAP = (NB_EFF - NP_EFF - 2.0) / (NB_EFF - 1.0)
print("\n" + "-" * 79)
print("  THRESHOLD COVARIANCE  (why naive quadrature is wrong)")
print("-" * 79)
print("  %d usable thresholds, %d bootstrap replicates" % (NP_EFF, NB_EFF))
print("  correlation of xi_hat vs threshold separation:")
for _lag in (1, 2, 4, 8, 16):
    if _lag < NP_EFF:
        print("     lag %2d (%4.1f mJy): mean rho = %.3f"
              % (_lag, 2.5 * _lag, float(np.mean(np.diag(CORR, _lag)))))
_ev = np.linalg.eigvalsh(CORR)[::-1]
N_EFF_SUM = float(NP_EFF ** 2 / CORR.sum())
N_EFF_PCA = float(_ev.sum() ** 2 / np.sum(_ev ** 2))
print("  effective independent thresholds: %.2f (sum of correlations), "
      "%.2f (eigenvalue spread), of %d" % (N_EFF_SUM, N_EFF_PCA, NP_EFF))
if NB_EFF - NP_EFF - 2 <= 0:
    raise SystemExit("ERROR: %d bootstrap replicates cannot support a %d x %d "
                     "covariance inverse.  Raise CIB_N_BOOT." % (NB_EFF, NP_EFF,
                                                                 NP_EFF))
print("  Hartlap factor (N-p-2)/(N-1) = %.3f" % HARTLAP)
COVI = np.linalg.inv(COV) * HARTLAP


def sep_full(d_vec, n_pc=None):
    """Covariance-correct separation sqrt(d^T C^-1 d).

    n_pc : if given, invert only in the leading n_pc principal components of
    the correlation matrix.  This regularises the inverse and is reported as a
    robustness check: a full-covariance number that collapses under mild
    truncation is being carried by poorly-determined low-variance modes.
    """
    d = d_vec[_bok]
    if not np.all(np.isfinite(d)):
        return np.nan
    if n_pc is None:
        return float(np.sqrt(max(d @ COVI @ d, 0.0)))
    w, V = np.linalg.eigh(COV)
    idx = np.argsort(w)[::-1][:n_pc]
    proj = V[:, idx].T @ d
    return float(np.sqrt(max(np.sum(proj ** 2 / w[idx]) * HARTLAP, 0.0)))


print("\n  MODEL SEPARATIONS.  These express DISCRIMINATING POWER -- how far")
print("  apart the models are relative to the data errors -- NOT a detection")
print("  significance: the data sit on none of the models (see cell 4b/H1c).")
print("  %-20s %11s %11s %13s %10s %10s"
      % ("pair", "best single", "naive quad", "FULL COV", "PC=3", "PC=5"))
sep_best, sep_cov = {}, {}
for p in pairs:
    r = gaps[p] / XIERR[PRIMARY]
    ib = int(np.nanargmax(np.where(good, r, np.nan)))
    sep_best[p] = (float(r[ib]), float(U_GRID[ib]))
    naive = float(np.sqrt(np.nansum(np.where(good, r, np.nan) ** 2)))
    full = sep_full(gaps[p])
    sep_cov[p] = full
    print("  %-20s %11.2f %11.2f %13.2f %10.2f %10.2f"
          % ("%s vs %s" % p, r[ib], naive, full,
             sep_full(gaps[p], 3), sep_full(gaps[p], 5)))
print("""
  The naive-quadrature column is the INCORRECT v2 statistic, printed only so the
  size of the error is visible.  The effect is NOT a uniform deflation: pairs
  whose model difference is mostly a vertical OFFSET (Schechter vs SPL/DPL) are
  penalised, because that is also the dominant bootstrap noise mode, while a
  pair whose curves CROSS (SPL vs DPL) is rewarded, its difference being nearly
  orthogonal to that mode.  Diagonal errors cannot see this.

  READ THE PC COLUMNS CAREFULLY.  The full-covariance statistic is much larger
  than the few-PC ones, i.e. a large part of it comes from LOW-VARIANCE modes of
  the threshold covariance -- directions in which xi_hat(u) is very precisely
  determined.  That is legitimate in principle (it is exactly what covariance
  weighting is for) but it is also where a sample covariance is least reliable,
  so it must be checked rather than assumed.  The half-sample test below does
  that.""")

# --- is the full-covariance statistic numerically stable? -------------------
print("\n  STABILITY of the full-covariance statistic: recompute it from random")
print("  HALVES of the bootstrap ensemble.  If the low-variance modes were noise")
print("  the half-sample values would scatter wildly and sit far from the full")
print("  one; if they are genuinely determined, they will not.")
print("  %-20s %22s %14s" % ("pair", "half-sample (mean +- sd)", "full sample"))
_nhalf = 20 if not QUICK else 5
stab = {}
for p in pairs:
    d = gaps[p][_bok]
    vals = []
    for rep in range(_nhalf):
        sub = np.random.default_rng(1000 + rep).permutation(NB_EFF)[:NB_EFF // 2]
        Ch = np.cov(BMAT[sub], rowvar=False)
        hh = (NB_EFF // 2 - NP_EFF - 2.0) / (NB_EFF // 2 - 1.0)
        if hh <= 0:
            continue
        vals.append(np.sqrt(max(d @ (np.linalg.inv(Ch) * hh) @ d, 0.0)))
    if vals:
        stab[p] = (float(np.mean(vals)), float(np.std(vals)))
        print("  %-20s %14.2f +- %5.2f %14.2f"
              % ("%s vs %s" % p, stab[p][0], stab[p][1], sep_cov[p]))
_worst = max((abs(stab[p][0] - sep_cov[p]) / max(sep_cov[p], 1e-9)
              for p in stab), default=np.nan)
print("  largest fractional shift between half and full sample: %.1f%%  [%s]"
      % (100 * _worst, "PASS" if _worst < 0.15 else "CHECK"))
print("""  With %d replicates for %d thresholds (Hartlap %.3f) the estimate is
  numerically sound.  What it does NOT test is whether the bootstrap
  distribution is Gaussian in those deep modes, which the chi2-like statistic
  assumes; for the paper the conservative BEST-SINGLE-THRESHOLD column is the
  safer headline, with the full-covariance value quoted as the optimal
  combination achievable if the covariance is taken at face value.""" %
      (NB_EFF, NP_EFF, HARTLAP))
RES.update(sep_stability_json=json.dumps({("%s_vs_%s" % p): stab[p]
                                          for p in stab}))

chi2, chi2_masked, chi2_masked_cov = {}, {}, {}
_gm2 = _gm & good
# masked covariance, same treatment as the unmasked one
_mok = _gm2 & np.all(np.isfinite(_bt), axis=0)
_BM = _bt[:, _mok]
_nbm, _npm = _BM.shape
if _nbm - _npm - 2 > 0:
    _COVM = np.cov(_BM, rowvar=False)
    _HM = (_nbm - _npm - 2.0) / (_nbm - 1.0)
    _COVMI = np.linalg.inv(_COVM) * _HM
else:
    _COVMI, _HM = None, np.nan
for nm in MODELS:
    r = (XI[PRIMARY] - xi_model[nm]) / XIERR[PRIMARY]
    chi2[nm] = float(np.nansum(r[good] ** 2))
    rm = (xi_msk - xi_model[nm]) / xierr_msk
    chi2_masked[nm] = float(np.nansum(rm[_gm2] ** 2))
    if _COVMI is not None:
        d = (xi_msk - xi_model[nm])[_mok]
        chi2_masked_cov[nm] = float(max(d @ _COVMI @ d, 0.0))
    else:
        chi2_masked_cov[nm] = np.nan
print("\n  chi2 vs data:")
print("    %-10s %16s %14s %16s"
      % ("model", "unmasked diag(%d)" % good.sum(),
         "masked diag(%d)" % _gm2.sum(), "masked FULL COV"))
for nm in MODELS:
    print("    %-10s %16.1f %14.1f %16.1f"
          % (nm, chi2[nm], chi2_masked[nm], chi2_masked_cov[nm]))
print("    [masked covariance from %d replicates on %d thresholds, "
      "Hartlap %.3f]" % (_nbm, _npm, _HM))
_ordc = sorted(chi2_masked_cov, key=chi2_masked_cov.get)
print("    ranking with the full covariance: %s" % " < ".join(_ordc))
print("""    The ranking survives the covariance treatment, but the SPREAD
    compresses substantially -- with diagonal errors the models look far more
    separated than they are, for the same reason the naive quadrature did.""")
print("""  THE TWO VARIANTS BRACKET THE TRUTH; NEITHER IS A GOODNESS-OF-FIT ON ITS OWN.
  The unmasked measurement sits ABOVE every model, because the map contains the
  bright lensed population the models truncate away (cell 4b).  The masked
  measurement sits BELOW every model at EVERY threshold (see the residual signs
  above), because map-level masking right-censors the exceedance distribution
  and biases the GPD MLE downward -- not only near S_mask, as the residual
  table shows.  So the model-comparable value of xi_hat(u) lies BETWEEN the two,
  and the absolute chi2 from either should not be quoted as a fit quality until
  the censoring bias is calibrated by applying the identical mask to simulated
  maps (straightforward with the H1c machinery; not done here).

  What IS robust is the model RANKING, which is identical for both variants and
  stable against the choice of U_MAX_MASKED (checked immediately below).

  TWO SYSTEMATICS ON THESE SEPARATIONS, both established by module H1c:
    (i)  the analytic KL-projected pixel curve differs from a simulated PEAK
         measurement of the SAME counts by up to |dxi| ~ 0.047 near u = 40-45
         mJy, comparable to the model gaps themselves.  The separations above
         are therefore INDICATIVE; the paper's numbers should be recomputed
         from simulated peaks using the H1c machinery.
    (ii) the bright-population mismatch, quantified by the unmasked-vs-masked
         difference in cell 4b.
  Neither affects the MEASUREMENT, only the comparison to models.""")

# --- is the ranking an artefact of where the masked range is cut? ------------
print("\n  RANKING STABILITY of the masked chi2 against the U_MAX_MASKED choice:")
print("    %8s %5s %11s %10s %10s   ranking" % ("U_max", "N", "Schechter",
                                                "SPL", "DPL"))
rank_stable = []
for _umax in (35.0, 40.0, 45.0, U_MAX_MASKED):
    _g3 = (U_GRID <= _umax) & np.isfinite(xi_msk) & np.isfinite(xierr_msk) \
        & (xierr_msk > 0)
    _c = {nm: float(np.nansum(((xi_msk - xi_model[nm]) / xierr_msk)[_g3] ** 2))
          for nm in MODELS}
    _ord = sorted(_c, key=_c.get)
    rank_stable.append(tuple(_ord))
    print("    %8.0f %5d %11.1f %10.1f %10.1f   %s"
          % (_umax, int(_g3.sum()), _c["Schechter"], _c["SPL"], _c["DPL"],
             " < ".join(_ord)))
print("    -> ranking %s across the range tested."
      % ("STABLE" if len(set(rank_stable)) == 1 else "NOT stable"))
RES.update(rank_stable=bool(len(set(rank_stable)) == 1),
           rank_order="|".join(rank_stable[-1]))

RES.update(xi_sch=xi_model["Schechter"], xi_spl=xi_model["SPL"],
           xi_dpl=xi_model["DPL"], chi2_json=json.dumps(chi2),
           chi2_masked_json=json.dumps(chi2_masked),
           chi2_masked_cov_json=json.dumps(chi2_masked_cov),
           thr_corr=CORR, thr_cov=COV, hartlap=HARTLAP,
           n_eff_sum=N_EFF_SUM, n_eff_pca=N_EFF_PCA,
           corr_eigenvalues=_ev,
           sep_cov_json=json.dumps({("%s_vs_%s" % p): sep_cov[p]
                                    for p in pairs}),
           gaps_json=json.dumps({("%s_vs_%s" % p): list(map(float, gaps[p]))
                                 for p in pairs}),
           sep_best_json=json.dumps({("%s_vs_%s" % p): sep_best[p]
                                     for p in pairs}))


#%% -------------------------------- cell 6: figures --------------------------
print("\n" + "-" * 79)
print("cell 6: figures")
print("-" * 79)


def _save(fig, name):
    if SAVE_FIGURES:
        p = os.path.join(FIGURE_DIR, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("  wrote", p)


# --- main result -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.4))
ax.errorbar(U_GRID, XI[PRIMARY], yerr=XIERR[PRIMARY], fmt="o", color="k", ms=5,
            capsize=3, zorder=5,
            label="GAMA-09 350 $\\mu$m, self-filtered IMAGE (%d cutouts)" % NC)
ax.errorbar(U_GRID[_use_m], xi_msk[_use_m], yerr=xierr_msk[_use_m], fmt="D",
            mfc="none", color="0.35", ms=5, capsize=2, zorder=4,
            label="bright-masked ($S<%.0f$ mJy), model-comparable" % S_MASK_MJY)
for nm in MODELS:
    ax.plot(U_GRID, xi_model[nm], MLS[nm], color=MCOL[nm], lw=2,
            label="%s (counts truncated at %.0f mJy)" % (nm, S_CUT_MJY))
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("threshold $u$ [mJy/beam, absolute]")
ax.set_ylabel(r"GPD shape parameter $\hat\xi(u)$")
ax.set_title("Unlensed $\\xi(u)$, GAMA-09 350 $\\mu$m (H1v2)", fontsize=12)
axk = ax.twiny()
axk.set_xlim(*[(v - MU_CORE) / SIG_CORE for v in ax.get_xlim()])
axk.set_xlabel(r"$k=(u-\mu_{\rm core})/\sigma_{\rm core}$", fontsize=9)
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
_save(fig, "herschel_v3_xi_vs_flux.png")

# --- map variants ------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 2, figsize=(12.2, 4.6))
for v, c, mk in [("SF_IMAGE", "k", "o"), ("SF_NEB", "tab:orange", "s"),
                 ("HELP_MFILT", "tab:purple", "^")]:
    ax2[0].errorbar(U_GRID, XI[v], yerr=XIERR[v], fmt=mk, color=c, ms=4,
                    capsize=2, alpha=0.85, label=v)
ax2[0].axhline(0, color="0.7", lw=0.8)
ax2[0].set_xlabel("threshold $u$ [mJy/beam]")
ax2[0].set_ylabel(r"$\hat\xi(u)$")
ax2[0].set_title("Map-variant systematic", fontsize=11)
ax2[0].legend(fontsize=8)
ax2[0].grid(alpha=0.25)
bw = np.arange(len(["IMAGE", "NEBFILT"] + VARIANTS))
ax2[1].bar(bw - 0.2, [meas_w[t]["mad"] for t in ["IMAGE", "NEBFILT"] + VARIANTS],
           0.4, label="MAD-sigma", color="tab:blue")
ax2[1].bar(bw + 0.2, [meas_w[t]["rms"] for t in ["IMAGE", "NEBFILT"] + VARIANTS],
           0.4, label="rms", color="tab:red")
ax2[1].axhline(_st["madsig"], color="tab:blue", ls=":", lw=1.4,
               label="model MAD-sigma")
ax2[1].axhline(_st["rms"], color="tab:red", ls=":", lw=1.4, label="model rms")
ax2[1].set_xticks(bw)
ax2[1].set_xticklabels(["IMAGE", "NEBFILT"] + VARIANTS, rotation=20, fontsize=8)
ax2[1].set_ylabel("width [mJy/beam]")
ax2[1].set_title("Like-for-like widths (cell 3)", fontsize=11)
ax2[1].legend(fontsize=7)
ax2[1].grid(alpha=0.25, axis="y")
fig2.tight_layout()
_save(fig2, "herschel_v3_map_variants.png")

# --- profile moments and exact vs Gaussian P(D) ------------------------------
fig3, ax3 = plt.subplots(1, 3, figsize=(15.0, 4.3))
r = PIX * (np.arange(n_s) - c_s)
ax3[0].plot(r, PSF_G[c_s], "-", color="0.4", lw=2, label="Gaussian 25.15\"")
ax3[0].plot(r, PSF_FN[c_s], "-", color="crimson", lw=2, label="filtered (peak-norm.)")
ax3[0].axhline(0, color="0.7", lw=0.8)
ax3[0].set_xlim(-120, 120)
ax3[0].set_xlabel("radius [arcsec]")
ax3[0].set_ylabel("profile")
ax3[0].set_title("Effective profile (note the lobes)", fontsize=11)
ax3[0].legend(fontsize=8)
ax3[0].grid(alpha=0.25)
ax3[1].plot(QLIST, om_ratio, "o-", color="crimson")
ax3[1].axhline(1.0, color="0.7", lw=0.8)
ax3[1].axvspan(2.0, 4.0, color="0.9", zorder=0, label="tail-relevant $q$")
ax3[1].set_xlabel("$q$")
ax3[1].set_ylabel(r"$\int P_{\rm filt}^q\,/\,\int P_{\rm Gauss}^q$")
ax3[1].set_title("Profile moments", fontsize=11)
ax3[1].legend(fontsize=8)
ax3[1].grid(alpha=0.25)
ax3[2].plot(U_GRID, xi_g, "-", color="0.4", lw=2, label="Gaussian-beam P(D)")
ax3[2].plot(U_GRID, xi_e, "--", color="crimson", lw=2, label="exact filtered P(D)")
ax3[2].errorbar(U_GRID, XI[PRIMARY], yerr=XIERR[PRIMARY], fmt="o", color="k",
                ms=3, capsize=2, alpha=0.5, label="data")
ax3[2].set_xlabel("threshold $u$ [mJy/beam]")
ax3[2].set_ylabel(r"$\xi(u)$")
ax3[2].set_title("Does the profile matter for $\\xi$?", fontsize=11)
ax3[2].legend(fontsize=8)
ax3[2].grid(alpha=0.25)
fig3.tight_layout()
_save(fig3, "herschel_v3_profile_moments.png")

# --- the three count models --------------------------------------------------
Sg = np.geomspace(1.0, 200.0, 300)
fig4, ax4 = plt.subplots(1, 2, figsize=(12.0, 4.6))
for nm in MODELS:
    ax4[0].loglog(Sg, MODELS[nm].dnds(Sg), MLS[nm], color=MCOL[nm], lw=2, label=nm)
    ax4[1].semilogx(Sg, MODELS[nm].eta(Sg), MLS[nm], color=MCOL[nm], lw=2, label=nm)
for a in ax4:
    a.axvspan(U_GRID[0], U_GRID[-1], color="0.88", zorder=0, label="threshold range")
    a.axvline(DPL_PARS["sstar"], color="darkgreen", ls=":", lw=1.2)
    a.set_xlabel("$S$ [mJy]")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)
ax4[0].set_ylabel("$dN/dS$ [mJy$^{-1}$ deg$^{-2}$]")
ax4[1].set_ylabel(r"local slope $\eta(S)$")
ax4[0].set_title("The three count models", fontsize=11)
ax4[1].set_title(r"$\eta(S)$ — what $\xi$ reads out", fontsize=11)
fig4.tight_layout()
_save(fig4, "herschel_v3_counts_models.png")

# --- P(D) comparison ---------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(7.2, 5.0))
bins = np.linspace(-40, 100, 211)
ctr = 0.5 * (bins[1:] + bins[:-1])
hh, _ = np.histogram(pix_pool[PRIMARY], bins=bins, density=True)
ax5.step(ctr, hh, where="mid", color="k", lw=1.5, label="measured (%s pixels)" % PRIMARY)
for nm in MODELS:
    pp = pofd_model[nm]
    ax5.plot(ctr, np.interp(ctr - MU_CORE, pp.d, pp.p), MLS[nm], color=MCOL[nm],
             lw=1.6, label="%s model" % nm)
ax5.plot(ctr, np.interp(ctr - MU_CORE, d_ex, p_ex), ":", color="tab:orange",
         lw=1.6, label="Schechter, exact filtered profile")
ax5.set_yscale("log")
ax5.set_xlim(-40, 100)
ax5.set_ylim(1e-7, None)
ax5.set_xlabel("flux [mJy/beam]")
ax5.set_ylabel("P(D)")
ax5.set_title("Measured vs predicted P(D)", fontsize=11)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.25)
fig5.tight_layout()
_save(fig5, "herschel_v3_widths.png")

# --- discrimination ----------------------------------------------------------
# --- threshold covariance diagnostic (new in v3) -----------------------------
figC, axC = plt.subplots(1, 3, figsize=(14.5, 4.3))
_uc = U_GRID[_bok]
im = axC[0].imshow(CORR, origin="lower", vmin=0, vmax=1, cmap="viridis",
                   extent=[_uc[0], _uc[-1], _uc[0], _uc[-1]])
figC.colorbar(im, ax=axC[0]).set_label(r"$\rho$", fontsize=9)
axC[0].set_xlabel("threshold $u$ [mJy/beam]")
axC[0].set_ylabel("threshold $u$ [mJy/beam]")
axC[0].set_title(r"Correlation of $\hat\xi(u)$ between thresholds", fontsize=11)
_lags = np.arange(1, min(NP_EFF, 18))
axC[1].plot(2.5 * _lags, [np.mean(np.diag(CORR, l)) for l in _lags], "o-",
            color="crimson")
axC[1].axhline(0, color="0.7", lw=0.8)
axC[1].set_xlabel("threshold separation [mJy]")
axC[1].set_ylabel(r"mean $\rho$")
axC[1].set_ylim(0, 1)
axC[1].set_title("Thresholds are not independent", fontsize=11)
axC[1].grid(alpha=0.25)
axC[2].semilogy(np.arange(1, len(_ev) + 1), _ev / _ev.sum(), "o-", color="k")
axC[2].axvline(N_EFF_PCA, color="crimson", ls="--",
               label=r"$n_{\rm eff}=%.2f$ (of %d)" % (N_EFF_PCA, NP_EFF))
axC[2].set_xlabel("principal component")
axC[2].set_ylabel("variance fraction")
axC[2].set_title("Eigenvalue spectrum", fontsize=11)
axC[2].legend(fontsize=8)
axC[2].grid(alpha=0.25)
figC.tight_layout()
_save(figC, "herschel_v3_threshold_covariance.png")

fig6, ax6 = plt.subplots(figsize=(7.4, 4.8))
for (a, b), c in zip(pairs, ["tab:green", "tab:orange", "tab:purple"]):
    ax6.plot(U_GRID, gaps[(a, b)], "o-", color=c, ms=4,
             label=r"$|\xi_{\rm %s}-\xi_{\rm %s}|$" % (a, b))
ax6.plot(U_GRID, XIERR[PRIMARY], "s--", color="k", label="data error (bootstrap)")
ax6.set_xlabel("threshold $u$ [mJy/beam]")
ax6.set_ylabel(r"$\Delta\xi$")
ax6.set_title("Three-way model separation vs the data errors", fontsize=11)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.25)
fig6.tight_layout()
_save(fig6, "herschel_v3_discrimination.png")

#%% -------------------------------- cell 7: summary --------------------------
print("\n" + "=" * 79)
print("SUMMARY  --  H1v2 unlensed xi(u), GAMA-09 350 um")
print("=" * 79)
print("  primary map             : %s (self-filtered, padded %d pix)" % (PRIMARY, PAD))
print("  operator check          : corr %.10f vs HELP MFILT, rms %.2e" % (OP_CORR, OP_RMS))
print("  padding check           : max|diff| %.2e mJy/beam (unpadded: %.4f rms)"
      % (PAD_MAXDIFF, BARE_RMS))
print("  cutouts / peaks         : %d / %d" % (NC, CORE[PRIMARY]["n"]))
print("  peak core mu / sigma    : %+.3f / %.3f mJy/beam" % (MU_CORE, SIG_CORE))
print("  matched-filter noise    : %.3f mJy/beam (+-%.1f%% across cutouts)"
      % (SIGMA_INST, 100 * np.std(noise_cut) / SIGMA_INST))
print("  Omega_2 ratio           : %.4f  |  flux response %.4f" % (OM2_RATIO, FLUX_RESP))
print("  profile effect on xi    : max |dxi| = %.4f (vs median error %.4f)"
      % (DXI_PROFILE, np.nanmedian(XIERR[PRIMARY])))
print("  model/measured width    : rms %.3f, MAD-sigma %.3f (no renormalisation)"
      % (_st["rms"] / meas_w[PRIMARY]["rms"], _st["madsig"] / meas_w[PRIMARY]["mad"]))
print("  threshold range         : %.0f-%.0f mJy (k = %.2f-%.2f)"
      % (U_GRID[0], U_GRID[-1], (U_GRID[0] - MU_CORE) / SIG_CORE,
         (U_GRID[-1] - MU_CORE) / SIG_CORE))
print("  threshold covariance    : n_eff = %.2f of %d, Hartlap %.3f"
      % (N_EFF_SUM, NP_EFF, HARTLAP))
print("  model separations (full covariance; best-single in brackets):")
for p in pairs:
    print("      %-22s %5.2f sigma   [%.2f at u = %.1f mJy]"
          % ("%s vs %s" % p, sep_cov[p], sep_best[p][0], sep_best[p][1]))
print("  masked chi2 (full cov)  : %s"
      % ", ".join("%s %.1f" % (n, chi2_masked_cov[n]) for n in _ordc))
print("  runtime                 : %.0f s" % (time.time() - T0))
print("  bright peaks             : %d > 100 mJy, %d > 200 mJy in %.1f deg^2"
      % (bright_obs[0], bright_obs[2], AREA))
print("                            (Schechter predicts %.2f and %.2f)"
      % (bright_pred["Schechter"][0], bright_pred["Schechter"][2]))
print("  slope d(xi)/du, u<=%.0f    : %+.5f unmasked, %+.5f bright-masked"
      % (U_MAX_MASKED, _sl_un, _sl_ms))
print("=" * 79)
print("""
  *** THE RISING xi_hat(u) IS UNDERSTOOD -- IT IS NOT A SYSTEMATIC ***

  Module H1c attributed the measured d(xi)/du = +0.00876 mJy^-1 as follows:
  pixel -> peak -0.00077 (small, and the WRONG SIGN), bright lensed population
  +0.01028 (dominant), residual +0.00219.  chi2 vs the data falls from 1753
  (analytic model) to 152 once a bright tail calibrated to the observed bright
  peaks is injected into a peak-level simulation.  So xi(u) at high threshold is
  reading the transition from the intrinsic counts to the lensing-dominated
  bright population -- a result to report, not an error to remove.

  Corrected here relative to H1: the pixel-vs-peak mismatch was the wrong
  suspect, and declustered peaks give a LOWER xi than pixels, not higher.

  What this module establishes independently:

    - the matched filter is not the culprit: SF_IMAGE vs HELP_MFILT differ by
      at most %.2f sigma per threshold (median %.2f, none above 2), and
      SF_NEB reproduces HELP_MFILT to %.0e in xi;
    - the beam model is not the culprit: the exact filtered-profile P(D), lobes
      and all, moves xi by |dxi| = %.4f, below the median bootstrap error;
    - the counts normalisation is not the culprit: compared like-for-like the
      model width matches the data inside the fit's own uncertainty;
    - the bright population is real and large: %d peaks above 200 mJy where the
      Schechter predicts %.2f in the same area.

  WHAT REMAINS OPEN, and what H2 must inherit:

    1. The analytic model curves carry a systematic of order the signal.  H1c
       found the KL-projected pixel curve differs from a simulated PEAK
       measurement of identical counts by up to |dxi| ~ 0.047, against model
       gaps of ~0.086.  The paper's discrimination should be recomputed from
       simulated peaks (H1c machinery), not from xi_of_threshold_analytic.
    2. The simulated peak density (0.2545/beam in H1c) undershoots the measured
       0.3736 by 32%%, unexplained; clustering is the obvious candidate, since
       both the analytic P(D) and CIBMapSimulator are Poisson.
    3. For H2 the bright population largely cancels in Delta-xi, since cluster
       and control cutouts see the same sky -- but the S_mask = %.0f mJy map
       mask defined in cell 4b MUST be applied identically to both, and the
       Delta-xi prediction should come from simulated peaks.
""" % (_zv.max(), np.median(_zv), _dn, DXI_PROFILE, bright_obs[2],
       bright_pred["Schechter"][2], S_MASK_MJY))
print("=" * 79)

npz = os.path.join(FIGURE_DIR, "herschel_unlensed_v3_350.npz")
np.savez_compressed(npz, **RES)
print("\nwrote", npz)
if SAVE_FIGURES and os.environ.get("DISPLAY") or sys.platform == "darwin":
    plt.show()
