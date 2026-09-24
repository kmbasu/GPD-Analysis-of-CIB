#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_analysis.py -- Noise budget and P(D) characterisation of the HELP dmu19
                     Herschel/SPIRE 350 um map of the GAMA-09 field.
===============================================================================

PURPOSE
-------
This is module **H0** of the Herschel arm of the CIB-lensing / GPD project.  It
is the Herschel counterpart of the noise-budget part of ``planck_unlensed_pd.py``
(module M4a of the Planck task) and it must be run and understood *before* any
POT/GPD machinery is applied to these data.  It answers four questions:

  (A) What are the actual instrument and confusion noise levels in the GAMA-09
      350 um map, and do they agree with (i) the values HELP stored in the
      ``Matchedfilter`` header, and (ii) the published SPIRE confusion limit of
      Nguyen et al. (2010, A&A 518, L5), sigma_conf = 6.3 mJy/beam at 350 um?

  (B) Exactly what operation produced the ``MFILT`` extension?  Unless we can
      reproduce it from first principles we cannot build a forward model that
      maps a source-count model n(S) onto the observed P(D) of the filtered map,
      and the whole GPD programme would be stuck with an uncalibrated observable.

  (C) Is the matched-filtered map a *suitable* primary map for peaks-over-
      threshold (POT) work?  The raw ``IMAGE`` extension is not: at 8"/pixel
      under a 25.15" beam the instrument noise is essentially white per pixel
      while the sky is beam-correlated, so local-maximum declustering saturates
      and one ends up characterising the detectors rather than the CIB.

  (D) Can we reproduce, for GAMA-09 at 350 um, the diagnostic plot of
      Nguyen et al. (2010) Fig. 3 -- the pixel-intensity histogram of the map
      with the instrument-noise-only distribution overlaid?  This is the single
      most legible demonstration that the bright-side asymmetry of the P(D) is
      sky (confusion), not detector noise.

WHAT THE SCRIPT DOES NOT DO
---------------------------
No GPD/EVT fitting.  No xi(u), no Delta-xi(u), no cluster stacking.  Those come
in later modules.  This script is deliberately confined to second-moment noise
accounting plus the P(D) histogram.

METHOD SUMMARY
--------------
* **Matched filter.**  We recover the HELP matched filter as the standard
  inverse-variance-weighted form (Chapin et al. 2011, MNRAS 411, 505):

      MFILT = [ (d / sigma^2) * K ] / [ (1 / sigma^2) * K^2 ]

  where ``d`` = NEBFILT (HDU 2, the nebulosity-filtered signal map), ``sigma`` =
  ERROR (HDU 3), ``K`` = the 101x101 kernel stored in HDU 8, and ``*`` denotes
  2-D convolution.  The matching error map is

      MFILT_ERROR = 1 / sqrt( (1 / sigma^2) * K^2 ).

  Note the filter is built on NEBFILT, *not* on IMAGE -- consistent with the
  header comment ``FWHM = 25.15 / FWHM of PSF in NEBFILT map``.  Cell 4 verifies
  both identities numerically on an interior test region.

* **Noise decomposition, route 1 (Nguyen's own method).**  Spatial fluctuations
  in a map come from the detectors, which integrate down with time, and from the
  sky, which does not:

      sigma_total^2(t) = sigma_conf^2 + sigma_inst^2 / t

  with ``sigma_conf`` in mJy/beam and ``sigma_inst`` in mJy/beam/sqrt(s).  Binning
  pixels by their EXPOSURE value and fitting the binned variance against 1/t
  gives ``sigma_conf^2`` as the intercept and ``sigma_inst^2`` as the slope.  This
  is Nguyen et al. (2010) Fig. 2 and it uses no external noise information at all.

* **Noise decomposition, route 2 (ERROR-map route).**  The ERROR extension is the
  pipeline's own per-pixel instrument-noise estimate.  Taking it at face value,
  the confusion term follows by quadrature subtraction from the robust width of
  the map.  Route 2 is a consistency check on route 1, not an independent
  measurement, because both ultimately trace back to the same timeline data.

* **Instrument-noise realisation.**  Nguyen et al. built their instrument-noise
  histogram from a jackknife (null) map, i.e. the difference of two halves of the
  data.  HELP does not ship half-maps for GAMA-09, so we synthesise a noise
  realisation ``n ~ N(0, ERROR_i)`` per pixel and, for the MFILT panel, push that
  realisation through the *same* matched filter.  This is a fair substitute for
  the purpose at hand (it has the right per-pixel variance and the right response
  to the filter), but it does assume the pipeline noise is white between pixels
  and correctly normalised; cell 6 tests exactly that assumption rather than
  taking it on trust.

* **Robust widths.**  Every "sigma" quoted for a map is a MAD-based estimate,
  ``1.4826 * median(|x - median(x)|)``, so that the bright confusion tail does not
  inflate the core width.  Plain rms values are reported alongside for contrast;
  the gap between them *is* the non-Gaussianity we care about.

INPUTS
------
``<CIB_HERSCHEL_DIR>/GAMA-09_SPIRE350_v1.0.fits`` -- the HELP dmu19 product.
Use the ``v1.0`` file, not ``v0.9``: the latter has only 5 HDUs and lacks the
MFILT, MFILT_ERROR and Matchedfilter extensions this script depends on.

HDU layout of the v1.0 files::

    0 PRIMARY
    1 IMAGE          signal map,          Jy/beam, float32
    2 NEBFILT        nebulosity-filtered, Jy/beam, float32
    3 ERROR          error map,           Jy/beam, float32
    4 EXPOSURE       exposure map,        seconds, float64
    5 MASK           0 = good, 1 = low relative depth, int32
    6 MFILT          matched-filtered,    Jy/beam, float64
    7 MFILT_ERROR    error on MFILT,      Jy/beam, float64
    8 Matchedfilter  101x101 kernel; header carries FWHM, NCONF, NINS, PIXSIZE

OUTPUTS
-------
``results/herschel_noise_analysis_350.npz``      all derived quantities
``results/herschel_noise_pd_histogram.png``      the Nguyen Fig. 3 analogue
``results/herschel_noise_variance_vs_invt.png``  the Nguyen Fig. 2 analogue
``results/herschel_noise_peak_density.png``      the POT-suitability diagnostic
``results/herschel_noise_psf_response.png``      matched-filter PSF calibration

ENVIRONMENT SWITCHES
--------------------
``CIB_HERSCHEL_DIR``  directory holding the dmu19 FITS files (default:
                      ``Herschel_analysis/data/``)
``CIB_SAVE_FIGURES``  "1" to write PNGs (default "1")
``CIB_FIGURE_DIR``    where to write them (default ``./results``)
``CIB_QUICK_TEST``    "1" to run on a 2000x2000 sub-region for a fast smoke test
``CIB_RANDOM_SEED``   seed for the noise realisation (default 1234)

USAGE
-----
Run top-to-bottom (``python noise_analysis.py``) or cell-by-cell in Spyder; the
``#%%`` separators mark independently executable cells.

DEPENDENCIES
------------
numpy, scipy, matplotlib, astropy.  All standard; nothing exotic.

Author: CIB-lensing / GPD project, Herschel task, July 2026.
Companion documents: ``Planck_analysis/Planck_analysis_summary.md`` (the template
this module follows), ``GPD_tail_modelling_manual_v4.md`` sections 13.2/13.6/13.7
(floor and peak-shift arithmetic that the next module will need).
"""

#%% ---------------------------------------------------------------- imports --
import os
import sys
import json
import warnings

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.signal import oaconvolve
from scipy.ndimage import maximum_filter
from scipy.stats import skew, kurtosis, norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------ config --
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

DATA_DIR = os.environ.get(
    "CIB_HERSCHEL_DIR", os.path.join(HERE, "data")
)
MAP_FILE = os.path.join(DATA_DIR, "GAMA-09_SPIRE350_v1.0.fits")

SAVE_FIGURES = os.environ.get("CIB_SAVE_FIGURES", "1") == "1"
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", os.path.join(HERE, "results"))
QUICK_TEST = os.environ.get("CIB_QUICK_TEST", "0") == "1"
SEED = int(os.environ.get("CIB_RANDOM_SEED", "1234"))

os.makedirs(FIGURE_DIR, exist_ok=True)

JY2MJY = 1.0e3          # the maps are in Jy/beam; we work throughout in mJy/beam
MAD2SIG = 1.4826        # MAD -> Gaussian sigma

# Published reference values, for comparison only (Nguyen et al. 2010, Table 1).
NGUYEN_CONF = {"250": 5.8, "350": 6.3, "500": 6.8}          # mJy/beam
NGUYEN_CONF_ERR = {"250": 0.3, "350": 0.4, "500": 0.4}
NGUYEN_INST = {"250": 8.5, "350": 9.4, "500": 13.3}          # mJy/beam/sqrt(s)
NGUYEN_INST_ERR = {"250": 0.4, "350": 0.5, "500": 0.7}
BAND = "350"

RESULTS = {}            # everything worth keeping ends up here


def robust_sigma(x):
    """MAD-based Gaussian-equivalent sigma, ignoring NaNs.

    Uses ``1.4826 * median(|x - median(x)|)``.  Preferred over the plain rms
    everywhere in this project because the confusion tail is exactly the thing we
    do not want contaminating our estimate of the *core* width.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return MAD2SIG * np.median(np.abs(x - np.median(x)))


def describe(name, x, mask=None):
    """Return (and print) a dict of robust + non-robust summary statistics.

    Reports mean/median/rms/MAD-sigma/skewness/excess-kurtosis/extrema.  The
    rms-to-MAD ratio and the skewness are the two numbers that tell us at a
    glance how non-Gaussian the distribution is -- for a pure Gaussian the ratio
    is 1 and the skewness 0.
    """
    v = x[mask] if mask is not None else np.asarray(x).ravel()
    v = v[np.isfinite(v)]
    d = dict(
        n=int(v.size),
        mean=float(v.mean()),
        median=float(np.median(v)),
        rms=float(v.std()),
        mad_sigma=float(robust_sigma(v)),
        skew=float(skew(v)),
        kurtosis=float(kurtosis(v)),
        min=float(v.min()),
        max=float(v.max()),
    )
    d["rms_over_mad"] = d["rms"] / d["mad_sigma"] if d["mad_sigma"] > 0 else np.nan
    print(
        "  %-14s n=%9d  mean %8.4f  med %8.4f  rms %7.3f  MADsig %7.3f  "
        "rms/MAD %5.2f  skew %6.2f  kurt %8.2f  max %9.2f"
        % (name, d["n"], d["mean"], d["median"], d["rms"], d["mad_sigma"],
           d["rms_over_mad"], d["skew"], d["kurtosis"], d["max"])
    )
    return d


#%% ------------------------------------------------- cell 1: load the map ----
print("=" * 79)
print("H0  Herschel/SPIRE 350 um GAMA-09 noise analysis")
print("=" * 79)
print("map file :", MAP_FILE)
if not os.path.exists(MAP_FILE):
    raise SystemExit(
        "\nERROR: %s not found.\n"
        "Set CIB_HERSCHEL_DIR to the directory holding the dmu19 FITS files.\n"
        "Note this script requires the v1.0 product; v0.9 lacks the MFILT and\n"
        "Matchedfilter extensions." % MAP_FILE
    )

hdul = fits.open(MAP_FILE, memmap=True)
if len(hdul) < 9:
    raise SystemExit(
        "ERROR: this file has only %d HDUs. You are probably looking at the v0.9\n"
        "product; the v1.0 file has 9 (PRIMARY + 8 extensions) including MFILT." % len(hdul)
    )

hdr_pri = hdul[0].header
hdr_img = hdul[1].header
hdr_mf = hdul[8].header

FWHM_ARCSEC = float(hdr_mf["FWHM"])          # 25.15" at 350 um
PIXSIZE = float(hdr_mf["PIXSIZE"])           # 8.0"
NCONF_HDR = float(hdr_mf["NCONF"]) * JY2MJY  # mJy/beam
NINS_HDR = float(hdr_mf["NINS"]) * JY2MJY    # mJy/beam

# Beam solid angle for a Gaussian: Omega = 1.133 * FWHM^2 (i.e. 2*pi*sigma^2).
BEAM_ARCSEC2 = 1.133 * FWHM_ARCSEC ** 2
BEAM_PIX = BEAM_ARCSEC2 / PIXSIZE ** 2
PIX_PER_FWHM = FWHM_ARCSEC / PIXSIZE

print("\nfield %s / filter %s / HELP version %s"
      % (hdr_pri.get("FIELD"), hdr_pri.get("FILTER"), hdr_pri.get("VERSION")))
print("FWHM      = %.2f arcsec        (%.2f pixels per FWHM)" % (FWHM_ARCSEC, PIX_PER_FWHM))
print("pixel     = %.1f arcsec         (beam = %.2f pixels = %.1f arcsec^2)"
      % (PIXSIZE, BEAM_PIX, BEAM_ARCSEC2))
print("NCONF     = %.3f mJy/beam      (HELP header: confusion noise)" % NCONF_HDR)
print("NINS      = %.3f mJy/beam      (HELP header: instrumental noise)" % NINS_HDR)

sl = (slice(0, 2000), slice(0, 2000)) if QUICK_TEST else (slice(None), slice(None))
if QUICK_TEST:
    print("\n*** CIB_QUICK_TEST=1: running on a 2000x2000 sub-region ***")

# Loaded as float32 deliberately.  The full array is 6742x2707 = 18.3e6 pixels;
# at float64 the six maps alone would occupy ~900 MB before any working copies,
# which will kill the process on a 16 GB laptop once the convolutions start.
# float32 carries ~7 significant digits, far more than these data warrant (the
# FITS storage class for IMAGE/NEBFILT/ERROR is itself float32).  Cell 4, which
# needs to demonstrate an exact operator identity, promotes its sub-region back
# to float64.
FDT = np.float32
IMAGE = hdul[1].data[sl].astype(FDT) * JY2MJY
NEBFILT = hdul[2].data[sl].astype(FDT) * JY2MJY
ERROR = hdul[3].data[sl].astype(FDT) * JY2MJY
EXPOSURE = hdul[4].data[sl].astype(FDT)
MASK = hdul[5].data[sl]
MFILT = hdul[6].data[sl].astype(FDT) * JY2MJY
MFILT_ERROR = hdul[7].data[sl].astype(FDT) * JY2MJY
KERNEL = hdul[8].data.astype(np.float64)

# "good" = on the observed footprint, unmasked, with a usable error estimate.
GOOD = (
    np.isfinite(IMAGE) & np.isfinite(NEBFILT) & np.isfinite(MFILT)
    & np.isfinite(ERROR) & (ERROR > 0) & (MASK == 0)
)
area_deg2 = GOOD.sum() * (PIXSIZE / 3600.0) ** 2
n_beams = GOOD.sum() / BEAM_PIX

print("\ngood pixels    : %d  (%.1f%% of the array)" % (GOOD.sum(), 100 * GOOD.mean()))
print("sky area       : %.2f deg^2   -> %.3e independent beams" % (area_deg2, n_beams))
print("MASK values    : %s" % dict(zip(*[a.tolist() for a in np.unique(MASK, return_counts=True)])))
print("EXPOSURE (s)   : median %.2f, range %.2f - %.2f"
      % (np.median(EXPOSURE[GOOD]), EXPOSURE[GOOD].min(), EXPOSURE[GOOD].max()))

RESULTS.update(
    fwhm_arcsec=FWHM_ARCSEC, pixsize=PIXSIZE, beam_pix=BEAM_PIX,
    beam_arcsec2=BEAM_ARCSEC2, nconf_hdr=NCONF_HDR, nins_hdr=NINS_HDR,
    area_deg2=area_deg2, n_beams=n_beams, n_good=int(GOOD.sum()),
)

#%% -------------------------------------- cell 2: basic map statistics -------
print("\n" + "-" * 79)
print("cell 2: pixel statistics on good pixels  [mJy/beam]")
print("-" * 79)

stats = {}
stats["IMAGE"] = describe("IMAGE", IMAGE, GOOD)
stats["NEBFILT"] = describe("NEBFILT", NEBFILT, GOOD)
stats["ERROR"] = describe("ERROR", ERROR, GOOD)
stats["MFILT"] = describe("MFILT", MFILT, GOOD)
stats["MFILT_ERROR"] = describe("MFILT_ERROR", MFILT_ERROR, GOOD)
stats["IMAGE-NEBFILT"] = describe("IMAGE-NEBFILT", IMAGE - NEBFILT, GOOD)

# The nebular filter's footprint on the variance budget: if this is small, the
# cirrus contribution to the P(D) is small, and the choice between IMAGE and
# NEBFILT as the basis of the analysis is not a leading-order decision.
frac_var_neb = stats["IMAGE-NEBFILT"]["rms"] ** 2 / stats["IMAGE"]["rms"] ** 2
print("\n  variance removed by the nebular filter: %.1f%% of the IMAGE variance"
      % (100 * frac_var_neb))
print("  quadrature of header values sqrt(NCONF^2 + NINS^2) = %.3f mJy/beam"
      % np.hypot(NCONF_HDR, NINS_HDR))
print("  measured IMAGE MAD-sigma                           = %.3f mJy/beam"
      % stats["IMAGE"]["mad_sigma"])
print("  -> ratio %.4f  (a value near 1 means the variance budget closes on\n"
      "     Poisson confusion + instrument noise alone, with no unexplained\n"
      "     clustered-CIB/cirrus excess of the kind that dominated at Planck)"
      % (stats["IMAGE"]["mad_sigma"] / np.hypot(NCONF_HDR, NINS_HDR)))

RESULTS["stats"] = stats
RESULTS["frac_var_nebular"] = frac_var_neb

#%% ------------------- cell 3: matched-filter kernel characterisation --------
print("\n" + "-" * 79)
print("cell 3: the matched-filter kernel (HDU 8)")
print("-" * 79)

K = KERNEL
sumK, sumK2 = K.sum(), (K ** 2).sum()
print("  shape %s   sum(K) = %.4f   sum(K^2) = %.4f   peak = %.4f   min = %.4f"
      % (K.shape, sumK, sumK2, K.max(), K.min()))
print("  the kernel has negative side lobes (min < 0): it is a *whitening* filter,")
print("  not a smoothing kernel -- it suppresses the white per-pixel detector")
print("  noise while matching the beam-scale source profile.")

# For a spatially uniform error map, MFILT_ERROR reduces to sigma_pix/sqrt(sum K^2).
pred_mfe_uniform = np.median(ERROR[GOOD]) / np.sqrt(sumK2)
print("\n  predicted MFILT_ERROR for uniform sigma: median(ERROR)/sqrt(sum K^2)")
print("      = %.4f / %.4f = %.4f mJy/beam"
      % (np.median(ERROR[GOOD]), np.sqrt(sumK2), pred_mfe_uniform))
print("  measured median(MFILT_ERROR)                     = %.4f mJy/beam"
      % np.median(MFILT_ERROR[GOOD]))

RESULTS.update(kernel_sum=float(sumK), kernel_sum2=float(sumK2),
               kernel_peak=float(K.max()), kernel_min=float(K.min()),
               mfilt_error_pred_uniform=float(pred_mfe_uniform),
               mfilt_error_median=float(np.median(MFILT_ERROR[GOOD])))


def apply_matched_filter(data, sigma, kernel, valid=None):
    """Apply the HELP/Chapin inverse-variance-weighted matched filter.

        MFILT = [ (d / sigma^2) * K ] / [ (1 / sigma^2) * K^2 ]

    Parameters
    ----------
    data : 2-D array
        Signal map (any units); NaNs are treated as zero-signal, zero-weight.
    sigma : 2-D array
        Per-pixel noise map, same units as ``data``.
    kernel : 2-D array
        The matched-filter kernel, HDU 8.
    valid : 2-D bool array, optional
        Pixels to include.  Defaults to all finite, positive-sigma pixels.

    Returns
    -------
    filtered : 2-D array
        The filtered map, NaN outside ``valid``'s support.
    err : 2-D array
        The propagated error map, ``1 / sqrt((1/sigma^2) * K^2)``.

    Notes
    -----
    ``oaconvolve`` (overlap-add) is used rather than ``fftconvolve`` because the
    kernel is small (101x101) relative to the map (6742x2707) and overlap-add is
    far cheaper in peak memory for that regime.
    """
    if valid is None:
        valid = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
    w = np.where(valid, 1.0 / np.where(sigma > 0, sigma, 1.0) ** 2, 0.0)
    d = np.where(valid, data, 0.0) * w
    num = oaconvolve(d, kernel, mode="same")
    den = oaconvolve(w, kernel ** 2, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = np.where(den > 0, num / den, np.nan)
        err = np.where(den > 0, 1.0 / np.sqrt(den), np.nan)
    return filtered, err


def apply_matched_filter_tiled(data, sigma, kernel, valid=None, rows=512,
                               dtype=np.float32):
    """Memory-bounded version of :func:`apply_matched_filter`.

    Convolution is a local operation with a finite footprint, so the map can be
    processed in horizontal strips provided each strip is padded by one kernel
    half-width on each side and the padding is discarded afterwards.  Peak memory
    then scales with ``rows`` rather than with the map height, which is what makes
    this runnable on a laptop: the full GAMA-09 array is 6742 x 2707, and doing
    the convolution in one piece needs several gigabytes of float64 workspace.

    Results are bit-comparable to the un-tiled routine to within float32
    rounding; cell 4 verifies the operator itself at float64 on a sub-region.
    """
    ny_, nx_ = data.shape
    kh = kernel.shape[0] // 2
    if valid is None:
        valid = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
    out = np.full((ny_, nx_), np.nan, dtype=dtype)
    out_e = np.full((ny_, nx_), np.nan, dtype=dtype)
    for y0 in range(0, ny_, rows):
        y1 = min(y0 + rows, ny_)
        ys_ = max(y0 - kh, 0)
        ye_ = min(y1 + kh, ny_)
        v = valid[ys_:ye_]
        s = np.asarray(sigma[ys_:ye_], dtype=np.float64)
        d = np.asarray(data[ys_:ye_], dtype=np.float64)
        w = np.where(v, 1.0 / np.where(s > 0, s, 1.0) ** 2, 0.0)
        num = oaconvolve(np.where(v, d, 0.0) * w, kernel, mode="same")
        den = oaconvolve(w, kernel ** 2, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(den > 0, num / den, np.nan)
            e = np.where(den > 0, 1.0 / np.sqrt(den), np.nan)
        out[y0:y1] = f[y0 - ys_:y0 - ys_ + (y1 - y0)].astype(dtype)
        out_e[y0:y1] = e[y0 - ys_:y0 - ys_ + (y1 - y0)].astype(dtype)
        del v, s, d, w, num, den, f, e
    return out, out_e


#%% ------------------- cell 4: verify the matched-filter definition ----------
print("\n" + "-" * 79)
print("cell 4: reproducing MFILT and MFILT_ERROR from NEBFILT + ERROR + K")
print("-" * 79)
print("  If this succeeds we know the exact operator relating the counts model")
print("  to the observed filtered map, which is what the GPD forward model needs.")

# An interior test region, padded so the convolution is not edge-contaminated.
ny, nx = IMAGE.shape
cy, cx = ny // 2, nx // 2
half = min(400, ny // 2 - 150, nx // 2 - 150)
pad = K.shape[0]           # one full kernel width of padding
ys, ye = cy - half - pad, cy + half + pad
xs, xe = cx - half - pad, cx + half + pad
core = (slice(pad, -pad), slice(pad, -pad))

sub_neb = NEBFILT[ys:ye, xs:xe].astype(np.float64)
sub_err = ERROR[ys:ye, xs:xe].astype(np.float64)
sub_valid = np.isfinite(sub_neb) & np.isfinite(sub_err) & (sub_err > 0)
rec, rec_err = apply_matched_filter(sub_neb, sub_err, K, sub_valid)
rec, rec_err = rec[core], rec_err[core]

# Read the reference back from the file at full float64 precision: comparing
# against the float32 working copy would floor the achievable correlation at the
# float32 rounding level and obscure whether the operator is exactly right.
# (``sl`` always starts at index 0, so sliced and file coordinates coincide.)
ref = hdul[6].data[ys + pad:ye - pad, xs + pad:xe - pad].astype(np.float64) * JY2MJY
ref_err = hdul[7].data[ys + pad:ye - pad, xs + pad:xe - pad].astype(np.float64) * JY2MJY
m = np.isfinite(rec) & np.isfinite(ref)

resid = rec[m] - ref[m]
corr = np.corrcoef(rec[m], ref[m])[0, 1]
print("\n  test region %dx%d, %d valid pixels" % (ref.shape[0], ref.shape[1], m.sum()))
print("  reconstruction vs stored MFILT : corr = %.8f" % corr)
print("      residual rms = %.3e mJy/beam  (map rms = %.3f)"
      % (resid.std(), ref[m].std()))
print("      max |residual| = %.3e mJy/beam" % np.abs(resid).max())

me = np.isfinite(rec_err) & np.isfinite(ref_err)
resid_e = rec_err[me] - ref_err[me]
print("  reconstruction vs stored MFILT_ERROR : corr = %.8f"
      % np.corrcoef(rec_err[me], ref_err[me])[0, 1])
print("      residual rms = %.3e mJy/beam" % resid_e.std())

# A plain (unweighted) convolution is shown for contrast: it is *not* the filter.
plain = oaconvolve(np.nan_to_num(sub_neb), K, mode="same")[core]
mp = np.isfinite(plain) & np.isfinite(ref)
print("\n  for contrast, a plain unweighted convolution NEBFILT*K:")
print("      corr = %.6f (slope %.4f) -- close but demonstrably not the operator"
      % (np.corrcoef(plain[mp], ref[mp])[0, 1], np.polyfit(plain[mp], ref[mp], 1)[0]))

# Same test against IMAGE, to confirm the filter is built on NEBFILT.
sub_img = IMAGE[ys:ye, xs:xe].astype(np.float64)
rec_img, _ = apply_matched_filter(sub_img, sub_err, K,
                                  np.isfinite(sub_img) & sub_valid)
rec_img = rec_img[core]
mi = np.isfinite(rec_img) & np.isfinite(ref)
print("  same filter applied to IMAGE instead of NEBFILT:")
print("      corr = %.6f -- confirms HELP built MFILT on the NEBFILT extension"
      % np.corrcoef(rec_img[mi], ref[mi])[0, 1])

MF_VERIFIED = bool(corr > 0.9999 and resid.std() < 1e-6 * max(ref[m].std(), 1e-12) + 1e-6)
print("\n  => matched-filter definition VERIFIED: %s" % MF_VERIFIED)

RESULTS.update(mf_corr=float(corr), mf_resid_rms=float(resid.std()),
               mf_verified=MF_VERIFIED,
               mf_corr_plain=float(np.corrcoef(plain[mp], ref[mp])[0, 1]),
               mf_corr_on_image=float(np.corrcoef(rec_img[mi], ref[mi])[0, 1]))

#%% ---------------- cell 5: point-source response of the matched filter ------
print("\n" + "-" * 79)
print("cell 5: point-source calibration and effective beam of the filtered map")
print("-" * 79)
print("  We inject a known-flux point source into a blank, uniform-noise map and")
print("  push it through the filter.  This fixes (a) whether MFILT preserves")
print("  point-source flux, and (b) the *effective* beam of the filtered map,")
print("  which sets the 'per beam' normalisation for all later P(D) work.")

S_IN = 100.0                                   # mJy, injected point-source flux
npix_sim = 401
yy, xx = np.mgrid[:npix_sim, :npix_sim]
c0 = npix_sim // 2
sigma_pix = (FWHM_ARCSEC / PIXSIZE) / (2 * np.sqrt(2 * np.log(2)))

# A Gaussian PSF normalised to unit *peak*, so that a source of flux S has peak
# S in mJy/beam -- the convention the SPIRE maps use.
psf = np.exp(-((yy - c0) ** 2 + (xx - c0) ** 2) / (2 * sigma_pix ** 2))
src_map = S_IN * psf
sigma_uniform = np.full_like(src_map, np.median(ERROR[GOOD]))

filt_src, _ = apply_matched_filter(src_map, sigma_uniform, K,
                                   np.ones_like(src_map, dtype=bool))
peak_out = np.nanmax(filt_src)
print("\n  injected point source  : S = %.1f mJy  (peak %.1f mJy/beam in map space)"
      % (S_IN, src_map.max()))
print("  peak after filtering   : %.3f mJy/beam" % peak_out)
print("  flux response factor   : %.4f" % (peak_out / S_IN))

# Effective beam of the filtered map, from the filtered PSF profile.
prof = filt_src[c0, :]
above = prof >= 0.5 * prof.max()
fwhm_eff_pix = np.count_nonzero(above)
# sub-pixel FWHM by interpolation on the rising edge
idx = np.where(above)[0]
if len(idx) > 1 and idx[0] > 0:
    lo = idx[0] - 1 + (0.5 * prof.max() - prof[idx[0] - 1]) / (prof[idx[0]] - prof[idx[0] - 1])
    hi = idx[-1] + (prof[idx[-1]] - 0.5 * prof.max()) / (prof[idx[-1]] - prof[idx[-1] + 1])
    fwhm_eff_pix = hi - lo
fwhm_eff = fwhm_eff_pix * PIXSIZE
# Solid angle of the filtered response, normalised to peak (the "beam area" that
# converts a surface-brightness variance into a per-beam variance).
omega_eff_pix = np.nansum(filt_src) / np.nanmax(filt_src)
omega_eff = omega_eff_pix * PIXSIZE ** 2

print("\n  effective FWHM of the filtered PSF : %.2f arcsec  (map-space %.2f)"
      % (fwhm_eff, FWHM_ARCSEC))
print("  effective solid angle              : %.1f arcsec^2 = %.2f pixels"
      % (omega_eff, omega_eff_pix))
print("      (map-space beam: %.1f arcsec^2 = %.2f pixels)" % (BEAM_ARCSEC2, BEAM_PIX))
print("  -> ratio FWHM_eff / FWHM = %.3f" % (fwhm_eff / FWHM_ARCSEC))
print("""
  Interpretation.  The filter leaves BOTH the point-source flux scale and the
  effective beam width essentially unchanged.  That is the signature of a
  properly normalised matched filter: its negative side lobes cancel the
  broadening that a plain smoothing kernel would introduce, so the output is
  still 'peak flux in mJy/beam' on a beam of the original width.  The noise
  suppression therefore does NOT come from smoothing.  It comes from combining
  the ~%.0f pixels across a beam with optimal (inverse-variance, beam-matched)
  weights, which reduces white per-pixel noise by sqrt(sum K^2) = %.2f while
  leaving a beam-shaped source untouched.  This is exactly why MFILT is the
  right map for POT work: it improves the sky-to-detector ratio without
  degrading the resolution or rescaling the flux axis that xi(u) is measured on.
""" % (BEAM_PIX, np.sqrt(sumK2)))

RESULTS.update(psf_flux_response=float(peak_out / S_IN),
               fwhm_eff_arcsec=float(fwhm_eff),
               omega_eff_arcsec2=float(omega_eff),
               omega_eff_pix=float(omega_eff_pix))

#%% ------------- cell 6: noise decomposition, route 1 (Nguyen's method) ------
print("\n" + "-" * 79)
print("cell 6: sigma_total^2 vs 1/t  --  the Nguyen et al. (2010) Fig. 2 method")
print("-" * 79)
print("  sigma_total^2(t) = sigma_conf^2 + sigma_inst^2 / t")
print("  intercept -> confusion noise [mJy/beam]; slope -> instrument noise")
print("  [mJy/beam/sqrt(s)].  Uses no external noise information whatsoever.")

# Restrict to the well-covered interior so that map edges (low t, high cirrus
# gradient, partial beams) do not drive the fit.
t = EXPOSURE.copy()
t[~GOOD] = np.nan
t_lo, t_hi = np.nanpercentile(t, [2, 98])
sel = GOOD & (EXPOSURE > t_lo) & (EXPOSURE < t_hi)
print("\n  exposure range used: %.3f - %.3f s  (%d pixels)" % (t_lo, t_hi, sel.sum()))

NBIN = 14
edges = np.percentile(EXPOSURE[sel], np.linspace(0, 100, NBIN + 1))
edges = np.unique(edges)
bin_t, bin_var, bin_n, bin_var_err = [], [], [], []
for i in range(len(edges) - 1):
    b = sel & (EXPOSURE >= edges[i]) & (EXPOSURE < edges[i + 1])
    if b.sum() < 500:
        continue
    v = IMAGE[b]
    # A robust variance: MAD-based, so the bright confusion tail (which is a
    # *sky* signal but a wildly non-Gaussian one) does not dominate the fit.
    s = robust_sigma(v)
    bin_t.append(np.median(EXPOSURE[b]))
    bin_var.append(s ** 2)
    bin_n.append(int(b.sum()))
    # variance of a variance estimate ~ 2 sigma^4 / (n-1)
    bin_var_err.append(s ** 2 * np.sqrt(2.0 / (b.sum() - 1)))
bin_t = np.array(bin_t); bin_var = np.array(bin_var)
bin_n = np.array(bin_n); bin_var_err = np.array(bin_var_err)
invt = 1.0 / bin_t

print("\n   bin   <t>[s]    1/t     MAD-sigma   sigma^2      N_pix")
for i in range(len(bin_t)):
    print("   %3d  %7.3f  %7.3f   %8.3f  %9.2f  %9d"
          % (i, bin_t[i], invt[i], np.sqrt(bin_var[i]), bin_var[i], bin_n[i]))

W = 1.0 / bin_var_err ** 2
A = np.vstack([np.ones_like(invt), invt]).T
ATW = A.T * W
cov = np.linalg.inv(ATW @ A)
coef = cov @ (ATW @ bin_var)
intercept, slope = coef
d_intercept, d_slope = np.sqrt(np.diag(cov))

sigma_conf_r1 = np.sqrt(max(intercept, 0.0))
sigma_inst_r1 = np.sqrt(max(slope, 0.0))
d_conf_r1 = 0.5 * d_intercept / sigma_conf_r1 if sigma_conf_r1 > 0 else np.nan
d_inst_r1 = 0.5 * d_slope / sigma_inst_r1 if sigma_inst_r1 > 0 else np.nan
chi2 = np.sum(W * (bin_var - A @ coef) ** 2)
ndf = len(bin_t) - 2

print("\n  weighted linear fit:")
print("    intercept = %8.3f +- %.3f  ->  sigma_conf = %.3f +- %.3f mJy/beam"
      % (intercept, d_intercept, sigma_conf_r1, d_conf_r1))
print("    slope     = %8.3f +- %.3f  ->  sigma_inst = %.3f +- %.3f mJy/beam/sqrt(s)"
      % (slope, d_slope, sigma_inst_r1, d_inst_r1))
print("    chi2/ndf  = %.2f / %d = %.2f" % (chi2, ndf, chi2 / max(ndf, 1)))

# The per-bin statistical errors are tiny (each bin holds ~1e5-1e6 pixels), so a
# chi2/ndf well above 1 does not mean the model is wrong -- it means bin-to-bin
# SYSTEMATICS dominate. Exposure correlates with position on the sky, and so
# does the cirrus amplitude, so different t-bins sample slightly different sky.
# We therefore inflate the errors by sqrt(chi2/ndf), the standard remedy, and
# quote the inflated values as the honest uncertainty.
scale = np.sqrt(max(chi2 / max(ndf, 1), 1.0))
d_conf_r1 *= scale
d_inst_r1 *= scale
print("\n  chi2/ndf > 1 reflects bin-to-bin systematics (exposure correlates with")
print("  sky position, hence with cirrus amplitude), not a failure of the model:")
print("  the per-bin statistical errors are negligible at ~1e5-1e6 pixels/bin.")
print("  Errors below are inflated by sqrt(chi2/ndf) = %.2f accordingly." % scale)
print("    sigma_conf = %.3f +- %.3f mJy/beam" % (sigma_conf_r1, d_conf_r1))
print("    sigma_inst = %.3f +- %.3f mJy/beam/sqrt(s)" % (sigma_inst_r1, d_inst_r1))

print("\n  literature (Nguyen+2010 Table 1, %s um):" % BAND)
print("    sigma_conf = %.1f +- %.1f mJy/beam" % (NGUYEN_CONF[BAND], NGUYEN_CONF_ERR[BAND]))
print("    sigma_inst = %.1f +- %.1f mJy/beam/sqrt(s)" % (NGUYEN_INST[BAND], NGUYEN_INST_ERR[BAND]))
print("  HELP header NCONF = %.2f mJy/beam" % NCONF_HDR)
print("  HELP header NINS  = %.2f mJy/beam (already divided by sqrt(t); at the"
      % NINS_HDR)
print("      median exposure %.2f s this corresponds to %.2f mJy/beam/sqrt(s))"
      % (np.median(EXPOSURE[GOOD]), NINS_HDR * np.sqrt(np.median(EXPOSURE[GOOD]))))
print("\n  Caveat worth carrying forward: route 1 measures the confusion noise of")
print("  the IMAGE extension, which still contains cirrus. Its intercept is")
print("  therefore an UPPER limit on the extragalactic confusion noise, while")
print("  any mismatch with Nguyen+2010 also reflects their use of deep HerMES")
print("  fields against this shallow, wide H-ATLAS-style field.")

RESULTS.update(
    route1_bin_t=bin_t, route1_bin_var=bin_var, route1_bin_n=bin_n,
    route1_intercept=float(intercept), route1_slope=float(slope),
    route1_sigma_conf=float(sigma_conf_r1), route1_sigma_inst=float(sigma_inst_r1),
    route1_d_sigma_conf=float(d_conf_r1), route1_d_sigma_inst=float(d_inst_r1),
    route1_chi2=float(chi2), route1_ndf=int(ndf),
)

#%% ------------- cell 7: noise decomposition, route 2 (ERROR-map route) ------
print("\n" + "-" * 79)
print("cell 7: ERROR-map route and the noise budget of the FILTERED map")
print("-" * 79)

sig_inst_img = float(np.median(ERROR[GOOD]))
sig_tot_img = stats["IMAGE"]["mad_sigma"]
sig_conf_img = np.sqrt(max(sig_tot_img ** 2 - sig_inst_img ** 2, 0.0))

sig_inst_mf = float(np.median(MFILT_ERROR[GOOD]))
sig_tot_mf = stats["MFILT"]["mad_sigma"]
sig_conf_mf = np.sqrt(max(sig_tot_mf ** 2 - sig_inst_mf ** 2, 0.0))

print("\n  IMAGE (map space):")
print("    total  MAD-sigma      = %6.3f mJy/beam" % sig_tot_img)
print("    instrument (ERROR)    = %6.3f mJy/beam" % sig_inst_img)
print("    confusion (quadrature)= %6.3f mJy/beam" % sig_conf_img)
print("    sigma_inst/sigma_conf = %6.3f   (>1: detector-dominated)"
      % (sig_inst_img / sig_conf_img))
print("\n  MFILT (filtered space):")
print("    total  MAD-sigma      = %6.3f mJy/beam" % sig_tot_mf)
print("    instrument (MFILT_ERR)= %6.3f mJy/beam" % sig_inst_mf)
print("    confusion (quadrature)= %6.3f mJy/beam" % sig_conf_mf)
print("    sigma_inst/sigma_conf = %6.3f   (>1: detector-dominated)"
      % (sig_inst_mf / sig_conf_mf))
print("\n  instrument noise suppressed by the matched filter : %.2fx"
      % (sig_inst_img / sig_inst_mf))
print("  confusion retained                                : %.2fx"
      % (sig_conf_mf / sig_conf_img))
print("  => sky-to-detector VARIANCE ratio improves by      : %.2fx"
      % ((sig_conf_mf / sig_inst_mf) ** 2 / (sig_conf_img / sig_inst_img) ** 2))
print("""
  Read this honestly.  The matched filter improves the sky-to-detector variance
  ratio by roughly a factor 2, but it does NOT make the map sky-dominated: the
  filtered instrument and confusion terms end up comparable (ratio ~1).  It also
  suppresses the confusion term itself, because a filter matched to a single
  point source partially averages down the fluctuations from the many faint
  sources per beam.  So the case for MFILT as the primary map does not rest on
  this cell -- it rests on cells 8b and 11, where the raw IMAGE is shown to have
  white per-pixel noise that makes local-maximum declustering meaningless.  A
  residual Gaussian instrument component of ~5 mJy/beam survives in MFILT and
  must be carried explicitly into the xi(u) forward model as a noise floor (the
  Planck analysis's 'sigma_extra' device, manual sections 13.2 and 13.6).""")

RESULTS.update(
    sigma_inst_image=sig_inst_img, sigma_total_image=sig_tot_img,
    sigma_conf_image=float(sig_conf_img),
    sigma_inst_mfilt=sig_inst_mf, sigma_total_mfilt=sig_tot_mf,
    sigma_conf_mfilt=float(sig_conf_mf),
)

#%% -------------- cell 8: synthetic instrument-noise realisations ------------
print("\n" + "-" * 79)
print("cell 8: instrument-noise realisations (jackknife substitute)")
print("-" * 79)
print("  Nguyen et al. used a jackknife null map.  HELP ships no half-maps for")
print("  GAMA-09, so we draw n ~ N(0, ERROR_i) per pixel and push it through the")
print("  same matched filter.  Cell 9 checks this against MFILT_ERROR, which is")
print("  an independent, pipeline-derived prediction of the filtered noise.")

rng = np.random.default_rng(SEED)
noise_img = np.where(
    GOOD, rng.normal(0.0, np.where(GOOD, ERROR, 1.0)).astype(FDT), np.nan
).astype(FDT)
# Tiled, so that peak memory stays bounded regardless of map size.
noise_mf, _ = apply_matched_filter_tiled(
    np.where(GOOD, noise_img, 0.0).astype(FDT),
    np.where(GOOD, ERROR, 1.0).astype(FDT), K, GOOD, dtype=FDT)
noise_mf = np.where(GOOD, noise_mf, np.nan).astype(FDT)

print()
stats["NOISE_image"] = describe("NOISE(image)", noise_img, GOOD)
stats["NOISE_mfilt"] = describe("NOISE(mfilt)", noise_mf, GOOD)

sig_noise_mf = robust_sigma(noise_mf[GOOD])
print("\n  filtered noise realisation MAD-sigma = %.4f mJy/beam" % sig_noise_mf)
print("  pipeline MFILT_ERROR median          = %.4f mJy/beam" % sig_inst_mf)
print("  pooled ratio                         = %.4f" % (sig_noise_mf / sig_inst_mf))

# The pooled comparison above is only approximate: pooling pixels with different
# sigma gives a mixture whose MAD is not the median sigma.  The sharp test is
# pixelwise -- divide the realisation by its own predicted error and check that
# the result is a unit Gaussian.  Edge pixels, where the convolution reaches
# into the zero-padded region outside the footprint, are excluded by eroding the
# good-pixel mask by one kernel half-width.
# A direct binary_erosion with a 101x101 structuring element would be O(N*101^2)
# and is far too slow on an 18-megapixel mask.  A separable uniform filter on the
# mask gives the identical result in O(N): a pixel survives only if every pixel
# in its 101x101 neighbourhood is good, i.e. if the local mean of the mask is 1.
from scipy.ndimage import uniform_filter
_kw = K.shape[0]
_frac = uniform_filter(GOOD.astype(np.float32), size=_kw, mode="constant", cval=0.0)
INTERIOR = _frac > 0.9999
del _frac
print("\n  interior pixels (eroded by one kernel width): %d of %d good"
      % (INTERIOR.sum(), GOOD.sum()))
if INTERIOR.sum() > 1000:
    z_img = (noise_img / ERROR)[INTERIOR]
    z_mf = (noise_mf / MFILT_ERROR)[INTERIOR]
    for nm, z in [("noise/ERROR      ", z_img), ("noise/MFILT_ERROR", z_mf)]:
        z = z[np.isfinite(z)]
        print("    %s : mean %+.4f  MAD-sigma %.4f  rms %.4f"
              % (nm, z.mean(), robust_sigma(z), z.std()))
    pix_ratio = float(robust_sigma(z_mf[np.isfinite(z_mf)]))
    print("\n  => a MAD-sigma of 1.00 on the second line confirms both that the")
    print("     pipeline noise is white between pixels and that MFILT_ERROR is")
    print("     correctly normalised, so the synthetic realisation is a valid")
    print("     stand-in for Nguyen's jackknife map.")
else:
    pix_ratio = np.nan

RESULTS.update(sigma_noise_mfilt_realisation=float(sig_noise_mf),
               noise_realisation_ratio=float(sig_noise_mf / sig_inst_mf),
               noise_pixelwise_z_sigma=pix_ratio)

#%% ------------ cell 8b: autocorrelation -- white vs beam-correlated ---------
print("\n" + "-" * 79)
print("cell 8b: pixel-pixel autocorrelation of IMAGE, MFILT and the noise")
print("-" * 79)
print("  This is the direct evidence for the claim that drives the whole choice")
print("  of primary map.  A beam-correlated field has an ACF whose width is the")
print("  beam; white per-pixel noise has an ACF that is a delta function at zero")
print("  lag.  If IMAGE shows a sharp zero-lag spike on top of a beam-width")
print("  pedestal, its local maxima are noise spikes and POT on it is unsafe.")


def acf_1d(a, valid, maxlag=12, nsamp=600):
    """Mean row-wise autocorrelation as a function of pixel lag.

    Computed on ``nsamp`` randomly chosen fully-valid row segments to keep the
    cost independent of map size.  Returns lags and the normalised ACF.
    """
    rng_l = np.random.default_rng(7)
    ny_, nx_ = a.shape
    seglen = 4 * maxlag + 1
    out = np.zeros(maxlag + 1)
    cnt = 0
    tries = 0
    while cnt < nsamp and tries < nsamp * 60:
        tries += 1
        j = rng_l.integers(0, ny_)
        i = rng_l.integers(0, max(nx_ - seglen, 1))
        seg = a[j, i:i + seglen]
        vseg = valid[j, i:i + seglen]
        if not vseg.all() or not np.isfinite(seg).all():
            continue
        s = seg - seg.mean()
        denom = np.dot(s, s)
        if denom <= 0:
            continue
        for L in range(maxlag + 1):
            out[L] += np.dot(s[:seglen - L], s[L:]) / denom
        cnt += 1
    return np.arange(maxlag + 1), (out / cnt if cnt else out), cnt


lags, acf_img, n1 = acf_1d(IMAGE, GOOD)
_, acf_mf, n2 = acf_1d(MFILT, GOOD)
_, acf_noi, n3 = acf_1d(noise_img, GOOD)
_, acf_noimf, n4 = acf_1d(noise_mf, GOOD)

print("\n  lag[pix]  lag[\"]     IMAGE     MFILT   noise(img)  noise(mf)")
for L in range(0, 9):
    print("  %6d   %6.1f   %8.4f  %8.4f   %8.4f   %8.4f"
          % (L, L * PIXSIZE, acf_img[L], acf_mf[L], acf_noi[L], acf_noimf[L]))
print("\n  one FWHM = %.2f pixels. A beam-correlated field should still have" % PIX_PER_FWHM)
print("  ACF ~ 0.5 at that lag; white noise drops to ~0 by lag 1.")
print("  IMAGE  ACF at lag 1 = %.3f" % acf_img[1])
print("  MFILT  ACF at lag 1 = %.3f" % acf_mf[1])

RESULTS.update(acf_lags=lags, acf_image=acf_img, acf_mfilt=acf_mf,
               acf_noise_image=acf_noi, acf_noise_mfilt=acf_noimf)

fig5, ax5 = plt.subplots(figsize=(6.4, 4.6))
ax5.plot(lags * PIXSIZE, acf_img, "o-", color="k", label="IMAGE")
ax5.plot(lags * PIXSIZE, acf_mf, "s-", color="crimson", label="MFILT")
ax5.plot(lags * PIXSIZE, acf_noi, "^--", color="0.5", label="noise (map space)")
ax5.plot(lags * PIXSIZE, acf_noimf, "v--", color="tab:blue",
         label="noise (filtered)")
ax5.axvline(FWHM_ARCSEC, color="tab:green", ls="-.", lw=1.2,
            label="beam FWHM = %.1f\"" % FWHM_ARCSEC)
ax5.axhline(0.0, color="0.7", lw=0.8)
ax5.set_xlabel("lag [arcsec]"); ax5.set_ylabel("normalised autocorrelation")
ax5.set_title("Spatial correlation: sky is beam-wide, detector noise is white",
              fontsize=11)
ax5.legend(fontsize=8); ax5.grid(alpha=0.25)
fig5.tight_layout()
if SAVE_FIGURES:
    p = os.path.join(FIGURE_DIR, "herschel_noise_autocorrelation.png")
    fig5.savefig(p, dpi=150, bbox_inches="tight")
    print("  wrote", p)

#%% ------------------- cell 9: the Nguyen Fig. 3 analogue --------------------
print("\n" + "-" * 79)
print("cell 9: P(D) histograms  --  the Nguyen et al. (2010) Fig. 3 analogue")
print("-" * 79)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

panels = [
    ("IMAGE (map space)", IMAGE, noise_img, sig_inst_img, (-40, 80)),
    ("MFILT (matched-filtered)", MFILT, noise_mf, sig_inst_mf, (-30, 60)),
]

for ax, (title, mp, npz_, sig_i, xr) in zip(axes, panels):
    bins = np.linspace(xr[0], xr[1], 141)
    ctr = 0.5 * (bins[1:] + bins[:-1])

    h_map, _ = np.histogram(mp[GOOD], bins=bins)
    h_noi, _ = np.histogram(npz_[GOOD], bins=bins)

    ax.step(ctr, h_map, where="mid", color="k", lw=1.4, label="map")
    ax.step(ctr, h_noi, where="mid", color="0.35", lw=1.1, ls=":",
            label="instrument noise")

    # Gaussian fit to the noise histogram (Nguyen's red curve).
    v = npz_[GOOD]
    v = v[np.isfinite(v)]
    mu_n, sd_n = float(np.mean(v)), float(np.std(v))
    ax.plot(ctr, v.size * np.diff(bins)[0] * norm.pdf(ctr, mu_n, sd_n),
            color="crimson", lw=1.6,
            label=r"Gaussian fit, $\sigma=%.2f$" % sd_n)

    # A Gaussian fitted to the NEGATIVE side of the map distribution only.
    # The faint-source tail is one-sided (sources add flux), so the side of the
    # P(D) below the mode is the closest thing the map itself offers to a
    # "source-free" reference.  Fitting there and extrapolating to positive flux
    # isolates the confusion excess cleanly; a fit to the whole distribution
    # would be dragged wide by the very tail we are trying to display.
    vm = mp[GOOD]; vm = vm[np.isfinite(vm)]
    mode_m = float(np.median(vm))
    lower = vm[vm < mode_m]
    sd_lo = float(np.sqrt(np.mean((lower - mode_m) ** 2)))   # half-Gaussian width
    ax.plot(ctr, vm.size * np.diff(bins)[0] * norm.pdf(ctr, mode_m, sd_lo),
            color="tab:blue", lw=1.3, ls="--",
            label=r"neg.-side Gaussian, $\sigma=%.2f$" % sd_lo)
    print("    %-26s mode %+6.2f  neg-side sigma %5.2f  MAD-sigma %5.2f"
          % (title, mode_m, sd_lo, robust_sigma(vm)))

    ax.set_yscale("log")
    ax.set_xlim(*xr)
    ax.set_ylim(1, max(h_map.max(), h_noi.max()) * 3)
    ax.set_xlabel("Pixel intensity [mJy/beam]")
    ax.set_ylabel(r"$N_{\rm pix}$")
    ax.set_title("GAMA-09 350 $\\mu$m: " + title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

fig.suptitle(
    "P(D) of the GAMA-09 SPIRE 350 $\\mu$m map "
    "(cf. Nguyen et al. 2010, Fig. 3)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
if SAVE_FIGURES:
    p = os.path.join(FIGURE_DIR, "herschel_noise_pd_histogram.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("  wrote", p)

#%% ------------- cell 10: variance-vs-1/t figure (Nguyen Fig. 2) -------------
fig2, ax = plt.subplots(figsize=(6.4, 4.8))
ax.errorbar(invt, bin_var, yerr=bin_var_err, fmt="o", color="k", ms=5,
            capsize=3, label="GAMA-09 350 $\\mu$m, MAD variance")
xs_ = np.linspace(0, invt.max() * 1.08, 100)
ax.plot(xs_, intercept + slope * xs_, "-", color="crimson", lw=1.6,
        label=r"fit: $\sigma_{\rm conf}^2 + \sigma_{\rm inst}^2/t$")
ax.axhline(intercept, color="tab:blue", ls="--", lw=1.2,
           label=r"$\sigma_{\rm conf}=%.2f$ mJy/beam" % sigma_conf_r1)
ax.axhline(NGUYEN_CONF[BAND] ** 2, color="tab:green", ls=":", lw=1.4,
           label=r"Nguyen+2010: $\sigma_{\rm conf}=%.1f$" % NGUYEN_CONF[BAND])
ax.axhline(NCONF_HDR ** 2, color="tab:orange", ls="-.", lw=1.2,
           label=r"HELP NCONF $=%.2f$" % NCONF_HDR)
ax.set_xlabel(r"$1/t$  [s$^{-1}$]")
ax.set_ylabel(r"$\sigma_{\rm total}^2$  [(mJy/beam)$^2$]")
ax.set_xlim(0, invt.max() * 1.08)
ax.set_title("Noise decomposition (cf. Nguyen et al. 2010, Fig. 2)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig2.tight_layout()
if SAVE_FIGURES:
    p = os.path.join(FIGURE_DIR, "herschel_noise_variance_vs_invt.png")
    fig2.savefig(p, dpi=150, bbox_inches="tight")
    print("  wrote", p)

#%% ------------- cell 11: POT suitability -- declustered peak density --------
print("\n" + "-" * 79)
print("cell 11: declustered peak density  --  is the map usable for POT?")
print("-" * 79)
print("  A w x w local-maximum filter can return at most 1 peak per w^2 pixels,")
print("  i.e. a saturation density of BEAM_PIX/w^2 peaks per beam.  A map whose")
print("  measured density approaches that ceiling is telling us its local maxima")
print("  are being created by white per-pixel noise, not by beam-scale sources,")
print("  and POT applied to it would characterise the detectors.")

windows = [3, 5, 7, 9]
peak_tab = {}
for label, mp in [("IMAGE", IMAGE), ("MFILT", MFILT)]:
    row = []
    filled = np.where(GOOD, mp, -np.inf)
    for w in windows:
        mx = maximum_filter(filled, size=w, mode="constant", cval=-np.inf)
        pk = GOOD & (filled == mx)
        dens = pk.sum() / (GOOD.sum() / BEAM_PIX)
        sat = BEAM_PIX / w ** 2
        row.append((w, int(pk.sum()), dens, sat, dens / sat))
    peak_tab[label] = row

print("\n  map      w   N_peaks     peaks/beam   saturation   ratio")
for label, row in peak_tab.items():
    for (w, n, dens, sat, ratio) in row:
        flag = "  <-- SATURATED" if ratio > 0.7 else ""
        print("  %-7s %2d %9d %12.4f %12.4f %7.3f%s"
              % (label, w, n, dens, sat, ratio, flag))
print("\n  (Planck 857 GHz, for reference, measured 0.23-0.24 peaks per beam with")
print("   a one-FWHM window -- comfortably below saturation.)")
print("\n  NOTE on reading this table: the ratio is only diagnostic for windows")
print("  at or below one FWHM (%.1f pix here). For larger w every map is pushed"
      % PIX_PER_FWHM)
print("  toward the ceiling by construction, because the filter is then forcing")
print("  roughly one maximum per window regardless of the field's correlation")
print("  structure. The decisive comparison is therefore the w = 3 row, where")
print("  IMAGE sits at %.2f of saturation and MFILT at %.2f."
      % (peak_tab["IMAGE"][0][4], peak_tab["MFILT"][0][4]))

fig3, ax = plt.subplots(figsize=(6.2, 4.6))
for label, row in peak_tab.items():
    w_ = [r[0] for r in row]
    ratio_ = [r[4] for r in row]
    ax.plot(w_, ratio_, "o-", label=label)
ax.axhline(1.0, color="k", ls="--", lw=1, label="saturation ceiling")
ax.axhline(0.7, color="crimson", ls=":", lw=1, label="0.7 (unsafe above)")
ax.axvline(PIX_PER_FWHM, color="tab:green", ls="-.", lw=1.2,
           label="one FWHM = %.2f pix" % PIX_PER_FWHM)
ax.set_xlabel("declustering window [pixels]")
ax.set_ylabel("measured density / saturation density")
ax.set_title("Local-maximum saturation: IMAGE vs MFILT", fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig3.tight_layout()
if SAVE_FIGURES:
    p = os.path.join(FIGURE_DIR, "herschel_noise_peak_density.png")
    fig3.savefig(p, dpi=150, bbox_inches="tight")
    print("  wrote", p)

RESULTS["peak_density"] = {k: np.array(v) for k, v in peak_tab.items()}

#%% -------------------- cell 12: PSF-response figure -------------------------
fig4, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
r_ax = (np.arange(npix_sim) - c0) * PIXSIZE
axs[0].plot(r_ax, src_map[c0, :] / src_map.max(), "k-", lw=1.6,
            label="map-space PSF (FWHM %.1f\")" % FWHM_ARCSEC)
axs[0].plot(r_ax, filt_src[c0, :] / np.nanmax(filt_src), "-", color="crimson",
            lw=1.6, label="matched-filtered (FWHM %.1f\")" % fwhm_eff)
axs[0].axhline(0.5, color="0.6", ls=":", lw=1)
axs[0].axhline(0.0, color="0.6", ls="-", lw=0.6)
axs[0].set_xlim(-90, 90)
axs[0].set_xlabel("offset [arcsec]"); axs[0].set_ylabel("normalised response")
axs[0].set_title("Point-source response", fontsize=11)
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)

kc = K.shape[0] // 2
k_ax = (np.arange(K.shape[0]) - kc) * PIXSIZE
axs[1].plot(k_ax, K[kc, :], "k-", lw=1.4)
axs[1].axhline(0.0, color="crimson", ls="--", lw=1)
axs[1].set_xlim(-120, 120)
axs[1].set_xlabel("offset [arcsec]"); axs[1].set_ylabel("kernel amplitude")
axs[1].set_title("Matched-filter kernel (central cut); note negative lobes",
                 fontsize=11)
axs[1].grid(alpha=0.25)
fig4.tight_layout()
if SAVE_FIGURES:
    p = os.path.join(FIGURE_DIR, "herschel_noise_psf_response.png")
    fig4.savefig(p, dpi=150, bbox_inches="tight")
    print("  wrote", p)

#%% -------------------------- cell 13: summary ------------------------------
print("\n" + "=" * 79)
print("SUMMARY  --  GAMA-09 SPIRE 350 um noise budget")
print("=" * 79)
print("""
  quantity                                value        reference / comparison
  -------------------------------------------------------------------------""")
print("  sky area                          %8.2f deg^2" % area_deg2)
print("  beams on sky                      %8.3e" % n_beams)
print("  FWHM                              %8.2f \"      HELP header" % FWHM_ARCSEC)
print("  pixel size                        %8.2f \"      %.2f pix/FWHM"
      % (PIXSIZE, PIX_PER_FWHM))
print("  -------------------------------------------------------------------------")
print("  sigma_conf  (route 1, var vs 1/t) %8.3f mJy/beam  Nguyen+2010: %.1f +- %.1f"
      % (sigma_conf_r1, NGUYEN_CONF[BAND], NGUYEN_CONF_ERR[BAND]))
print("  sigma_conf  (route 2, quadrature) %8.3f mJy/beam  HELP NCONF: %.2f"
      % (sig_conf_img, NCONF_HDR))
print("  sigma_inst  (route 1)             %8.3f mJy/beam/sqrt(s)  Nguyen+2010: %.1f +- %.1f"
      % (sigma_inst_r1, NGUYEN_INST[BAND], NGUYEN_INST_ERR[BAND]))
print("  sigma_inst  (ERROR map)           %8.3f mJy/beam  HELP NINS: %.2f"
      % (sig_inst_img, NINS_HDR))
print("  -------------------------------------------------------------------------")
print("  IMAGE  total MAD-sigma            %8.3f mJy/beam  skew %.2f"
      % (sig_tot_img, stats["IMAGE"]["skew"]))
print("  MFILT  total MAD-sigma            %8.3f mJy/beam  skew %.2f"
      % (sig_tot_mf, stats["MFILT"]["skew"]))
print("  MFILT  instrument                 %8.3f mJy/beam" % sig_inst_mf)
print("  MFILT  confusion                  %8.3f mJy/beam" % sig_conf_mf)
print("  sky/detector variance gain        %8.2f x" %
      ((sig_conf_mf / sig_inst_mf) ** 2 / (sig_conf_img / sig_inst_img) ** 2))
print("  -------------------------------------------------------------------------")
print("  matched filter reproduced         %8s        corr = %.8f"
      % ("YES" if MF_VERIFIED else "NO", corr))
print("  point-source flux response        %8.4f" % (peak_out / S_IN))
print("  effective FWHM after filtering    %8.2f \"" % fwhm_eff)
print("=" * 79)

npz_path = os.path.join(FIGURE_DIR, "herschel_noise_analysis_%s.npz" % BAND)
np.savez_compressed(
    npz_path,
    **{k: v for k, v in RESULTS.items() if not isinstance(v, dict)},
    stats_json=json.dumps(stats),
    peak_density_json=json.dumps({k: np.array(v).tolist()
                                  for k, v in peak_tab.items()}),
)
print("\nwrote", npz_path)

if SAVE_FIGURES:
    plt.show()
