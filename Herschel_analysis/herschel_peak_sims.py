#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
herschel_peak_sims.py -- Module H1c: is the rising xi_hat(u) a peak-statistics
                         artefact, or the bright lensed source population?
===============================================================================

THE PROBLEM THIS MODULE EXISTS TO SOLVE
----------------------------------------
H1v2 (`herschel_unlensed_v2.py`) measured xi_hat(u) rising monotonically from
0.14 at u = 25 mJy/beam to ~0.48 at 75 mJy, at 8.1 sigma from a flat line, while
all three count models PREDICT xi falling through zero.  The disagreement is in
the SIGN of dxi/du, so it cannot be absorbed by any renormalisation.  H1v2 ruled
out three candidate causes (map/filter choice, beam profile, counts
normalisation).  Two survive, and this module separates them.

  CAUSE A -- "PIXEL vs PEAK".
  ---------------------------
  The model curve is built like this:
      analytic one-point P(D)  =  distribution of the map value at a RANDOM
      PIXEL  ->  its exceedance density above u  ->  KL projection onto the GPD
      family  ->  xi_model(u).
  The measurement is built like this:
      find LOCAL MAXIMA of the map (declustered over one FWHM)  ->  their
      exceedances above u  ->  GPD maximum likelihood  ->  xi_hat(u).

  These are different random variables.  Local maxima are not a random sample of
  pixels: they are a size-biased sample, and the bias depends on u.  For a
  smooth Gaussian random field in 2-D the density of maxima above a level u goes
  as ~ u exp(-u^2/2 sigma^2) (Kac-Rice / Bardeen et al. 1986), whereas the pixel
  exceedance probability goes as ~ (sigma/u) exp(-u^2/2 sigma^2).  Their ratio
  is ~ u^2: peaks are progressively over-represented relative to pixels as the
  threshold rises, so the peak-value distribution has a different tail SHAPE,
  not merely a different normalisation or a constant offset.  H1 and H1v2 both
  handle this by shifting the model grid by the measured core centroid mu_core,
  which absorbs the leading constant term and nothing else.

  This project has already measured the effect once, in a different context:
  `GPD_beam_and_declustering_v2.md` Sec. 4 reports that the declustered-peak
  marginal sits well above the pixel P(D), and that with white (rather than
  beam-correlated) noise the declustering retains ~4x more "peaks", most of them
  noise spikes.  What has never been done is to propagate it through to xi(u)
  for THIS beam, THIS noise level and THIS declustering window.  There is no
  useful closed form for a non-Gaussian, source-dominated field, so the only
  reliable route is forward simulation -- hence this module.

  CAUSE B -- THE BRIGHT, STRONGLY-LENSED SOURCE POPULATION.
  ---------------------------------------------------------
  The count models are truncated at S_cut = 100 mJy, because above ~100 mJy the
  350 um counts are dominated by strongly lensed sources (Lima, Jain, Devlin &
  Aguirre 2010, ApJL 717, L31) and no longer trace the intrinsic dN/dS; the
  counts memo excluded Bethermin's 133.7 mJy point for exactly this reason.  But
  the MAP is not truncated.  A diagnostic run on the H1v2 peak sample found:

      peaks above 100 mJy :  70   (0.043% of 161,810 peaks, 25.75 deg^2)
      peaks above 150 mJy :  18
      peaks above 200 mJy :  12   brightest peak 368 mJy/beam

  against the adopted Schechter's prediction for the same area:

      N(>100) = 30.9    observed 70   ->   2.3x
      N(>150) =  1.12   observed 18   ->  16x
      N(>200) =  0.05   observed 12   -> 246x

  So there is a real, large, and completely unmodelled bright population.  Its
  own slope is shallow: fitting N(>S) ~ S^-a to the observed points between 100
  and 200 mJy gives a ~ 2.54, i.e. dN/dS ~ S^-3.54, eta ~ 3.54, and hence an
  asymptotic xi = 1/(eta-1) ~ 0.39 -- remarkably close to the plateau the
  measured xi_hat(u) reaches at high threshold (0.45-0.48).

  If Cause B dominates, then the rising xi_hat(u) IS NOT A SYSTEMATIC.  It is
  the GPD detecting the transition from the intrinsic counts to the
  lensing-dominated bright tail, which is a result worth reporting rather than a
  bug worth fixing.

  A NOTE ON A THIRD, WEAKER SUSPECT: clustering.  The analytic P(D) is Poisson
  and so is `CIBMapSimulator`, so this module cannot test clustering.  It is
  left for H1d if the two causes above fail to account for the rise.

WHY A NAIVE "CAP THE PEAKS AT 100 mJy" TEST IS NOT ENOUGH
----------------------------------------------------------
The obvious quick test -- discard measured peaks above 100 mJy so the data are
truncated like the model -- gives dxi/du = -0.0105 (versus +0.0081 uncapped),
which looks like a clean confirmation of Cause B.  It is NOT clean, for two
reasons, and this module exists partly to do the test properly:

  1. Truncating the PEAK sample is right-censoring, which biases the GPD MLE
     DOWNWARD by construction: at u = 75 mJy a cap at 100 leaves only
     exceedances in a 25 mJy-wide window, and the fitted xi is then driven by
     the width of the window rather than by the tail.  Some of the -0.0105 is
     this artefact.
  2. Truncating the SOURCE counts at 100 mJy (what the model does) is not the
     same operation as truncating the PEAK values at 100 mJy (what the test
     does): a 90 mJy source sitting on a positive confusion fluctuation makes a
     >100 mJy peak, and a 110 mJy source in a hole makes a <100 mJy peak.

Simulation avoids both problems, because we can truncate the SOURCE counts --
exactly as the model does -- and then measure xi from the resulting peaks with
no censoring at all.

THE EXPERIMENT
--------------
A 2 x 2 factorial, plus a validation leg.  Every leg goes through the IDENTICAL
baselining, declustering and edge exclusion used on the real data in H1v2, and
xi is estimated by the same GPD MLE on the same threshold grid.

  leg                     counts injected            xi measured from
  ---------------------   ------------------------   ------------------
  A_pixel_truncated       Schechter, S < 100 mJy     simulated PIXELS
  B_peak_truncated        Schechter, S < 100 mJy     simulated PEAKS
  C_peak_brighttail       Schechter + bright tail    simulated PEAKS
  D_pixel_brighttail      Schechter + bright tail    simulated PIXELS

Then:
  * (A vs the ANALYTIC model curve)  validates the simulator: they must agree,
    since both are pixel-level with the same counts.  If they do not, stop --
    something is wrong with the simulation, not with the data.
  * (B - A)  is CAUSE A, the pure pixel-to-peak effect, isolated.
  * (C - B)  is CAUSE B, the pure bright-population effect, isolated.
  * (C vs the DATA) is the end-to-end test: if leg C reproduces the measured
    rise, the rise is explained.

SIMULATION CHOICES, AND WHY
---------------------------
  * We simulate the map ALREADY AT THE MATCHED-FILTERED STAGE: a Gaussian beam
    of 25.15" with sigma_noise = the measured filtered noise.  This is justified
    because H1v2 cell 2 verified that the filtered profile's Condon-equivalent
    FWHM is 25.15" to 4 significant figures (int P^2 = 358.24 vs 358.35
    arcsec^2), and cell 2b verified that using the exact signed profile instead
    of a Gaussian moves xi by at most 0.0165, below the bootstrap errors.
    Simulating the unfiltered map and then filtering it would add a second
    difference between the legs and is left as an optional cross-check
    (CIB_SIM_FILTER=1).
  * noise_mode="beam", NOT "white".  The real map is matched filtered, so its
    noise is beam-correlated (H0 measured lag-1 pixel ACF 0.719 in the filtered
    map versus 0.245 unfiltered).  `simulate_maps` documents that "white" mode
    makes the declustering retain ~4x more peaks, nearly all noise spikes --
    which would manufacture exactly the kind of peak-statistics artefact we are
    trying to test for.  Getting this switch wrong would invalidate the module.
  * Maps are the SAME SIZE as the real cutouts (225 pix = 30'), and many of
    them, rather than a few large maps.  This reproduces the real baseline
    degrees of freedom and the real edge exclusion exactly.
  * The simulated peak density per beam is compared against the measured 0.374
    as an independent validation that the field statistics match.

THE BRIGHT-TAIL MODEL
---------------------
Calibrated directly to the observed bright peak counts, not taken from the
literature, since the point is to ask whether THIS population accounts for the
rise.  A power law N(>S) = N_100 (S/100)^-a is fitted to the observed
N(>100), N(>150), N(>200) in the cutout area, converted to a differential
dN/dS = a N_100 (S/100)^-(a+1) / 100 mJy^-1 deg^-2, and injected ABOVE
S_BRIGHT_LO = 100 mJy on top of the truncated Schechter.  Caveat recorded: peak
values are not source fluxes (confusion boosts them), so the calibration is
approximate; at S > 200 mJy, with sigma_core ~ 8 mJy, the correction is small.

OUTPUTS
-------
results/herschel_peak_sims_350.npz
results/herschel_h1c_legs.png            the four legs and the data
results/herschel_h1c_decomposition.png   (B-A) and (C-B) side by side
results/herschel_h1c_validation.png      simulator vs analytic, peak density

HOW TO RUN
----------
    cd Herschel_analysis
    python3 herschel_peak_sims.py                 # full run, ~2-4 min

    CIB_QUICK_TEST=1 python3 herschel_peak_sims.py   # ~20 s smoke run

Requires `results/herschel_unlensed_v2_350.npz` (run herschel_unlensed_v2.py
first) for the measured xi_hat(u) it compares against.

ENVIRONMENT SWITCHES
--------------------
CIB_N_SIM_MAPS   simulated maps per leg (default 500; data used 103)
CIB_SIM_SEED     base RNG seed (default 20260729)
CIB_SCUT_MJY     source-counts truncation, mJy (default 100)
CIB_SMIN_MJY     faint integration floor, mJy (default 1.0)
CIB_SIM_FILTER   "1" to simulate unfiltered + apply the matched filter
CIB_FIGURE_DIR, CIB_SAVE_FIGURES, CIB_QUICK_TEST  as elsewhere

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

from scipy.ndimage import maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis_modules"))

from counts import BaseCounts, Schechter                      # noqa: E402
from pofd_analytic import PofD                                # noqa: E402
from simulate_maps import CIBMapSimulator                     # noqa: E402
from gpd_tail import fit_gpd, xi_of_threshold_analytic        # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ------------------------------------------------------------------ config --
FIGURE_DIR = os.environ.get("CIB_FIGURE_DIR", os.path.join(HERE, "results"))
SAVE_FIGURES = os.environ.get("CIB_SAVE_FIGURES", "1") == "1"
QUICK = os.environ.get("CIB_QUICK_TEST", "0") == "1"
SEED = int(os.environ.get("CIB_SIM_SEED", "20260729"))
N_SIM = int(os.environ.get("CIB_N_SIM_MAPS", "40" if QUICK else "500"))
S_CUT_MJY = float(os.environ.get("CIB_SCUT_MJY", "100.0"))
S_MIN_MJY = float(os.environ.get("CIB_SMIN_MJY", "1.0"))
V2_NPZ = os.path.join(FIGURE_DIR, "herschel_unlensed_v2_350.npz")

os.makedirs(FIGURE_DIR, exist_ok=True)
MAD2SIG = 1.4826
SCH_PARS = dict(alpha=-1.890, sstar=19.01, nstar=7014.0)
S_BRIGHT_LO = 100.0
T0 = time.time()
RES = {}


def robust_sigma(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return MAD2SIG * np.median(np.abs(x - np.median(x))) if x.size else np.nan


class TruncatedAbove(BaseCounts):
    """Wrap a counts model, zeroing dN/dS above s_cut.

    This is the SOURCE-plane truncation the analytic model applies, and is the
    operation the naive 'cap the measured peaks' test fails to reproduce.
    """

    def __init__(self, base, s_cut):
        self.base, self.s_cut = base, float(s_cut)
        self.s_min = base.s_min
        self.s_max = min(getattr(base, "s_max", 1e4), float(s_cut))

    def dnds(self, S):
        S = np.asarray(S, float)
        return np.where(S <= self.s_cut, self.base.dnds(S), 0.0)


class SchechterPlusBrightTail(BaseCounts):
    """Truncated Schechter below S_b, plus a power-law bright tail above it.

        dN/dS = Schechter(S)                                    S <= S_b
              = a * N_b * (S/S_b)^-(a+1) / S_b                  S >  S_b

    so that N(>S) = N_b (S/S_b)^-a for S > S_b.  `N_b` and `a` are calibrated to
    the observed bright peak counts (see module docstring); they are NOT taken
    from a published bright-end fit, because the question being asked is whether
    the observed bright population accounts for the observed xi(u) rise.
    """

    def __init__(self, base, s_b, n_b, a, s_min=1.0, s_max=1e4):
        self.base, self.s_b, self.n_b, self.a = base, float(s_b), float(n_b), float(a)
        self.s_min, self.s_max = float(s_min), float(s_max)

    def dnds(self, S):
        S = np.asarray(S, float)
        faint = np.where(S <= self.s_b, self.base.dnds(S), 0.0)
        bright = np.where(S > self.s_b,
                          self.a * self.n_b * (S / self.s_b) ** (-(self.a + 1.0))
                          / self.s_b, 0.0)
        return faint + bright


def poly_baseline(cut, order, valid):
    """Identical to herschel_unlensed_v2.poly_baseline -- kept local so the two
    modules cannot silently drift apart."""
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


def peaks_of(cut, decl, edge, order=1):
    """Baseline-subtract, decluster over `decl` pixels, drop the edge.

    Byte-for-byte the operation applied to the real cutouts in H1v2 cell 1d.
    """
    side = cut.shape[0]
    valid = np.isfinite(cut)
    res = poly_baseline(cut, order, valid)
    interior = np.zeros((side, side), bool)
    interior[edge:side - edge, edge:side - edge] = True
    filled = np.where(valid, res, -np.inf)
    pk = valid & interior & (filled == maximum_filter(filled, size=decl,
                                                      mode="constant",
                                                      cval=-np.inf))
    return res[pk], res[valid & interior]


def xi_curve(samples_by_map, u_grid, min_exc=50):
    """GPD MLE xi at each threshold, and the exceedance count."""
    y_all = np.concatenate(samples_by_map)
    xi = np.full(len(u_grid), np.nan)
    nex = np.zeros(len(u_grid), int)
    for i, u in enumerate(u_grid):
        y = y_all[y_all > u] - u
        nex[i] = y.size
        if y.size >= min_exc:
            xi[i] = fit_gpd(y)["xi"]
    return xi, nex


#%% -------------- cell 1: recover the measured result and the geometry -------
print("=" * 79)
print("H1c  Peak-level simulations: pixel-vs-peak, or the bright lensed tail?")
print("=" * 79)
if not os.path.exists(V2_NPZ):
    raise SystemExit("ERROR: %s not found.  It ships with the release (it is the"
                     " output of the development version v2 of"
                     " herschel_unlensed_v3.py); see Herschel_analysis/README.md."
                     % V2_NPZ)
v2 = np.load(V2_NPZ, allow_pickle=True)
U_GRID = v2["u_grid"]
XI_DATA, XIERR_DATA = v2["xi_meas"], v2["xi_err"]
FWHM, PIX = float(v2["fwhm"]), float(v2["pix"])
SIGMA_INST = float(v2["sigma_inst"])
SIDE = int(v2["side_pix"])
MU_CORE, SIG_CORE = float(v2["mu_core"]), float(v2["sig_core"])
NC_DATA, NPK_DATA = int(v2["n_cutouts"]), int(v2["n_peaks"])
EDGE = int(np.ceil(FWHM / PIX))
DECL = int(round(FWHM / PIX)) | 1
BEAM_PIX = 1.133 * (FWHM / PIX) ** 2
AREA_DATA = NC_DATA * (SIDE * PIX / 3600.0) ** 2

print("  recovered from H1v2:")
print("    beam %.2f\" | pixel %.1f\" | filtered noise %.3f mJy/beam"
      % (FWHM, PIX, SIGMA_INST))
print("    cutouts %d of %d pix (%.2f deg^2) | %d peaks | core mu %+.3f sigma %.3f"
      % (NC_DATA, SIDE, AREA_DATA, NPK_DATA, MU_CORE, SIG_CORE))
print("    measured peak density %.4f per beam"
      % (NPK_DATA / (NC_DATA * (SIDE - 2 * EDGE) ** 2 / BEAM_PIX)))
print("  simulating %d maps of %d pix per leg (%.2f deg^2, %.1fx the data)"
      % (N_SIM, SIDE, N_SIM * (SIDE * PIX / 3600.) ** 2, N_SIM / NC_DATA))

#%% ------------- cell 2: the two counts models to inject ---------------------
print("\n" + "-" * 79)
print("cell 2: counts models -- truncated Schechter, and Schechter + bright tail")
print("-" * 79)
sch_full = Schechter(alpha=SCH_PARS["alpha"],
                     log_sstar=np.log10(SCH_PARS["sstar"]),
                     log_phistar=np.log10(SCH_PARS["nstar"]), s_min=S_MIN_MJY)
cnts_trunc = TruncatedAbove(sch_full, S_CUT_MJY)

# --- calibrate the bright tail to the observed bright peak counts ------------
OBS_BRIGHT = {100.0: 70, 150.0: 18, 200.0: 12}      # counts in AREA_DATA deg^2
sb = np.array(sorted(OBS_BRIGHT))
nb = np.array([OBS_BRIGHT[s] for s in sb], float) / AREA_DATA
A_FIT, LNN_FIT = np.polyfit(np.log(sb / S_BRIGHT_LO), np.log(nb), 1)
A_FIT = -A_FIT
N_B = float(np.exp(LNN_FIT))
print("  observed bright PEAK counts in %.2f deg^2 (from the H1v2 sample):"
      % AREA_DATA)
for s in sb:
    print("     N(>%3.0f mJy) = %3d  =  %.3f deg^-2" % (s, OBS_BRIGHT[s],
                                                        OBS_BRIGHT[s] / AREA_DATA))
print("  power-law fit  N(>S) = %.3f (S/100)^-%.3f deg^-2" % (N_B, A_FIT))
print("  -> dN/dS ~ S^-%.3f, i.e. eta = %.3f, asymptotic xi = 1/(eta-1) = %.3f"
      % (A_FIT + 1, A_FIT + 1, 1.0 / A_FIT))
print("  (compare the measured xi_hat plateau at high threshold, ~0.45-0.48)")
cnts_bright = SchechterPlusBrightTail(sch_full, S_BRIGHT_LO, N_B, A_FIT,
                                      s_min=S_MIN_MJY, s_max=1e4)

print("\n  Schechter prediction vs observation for the same area:")
for s in sb:
    S = np.geomspace(s, 1e5, 4000)
    pred = _trapz(sch_full.dnds(S), S) * AREA_DATA
    print("     N(>%3.0f): predicted %7.2f   observed %3d   ratio %6.1fx"
          % (s, pred, OBS_BRIGHT[s], OBS_BRIGHT[s] / max(pred, 1e-9)))
RES.update(bright_a=A_FIT, bright_nb=N_B, area_data=AREA_DATA,
           obs_bright=json.dumps({str(k): v for k, v in OBS_BRIGHT.items()}))

#%% -------------------- cell 3: run the four simulation legs -----------------
print("\n" + "-" * 79)
print("cell 3: simulating")
print("-" * 79)
LEGS = {
    "A_pixel_truncated": dict(counts=cnts_trunc, level="pixel",
                              label="Schechter S<%.0f, PIXELS" % S_CUT_MJY),
    "B_peak_truncated": dict(counts=cnts_trunc, level="peak",
                             label="Schechter S<%.0f, PEAKS" % S_CUT_MJY),
    "C_peak_brighttail": dict(counts=cnts_bright, level="peak",
                              label="Schechter + bright tail, PEAKS"),
    "D_pixel_brighttail": dict(counts=cnts_bright, level="pixel",
                               label="Schechter + bright tail, PIXELS"),
}
SIM = {}
for name, cfg in LEGS.items():
    sim = CIBMapSimulator(cfg["counts"], FWHM, PIX, npix=SIDE,
                          sigma_noise=SIGMA_INST, noise_mode="beam")
    pk_by_map, px_by_map, npk = [], [], 0
    for j in range(N_SIM):
        m = sim.make_map(seed=SEED + 100000 * list(LEGS).index(name) + j)
        pk, px = peaks_of(m, DECL, EDGE, order=1)
        pk_by_map.append(pk.astype(np.float32))
        px_by_map.append(px.astype(np.float32))
        npk += pk.size
    samples = pk_by_map if cfg["level"] == "peak" else px_by_map
    # Shift to the DATA's core centroid, exactly as H1v2 does for the analytic
    # model, so that model and data are read at the same absolute flux.
    pooled = np.concatenate(samples)
    shift = MU_CORE - float(np.median(np.concatenate(pk_by_map)))
    samples = [s + shift for s in samples]
    xi, nex = xi_curve(samples, U_GRID)
    dens = npk / (N_SIM * (SIDE - 2 * EDGE) ** 2 / BEAM_PIX)
    SIM[name] = dict(xi=xi, nex=nex, peak_density=dens,
                     core_mu=float(np.median(np.concatenate(pk_by_map))),
                     core_sig=robust_sigma(np.concatenate(pk_by_map)),
                     pix_sig=robust_sigma(np.concatenate(px_by_map)),
                     shift=shift, label=cfg["label"])
    print("  %-20s peaks %8d (%.4f/beam) | core sigma %.3f | %.0f s"
          % (name, npk, dens, SIM[name]["core_sig"], time.time() - T0))

print("\n  VALIDATION: measured peak density %.4f/beam vs simulated %.4f (leg B)"
      % (NPK_DATA / (NC_DATA * (SIDE - 2 * EDGE) ** 2 / BEAM_PIX),
         SIM["B_peak_truncated"]["peak_density"]))
print("  measured peak-core sigma %.3f vs simulated %.3f mJy/beam"
      % (SIG_CORE, SIM["B_peak_truncated"]["core_sig"]))

#%% ---------------- cell 4: the analytic reference and the decomposition -----
print("\n" + "-" * 79)
print("cell 4: decomposition")
print("-" * 79)
pp = PofD(sch_full, FWHM, mu=1.0, sigma_noise=SIGMA_INST, s_cut=S_CUT_MJY,
          d_span=300.0, n_fft=2 ** 17)
xi_analytic = np.asarray(xi_of_threshold_analytic(pp.d, pp.p,
                                                  U_GRID - MU_CORE)[0], float)
xA = SIM["A_pixel_truncated"]["xi"]
xB = SIM["B_peak_truncated"]["xi"]
xC = SIM["C_peak_brighttail"]["xi"]
xD = SIM["D_pixel_brighttail"]["xi"]

d_val = xA - xi_analytic          # simulator vs analytic  (must be ~0)
d_p2p = xB - xA                   # CAUSE A: pixel -> peak
d_bri = xC - xB                   # CAUSE B: bright population
d_res = XI_DATA - xC              # what is left unexplained

print("     u[mJy]   analytic   A:pix   B:peak   C:+bright    DATA |"
      " sim-anal  pix->peak  bright   residual")
for i in range(0, len(U_GRID), 2):
    print("   %8.1f  %+8.4f %+7.4f %+8.4f %+9.4f %+8.4f | %+8.4f %+9.4f %+8.4f %+9.4f"
          % (U_GRID[i], xi_analytic[i], xA[i], xB[i], xC[i], XI_DATA[i],
             d_val[i], d_p2p[i], d_bri[i], d_res[i]))

g = np.isfinite(XI_DATA) & np.isfinite(XIERR_DATA) & (XIERR_DATA > 0)


def slope(y):
    m = g & np.isfinite(y)
    return np.polyfit(U_GRID[m], y[m], 1, w=1.0 / XIERR_DATA[m])[0] if m.sum() > 4 else np.nan


print("\n  Weighted slope d(xi)/du over the threshold grid [per mJy]:")
for lab, y in (("DATA", XI_DATA), ("analytic pixel model", xi_analytic),
               ("A  sim pixels, truncated", xA),
               ("B  sim peaks,  truncated", xB),
               ("C  sim peaks,  + bright tail", xC),
               ("D  sim pixels, + bright tail", xD)):
    print("    %-30s %+.5f" % (lab, slope(y)))

print("\n  Attribution of the measured slope (%+.5f per mJy):" % slope(XI_DATA))
print("    simulator vs analytic (should be ~0)  %+.5f" % slope(d_val))
print("    CAUSE A, pixel -> peak                %+.5f" % slope(d_p2p))
print("    CAUSE B, bright lensed population     %+.5f" % slope(d_bri))
print("    residual, unexplained                 %+.5f" % slope(d_res))

chi2_C = float(np.nansum(((XI_DATA - xC) / XIERR_DATA)[g] ** 2))
chi2_B = float(np.nansum(((XI_DATA - xB) / XIERR_DATA)[g] ** 2))
chi2_an = float(np.nansum(((XI_DATA - xi_analytic) / XIERR_DATA)[g] ** 2))
print("\n  chi2 vs the data on %d thresholds (diagonal bootstrap errors):" % g.sum())
print("    analytic pixel model (H1v2 headline)   %9.1f" % chi2_an)
print("    B  sim peaks, truncated counts         %9.1f" % chi2_B)
print("    C  sim peaks, + bright tail            %9.1f" % chi2_C)

VAL_OK = np.nanmax(np.abs(d_val[g])) < 0.05
print("\n  [%s] simulator reproduces the analytic pixel model to max |d| = %.4f"
      % ("PASS" if VAL_OK else "FAIL", np.nanmax(np.abs(d_val[g]))))
if not VAL_OK:
    print("  *** The validation leg FAILED.  Do not interpret the decomposition")
    print("      below: the discrepancy is in the simulation, not in the data.")

RES.update(u_grid=U_GRID, xi_data=XI_DATA, xi_err=XIERR_DATA,
           xi_analytic=xi_analytic, xi_A=xA, xi_B=xB, xi_C=xC, xi_D=xD,
           d_validation=d_val, d_pixel_to_peak=d_p2p, d_bright=d_bri,
           d_residual=d_res, chi2_analytic=chi2_an, chi2_B=chi2_B,
           chi2_C=chi2_C, n_sim=N_SIM,
           peak_density_sim=SIM["B_peak_truncated"]["peak_density"],
           peak_density_data=NPK_DATA / (NC_DATA * (SIDE - 2 * EDGE) ** 2 / BEAM_PIX),
           validation_ok=bool(VAL_OK))


#%% -------------------------------- cell 5: figures --------------------------
print("\n" + "-" * 79)
print("cell 5: figures")
print("-" * 79)


def _save(fig, name):
    if SAVE_FIGURES:
        p = os.path.join(FIGURE_DIR, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("  wrote", p)


fig, ax = plt.subplots(figsize=(8.0, 5.6))
ax.errorbar(U_GRID, XI_DATA, yerr=XIERR_DATA, fmt="o", color="k", ms=5,
            capsize=3, zorder=6, label="GAMA-09 data (H1v2)")
ax.plot(U_GRID, xi_analytic, "-", color="0.55", lw=2,
        label="analytic PIXEL model (H1v2 headline)")
ax.plot(U_GRID, xA, "s--", color="tab:blue", ms=4, lw=1.6,
        label="A  sim PIXELS, $S<%.0f$" % S_CUT_MJY)
ax.plot(U_GRID, xB, "^-", color="tab:orange", ms=4, lw=1.8,
        label="B  sim PEAKS, $S<%.0f$" % S_CUT_MJY)
ax.plot(U_GRID, xC, "v-", color="crimson", ms=5, lw=2.2,
        label="C  sim PEAKS, + bright tail")
ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlabel("threshold $u$ [mJy/beam, absolute]")
ax.set_ylabel(r"GPD shape parameter $\xi(u)$")
ax.set_title("H1c: what makes $\\hat\\xi(u)$ rise?", fontsize=12)
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
_save(fig, "herschel_h1c_legs.png")

fig2, ax2 = plt.subplots(1, 2, figsize=(12.2, 4.6))
ax2[0].plot(U_GRID, d_p2p, "o-", color="tab:orange", label="B $-$ A: pixel $\\to$ peak")
ax2[0].plot(U_GRID, d_bri, "s-", color="crimson", label="C $-$ B: bright tail")
ax2[0].plot(U_GRID, d_res, "^--", color="k", label="data $-$ C: residual")
ax2[0].fill_between(U_GRID, -XIERR_DATA, XIERR_DATA, color="0.85", zorder=0,
                    label=r"$\pm 1\sigma$ data error")
ax2[0].axhline(0, color="0.6", lw=0.8)
ax2[0].set_xlabel("threshold $u$ [mJy/beam]")
ax2[0].set_ylabel(r"$\Delta\xi$")
ax2[0].set_title("Decomposition of the offset", fontsize=11)
ax2[0].legend(fontsize=8)
ax2[0].grid(alpha=0.25)
Sg = np.geomspace(5.0, 400.0, 400)
ax2[1].loglog(Sg, sch_full.dnds(Sg), "-", color="0.55", lw=2, label="Schechter")
ax2[1].loglog(Sg, cnts_bright.dnds(Sg), "-", color="crimson", lw=2,
              label="Schechter + bright tail")
ax2[1].axvline(S_CUT_MJY, color="k", ls=":", lw=1.2, label="$S_{\\rm cut}$")
ax2[1].axvspan(U_GRID[0], U_GRID[-1], color="0.9", zorder=0, label="threshold range")
ax2[1].set_xlabel("$S$ [mJy]")
ax2[1].set_ylabel("$dN/dS$ [mJy$^{-1}$ deg$^{-2}$]")
ax2[1].set_ylim(1e-6, None)
ax2[1].set_title("The injected bright population", fontsize=11)
ax2[1].legend(fontsize=8)
ax2[1].grid(alpha=0.25)
fig2.tight_layout()
_save(fig2, "herschel_h1c_decomposition.png")

fig3, ax3 = plt.subplots(1, 2, figsize=(12.0, 4.4))
ax3[0].plot(U_GRID, xi_analytic, "-", color="k", lw=2, label="analytic pixel P(D)")
ax3[0].plot(U_GRID, xA, "o--", color="tab:blue", ms=4, label="simulated pixels (leg A)")
ax3[0].set_xlabel("threshold $u$ [mJy/beam]")
ax3[0].set_ylabel(r"$\xi(u)$")
ax3[0].set_title("Validation: simulator vs analytic", fontsize=11)
ax3[0].legend(fontsize=8)
ax3[0].grid(alpha=0.25)
names = list(SIM)
ax3[1].bar(np.arange(len(names)), [SIM[n]["peak_density"] for n in names],
           0.55, color="tab:blue")
ax3[1].axhline(RES["peak_density_data"], color="k", ls="--", lw=1.5,
               label="measured (%.4f/beam)" % RES["peak_density_data"])
ax3[1].set_xticks(np.arange(len(names)))
ax3[1].set_xticklabels([n.split("_")[0] for n in names])
ax3[1].set_ylabel("peaks per beam")
ax3[1].set_title("Validation: peak density", fontsize=11)
ax3[1].legend(fontsize=8)
ax3[1].grid(alpha=0.25, axis="y")
fig3.tight_layout()
_save(fig3, "herschel_h1c_validation.png")

#%% -------------------------------- cell 6: summary --------------------------
print("\n" + "=" * 79)
print("SUMMARY  --  H1c peak-level simulations")
print("=" * 79)
print("  simulated maps per leg  : %d of %d pix (%.2f deg^2)"
      % (N_SIM, SIDE, N_SIM * (SIDE * PIX / 3600.) ** 2))
print("  validation, sim vs analytic (pixels): max |dxi| = %.4f  [%s]"
      % (np.nanmax(np.abs(d_val[g])), "PASS" if VAL_OK else "FAIL"))
print("  peak density  simulated %.4f/beam  vs measured %.4f"
      % (RES["peak_density_sim"], RES["peak_density_data"]))
print("  bright tail fitted from the data: N(>S) = %.3f (S/100)^-%.2f deg^-2,"
      % (N_B, A_FIT))
print("                                    asymptotic xi = %.3f" % (1.0 / A_FIT))
print("\n  SLOPE ATTRIBUTION  d(xi)/du [per mJy]:")
print("    measured                              %+.5f" % slope(XI_DATA))
print("    CAUSE A, pixel -> peak                %+.5f" % slope(d_p2p))
print("    CAUSE B, bright lensed population     %+.5f" % slope(d_bri))
print("    unexplained residual                  %+.5f" % slope(d_res))
print("\n  chi2 vs data: analytic %.1f -> sim peaks %.1f -> + bright tail %.1f"
      % (chi2_an, chi2_B, chi2_C))
print("=" * 79)
print("""
  HOW TO READ THIS.  Leg A must match the analytic curve -- if it does not, the
  simulation is wrong and nothing below means anything.  Given that, (B - A)
  is the pure pixel-to-peak effect and (C - B) is the pure bright-population
  effect, and the residual (data - C) is what neither explains.

  If CAUSE B dominates, the rising xi_hat(u) is NOT a systematic: it is the GPD
  reading the flattening of dN/dS caused by the strongly lensed bright
  population, and belongs in the paper as a result.  The consequence for H2 is
  benign, since lensed and control cutouts are drawn from the same sky and the
  bright population affects both.

  If CAUSE A dominates, the model curves in H1v2 must be replaced by
  simulation-derived PEAK-level predictions throughout, and the same must be
  done for the Delta-xi prediction in H2 -- a peak-level bias would otherwise
  propagate straight into the cluster measurement.""")

npz = os.path.join(FIGURE_DIR, "herschel_peak_sims_350.npz")
np.savez_compressed(npz, **RES)
print("\nwrote", npz)
print("total runtime %.0f s" % (time.time() - T0))
if SAVE_FIGURES and (os.environ.get("DISPLAY") or sys.platform == "darwin"):
    plt.show()
