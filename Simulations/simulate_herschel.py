"""
simulate_herschel.py — Herschel/SPIRE 350 um arm of the GPD/CIB-lensing suite
==============================================================================

Unlensed xi(u) and cluster-lensed Delta_xi(u) for Herschel-SPIRE at 350 um,
with every instrument parameter taken from the measured GAMA-09 analysis rather
than from a scaling of the count model.

Supersedes the Herschel panels of `main_sims_unlensed.py` and
`main_sims_lensed.py`.  All changes implement recommendations of
`instrument_consistency_audit.md` (2026-08-15); the audit tag is given for each.

WHAT THIS SCRIPT PRODUCES
-------------------------
  Sec. 4  Unlensed xi-hat(u) for Schechter / SPL / DPL against the analytic
          curves, on the DATA's own threshold ladder u = 25-75 mJy/beam.
  Sec. 5  Analytic Delta_xi(u) of the benchmark cluster, all models, all
          apertures.  Deterministic, single-cluster, no MC noise.
  Sec. 6  Three-arm lensed Monte Carlo (lens / control / null) with N identical
          copies of the benchmark, giving the per-cluster error on Delta_xi.
  Sec. 7  The forecast: how many such clusters a 3-sigma detection needs.
  Sec. 8  A pixel-scale diagnostic (audit R5) — the same unlensed run repeated
          at 5" pixels, to test whether the 48% peak-density shortfall against
          the data is a sampling artefact.

INSTRUMENT PARAMETERS, AND WHERE EACH COMES FROM
-------------------------------------------------
All from `Herschel_analysis/Full_Analysis_Summary_Herschel.md` (H3).

  beam   25.15"   Condon-equivalent FWHM from the HELP `Matchedfilter` header,
                  confirmed in H3 Sec. 4.5 by integral(P^2).  NOT the nominal
                  24.9" of Nguyen et al. (2010), and not 24.2".  The filtered
                  core has half-max width 25.42" and a Gaussian core fit of
                  24.47", but it is non-Gaussian and only the q=2 moment
                  matters for the P(D) variance.               [audit Sec. 3]

  pixel  8.0"     The HELP map pixel (H3 Sec. 2.1).  The old simulation used
                  5.0" = FWHM/5.  Using the data's pixel puts the simulation at
                  FWHM/pix = 3.14, the same regime as the real declustering.
                                                                  [audit R5]

  sigma_N 5.402   Matched-filter noise, H3 Sec. 5.1, with 11.3% cutout-to-cutout
          mJy/bm  scatter (range 3.789-5.476).  The old f_N = 0.5 gave 4.03,
                  34% low.  Expressed as a ratio this is f_N = 0.67.
                                                                  [audit R2]

  sigma_nonP 0.0  None needed.  H3 Sec. 5.5 shows model+noise reproduces the
                  measured SF_IMAGE width to 7% in rms, so unlike Planck there
                  is no missing sky component to inject.          [audit R3]

  S_cut  100 mJy  Same truncation both data pipelines use: above ~100 mJy the
                  350 um counts are lensing-dominated (Lima et al. 2010) and no
                  longer trace the intrinsic dN/dS, and this pipeline supplies
                  its own lensing.

  S_min  1.0 mJy  Matches `herschel_unlensed_v3.py:386`.  The old 0.1 mJy cost
                  nothing for Schechter/DPL but inflated the SPL's sigma_c by
                  49% and its predicted CIB monopole by a factor 19.
                                                                  [audit R1]

  u grid 25-75    The data's own ladder (H3 Sec. 5.3), in ABSOLUTE mJy/beam.
                  xi ~ 1/(eta(S_u)-1) is a function of absolute flux, so this is
                  the physically correct anchoring (H3 Sec. 5.9). [audit R6]

MEASURED NUMBERS THIS RUN SHOULD REPRODUCE
-------------------------------------------
  peak core        mu = 9.906, sigma = 8.080 mJy/beam   (H3 Sec. 5.1)
  pixel core       sigma = 7.665                        (H3 Sec. 5.1)
  SF_IMAGE         MAD-sigma 7.845, rms 9.003           (H3 Sec. 5.5)
  model+noise      MAD-sigma 8.565, rms 9.633           (H3 Sec. 5.5)
  peak density     0.374/beam                           (H3 Sec. 5.1)
  measured xi-hat  +0.14 at u=25 rising to +0.48 at u=75 (unmasked)
                   +0.04 at u=25 falling to -0.08 at u=50 (bright-masked)

The unmasked and masked variants BRACKET the model-comparable measurement
(H3 Sec. 5.7), so neither is a goodness-of-fit on its own.  This simulation
truncates the counts at 100 mJy, so it is comparable to the MASKED variant.

Usage
-----
    python simulate_herschel.py
    CIB_QUICK_TEST=1 python simulate_herschel.py     # ~1 min, not for science
    CIB_CACHE=1      python simulate_herschel.py     # resumable
    CIB_SHOW=1       python simulate_herschel.py     # display, do not save

Outputs: figures in ./figures/herschel/, results in ./results/herschel.npz,
full log in ./results/herschel.log.
"""

# %% Section 1 — imports
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_core as sc                                             # noqa: E402
from sim_core import (Instrument, Benchmark, Run, S_MIN, make_models,  # noqa: E402
                      MODEL_ORDER, budget, report_models, report_apertures,
                      stage_unlensed, stage_lensed_theory, stage_lensed_mc,
                      stage_forecast, figure_lensed_mc, figure_forecast, QUICK_TEST)

# %% Section 2 — instrument configuration  (THE ONLY INSTRUMENT-SPECIFIC BLOCK)
HERSCHEL = Instrument(
    name="Herschel",
    beam_arcsec=25.15,          # Condon-equivalent, H3 Sec. 2.2 / 4.5
    pix_arcsec=8.0,             # HELP map pixel, H3 Sec. 2.1        [R5]
    sigma_n=5.402,              # matched-filter noise, H3 Sec. 5.1  [R2]
    sigma_nonp=0.0,             # none needed, H3 Sec. 5.5           [R3]
    s_cut=100.0,
    u_grid=np.array([25.0, 30.0, 35.0, 40.0, 45.0,
                     50.0, 55.0, 65.0, 75.0]),          # H3 Sec. 5.3 [R6]
    core_measured=9.003,        # SF_IMAGE rms, H3 Sec. 5.5
    apertures_theta500=(0.5, 1.0, 1.5, 2.0),
    npix_unlensed=256,          # 34.1' cutouts, close to the data's 30'
    n_maps_unlensed=80,         # 25.9 deg^2, matching the data's 25.75
    n_clusters=300,
    note=("Full_Analysis_Summary_Herschel.md: beam 25.15\" (Matchedfilter "
          "header, Condon-equivalent); pixel 8.0\" (HELP map); sigma_N 5.402 "
          "mJy/beam (matched filter, 11.3% cutout scatter); no non-Poisson "
          "term needed (model+noise matches SF_IMAGE rms to 7%)"),
)

MC_MODEL = "Schechter"          # the model carried through the lensed MC
N_SIGMA = 3.0

if QUICK_TEST:
    HERSCHEL.n_maps_unlensed = 12
    HERSCHEL.n_clusters = 40
    sc.N_BOOT_DEFAULT = 60
    sc.MIN_EXCEED_DEFAULT = 25
    print("QUICK_TEST: reduced ensembles — not for science.")

run = Run("herschel", config_key=(
    HERSCHEL.beam, HERSCHEL.pix, HERSCHEL.sigma_n, HERSCHEL.sigma_nonp,
    HERSCHEL.s_cut, tuple(HERSCHEL.u_grid), HERSCHEL.npix_unlensed,
    HERSCHEL.n_maps_unlensed, HERSCHEL.n_clusters, S_MIN, QUICK_TEST))

# %% Section 3 — models, benchmark lens, noise budget
run.rule("simulate_herschel.py — SPIRE 350 um, GPD tail statistics of the CIB")
run.say(__doc__.split("Usage")[0].strip().split("WHAT THIS SCRIPT")[0].strip())

MODELS = make_models(s_min=S_MIN, s_max=HERSCHEL.s_cut)
BUDS = {nm: budget(MODELS[nm], HERSCHEL) for nm in MODEL_ORDER}
BENCH = Benchmark()

report_models(run, HERSCHEL, MODELS, BUDS)
run.say("\n  Cross-checks against the measured GAMA-09 decomposition "
        "(H3 Sec. 5.5):")
_sky = np.sqrt(9.003 ** 2 - 5.402 ** 2)
run.say(f"    measured total rms 9.003, minus sigma_N 5.402 "
        f"-> sky rms {_sky:.3f} mJy/beam")
run.say(f"    model sigma_c (Schechter)          {BUDS['Schechter']['sigma_c']:.3f}"
        f"  -> model/measured {BUDS['Schechter']['sigma_c'] / _sky:.3f}")
run.say("    Nguyen et al. (2010) 350 um: sigma_conf = 6.3 +- 0.4 mJy/beam "
        "(plain pixel variance,\n      24.9\" beam, effective cut ~80 mJy).  "
        "Our model is 25% high there; the CUT DEPENDENCE\n      is reproduced "
        "to 4% (audit Sec. 5.2), so the offset is an amplitude and is 1.1 "
        "sigma\n      of the joint counts-normalisation ⊕ SPIRE-calibration "
        "uncertainty.  NOT corrected.")

APER = report_apertures(run, HERSCHEL, BENCH)

# %% Section 4 — unlensed xi(u)
UNL = stage_unlensed(run, HERSCHEL, MODELS, BUDS)
run.say("\n  For comparison, the measured unmasked xi-hat(u) (H3 Sec. 5.3):")
run.say("    u=25 +0.1405  u=35 +0.2104  u=50 +0.3562  u=75 +0.4834")
run.say("  and the bright-masked variant, which is the model-comparable one "
        "here since\n  the simulation truncates at S_cut = 100 mJy:")
run.say("    u=25 +0.0432  u=35 +0.0266  u=50 -0.0803")

# %% Section 5 — analytic Delta_xi of the benchmark cluster
THEORY = stage_lensed_theory(run, HERSCHEL, MODELS, BUDS, BENCH)

# %% Section 6 — lensed Monte Carlo
MC = stage_lensed_mc(run, HERSCHEL, MODELS[MC_MODEL], BUDS[MC_MODEL], BENCH,
                     mc_model_name=MC_MODEL)

# %% Section 7 — forecast
FC = stage_forecast(run, HERSCHEL, THEORY, MC, BENCH,
                    mc_model_name=MC_MODEL, n_sigma=N_SIGMA)
figure_lensed_mc(run, HERSCHEL, THEORY, MC, BENCH, mc_model_name=MC_MODEL)
figure_forecast(run, HERSCHEL, FC, BENCH, n_sigma=N_SIGMA)
run.say("\n  Context: the real H2 measurement used 110 eFEDS clusters of "
        "median M_vir = 2.5e14\n  (a factor 7 less massive than this "
        "benchmark) and reached sigma(Delta_xi) = 0.022-0.165.")

# %% Section 8 — pixel-scale diagnostic (audit R5)
run.rule("Herschel — pixel-scale diagnostic (audit R5)", ch="-")
run.say("  The data run at FWHM/pix = 25.15/8.0 = 3.14, outside the 4-10 range "
        "over which\n  the declustering peak-density calibration was "
        "established, and the measured peak\n  density (0.374/beam) exceeds "
        "the old 5\" simulation's (0.252/beam) by 48%.  Here the\n  same "
        "unlensed run is repeated at 5\" to isolate the pixelisation effect "
        "from the\n  physical one (source clustering, which raises the real "
        "density above Poisson).")
FINE = Instrument(
    name="Herschel", beam_arcsec=25.15, pix_arcsec=5.0,
    sigma_n=HERSCHEL.sigma_n, sigma_nonp=HERSCHEL.sigma_nonp,
    s_cut=HERSCHEL.s_cut, u_grid=HERSCHEL.u_grid,
    core_measured=HERSCHEL.core_measured,
    apertures_theta500=HERSCHEL.apertures_theta500,
    npix_unlensed=410,          # same sky area as the 8" run
    n_maps_unlensed=HERSCHEL.n_maps_unlensed,
    n_clusters=HERSCHEL.n_clusters, note="5\" pixel diagnostic leg")
UNL5 = stage_unlensed(run, FINE, MODELS, BUDS, label="pix5")
run.say("")
_H8, _H5 = 'pk/beam @8as', 'pk/beam @5as'
run.say(f"{'model':>10s} {_H8:>13s} {_H5:>13s} {'data':>8s}")
for nm in MODEL_ORDER:
    run.say(f"{nm:>10s} {UNL[nm]['peaks_per_beam']:12.4f} "
            f"{UNL5[nm]['peaks_per_beam']:12.4f} {0.374:8.3f}")
run.say("\n  If the 5\" and 8\" densities agree, pixelisation is not the "
        "cause and the residual\n  is physical (clustering); if they differ, "
        "the declustering window is resolution-\n  dependent and the "
        "exceedance counts need a correction.")

# %% Section 9 — save
_flat = {}
for nm in MODEL_ORDER:
    _flat[f"unl_xi_{nm}"] = UNL[nm]["scan"]["xi"]
    _flat[f"unl_lo_{nm}"] = UNL[nm]["scan"]["lo"]
    _flat[f"unl_hi_{nm}"] = UNL[nm]["scan"]["hi"]
    _flat[f"unl_nexc_{nm}"] = UNL[nm]["scan"]["n_exc"]
    _flat[f"unl_theory_{nm}"] = UNL[nm]["theory"]
    _flat[f"unl_pix5_xi_{nm}"] = UNL5[nm]["scan"]["xi"]
for R in HERSCHEL.apertures_theta500:
    for nm in MODEL_ORDER:
        _flat[f"dxi_theory_{R}_{nm}"] = THEORY[(R, nm)]["delta"]
    _flat[f"dxi_mc_{R}"] = MC[R]["delta"]["delta"]
    _flat[f"dxi_mc_sig_{R}"] = MC[R]["delta"]["sigma"]
    _flat[f"dxi_null_{R}"] = MC[R]["null"]["delta"]
    _flat[f"n3sig_{R}"] = FC[R]["n_needed"]
    _flat[f"n3sig_infl_{R}"] = FC[R]["n_needed_infl"]
run.save(u_grid=HERSCHEL.u_grid, s_min=S_MIN, s_cut=HERSCHEL.s_cut,
         beam=HERSCHEL.beam, pix=HERSCHEL.pix, sigma_n=HERSCHEL.sigma_n,
         sigma_nonp=HERSCHEL.sigma_nonp, theta500=BENCH.theta500,
         m500=BENCH.m500, m_vir=BENCH.m_vir, z_lens=BENCH.z_l,
         apertures=np.array(HERSCHEL.apertures_theta500),
         n_clusters=MC["n_clusters"], note=HERSCHEL.note, **_flat)
