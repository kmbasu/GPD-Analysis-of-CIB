"""
simulate_planck.py — Planck 857 GHz arm of the GPD/CIB-lensing suite
=====================================================================

Unlensed xi(u) and cluster-lensed Delta_xi(u) at Planck's 5' resolution, with
every instrument parameter taken from the measured Lenz et al. (2019) CIB-map
analysis rather than from a scaling of the count model.

Supersedes the Planck panels of `main_sims_unlensed.py` and
`main_sims_lensed.py`.  All changes implement recommendations of
`instrument_consistency_audit.md` (2026-08-15).

THE POINT OF THE PLANCK ARM
----------------------------
It is a demonstration of BEAM DILUTION, not a detection forecast.  A 5' beam
contains ~10^2-10^3 sources above 1 mJy, so the CLT has all but Gaussianized
the confusion P(D): the brightest source the intrinsic counts allow raises a
single beam by about 1 sigma_c, against 12.6 sigma_c at SPIRE's 25"
(`Revised_Planck_analysis.md` Sec. 6).  A null result here is a measurement,
and the forecast section is expected to return an astronomically large cluster
requirement.  That number IS the deliverable.

WHAT THIS SCRIPT PRODUCES
-------------------------
  Sec. 4  Unlensed xi-hat(u), three models, on the data's own k-ladder in
          absolute mJy/beam (u = 178-1105, science window 549-827).
  Sec. 5  The bright-extended leg — the same run with the Schechter counts
          extended to 1500 mJy, reproducing the construction of
          `Revised_Planck_analysis.md` Sec. 7.5.  This is Planck's substitute
          for the bright mask that Herschel can apply and Planck cannot.
  Sec. 6  Analytic Delta_xi(u) of the benchmark cluster.
  Sec. 7  Three-arm lensed Monte Carlo.
  Sec. 8  The forecast: clusters needed for 3 sigma.

INSTRUMENT PARAMETERS, AND WHERE EACH COMES FROM
-------------------------------------------------
All from `Planck_analysis/Revised_Planck_analysis.md` (M4a-v2 / M5-v2),
mask variant `4.0e+20_gp40`, f_sky = 0.3383 = 13,955 deg^2.

  beam   300"     Lenz et al. 857 GHz maps, effective beam homogenised to the
                  SMICA window, 5' FWHM Gaussian for our purposes (Sec. 4.1).

  pixel  60"      The analysis cutout pixel (Sec. 7.1), chosen so the beam FWHM
                  spans exactly 5 cutout pixels.  Native HEALPix is Nside=2048
                  = 1.72', so this is mild oversampling.

  sigma_N 30.4    From (odd-even)/2, MAD (Sec. 4.1).  Plain std is 31.9.  The
          mJy/bm  old f_N = 0.30 gave 29.1, so Planck was the one instrument the
                  old parametrisation got right (f_N implied = 0.313).  [R2]

  sigma_nonP      THE BIG CHANGE.  The measured post-baseline core is
   159.2 mJy/bm   sigma_core = 185.4 mJy/beam while Poisson confusion is only
                  ~95, so 159.2 mJy/beam of clustered CIB plus residual cirrus
                  is missing from a Poisson injector.  The old simulation ran
                  with Sigma_tot = 101.3, i.e. a core 1.83x too narrow, which
                  displaced its whole threshold ladder downward.
                  Injected here as an extra BEAM-CORRELATED Gaussian, which
                  reproduces the width exactly; `simulate_maps.py` also offers
                  a physical `clustering` mode (v2 memo Sec. 11), left as a
                  switch because v2 Sec. 10.6 warns it makes simulation-derived
                  error bars ~2x too small.                          [audit R3]

  S_cut  100 mJy  Follows the DATA memo, not the old simulation.  Sec. 5 of the
                  Planck memo truncates the model at 100 mJy on the physical
                  grounds that the brighter counts are lensing-dominated; the
                  old simulation instead used 650 mJy (the point-source mask
                  limit) to keep a measurable tail.  Those are different
                  experiments.  Since 100 mJy is only 0.54 sigma_core, the map
                  cannot be masked there — so the bracket runs the other way,
                  by EXTENDING the model (Sec. 5 below).            [audit Sec. 2]

  S_min  1.0 mJy  Matches `planck_unlensed_pd_v2.py:266`.            [audit R1]

  u grid          The data's own ladder u = mu_core + k sigma_core with
                  sigma_core = 185.4, k = 1.0...6.0 (Sec. 7.2), in absolute
                  mJy/beam.  Science window k in [3, 4.5] = 549-827.  Below
                  k ~ 3 the measurement is floor-dominated: a residual there is
                  a statement about the foreground model, not about dN/dS.
                                                                     [audit R6]

MEASURED NUMBERS THIS RUN SHOULD REPRODUCE
-------------------------------------------
  masked map rms          254.4 mJy/beam
  post-baseline core      185.4 (negative side 181.0)
  non-Poisson budget      159.2 (Sch) / 141.9 (SPL) / 161.9 (DPL)
  declustered peaks       53,977 over 1820 deg^2 = 0.233/beam
  measured xi-hat         -0.092 at k=1, crossing zero at k~1.9,
                          +0.075 / +0.093 / +0.083 at k = 3.0 / 3.5 / 4.0
  PS-mask ceiling         650 mJy/beam = 3.55 sigma_core

CAUTION ON INTERPRETATION.  Sec. 7.5 of the Planck memo establishes that the
measured positive xi-hat is driven by sources above the models' 100 mJy
truncation and by clustered/cirrus non-Gaussianity, NOT by the sub-100 mJy
counts.  This simulation contains the second effect only as a Gaussian, so the
data-minus-simulation residual is an upper bound on the bright population's
contribution, not a model test.  Do not rank the count models on this arm.

Usage
-----
    python simulate_planck.py
    CIB_QUICK_TEST=1 python simulate_planck.py
    CIB_CACHE=1      python simulate_planck.py
    CIB_PLANCK_CLUSTERING=1 python simulate_planck.py   # physical R_core mode

Outputs: figures in ./figures/planck/, results in ./results/planck.npz,
log in ./results/planck.log.
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
SIGMA_CORE_MEASURED = 185.4     # mJy/beam, Revised_Planck_analysis.md Sec. 7.1
MU_CORE_MEASURED = 85.2         # mJy/beam; u = mu_core + k sigma_core, Sec. 7.2
K_LADDER = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
SCIENCE_WINDOW_K = (3.0, 4.5)

PLANCK = Instrument(
    name="Planck",
    beam_arcsec=300.0,          # SMICA-homogenised 5', Sec. 4.1
    pix_arcsec=60.0,            # analysis cutout pixel, Sec. 7.1
    sigma_n=30.4,               # (odd-even)/2, MAD, Sec. 4.1          [R2]
    sigma_nonp=159.2,           # clustered CIB + cirrus, Sec. 7.1     [R3]
    s_cut=100.0,                # the DATA memo's truncation, Sec. 5
    u_grid=MU_CORE_MEASURED + K_LADDER * SIGMA_CORE_MEASURED,        # [R6]
    core_measured=SIGMA_CORE_MEASURED,
    apertures_theta500=(1.0, 1.5, 2.0, 3.0),   # <1 theta500 is sub-beam here
    npix_unlensed=256,          # 4.27 deg cutouts
    n_maps_unlensed=60,         # 1092 deg^2; data used 1820
    n_clusters=3000,            # 0.7 peaks/cluster -> need a big stack
    note=("Revised_Planck_analysis.md: Lenz+2019 857 GHz, mask 4.0e+20_gp40; "
          "beam 5' SMICA-homogenised; pixel 60\"; sigma_N 30.4 mJy/beam from "
          "(odd-even)/2; sigma_nonP 159.2 mJy/beam so the core reproduces the "
          "measured 185.4; S_cut 100 mJy per the data memo Sec. 5"),
)

MC_MODEL = "Schechter"
N_SIGMA = 3.0
S_CUT_EXTENDED = 1500.0         # bright-extended leg, data memo Sec. 7.5
USE_CLUSTERING = bool(int(os.environ.get("CIB_PLANCK_CLUSTERING", "0")))

if QUICK_TEST:
    PLANCK.n_maps_unlensed = 12
    PLANCK.n_clusters = 400
    sc.N_BOOT_DEFAULT = 60
    sc.MIN_EXCEED_DEFAULT = 25
    print("QUICK_TEST: reduced ensembles — not for science.")

run = Run("planck", config_key=(
    PLANCK.beam, PLANCK.pix, PLANCK.sigma_n, PLANCK.sigma_nonp, PLANCK.s_cut,
    tuple(PLANCK.u_grid), PLANCK.npix_unlensed, PLANCK.n_maps_unlensed,
    PLANCK.n_clusters, S_MIN, S_CUT_EXTENDED, QUICK_TEST))

# %% Section 3 — models, benchmark lens, noise budget
run.rule("simulate_planck.py — 857 GHz, GPD tail statistics of the CIB")
run.say("BEAM DILUTION ARM.  A 5' beam Gaussianizes the confusion P(D); a null")
run.say("here is a measurement of that, not a failure of the method.")

MODELS = make_models(s_min=S_MIN, s_max=PLANCK.s_cut)
BUDS = {nm: budget(MODELS[nm], PLANCK) for nm in MODEL_ORDER}
BENCH = Benchmark()

report_models(run, PLANCK, MODELS, BUDS)

run.say("\n  The non-Poisson budget, checked against the data (Sec. 7.1):")
run.say(f"    {'model':>10s} {'sigma_c':>9s} {'measured nonP':>14s} "
        f"{'injected':>10s} {'Sigma_tot':>10s} {'vs 185.4':>9s}")
for nm, meas in (("Schechter", 159.2), ("SPL", 141.9), ("DPL", 161.9)):
    b = BUDS[nm]
    run.say(f"    {nm:>10s} {b['sigma_c']:9.2f} {meas:14.1f} "
            f"{PLANCK.sigma_nonp:10.1f} {b['sigma_tot']:10.2f} "
            f"{b['sigma_tot'] / SIGMA_CORE_MEASURED:9.3f}")
run.say("    The single injected value 159.2 is the Schechter's measured "
        "residual; the SPL and\n    DPL therefore land a few percent off "
        "185.4, which is the honest consequence of\n    treating the "
        "foreground as a model-independent property of the sky.")
run.say(f"\n  Beam-dilution number: S_cut / sigma_c = "
        f"{PLANCK.s_cut / BUDS['Schechter']['sigma_c']:.2f} here, against "
        f"12.6 at SPIRE 25\" (data memo Sec. 6).")
run.say(f"  Sources per beam above {S_MIN} mJy: "
        + ", ".join(f"{nm} {BUDS[nm]['n_beam']:.0f}" for nm in MODEL_ORDER)
        + "  -> the CLT has done its work.")

if USE_CLUSTERING:
    run.say("\n  CIB_PLANCK_CLUSTERING=1: the physical clustering mode is NOT "
            "wired into this\n  run; see v2 memo Sec. 11 for "
            "`simulate_maps.CIBMapSimulator(clustering=...)`.  The Gaussian\n"
            "  floor is used instead.  Remove the flag to silence this note.")

APER = report_apertures(run, PLANCK, BENCH)
run.say(f"\n  theta_500 = {BENCH.theta500:.2f}' against a 5.00' beam: the "
        f"benchmark cluster is SMALLER\n  than one beam, so no aperture below "
        f"~1.5 theta_500 contains even one independent peak.")

# %% Section 4 — unlensed xi(u)
UNL = stage_unlensed(run, PLANCK, MODELS, BUDS)
run.say("\n  Measured xi-hat(u) for comparison (data memo Sec. 7.2):")
run.say("    k=1.0 u=177.9  -0.0916 +- 0.0066     k=3.0 u=548.8  +0.0747 +- 0.0270")
run.say("    k=2.0 u=363.3  +0.0168 +- 0.0126     k=3.5 u=641.5  +0.0929 +- 0.0371")
run.say("    k=2.5 u=456.0  +0.0549 +- 0.0194     k=4.0 u=734.2  +0.0829 +- 0.0469")
run.say(f"  Science window k in {SCIENCE_WINDOW_K}; below k~3 the measurement "
        "is floor-dominated.")
run.say("  Measured declustered peak density 0.233/beam; simulated: "
        + ", ".join(f"{nm} {UNL[nm]['peaks_per_beam']:.3f}"
                    for nm in MODEL_ORDER))

# %% Section 5 — the bright-extended leg (data memo Sec. 7.5)
run.rule("Planck — bright-extended leg: Schechter counts to "
         f"{S_CUT_EXTENDED:.0f} mJy", ch="-")
run.say("  At Herschel the bright-source audit is done by masking the map above")
run.say("  S_cut and comparing with identically truncated models.  At Planck "
        "that is\n  impossible: S_cut = 100 mJy is 0.54 sigma_core, so masking "
        "there would remove\n  most of the sky.  The bracket must run the "
        "other way — extend the model.")
run.say("  CAVEAT (data memo Sec. 7.5): this extrapolates the fitted Schechter "
        "far above its\n  fitted range, and the real bright counts are "
        "lensing-dominated and shallower, so\n  it UNDER-states the bright "
        "population's contribution.  Read it as a lower bound.")

PLANCK_EXT = Instrument(
    name="Planck", beam_arcsec=PLANCK.beam, pix_arcsec=PLANCK.pix,
    sigma_n=PLANCK.sigma_n, sigma_nonp=PLANCK.sigma_nonp,
    s_cut=S_CUT_EXTENDED, u_grid=PLANCK.u_grid,
    core_measured=PLANCK.core_measured,
    apertures_theta500=PLANCK.apertures_theta500,
    npix_unlensed=PLANCK.npix_unlensed,
    n_maps_unlensed=PLANCK.n_maps_unlensed, n_clusters=PLANCK.n_clusters,
    note="bright-extended leg, counts to 1500 mJy")
MODELS_EXT = make_models(s_min=S_MIN, s_max=S_CUT_EXTENDED)
BUDS_EXT = {nm: budget(MODELS_EXT[nm], PLANCK_EXT) for nm in MODEL_ORDER}
UNL_EXT = stage_unlensed(run, PLANCK_EXT, MODELS_EXT, BUDS_EXT, label="bright")

run.say("")
run.say(f"{'model':>10s} {'xi truncated':>13s} {'xi extended':>12s} "
        f"{'shift':>9s}   (at each threshold)")
for nm in MODEL_ORDER:
    a, b = UNL[nm]["scan"]["xi"], UNL_EXT[nm]["scan"]["xi"]
    run.say(f"{nm:>10s} " + "  ".join(
        f"{x:+.3f}/{y:+.3f}" for x, y in zip(a, b)))
run.say("  Extending the counts moves the simulated xi UPWARD; the data memo "
        "measures that\n  shift to be roughly half the data-minus-null offset "
        "at k = 3.5.")

# %% Section 6 — analytic Delta_xi of the benchmark cluster
THEORY = stage_lensed_theory(run, PLANCK, MODELS, BUDS, BENCH)

# %% Section 7 — lensed Monte Carlo
MC = stage_lensed_mc(run, PLANCK, MODELS[MC_MODEL], BUDS[MC_MODEL], BENCH,
                     mc_model_name=MC_MODEL)

# %% Section 8 — forecast
FC = stage_forecast(run, PLANCK, THEORY, MC, BENCH,
                    mc_model_name=MC_MODEL, n_sigma=N_SIGMA)
figure_lensed_mc(run, PLANCK, THEORY, MC, BENCH, mc_model_name=MC_MODEL)
figure_forecast(run, PLANCK, FC, BENCH, n_sigma=N_SIGMA)
run.say("\n  Context: the real M5-v2 measurement used 378 PSZ2 pairs and "
        "reached sigma(Delta_xi)\n  = 0.158-0.296, against an NFW prediction "
        "of 3e-4 to 7e-4 — three orders of magnitude\n  inside the error.  A "
        "large N here is the expected and correct answer: it is the\n  "
        "quantitative form of the beam-dilution statement, and it is why the "
        "paper needs\n  the Herschel and CCAT arms.")

# %% Section 9 — save
_flat = {}
for nm in MODEL_ORDER:
    _flat[f"unl_xi_{nm}"] = UNL[nm]["scan"]["xi"]
    _flat[f"unl_lo_{nm}"] = UNL[nm]["scan"]["lo"]
    _flat[f"unl_hi_{nm}"] = UNL[nm]["scan"]["hi"]
    _flat[f"unl_nexc_{nm}"] = UNL[nm]["scan"]["n_exc"]
    _flat[f"unl_theory_{nm}"] = UNL[nm]["theory"]
    _flat[f"unl_bright_xi_{nm}"] = UNL_EXT[nm]["scan"]["xi"]
    _flat[f"unl_bright_theory_{nm}"] = UNL_EXT[nm]["theory"]
for R in PLANCK.apertures_theta500:
    for nm in MODEL_ORDER:
        _flat[f"dxi_theory_{R}_{nm}"] = THEORY[(R, nm)]["delta"]
    _flat[f"dxi_mc_{R}"] = MC[R]["delta"]["delta"]
    _flat[f"dxi_mc_sig_{R}"] = MC[R]["delta"]["sigma"]
    _flat[f"dxi_null_{R}"] = MC[R]["null"]["delta"]
    _flat[f"n3sig_{R}"] = FC[R]["n_needed"]
    _flat[f"n3sig_infl_{R}"] = FC[R]["n_needed_infl"]
run.save(u_grid=PLANCK.u_grid, k_ladder=K_LADDER, s_min=S_MIN,
         s_cut=PLANCK.s_cut, s_cut_extended=S_CUT_EXTENDED, beam=PLANCK.beam,
         pix=PLANCK.pix, sigma_n=PLANCK.sigma_n, sigma_nonp=PLANCK.sigma_nonp,
         sigma_core_measured=SIGMA_CORE_MEASURED, theta500=BENCH.theta500,
         m500=BENCH.m500, m_vir=BENCH.m_vir, z_lens=BENCH.z_l,
         apertures=np.array(PLANCK.apertures_theta500),
         n_clusters=MC["n_clusters"], note=PLANCK.note, **_flat)
