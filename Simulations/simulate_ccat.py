"""
simulate_ccat.py — CCAT / FYST Prime-Cam 850 GHz arm of the GPD/CIB-lensing suite
==================================================================================

Unlensed xi(u) and cluster-lensed Delta_xi(u) for the Fred Young Submillimeter
Telescope (FYST) with Prime-Cam.  Unlike the Planck and Herschel arms this is a
pure FORECAST — the instrument is not yet on sky — so every number traces to
the CCAT-prime Collaboration (2023) science paper rather than to a measurement.

Supersedes the CCAT panels of `main_sims_unlensed.py` and `main_sims_lensed.py`.
All changes implement recommendations of `instrument_consistency_audit.md`
(2026-08-15).

WHICH BAND THIS IS — READ THIS BEFORE CHANGING THE BEAM
--------------------------------------------------------
350 um = 857 GHz, so the Prime-Cam module we want is the **850 GHz** one
(353 um), Table 1 beam **15"**.  Prime-Cam ALSO has a broadband module named
"350 GHz", which is at 857 um with a **37"** beam — a factor 2.5 in beam and a
completely different confusion regime.  The old `counts_350um.py` note string
said "CCAT-prime 350 um class resolution", which is numerically right but reads
ambiguously.  The beam here is the 850 GHz value.            [audit Sec. 4, R7]

WHAT THIS SCRIPT PRODUCES
-------------------------
  Sec. 4  Unlensed xi-hat(u), three models, Q1 (best-quartile) noise.
  Sec. 5  The Q1-Q3 weather leg — the same run at the weather-averaged depth.
  Sec. 6  Analytic Delta_xi(u) of the benchmark cluster.
  Sec. 7  Three-arm lensed Monte Carlo.
  Sec. 8  The forecast: clusters needed for 3 sigma, and what the planned
          100 deg^2 survey would actually deliver.

INSTRUMENT PARAMETERS, AND WHERE EACH COMES FROM
-------------------------------------------------
All from `relevant_papers/CCAT_Science_Paper_2023.pdf`, Sec. 2.4 and Table 1.

  beam   15"      Table 1, 850 GHz broadband channel.  The old simulation's 15"
                  was already correct.                              [audit Sec. 4]

  pixel  3.0"     = FWHM/5.  No map pixel is fixed by the project yet, so the
                  oversampling convention of the declustering calibration
                  (FWHM/pix in 4-10) is used.

  sigma_N 3.1     Table 1, middle block: the 100 deg^2 / 500 h DSFG survey
          mJy/bm  reaches sigma_survey/beam = 3100 uJy/beam in first-quartile
                  PWV, and 4900 uJy/beam averaged over Q1-Q3.  The old
                  f_N = 0.5 gave 2.42 mJy/beam, optimistic by 1.3-2x.
                  Q1 is the fiducial because the paper states the DSFG survey
                  will use best-quartile weather; Q1-Q3 is Sec. 5.      [R2]

                  NOTE the WIDE 20,000 deg^2 survey is useless at 850 GHz
                  (NEI 4.8e5 Jy/sr/sqrt(s)); any CCAT forecast for this project
                  must assume the 100 deg^2 DSFG/CIB-Mid survey or the
                  4-16 deg^2 ultra-deep fields.

  sigma_nonP 0.0  No measurement exists.  At a 15" beam the v2 memo measures
                  R_core = 0.15, i.e. clustering adds only sqrt(1.15) = 1.07 to
                  the core width, versus 1.96 at Planck's 5'.  Omitting it is a
                  7% effect on the core here, against 83% at Planck.     [R3]

  S_cut  100 mJy  Same as Herschel: above ~100 mJy the counts are
                  lensing-dominated and the pipeline supplies its own lensing.

  S_min  1.0 mJy  Common with both data pipelines.                      [R1]

  u grid          Built from this instrument's own Sigma_tot, since there is no
                  measured core to anchor to.  Reported in absolute mJy/beam so
                  it can be compared with the Herschel ladder directly.  [R6]

CONFUSION CROSS-CHECK
---------------------
The CCAT paper gives no explicit sigma_conf at 850 GHz; its only quantitative
statement is that FYST reaches "about 3 times deeper into the source confusion"
than Herschel.  Solving the standard self-consistent criterion
S_lim = 5 sigma_c(S_lim) with our models at 15" gives sigma_c = 3.07 (DPL) to
3.49 (Schechter), which happens to vindicate the placeholder sigma_c = 3.0 that
the old simulation carried.  The same calculation at 25.15" gives 33.4-36.2 mJy
against Nguyen et al.'s quoted 31.5 mJy, so the criterion is calibrated.  Our
implied Herschel-to-CCAT flux gain is 2.1x, somewhat short of the paper's ~3x
— expected, since theirs rests on Bethermin et al. (2017) rather than on our
extrapolated Bethermin et al. (2012) fits.                    [audit Sec. 5.3]

Usage
-----
    python simulate_ccat.py
    CIB_QUICK_TEST=1 python simulate_ccat.py
    CIB_CACHE=1      python simulate_ccat.py

Outputs: figures in ./figures/ccat/, results in ./results/ccat.npz,
log in ./results/ccat.log.
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
SIGMA_N_Q1 = 3.1                # mJy/beam, Table 1, 100 deg^2 survey, Q1
SIGMA_N_Q13 = 4.9               # mJy/beam, same survey, Q1-Q3 average
SURVEY_AREA_DEG2 = 100.0        # the planned DSFG / CIB-Mid pilot

CCAT = Instrument(
    name="CCAT",
    beam_arcsec=15.0,           # Table 1, 850 GHz = 353 um       [audit Sec. 4]
    pix_arcsec=3.0,             # = FWHM/5, no map pixel fixed yet
    sigma_n=SIGMA_N_Q1,         # Q1 weather, 100 deg^2 survey          [R2]
    sigma_nonp=0.0,             # R_core = 0.15 at 15", a 7% effect     [R3]
    s_cut=100.0,
    # Absolute ladder.  EXTENDED DOWNWARD 2026-08-17 (must-close item 1 of
    # `00_Paper_Draft/revised_outline.md` Sec. 4; three-experiments Sec. 8.1):
    # the previous grid started at 12.0 and the forecast optimum sat on that
    # first grid point for all four apertures, so N_3sigma = 3.7e3 was an
    # upper bound on the requirement rather than a minimum.  The six new
    # points reach k = 1.06 of Sigma_tot, well into the Gaussian core, so
    # that the minimum is now interior and can be located rather than
    # bounded.  See Sec. 8b for the validity floor that governs which
    # optimum is quotable.
    u_grid=np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                     12.0, 15.0, 18.0, 22.0, 26.0, 31.0, 37.0, 45.0, 55.0]),
    core_measured=None,         # no data exist
    apertures_theta500=(0.5, 1.0, 1.5, 2.0),
    npix_unlensed=512,          # 25.6' cutouts
    n_maps_unlensed=50,         # 9.1 deg^2 simulated; the survey is 100 (Sec. 8)
    n_clusters=120,             # 293 peaks/cluster at 1.5 theta500 -> ample
    note=("CCAT-prime Collaboration (2023) Table 1: 850 GHz (353 um) "
          "broadband module, beam 15 arcsec; sigma_survey/beam 3100 uJy (Q1) "
          "and 4900 uJy (Q1-Q3) for the 100 deg^2 / 500 h DSFG survey. "
          "FORECAST ONLY — no data exist."),
)

MC_MODEL = "Schechter"
N_SIGMA = 3.0

if QUICK_TEST:
    CCAT.n_maps_unlensed = 10
    CCAT.n_clusters = 30
    sc.N_BOOT_DEFAULT = 60
    sc.MIN_EXCEED_DEFAULT = 25
    print("QUICK_TEST: reduced ensembles — not for science.")

run = Run("ccat", config_key=(
    CCAT.beam, CCAT.pix, CCAT.sigma_n, CCAT.s_cut, tuple(CCAT.u_grid),
    CCAT.npix_unlensed, CCAT.n_maps_unlensed, CCAT.n_clusters, S_MIN,
    SIGMA_N_Q13, QUICK_TEST))

# %% Section 3 — models, benchmark lens, noise budget
run.rule("simulate_ccat.py — FYST/Prime-Cam 850 GHz (353 um), GPD tail of the CIB")
run.say("FORECAST ARM.  No data exist; every instrument number traces to")
run.say("CCAT-prime Collaboration (2023) Table 1.  Band = 850 GHz, NOT the")
run.say("Prime-Cam channel named '350 GHz' (857 um, 37\" beam).")

MODELS = make_models(s_min=S_MIN, s_max=CCAT.s_cut)
BUDS = {nm: budget(MODELS[nm], CCAT) for nm in MODEL_ORDER}
BENCH = Benchmark()

report_models(run, CCAT, MODELS, BUDS)

run.say("\n  Self-consistent confusion limit, S_lim = 5 sigma_c(S_lim) "
        "(audit Sec. 5.3):")
run.say("    at 15\"     Schechter 17.5 / 3.49   SPL 27.2 / 5.45   "
        "DPL 15.4 / 3.07   [S_lim / sigma_c, mJy]")
run.say("    at 25.15\"  Schechter 36.2 / 7.24   SPL 47.8 / 9.56   "
        "DPL 33.4 / 6.68   vs Nguyen's 31.5 / 6.3")
run.say("    -> our Herschel-to-CCAT flux gain is 2.1x; the CCAT paper claims "
        "~3x from\n       Bethermin et al. (2017).  We are the more "
        "conservative of the two.")
run.say(f"\n  Dynamic range S_cut / sigma_c = "
        f"{CCAT.s_cut / BUDS['Schechter']['sigma_c']:.1f} (Schechter), "
        f"against 12.5 at Herschel and 1.1 at Planck.")

APER = report_apertures(run, CCAT, BENCH)
run.say(f"\n  theta_500 / FWHM = {BENCH.theta500 / (CCAT.beam / 60):.1f}: the "
        f"benchmark cluster spans ~14 beams,\n  so even the r < 0.5 "
        f"theta_500 aperture — where <mu> = 3.07 — is well resolved.  This\n"
        f"  is the structural advantage of the CCAT arm over both others.")

# %% Section 4 — unlensed xi(u), Q1 weather
UNL = stage_unlensed(run, CCAT, MODELS, BUDS)

# %% Section 5 — the Q1-Q3 weather leg
run.rule(f"CCAT — weather leg: sigma_N = {SIGMA_N_Q13} mJy/beam (Q1-Q3 "
         f"average)", ch="-")
run.say("  The Q1 value assumes the DSFG survey runs in best-quartile PWV, "
        "which the paper\n  states as the plan.  The weather-averaged depth is "
        "58% worse, and at 850 GHz the\n  atmosphere is the dominant term, so "
        "this leg brackets the realistic outcome.")
CCAT_Q13 = Instrument(
    name="CCAT", beam_arcsec=CCAT.beam, pix_arcsec=CCAT.pix,
    sigma_n=SIGMA_N_Q13, sigma_nonp=CCAT.sigma_nonp, s_cut=CCAT.s_cut,
    u_grid=CCAT.u_grid, core_measured=None,
    apertures_theta500=CCAT.apertures_theta500,
    npix_unlensed=CCAT.npix_unlensed,
    n_maps_unlensed=CCAT.n_maps_unlensed, n_clusters=CCAT.n_clusters,
    note="Q1-Q3 weather leg")
BUDS_Q13 = {nm: budget(MODELS[nm], CCAT_Q13) for nm in MODEL_ORDER}
UNL_Q13 = stage_unlensed(run, CCAT_Q13, MODELS, BUDS_Q13, label="q13")
run.say("")
run.say(f"{'model':>10s} {'Sig_tot Q1':>11s} {'Sig_tot Q1-Q3':>14s} "
        f"{'ratio':>7s}")
for nm in MODEL_ORDER:
    a, b = BUDS[nm]["sigma_tot"], BUDS_Q13[nm]["sigma_tot"]
    run.say(f"{nm:>10s} {a:11.3f} {b:14.3f} {b / a:7.3f}")

# %% Section 6 — analytic Delta_xi of the benchmark cluster
THEORY = stage_lensed_theory(run, CCAT, MODELS, BUDS, BENCH)

# %% Section 7 — lensed Monte Carlo (Q1 fiducial)
MC = stage_lensed_mc(run, CCAT, MODELS[MC_MODEL], BUDS[MC_MODEL], BENCH,
                     mc_model_name=MC_MODEL)

# %% Section 8 — forecast, and what 100 deg^2 would deliver
FC = stage_forecast(run, CCAT, THEORY, MC, BENCH,
                    mc_model_name=MC_MODEL, n_sigma=N_SIGMA)
figure_lensed_mc(run, CCAT, THEORY, MC, BENCH, mc_model_name=MC_MODEL)

# %% Section 8b — the validity floor, and which optimum is quotable
#
# WHY THIS BLOCK EXISTS.  Extending the ladder downward (Sec. 2) makes the
# forecast better without bound, because |Delta_xi_theory| rises toward the
# core while sigma_1 falls.  That improvement is not free: the project's own
# validity statement (`full_simulation_summary_v2.md` Sec. 10; carried into
# `revised_outline.md` Sec. 3) is that xi-hat is unbiased only above roughly
# 3 sigma_c, and the forecast divides an ANALYTIC signal (the KL projection
# of the pixel P(D)) by a MONTE CARLO error (measured on declustered peaks).
# Those are different distributions.  The pixel-vs-peak offset is documented
# to cancel in Delta_xi (three-experiments Sec. 8.3) — but the cancellation
# argument is a tail argument, and it degrades as u approaches the Gaussian
# core, exactly where the unconstrained optimum wants to sit.
#
# So two numbers are reported: the unconstrained minimum (what the grid
# says) and the minimum restricted to u >= 3 sigma_c (what is defensible).
# The paper quotes the floored one.
VALIDITY_K_SIGMA_C = 3.0
U_FLOOR = VALIDITY_K_SIGMA_C * BUDS[MC_MODEL]["sigma_c"]

run.rule(f"CCAT — validity floor: u >= {VALIDITY_K_SIGMA_C:.0f} sigma_c = "
         f"{U_FLOOR:.2f} mJy/beam ({MC_MODEL})", ch="-")
run.say(f"  sigma_c({MC_MODEL}) = {BUDS[MC_MODEL]['sigma_c']:.3f}, "
        f"Sigma_tot = {BUDS[MC_MODEL]['sigma_tot']:.3f} mJy/beam.")
run.say(f"  Of the {CCAT.u_grid.size} ladder points, "
        f"{int((CCAT.u_grid >= U_FLOOR).sum())} lie above the floor.")
run.say("  xi-hat is unbiased only above ~3 sigma_c (v2 memo Sec. 10); below")
run.say("  it, the forecast's analytic numerator and peak-level MC denominator")
run.say("  are no longer the same distribution.  QUOTE THE FLOORED OPTIMUM.")

_ok_u = CCAT.u_grid >= U_FLOOR
run.say("")
run.say(f"{'r/th500':>8s} {'u_free':>8s} {'N_infl free':>13s} "
        f"{'u_floor':>8s} {'N_infl floored':>15s} {'penalty':>9s}")
FLOOR_BEST = {}
for R in CCAT.apertures_theta500:
    ninf = np.asarray(FC[R]["n_needed_infl"], float)
    jf = FC[R]["j_best"]
    masked = np.where(_ok_u & np.isfinite(ninf), ninf, np.inf)
    if np.isfinite(masked).any() and np.min(masked) < np.inf:
        jr = int(np.argmin(masked))
        FLOOR_BEST[R] = (jr, float(ninf[jr]))
        pen = ninf[jr] / ninf[jf] if (jf >= 0 and np.isfinite(ninf[jf])) \
            else np.nan
        run.say(f"{R:8.2f} {CCAT.u_grid[jf]:8.1f} {ninf[jf]:13.3e} "
                f"{CCAT.u_grid[jr]:8.1f} {ninf[jr]:15.3e} {pen:8.2f}x")
    else:
        FLOOR_BEST[R] = (-1, np.nan)
        run.say(f"{R:8.2f} {CCAT.u_grid[jf]:8.1f} {ninf[jf]:13.3e} "
                f"{'--':>8s} {'(no valid threshold)':>15s} {'--':>9s}")

_fb_R, _fb_N, _fb_j = None, np.inf, -1
for R, (j, n) in FLOOR_BEST.items():
    if j >= 0 and np.isfinite(n) and n < _fb_N:
        _fb_R, _fb_N, _fb_j = R, n, j
if _fb_R is not None:
    run.say(f"\n  QUOTABLE HEADLINE (u >= 3 sigma_c): r < {_fb_R} theta_500 "
            f"(= {_fb_R * BENCH.theta500:.2f}'), u = {CCAT.u_grid[_fb_j]:.1f} "
            f"mJy/beam")
    run.say(f"    Delta_xi = "
            f"{THEORY[(_fb_R, MC_MODEL)]['delta'][_fb_j]:+.5f}   "
            f"sigma_1 = {FC[_fb_R]['sigma_1'][_fb_j]:.4f}")
    run.say(f"    N clusters for {N_SIGMA:.0f}-sigma = "
            f"{FC[_fb_R]['n_needed'][_fb_j]:.3e} (Poisson MC), "
            f"{_fb_N:.3e} (clustering-inflated)")
    run.say(f"  UNCONSTRAINED (grid minimum, NOT quotable): "
            f"{FC['best_N']:.3e} at r < {FC['best_R']} theta_500")

#  Drawn AFTER the floor is known, so the figure stars the quotable optimum
#  and shades the invalid region.  A figure that starred the unconstrained
#  minimum would contradict the text it illustrates.
figure_forecast(run, CCAT, FC, BENCH, n_sigma=N_SIGMA, u_floor=U_FLOOR)

run.rule("CCAT — translating the cluster requirement into a survey", ch="-")
run.say(f"  The planned DSFG / CIB-Mid pilot covers {SURVEY_AREA_DEG2:.0f} "
        f"deg^2.  Cluster surface densities for\n  a benchmark-class halo "
        f"(M_500 = 1e15, i.e. the very top of the mass function) are of\n"
        f"  order 0.01-0.05 per deg^2 out to z ~ 1, so such a field contains "
        f"only a few.  The\n  requirement below should therefore be read as "
        f"'how much sky, at what mass floor',\n  not as 'how many pointings'.")
if _fb_R is not None:
    n_req = _fb_N                       # the FLOORED number, per Sec. 8b
    run.say(f"\n  N required (clustering-inflated, u >= 3 sigma_c) = "
            f"{n_req:.3e} benchmark clusters at r < {_fb_R} theta_500.")
    for dens, lab in ((0.05, "optimistic"), (0.01, "conservative")):
        run.say(f"    at {dens:.2f} such clusters per deg^2 ({lab}): "
                f"{n_req / dens:.3e} deg^2 of sky")
    run.say(f"    NOTE Delta_xi scales roughly as <ln mu>, and <ln mu> falls "
            f"as M^(2/3)-ish, so a\n    sample of lower-mass clusters needs "
            f"more of them than the surface-density scaling\n    alone "
            f"suggests.  Use this as an order of magnitude.")

run.say("\n  Cross-experiment comparison is the point of the common benchmark:")
run.say("  run all three scripts and compare the 'BEST CONFIGURATION' lines.")

# %% Section 9 — save
_flat = {}
for nm in MODEL_ORDER:
    _flat[f"unl_xi_{nm}"] = UNL[nm]["scan"]["xi"]
    _flat[f"unl_lo_{nm}"] = UNL[nm]["scan"]["lo"]
    _flat[f"unl_hi_{nm}"] = UNL[nm]["scan"]["hi"]
    _flat[f"unl_nexc_{nm}"] = UNL[nm]["scan"]["n_exc"]
    _flat[f"unl_theory_{nm}"] = UNL[nm]["theory"]
    _flat[f"unl_q13_xi_{nm}"] = UNL_Q13[nm]["scan"]["xi"]
    _flat[f"unl_q13_theory_{nm}"] = UNL_Q13[nm]["theory"]
for R in CCAT.apertures_theta500:
    for nm in MODEL_ORDER:
        _flat[f"dxi_theory_{R}_{nm}"] = THEORY[(R, nm)]["delta"]
    _flat[f"dxi_mc_{R}"] = MC[R]["delta"]["delta"]
    _flat[f"dxi_mc_sig_{R}"] = MC[R]["delta"]["sigma"]
    _flat[f"dxi_null_{R}"] = MC[R]["null"]["delta"]
    _flat[f"n3sig_{R}"] = FC[R]["n_needed"]
    _flat[f"n3sig_infl_{R}"] = FC[R]["n_needed_infl"]
    _flat[f"jfloor_{R}"] = FLOOR_BEST[R][0]
    _flat[f"n3sig_infl_floor_{R}"] = FLOOR_BEST[R][1]
_flat["u_floor"] = U_FLOOR
_flat["floor_best_R"] = -1.0 if _fb_R is None else float(_fb_R)
_flat["floor_best_u"] = np.nan if _fb_j < 0 else float(CCAT.u_grid[_fb_j])
_flat["floor_best_N_infl"] = _fb_N
_flat["free_best_R"] = -1.0 if FC["best_R"] is None else float(FC["best_R"])
_flat["free_best_N_infl"] = FC["best_N"]
run.save(u_grid=CCAT.u_grid, s_min=S_MIN, s_cut=CCAT.s_cut, beam=CCAT.beam,
         pix=CCAT.pix, sigma_n_q1=SIGMA_N_Q1, sigma_n_q13=SIGMA_N_Q13,
         survey_area_deg2=SURVEY_AREA_DEG2, theta500=BENCH.theta500,
         m500=BENCH.m500, m_vir=BENCH.m_vir, z_lens=BENCH.z_l,
         apertures=np.array(CCAT.apertures_theta500),
         n_clusters=MC["n_clusters"], note=CCAT.note, **_flat)
