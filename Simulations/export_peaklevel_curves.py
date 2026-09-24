"""
export_peaklevel_curves.py -- freeze the peak-level model curves into one npz
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
`paper/make_paper_figures.py` is deliberately built to read ONLY
cached `.npz`/`.json` products, so that every paper figure can be redrawn in
seconds without touching survey maps, without a working `sim_core`
installation, and without re-running Monte Carlo.  Figs. 2, 3, 4, 5 and 10
are the exception: their model curves live in `sim_core`'s `Run` pickle cache
under `Simulations/tmp/cache_paperfig_*`, which is a private, per-config,
implementation-detail cache rather than a published product.

This script is the bridge.  It calls the four figure builders in
`paper_figures_peaklevel.py` (which hit that pickle cache and therefore do NO
new simulation, provided the config keys are unchanged), plus the ideal
flux-space projection (computed here, analytically), and writes every curve those
figures need into

    Simulations/results/paperfig_peaklevel_curves.npz

Figure drawing inside the imported module is suppressed: `_save` is patched to
a no-op, so no diagnostic PNGs are written.

RUNTIME.  Roughly 3-4 minutes on a warm cache, essentially all of it spent
un-pickling the ~300k-peak ensembles, not simulating.  If a config key has
drifted the underlying builders will start a fresh Monte Carlo run instead,
which takes hours -- the script therefore prints a per-stage timing so a cache
miss is obvious immediately.

THE CACHE IS NOT ON BY DEFAULT.  See the CIB_CACHE note in Section 1: without
it the exported curves are a fresh realization rather than the published one.
Every value written here is checked against the figure it reproduces before
the paper figures are redrawn.

Usage
-----
    python export_peaklevel_curves.py            # all five stages
    CIB_EXPORT=2,3 python export_peaklevel_curves.py

Stage keys: 2 = ideal flux-space fingerprints (draft Fig. 2)
            3 = peak-level fingerprints, SPIRE beam (draft Fig. 3)
            4 = beam sweep, illustrative DPL (draft Fig. 4)
            5 = Delta_xi validation, pixel and peak conventions (Fig. 5)
            10 = Herschel measured xi-hat with peak-level curves (draft Fig. 10)

Requires numpy, scipy, matplotlib, astropy, plus `sim_core`, `counts_350um`
and `analysis_modules` on the path.
"""

# %% Section 1 -- imports, path setup, stage selection
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#  Keep the imported module from running its own __main__ block.
os.environ.setdefault("CIB_FIGS", "none")

#  CRITICAL, AND EASY TO GET WRONG.  `sim_core.USE_CACHE` reads CIB_CACHE and
#  DEFAULTS TO OFF, so a plain `python paper_figures_peaklevel.py` silently
#  recomputes every `run.cached(...)` result -- including `blocked_scan`, whose
#  RNG drives the rate-thinning of over-large pools and therefore moves xi-hat
#  itself, not just the bootstrap band.  Running this exporter without the
#  cache produced Herschel chi^2 = 28.7 / 240.7 / 140.7 where the published
#  Fig. 10 and Sect. 5.2 quote 19.3 / 309.3 / 143.8: a different, equally valid
#  realization, but NOT the one the paper reports.  The environment variable
#  must therefore be set BEFORE `sim_core` is imported and its module-level
#  USE_CACHE is evaluated.  Set CIB_CACHE=0 explicitly only to deliberately
#  regenerate the ensembles from scratch (hours).
#
#  Release note (2026-09-24): the run-to-run difference traced to
#  `seed=hash(nm) % 9973`, which Python randomizes per session.  The release
#  uses `sim_core.stable_seed`, so a fresh (CIB_CACHE=0) run is now
#  deterministic, but it is still a different realization from the published
#  one, whose curves are shipped in results/paperfig_peaklevel_curves.npz.
os.environ.setdefault("CIB_CACHE", "1")

STAGES = [s.strip() for s in
          os.environ.get("CIB_EXPORT", "2,3,4,5,10").split(",") if s.strip()]

OUT = os.path.join(_HERE, "results", "paperfig_peaklevel_curves.npz")

import paper_figures_peaklevel as pfp                             # noqa: E402
from counts_350um import (make_models, MODEL_ORDER,               # noqa: E402
                          S_MIN_DEFAULT, FIT_RANGE_MJY)
from gpd_tail import xi_of_threshold_analytic                     # noqa: E402

#  Suppress every figure write inside the imported module.  The curves are
#  what we are after; the legacy PNGs must not be rewritten.
pfp._save = lambda fig, name: (plt.close(fig), "<suppressed>")[1]

BLOB = {}


def _stamp(t0, what):
    dt = time.time() - t0
    flag = "" if dt < 600 else "   <-- SLOW: probable cache miss"
    print(f"  [{dt:7.1f}s] {what}{flag}", flush=True)


# %% Section 2 -- stage 2: the ideal flux-space fingerprints (draft Fig. 2)
#  Pure analytic, no Monte Carlo: the GPD (Kullback-Leibler) projection of the
#  normalised source-flux density dN/dS over [S_MIN, S_CUT], evaluated on the
#  same threshold ladder `main_sims_unlensed.py` used, plus the asymptotic
#  1/(eta(u)-1) that it converges to.  Seconds, not minutes.
S_CUT_ILLUSTRATIVE = 100.0
U_IDEAL = np.geomspace(0.3, 60.0, 34)


def _xi_flux_space(cnts, u_grid, s_min=S_MIN_DEFAULT,
                   s_max=S_CUT_ILLUSTRATIVE):
    """GPD shape of the raw flux distribution (no beam, no noise)."""
    d = np.geomspace(s_min, s_max, 6000)
    p = cnts.dnds(d)
    p = p / np.trapezoid(p, d)
    return np.asarray(xi_of_threshold_analytic(d, p, u_grid)[0])


if "2" in STAGES:
    print("\n[stage 2] ideal flux-space fingerprints", flush=True)
    t0 = time.time()
    models_ill = make_models(s_min=S_MIN_DEFAULT, s_max=S_CUT_ILLUSTRATIVE)
    BLOB["f2_u"] = U_IDEAL
    BLOB["f2_fit_range"] = np.asarray(FIT_RANGE_MJY, float)
    for nm in MODEL_ORDER:
        BLOB[f"f2_proj_{nm}"] = _xi_flux_space(models_ill[nm], U_IDEAL)
        BLOB[f"f2_asym_{nm}"] = np.asarray(
            models_ill[nm].xi_asymptotic(U_IDEAL), float)
        print(f"    {nm:>10s}  xi(1) = {BLOB[f'f2_proj_{nm}'][np.argmin(abs(U_IDEAL - 1))]:6.3f}"
              f"   xi(10) = {BLOB[f'f2_proj_{nm}'][np.argmin(abs(U_IDEAL - 10))]:6.3f}",
              flush=True)
    _stamp(t0, "stage 2 done")


# %% Section 3 -- stage 3: peak-level fingerprints at the SPIRE beam (Fig. 3)
if "3" in STAGES:
    print("\n[stage 3] peak-level fingerprints, 25.15in beam", flush=True)
    t0 = time.time()
    r3 = pfp.figure_fingerprints_peak()
    BLOB["f3_u"] = np.asarray(r3["u"], float)
    BLOB["f3_sigma_c"] = float(r3["sigma_c"])
    BLOB["f3_beam"] = float(pfp.HERSCHEL.beam)
    BLOB["f3_sigma_n"] = float(pfp.HERSCHEL.sigma_n)
    for nm in MODEL_ORDER:
        sc = r3["curves"][nm]
        BLOB[f"f3_xi_{nm}"] = np.asarray(sc["xi"], float)
        BLOB[f"f3_lo_{nm}"] = np.asarray(sc["lo"], float)
        BLOB[f"f3_hi_{nm}"] = np.asarray(sc["hi"], float)
        #  the asymptotic overlay, on the same axis the figure draws it
        BLOB[f"f3_asym_{nm}"] = np.asarray(
            1.0 / (pfp.MODELS[nm].eta(BLOB["f3_u"]) - 1.0), float)
    _stamp(t0, "stage 3 done")


# %% Section 4 -- stage 4: the beam sweep on the illustrative DPL (Fig. 4)
if "4" in STAGES:
    print("\n[stage 4] beam sweep, illustrative DPL", flush=True)
    t0 = time.time()
    r4 = pfp.figure_beam_reshapes()
    beams = sorted(r4["legs"])
    BLOB["f4_beams"] = np.asarray(beams, float)
    #  the illustrative DPL's asymptotes, quoted in the figure
    BLOB["f4_alpha"] = 1.6
    BLOB["f4_beta"] = 3.8
    for b in beams:
        leg = r4["legs"][b]
        key = f"{int(b * 10)}"
        BLOB[f"f4_u_{key}"] = np.asarray(leg["u"], float)
        BLOB[f"f4_xi_{key}"] = np.asarray(leg["scan"]["xi"], float)
        BLOB[f"f4_sigma_c_{key}"] = float(leg["sigma_c"])
        BLOB[f"f4_n_beam_{key}"] = float(leg["n_beam"])
    _stamp(t0, "stage 4 done")


# %% Section 4b -- stage 5: the Delta_xi validation, pixel and peak (Fig. 5)
#  Analytic routes (linearized master relation, exact-slope evaluation, exact
#  mu-mixture P(D)) and the paired Monte Carlo Delta_xi at pixel and peak level
#  for the benchmark lens, aperture r < 1.0 theta_500 (800 cutouts per arm).
if "5" in STAGES:
    print("\n[stage 5] Delta_xi validation, pixel and peak", flush=True)
    t0 = time.time()
    r5 = pfp.figure_dxi_validation(R=1.0)
    BLOB["f5_R"] = 1.0
    BLOB["f5_u"] = np.asarray(r5["u"], float)
    BLOB["f5_lin"] = np.asarray(r5["lin"], float)
    BLOB["f5_exact"] = np.asarray(r5["exact"], float)
    BLOB["f5_mix"] = np.asarray(r5["mix"], float)
    for conv in ("pix", "pk"):
        BLOB[f"f5_delta_{conv}"] = np.asarray(r5[f"mc_{conv}"]["delta"], float)
        BLOB[f"f5_sigma_{conv}"] = np.asarray(r5[f"mc_{conv}"]["sigma"], float)
    BLOB["f5_xi_pix_ctrl"] = np.asarray(r5["xi_pix_ctrl"], float)
    BLOB["f5_xi_pk_ctrl"] = np.asarray(r5["xi_pk_ctrl"], float)
    _stamp(t0, "stage 5 done")


# %% Section 5 -- stage 10: Herschel measured xi-hat + peak-level curves
if "10" in STAGES:
    print("\n[stage 10] Herschel xi-hat vs peak-level model curves",
          flush=True)
    t0 = time.time()
    r10 = pfp.figure_herschel()
    BLOB["f10_u_fine"] = np.asarray(r10["u_fine"], float)
    for nm in MODEL_ORDER:
        BLOB[f"f10_curve_{nm}"] = np.asarray(r10["curves"][nm], float)
        BLOB[f"f10_chi2_{nm}"] = float(r10["chi2_peak"][nm])
        if nm in r10["chi2_pix"]:
            BLOB[f"f10_chi2pix_{nm}"] = float(r10["chi2_pix"][nm])
    BLOB["f10_p"] = int(r10["p"])
    BLOB["f10_hartlap"] = float(r10["hartlap"])
    BLOB["f10_order"] = "|".join(r10["order_peak"])
    _stamp(t0, "stage 10 done")


# %% Section 6 -- write
if BLOB:
    #  Merge into any existing export so a partial re-run (CIB_EXPORT=3)
    #  updates one stage without discarding the others.
    if os.path.exists(OUT):
        old = dict(np.load(OUT, allow_pickle=False))
        old.update(BLOB)
        BLOB = old
        print(f"\nmerged into existing export ({len(old)} keys total)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **BLOB)
    print(f"\nwrote {OUT}")
    print(f"      {len(BLOB)} arrays, {os.path.getsize(OUT) / 1024:.0f} kB")
else:
    print("\nnothing selected; set CIB_EXPORT")
