"""
paper_figures_peaklevel.py — regenerate the three model-curve figures at PEAK level
=====================================================================================

Closes must-close item 5 of `00_Paper_Draft/revised_outline.md` Sec. 4
(framing decision D2): *every model curve shown against data or Monte Carlo
must be a simulated peak-level curve; `xi_of_threshold_analytic` output
appears in no such figure.*

WHY (the one-paragraph version)
--------------------------------
Two different distributions are in play.  The analytic machinery
(`gpd_tail.xi_of_threshold_analytic`) is the Kullback-Leibler projection of
the *pixel* P(D) onto the GPD family — the value an MLE would converge to
with infinite data drawn from PIXELS.  What we actually fit is the
*declustered peak* distribution: `beam_declustered_peaks` keeps only local
maxima over a one-FWHM window, which is not a thinning but an ORDER-STATISTIC
SELECTION.  A pixel survives only by beating its neighbours, so

    p_pk(D)  ~  p_pix(D) * [F_pix(D)]^(n_eff - 1)

(`GPD_beam_and_declustering_v4.md` Sec. 3.2).  The factor F^(n-1) crushes the
core, where F < 1, and tends to unity in the far tail: the core is displaced
upward, the far tail only rescaled.  The two distributions therefore have
genuinely different shapes at intermediate thresholds.

Measured size of the offset (peak-level MC minus pixel-level analytic,
Schechter, from results/{planck,herschel,ccat}.npz):

    Herschel  median -0.059,  max |offset| 0.085
    CCAT      median -0.030,  max |offset| 0.216
    Planck    median -0.044,  max |offset| 0.138

against a maximum three-model separation of 0.185 (CCAT, u = 18 mJy/beam).
The convention offset is a third to a half of the signal being claimed, so
plotting a pixel curve against a peak measurement can move a model from
"favoured" to "excluded" by bookkeeping alone.

It largely CANCELS in Delta_xi, because it is a property of the estimator
convention rather than of the lensing, and it appears on both sides of the
cluster-minus-control difference.  Figure 2 of this script tests that
cancellation rather than assuming it.

WHAT THIS SCRIPT PRODUCES
--------------------------
  fig_herschel_xi_vs_flux.png   draft Fig. 9  — the measured Herschel xi-hat(u)
                                with the three model curves at PEAK level, and
                                the covariance-correct ranking RECOMPUTED
                                against those curves.
  fig_dxi_validation.png        draft Fig. 4  — linearized vs exact analytic
                                Delta_xi against BOTH pixel- and peak-level MC,
                                which validates the master relation and the
                                cancellation claim in one panel.
  fig_xi_fingerprints.png       draft Fig. 2  — the three models' xi(u)
                                fingerprints as peak-level simulated curves at
                                a stated beam, replacing the idealized
                                no-beam-no-noise illustration.

Usage
-----
    python paper_figures_peaklevel.py            # all three
    CIB_FIGS=9 python paper_figures_peaklevel.py # a subset, comma-separated

Figures are written to figures/paper/ and are NOT copied into
00_Paper_Draft/Figures/ by this script — that is a deliberate manual step, so
the draft never changes under the author's feet.

Requires numpy, scipy, matplotlib, plus sim_core, counts_350um and
analysis_modules.
"""

# %% Section 1 — imports and configuration
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_core as sc                                              # noqa: E402
from sim_core import (Instrument, Benchmark, Run, S_MIN, make_models,  # noqa: E402
                      MODEL_ORDER, MODEL_COLORS, budget, simulator,
                      declustered_peaks, blocked_scan, xi_theory,
                      binned_mixture, build_ensemble)

FIG_DIR = os.path.join(_HERE, "figures", "paper")
os.makedirs(FIG_DIR, exist_ok=True)
WHICH = os.environ.get("CIB_FIGS", "9,4,2").split(",")

HERSCHEL_NPZ = os.path.join(_ROOT, "Herschel_analysis", "results",
                            "herschel_unlensed_v3_350.npz")
SIM_HERSCHEL_NPZ = os.path.join(_HERE, "results", "herschel.npz")


def _save(fig, name):
    p = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# %% Section 2 — the Herschel instrument (verbatim from simulate_herschel.py)
HERSCHEL = Instrument(
    name="Herschel", beam_arcsec=25.15, pix_arcsec=8.0, sigma_n=5.402,
    sigma_nonp=0.0, s_cut=100.0,
    u_grid=np.array([25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 65.0, 75.0]),
    core_measured=9.003, apertures_theta500=(0.5, 1.0, 1.5, 2.0),
    npix_unlensed=256, n_maps_unlensed=80, n_clusters=300,
    note="peak-level paper figures")

MODELS = make_models(s_min=S_MIN, s_max=HERSCHEL.s_cut)
BUDS = {nm: budget(MODELS[nm], HERSCHEL) for nm in MODEL_ORDER}


# %% Section 3 — Figure 9: Herschel xi-hat(u) with peak-level model curves
def figure_herschel(u_fine=None, n_maps=200, seed=31000):
    """
    Draft Fig. 9.  The measured GAMA-09 xi-hat(u), masked and unmasked, with
    the three count models drawn as PEAK-LEVEL simulated curves.

    The bright-masked measurement is the model-comparable one: the simulation
    truncates the counts at S_cut = 100 mJy, which is exactly what the bright
    mask removes from the data (`summary_three_experiments.md` Sec. 5).  The
    unmasked curve is shown because it brackets the comparison from above and
    because its rise is itself a result (the lensed bright population).

    The model curves are simulated on a FINER threshold grid than the
    production run's 9-point ladder, purely so the curve is smooth; each
    point is an independent GPD fit to the same declustered-peak ensemble, so
    adjacent points are strongly correlated by construction and no error bar
    on the CURVE is implied.  Errors shown are the DATA's.
    """
    print("\n[Fig 9] Herschel xi-hat(u) with peak-level model curves")
    if not os.path.exists(HERSCHEL_NPZ):
        print(f"  SKIP — measured data not found: {HERSCHEL_NPZ}")
        return None
    D = np.load(HERSCHEL_NPZ, allow_pickle=True)
    u_meas = D["u_grid"]
    xi_meas, xi_err = D["xi_meas"], D["xi_err"]
    xi_mask, xi_mask_err = D["xi_masked"], D["xierr_masked"]
    u_max_masked = float(D["u_max_masked"])
    boot_masked = D["boot_masked"]                 # (n_boot, n_thr)

    #  ---- peak-level model curves, simulated on a fine grid ---------------
    u_fine = np.arange(25.0, 76.0, 2.5) if u_fine is None else u_fine
    inst = Instrument(
        name="Herschel", beam_arcsec=HERSCHEL.beam, pix_arcsec=HERSCHEL.pix,
        sigma_n=HERSCHEL.sigma_n, sigma_nonp=0.0, s_cut=HERSCHEL.s_cut,
        u_grid=u_fine, core_measured=HERSCHEL.core_measured,
        apertures_theta500=HERSCHEL.apertures_theta500,
        npix_unlensed=HERSCHEL.npix_unlensed, n_maps_unlensed=n_maps,
        n_clusters=HERSCHEL.n_clusters, note="fine-grid peak-level curves")
    run = Run("paperfig_herschel", config_key=(
        inst.beam, inst.pix, inst.sigma_n, inst.s_cut, tuple(u_fine),
        inst.npix_unlensed, n_maps, S_MIN, seed))

    curves, area = {}, inst.map_area_deg2(inst.npix_unlensed) * n_maps
    print(f"  simulating {n_maps} maps of {inst.npix_unlensed}^2 "
          f"= {area:.1f} deg^2 per model, {u_fine.size} thresholds")
    for nm in MODEL_ORDER:
        sim = simulator(MODELS[nm], inst, BUDS[nm], inst.npix_unlensed)
        bfp = inst.beam_fwhm_pix

        def one(i, sim=sim, bfp=bfp):
            return declustered_peaks(sim.make_map(seed=seed + 7919 * i), bfp)

        peaks = build_ensemble(run, f"pkcurve_{nm}", one, n_maps)
        scan = run.cached(f"pkscan_{nm}", blocked_scan, peaks, u_fine,
                          n_boot=120, min_exceed=80, seed=sc.stable_seed(nm))
        curves[nm] = scan["xi"]
        npk = sum(p.size for p in peaks)
        print(f"    {nm:>10s}  {npk:8d} peaks   "
              f"xi(25) = {scan['xi'][0]:+.4f}   xi(50) = "
              f"{scan['xi'][np.argmin(abs(u_fine - 50))]:+.4f}")

    #  ---- covariance-correct ranking, RECOMPUTED at peak level -------------
    #  The published ranking used pixel-level analytic curves; switching the
    #  curves changes chi^2, so the ranking must be re-derived, not carried.
    sel = (u_meas <= u_max_masked) & np.isfinite(xi_mask)
    p = int(sel.sum())
    Bm = boot_masked[:, sel]
    good = np.isfinite(Bm).all(axis=1)
    C = np.cov(Bm[good], rowvar=False)
    n_boot = int(good.sum())
    hartlap = (n_boot - p - 2) / (n_boot - 1)      # Hartlap et al. (2007)
    Cinv = np.linalg.inv(C) * hartlap
    print(f"  covariance: {p} thresholds, {n_boot} usable replicates, "
          f"Hartlap = {hartlap:.4f}")

    chi2_peak, chi2_pix = {}, {}
    for nm in MODEL_ORDER:
        mod_pk = np.interp(u_meas[sel], u_fine, curves[nm])
        r = xi_mask[sel] - mod_pk
        chi2_peak[nm] = float(r @ Cinv @ r)
        key = {"Schechter": "xi_sch", "SPL": "xi_spl", "DPL": "xi_dpl"}[nm]
        if key in D.files:
            r0 = xi_mask[sel] - np.asarray(D[key], float)[sel]
            chi2_pix[nm] = float(r0 @ Cinv @ r0)

    order_pk = sorted(MODEL_ORDER, key=lambda n: chi2_peak[n])
    print(f"  chi^2 (peak-level curves, {p} dof): "
          + ", ".join(f"{n} {chi2_peak[n]:.1f}" for n in MODEL_ORDER))
    if chi2_pix:
        order_px = sorted(chi2_pix, key=lambda n: chi2_pix[n])
        print(f"  chi^2 (pixel-level, as published): "
              + ", ".join(f"{n} {chi2_pix[n]:.1f}" for n in MODEL_ORDER))
        print(f"  ranking  peak: {' < '.join(order_pk)}")
        print(f"  ranking pixel: {' < '.join(order_px)}"
              + ("   [UNCHANGED]" if order_pk == order_px else
                 "   *** RANKING CHANGES ***"))

    #  ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for nm in MODEL_ORDER:
        ax.plot(u_fine, curves[nm], color=MODEL_COLORS[nm], lw=2.2,
                alpha=0.9, zorder=2,
                label=fr"{nm} (peak-level sim), $\chi^2$={chi2_peak[nm]:.0f}")
    ax.errorbar(u_meas, xi_meas, yerr=xi_err, fmt="s", ms=4.6, color="k",
                capsize=2.5, lw=1.2, zorder=4, label="measured, unmasked")
    ax.errorbar(u_meas[sel], xi_mask[sel], yerr=xi_mask_err[sel], fmt="o",
                ms=5.0, mfc="w", color="k", capsize=2.5, lw=1.4, zorder=5,
                label=r"measured, bright-masked ($S>100$ mJy removed)")
    ax.axvspan(u_max_masked, u_meas.max() * 1.03, color="0.92", zorder=0)
    ax.text(u_max_masked * 1.02, ax.get_ylim()[1],
            " masked variant\n censored", fontsize=7.6, color="0.35",
            va="top", ha="left")
    ax.axhline(0.0, color="0.4", lw=1.0, zorder=1)
    ax.set(xlabel=r"threshold $u$ [mJy/beam]", ylabel=r"$\hat\xi(u)$",
           xlim=(u_meas.min() * 0.96, u_meas.max() * 1.03))
    ax.set_title("Herschel/SPIRE 350 µm, GAMA-09: measured tail shape\n"
                 "against peak-level simulated count models", fontsize=11)
    ax.legend(fontsize=8.2, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = _save(fig, "fig_herschel_xi_vs_flux")
    return dict(path=path, chi2_peak=chi2_peak, chi2_pix=chi2_pix,
                order_peak=order_pk, u_fine=u_fine, curves=curves,
                p=p, hartlap=hartlap)


# %% Section 4 — Figure 4: the Delta_xi validation, pixel AND peak
def figure_dxi_validation(R=1.0, n_clusters=400, seed=42000):
    """
    Draft Fig. 4.  Validation of the master relation, upgraded.

    Three theory curves and TWO Monte Carlo measurements:

      linearized   Delta_xi = -ln(mu) * kappa / (eta - 1)^2   [manual Sec. 13.2]
      exact        1/(eta(S_u/mu) - 1) - 1/(eta(S_u) - 1)     [manual v4 Sec.]
      mixture      the full mu-mixture P(D) difference, KL-projected (pixel)
      MC pixel     simulated, xi fitted to raw PIXELS
      MC peak      simulated, xi fitted to DECLUSTERED PEAKS

    Showing both Monte Carlo conventions is the point.  The unlensed xi(u)
    differs between them by 0.03-0.09 (see the module docstring), yet
    Delta_xi should not, because the selection offset is common to the lensed
    and control arms and cancels in the difference.  This panel therefore
    validates the master relation AND the cancellation that licenses every
    other differential result in the paper — the claim that framing decision
    D2 rests on, tested rather than asserted.
    """
    print("\n[Fig 4] Delta_xi validation at pixel and peak level")
    bench = Benchmark()
    inst = HERSCHEL
    #  Truncated at 55 mJy/beam: above it the PEAK arm falls to ~84 exceedances
    #  against min_exceed = 80, i.e. into the conditionally-biased regime that
    #  the bootstrap guard exists to catch.  Points there are not quotable.
    u = np.array([25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    model, bud = MODELS["Schechter"], BUDS["Schechter"]

    mus, w, _ = bench.disc_mixture(R)
    mm, ww = binned_mixture(mus, w, n_bins=64)
    st = bench.disc_stats(R)
    mu_bar, lnmu_bar = st["mean_mu"], st["mean_ln_mu"]
    print(f"  aperture r < {R} theta_500: <mu> = {mu_bar:.4f}, "
          f"<ln mu> = {lnmu_bar:.4f}")

    #  ---- analytic curves ---------------------------------------------------
    eta = model.eta(u)
    dln = 1e-3
    kappa = -(model.eta(u * (1 + dln)) - model.eta(u * (1 - dln))) \
        / (np.log(1 + dln) - np.log(1 - dln))       # kappa = -d(eta)/d(ln S)
    dxi_lin = -lnmu_bar * kappa / (eta - 1.0) ** 2
    dxi_exact = 1.0 / (model.eta(u / mu_bar) - 1.0) - 1.0 / (eta - 1.0)
    dxi_mix = (xi_theory(model, inst, u, (mm, ww))
               - xi_theory(model, inst, u, 1.0))

    #  ---- Monte Carlo, both conventions -------------------------------------
    run = Run("paperfig_dxival", config_key=(
        inst.beam, inst.pix, inst.sigma_n, inst.s_cut, tuple(u), R,
        n_clusters, S_MIN, seed))
    r_max = R * bench.theta500
    npix = inst.npix_for_aperture(r_max)
    sim = simulator(model, inst, bud, npix)
    bfp = inst.beam_fwhm_pix
    print(f"  MC: {2 * n_clusters} cutouts of {npix}^2 at {inst.pix}\" "
          f"({npix * inst.pix / 60:.1f}')")

    from scipy.ndimage import maximum_filter

    def one(i, lens=None, off=0):
        m = sim.make_map(lens=lens, seed=seed + off + 104729 * i)
        cy = (m.shape[0] - 1) / 2.0
        yy, xx = np.indices(m.shape)
        th = np.hypot(yy - cy, xx - cy) * inst.pix / 60.0
        inap = th < r_max
        a = np.where(np.isfinite(m), m, -np.inf)
        size = max(1, int(round(bfp)))
        ismax = (a == maximum_filter(a, size=size)) & np.isfinite(m)
        return {"pix": m[inap], "pk": m[ismax & inap]}

    arms = {}
    for arm, lens, off in (("lens", bench.lens, 0), ("ctrl", None, 500_000)):
        arms[arm] = build_ensemble(run, f"dxival_{arm}",
                                   lambda i, l=lens, o=off: one(i, l, o),
                                   n_clusters)

    #  CRITICAL: `blocked_scan` silently thins any pooled sample above
    #  MAX_POOLED_PEAKS = 60_000.  The peak arm here holds ~18 500 values and is
    #  untouched; the PIXEL arm holds ~829 000 and would be thinned to 7%,
    #  which drops it to ~100 exceedances at u = 45 against min_exceed = 80 --
    #  i.e. straight into the conditional-bias regime, where the surviving fits
    #  are those that happened to draw a heavy tail.  Left at the default the
    #  pixel curve came out at Delta_xi = +0.32 at u = 45, with the wrong sign
    #  against the analytic; unthinned it tracks the analytic to 0.01-0.03.
    #  Thinning is unbiased in principle -- it is an i.i.d. subsample of the
    #  same marginal -- so this is a PRECISION failure that becomes a BIAS only
    #  through the exceedance floor.  Do not lower this cap.
    CAP = {"pix": 2_000_000, "pk": sc.MAX_POOLED_PEAKS}
    out, scans = {}, {}
    for conv in ("pix", "pk"):
        sc_ = {}
        for arm in ("lens", "ctrl"):
            sc_[arm] = run.cached(
                f"dxival_scan2_{conv}_{arm}", blocked_scan,
                [a[conv] for a in arms[arm]], u, n_boot=200,
                min_exceed=80, seed=17, max_pooled=CAP[conv])
        scans[conv] = sc_
        d = sc.paired_delta(sc_["lens"], sc_["ctrl"])
        out[conv] = d
        print(f"  MC {conv:>3s}  n_exc(ctrl): "
              + " ".join(f"{v:6d}" for v in sc_["ctrl"]["n_exc"]))
        print(f"  MC {conv:>3s}  Delta_xi   : "
              + " ".join(f"{v:+6.4f}" for v in d["delta"]))

    #  ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(u, dxi_lin, "-", color="C3", lw=2.0,
            label=r"linearized: $-\ln\mu\,\kappa/(\eta-1)^2$")
    ax.plot(u, dxi_exact, "--", color="C1", lw=2.0,
            label=r"exact slope: $1/(\eta(S_u/\mu)-1)-1/(\eta(S_u)-1)$")
    ax.plot(u, dxi_mix, ":", color="C0", lw=2.4,
            label=r"$\mu$-mixture $P(D)$ (pixel, KL)")
    for conv, mk, c, lab in (("pix", "s", "0.45", "MC, pixel-level"),
                             ("pk", "o", "k", "MC, peak-level")):
        d = out[conv]
        g = np.isfinite(d["delta"])
        ax.errorbar(u[g] * (1 + 0.004 * (conv == "pk")), d["delta"][g],
                    yerr=d["sigma"][g], fmt=mk, ms=5.0, color=c,
                    mfc="w" if conv == "pix" else c,
                    capsize=2.6, lw=1.3, label=lab, zorder=5)
    ax.axhline(0.0, color="0.5", lw=1.0)
    ax.set(xlabel=r"threshold $u$ [mJy/beam]", ylabel=r"$\Delta\xi(u)$")
    ax.set_title(rf"Benchmark lens, $r<{R}\,\theta_{{500}}$ "
                 rf"($\langle\mu\rangle={mu_bar:.2f}$), Schechter",
                 fontsize=10.5)
    ax.legend(fontsize=8.0)
    ax.grid(alpha=0.25)

    #  right panel: the cancellation, made explicit
    ax = axes[1]
    xi_pix_c = np.asarray(scans["pix"]["ctrl"]["xi"], float)
    xi_pk_c = np.asarray(scans["pk"]["ctrl"]["xi"], float)
    ax.plot(u, xi_pix_c - xi_pk_c, "d-", color="C4", lw=1.8, ms=5.5,
            label=r"unlensed: $\xi_{\rm pix}-\xi_{\rm pk}$")
    gg = np.isfinite(out["pix"]["delta"]) & np.isfinite(out["pk"]["delta"])
    ax.errorbar(u[gg], (out["pix"]["delta"] - out["pk"]["delta"])[gg],
                yerr=np.hypot(out["pix"]["sigma"], out["pk"]["sigma"])[gg],
                fmt="o", color="k", ms=5.0, capsize=2.6, lw=1.3,
                label=r"differential: $\Delta\xi_{\rm pix}-\Delta\xi_{\rm pk}$")
    ax.axhline(0.0, color="0.5", lw=1.0)
    ax.set(xlabel=r"threshold $u$ [mJy/beam]",
           ylabel="pixel $-$ peak convention")
    ax.set_title("The convention offset cancels in the difference",
                 fontsize=10.5)
    ax.legend(fontsize=8.4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = _save(fig, "fig_dxi_validation")
    return dict(path=path, u=u, lin=dxi_lin, exact=dxi_exact, mix=dxi_mix,
                mc_pix=out["pix"], mc_pk=out["pk"],
                xi_pix_ctrl=xi_pix_c, xi_pk_ctrl=xi_pk_c)


# %% Section 5 — Figure 2: peak-level model fingerprints
def figure_fingerprints(n_maps=200, seed=77000):
    """
    Draft Fig. 2.  The three models' xi(u) fingerprints, PEAK-LEVEL.

    NOTE THE CHANGE OF CHARACTER.  The figure this replaces was an idealized
    illustration with no beam and no noise, whose job was pedagogical: a pure
    power law gives a flat plateau, a Schechter cutoff a smooth decline, a
    double power law a step.  A peak-level simulated version is a physical
    PREDICTION at a stated beam and noise level, and the clean textbook
    shapes are partly obscured by the confusion core — which is itself the
    honest message, and is the same statement Sect. 2.3 makes about the
    usable window.  Both panels are therefore drawn: idealized shapes on the
    left as the concept, peak-level simulated curves on the right as what an
    instrument actually delivers.
    """
    print("\n[Fig 2] xi(u) fingerprints at peak level")
    #  left panel: the idealized fingerprint, xi = 1/(eta-1), no beam or noise
    s_ideal = np.logspace(np.log10(2.0), np.log10(200.0), 240)
    #  right panel: peak-level simulation at the Herschel configuration
    u_fine = np.concatenate([np.arange(20.0, 60.0, 2.5),
                             np.arange(60.0, 91.0, 5.0)])
    inst = Instrument(
        name="Herschel", beam_arcsec=HERSCHEL.beam, pix_arcsec=HERSCHEL.pix,
        sigma_n=HERSCHEL.sigma_n, sigma_nonp=0.0, s_cut=HERSCHEL.s_cut,
        u_grid=u_fine, core_measured=HERSCHEL.core_measured,
        apertures_theta500=HERSCHEL.apertures_theta500,
        npix_unlensed=HERSCHEL.npix_unlensed, n_maps_unlensed=n_maps,
        n_clusters=HERSCHEL.n_clusters, note="fingerprints")
    run = Run("paperfig_fingerprints", config_key=(
        inst.beam, inst.pix, inst.sigma_n, inst.s_cut, tuple(u_fine),
        inst.npix_unlensed, n_maps, S_MIN, seed))

    curves = {}
    for nm in MODEL_ORDER:
        sim = simulator(MODELS[nm], inst, BUDS[nm], inst.npix_unlensed)
        bfp = inst.beam_fwhm_pix

        def one(i, sim=sim, bfp=bfp):
            return declustered_peaks(sim.make_map(seed=seed + 7919 * i), bfp)

        peaks = build_ensemble(run, f"fp_{nm}", one, n_maps)
        scan = run.cached(f"fpscan_{nm}", blocked_scan, peaks, u_fine,
                          n_boot=120, min_exceed=80, seed=sc.stable_seed(nm))
        curves[nm] = scan
        print(f"    {nm:>10s}  xi(25) = {scan['xi'][2]:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    ax = axes[0]
    for nm in MODEL_ORDER:
        ax.plot(s_ideal, 1.0 / (MODELS[nm].eta(s_ideal) - 1.0),
                color=MODEL_COLORS[nm], lw=2.2, label=nm)
    ax.axhline(0.0, color="0.5", lw=1.0)
    ax.set(xscale="log", xlabel=r"flux $S$ [mJy]",
           ylabel=r"$\xi = 1/(\eta(S)-1)$", ylim=(-0.35, 0.85))
    ax.set_title("Idealized fingerprint: no beam, no noise\n"
                 "(flat plateau / smooth decline / step)", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for nm in MODEL_ORDER:
        s_ = curves[nm]
        g = np.isfinite(s_["xi"])
        ax.plot(u_fine[g], s_["xi"][g], "-", color=MODEL_COLORS[nm], lw=2.2,
                label=nm)
        ax.fill_between(u_fine[g], s_["lo"][g], s_["hi"][g],
                        color=MODEL_COLORS[nm], alpha=0.16, lw=0)
    ax.axhline(0.0, color="0.5", lw=1.0)
    sig_c = BUDS["Schechter"]["sigma_c"]
    ax.axvspan(u_fine.min(), 3.0 * sig_c, color="0.90", zorder=0)
    ax.axvline(3.0 * sig_c, color="crimson", ls="--", lw=1.3)
    ax.text(3.0 * sig_c * 1.02, ax.get_ylim()[1], r" $u=3\sigma_c$",
            fontsize=8.4, color="crimson", va="top")
    ax.set(xlabel=r"threshold $u$ [mJy/beam]", ylabel=r"$\hat\xi(u)$")
    ax.set_title(f"Peak-level simulation, {inst.beam:.1f}$''$ beam, "
                 fr"$\sigma_N$={inst.sigma_n:.1f} mJy/beam" "\n"
                 "(what an instrument actually delivers)", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = _save(fig, "fig_xi_fingerprints")
    return dict(path=path, u=u_fine, curves=curves)


# %% Section 5b — Figure 2b: peak-level fingerprints with 1/(eta-1) overlays
def figure_fingerprints_peak(n_maps=200, seed=88000, beam=25.15):
    """
    Draft Fig. 2b.  The three FITTED count models at peak level, one beam,
    with the asymptotic 1/(eta(S)-1) curves overlaid.

    WHAT THE OVERLAY MEANS, AND ITS ONE CAVEAT.  For a pure power-law tail of
    index eta, extreme-value theory gives exactly xi = 1/(eta-1).  That
    identity is the bridge from a statistic of the MAP to a property of the
    COUNTS, and it is the reason xi(u) carries astrophysical information at
    all.  Overlaying it therefore shows how much of the ideal count-slope
    signal survives the beam, the confusion, the noise and the declustering:
    the vertical gap is the whole degradation budget.

    The caveat is that the overlay lives on a SOURCE-FLUX axis while the
    simulated curve lives on a MAP-THRESHOLD axis, and the identification
    u <-> S_u is exact only for isolated sources.  At N_beam ~ 5 a map value
    is a sum over several sources and the correspondence blurs -- which is
    the same effect that makes the linearized master relation fail in
    `figure_dxi_validation`.  The overlay is therefore a reference curve, not
    a prediction, and is drawn dotted for that reason.

    Companion to Fig. 2a (the legacy no-beam, no-noise theory figure), which
    is kept unchanged: there the solid curves are the GPD projection of the
    ideal flux distribution, so the solid-vs-dotted gap is the
    finite-threshold correction alone, with no instrumental content.
    """
    print("\n[Fig 2b] peak-level fingerprints with 1/(eta-1) overlays")
    u_fine = np.concatenate([np.arange(15.0, 60.0, 2.5),
                             np.arange(60.0, 96.0, 5.0)])
    inst = Instrument(
        name="Herschel", beam_arcsec=beam, pix_arcsec=HERSCHEL.pix,
        sigma_n=HERSCHEL.sigma_n, sigma_nonp=0.0, s_cut=HERSCHEL.s_cut,
        u_grid=u_fine, core_measured=HERSCHEL.core_measured,
        apertures_theta500=HERSCHEL.apertures_theta500,
        npix_unlensed=HERSCHEL.npix_unlensed, n_maps_unlensed=n_maps,
        n_clusters=HERSCHEL.n_clusters, note="fingerprints, peak level")
    run = Run("paperfig_fp2b", config_key=(
        inst.beam, inst.pix, inst.sigma_n, inst.s_cut, tuple(u_fine),
        inst.npix_unlensed, n_maps, S_MIN, seed))

    curves = {}
    for nm in MODEL_ORDER:
        sim = simulator(MODELS[nm], inst, BUDS[nm], inst.npix_unlensed)
        bfp = inst.beam_fwhm_pix

        def one(i, sim=sim, bfp=bfp):
            return declustered_peaks(sim.make_map(seed=seed + 7919 * i), bfp)

        peaks = build_ensemble(run, f"fp2b_{nm}", one, n_maps)
        curves[nm] = run.cached(f"fp2bscan_{nm}", blocked_scan, peaks, u_fine,
                                n_boot=120, min_exceed=80,
                                seed=sc.stable_seed(nm),
                                max_pooled=5_000_000)
        print(f"    {nm:>10s}  {sum(p.size for p in peaks):8d} peaks")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    s_ax = np.logspace(np.log10(u_fine.min()), np.log10(u_fine.max()), 250)
    for nm in MODEL_ORDER:
        c = MODEL_COLORS[nm]
        ax.plot(s_ax, 1.0 / (MODELS[nm].eta(s_ax) - 1.0), ":", color=c,
                lw=1.9, alpha=0.95,
                label=fr"{nm}: $1/(\eta(S)-1)$ (asymptotic)")
        sn = curves[nm]
        g = np.isfinite(sn["xi"])
        ax.plot(u_fine[g], sn["xi"][g], "-", color=c, lw=2.3,
                label=f"{nm}: peak-level, declustered")
        ax.fill_between(u_fine[g], sn["lo"][g], sn["hi"][g], color=c,
                        alpha=0.15, lw=0)
    sig_c = BUDS["Schechter"]["sigma_c"]
    ax.axvspan(u_fine.min(), 3.0 * sig_c, color="0.90", zorder=0)
    ax.axvline(3.0 * sig_c, color="crimson", ls="--", lw=1.3, zorder=1)
    ax.text(3.0 * sig_c * 1.03, ax.get_ylim()[1],
            r" $u=3\sigma_c$" "\n validity floor", fontsize=8.2,
            color="crimson", va="top")
    ax.axhline(0.0, color="0.45", lw=1.0)
    ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]  "
                                r"(overlays: source flux $S$ [mJy])",
           ylabel=r"$\xi$")
    ax.set_title(f"Peak-level fingerprints, {beam:.2f}$''$ beam, "
                 fr"$\sigma_N$ = {inst.sigma_n:.2f} mJy/beam"
                 "\nsolid: declustered simulation   dotted: asymptotic "
                 r"$1/(\eta-1)$", fontsize=10.5)
    ax.legend(fontsize=8.0, ncol=1, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path = _save(fig, "fig_xi_fingerprints_peak")
    return dict(path=path, u=u_fine, curves=curves, sigma_c=sig_c)


# %% Section 5c — Figure 3: what the beam does, at peak level
def figure_beam_reshapes(n_maps=160, seed=99000,
                         beams=(5.0, 10.0, 20.0, 30.0)):
    """
    Draft Fig. 3, rebuilt at PEAK level and reduced to two panels.

    The legacy version plotted `xi_of_threshold_analytic` -- the KL
    projection of the PIXEL P(D) -- which is exactly the quantity framing
    decision D2 excludes from model-curve figures, and it is doubly wrong
    here because this is the figure whose subject IS the instrumental
    reshaping.  Both panels are now declustered peak-level simulations.

      (a) xi-hat(u) versus absolute threshold, one curve per beam.
      (b) the same curves versus k = u/sigma_c, which is the statement that
          the confusion core sets the scale of the low-u suppression: if the
          curves collapse, sigma_c is the correct scaling variable.

    Count model: the ILLUSTRATIVE DPL (alpha = 1.6, beta = 3.8, S_* = 4 mJy),
    not one of the three fitted models.  Its job here is morphological -- it
    has clean asymptotes 1/(alpha-1) = 1.667 and 1/(beta-1) = 0.357 and an
    unmistakable break, so the question "does the beam make xi rise or fall"
    is answered visually.  The fitted models carry the quantitative work in
    Figs. 2a and 2b.  sigma_N = 0 throughout, so panel (a) isolates the beam.
    """
    print("\n[Fig 3] beam reshaping at peak level (illustrative DPL)")
    from analysis_modules.counts import DoublePowerLaw
    #  S_min = 1e-4 mJy, NOT the project's S_MIN = 1.0.  The R1 convention
    #  applies to the three fitted 350 um models, where s_min is a physical
    #  assumption that propagates into sigma_c and the monopole.  The
    #  illustrative DPL is a pedagogical model defined down to sub-mJy (its
    #  own constructor default, tuned against Fujimoto+23 at 0.1-0.5 mJy),
    #  and it needs that range here: N_beam counts sources above s_min, and
    #  truncating at 1 mJy gives N_beam = 0.01-0.24, i.e. the ISOLATED-source
    #  regime at every beam, in which the figure has nothing to show.  With
    #  the native range N_beam spans ~0.8 to ~28 across 5"-30" and the curves
    #  cross the confusion transition, which is the figure's whole subject.
    #  sigma_c is nearly unaffected either way (it is set by the bright end),
    #  so this choice changes the source COUNT, not the noise budget.
    S_MIN_ILL = 1e-4
    dpl = DoublePowerLaw.illustrative(alpha=1.6, beta=3.8, sstar_mJy=4.0,
                                      s_min=S_MIN_ILL, s_max=100.0)
    alpha, beta_ = 1.6, 3.8
    out = {}
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for k, beam in enumerate(beams):
        pix = beam / 5.0
        inst = Instrument(name="ill", beam_arcsec=beam, pix_arcsec=pix,
                          sigma_n=0.0, sigma_nonp=0.0, s_cut=100.0,
                          u_grid=np.array([1.0]), npix_unlensed=256,
                          n_maps_unlensed=n_maps, n_clusters=10,
                          note="beam sweep")
        bud = budget(dpl, inst, s_min=S_MIN_ILL)
        sig_c = bud["sigma_c"]
        #  Stop at 14 sigma_c: beyond it the declustered exceedance count
        #  falls through `min_exceed` and the curve breaks up into
        #  conditionally-biased survivors rather than tracking anything.
        u = np.logspace(np.log10(0.8 * sig_c), np.log10(14.0 * sig_c), 20)
        inst.u_grid = u
        run = Run(f"paperfig_beam{int(beam * 10)}", config_key=(
            beam, pix, 0.0, 100.0, tuple(np.round(u, 6)), 256, n_maps,
            S_MIN_ILL, seed))
        sim = simulator(dpl, inst, bud, 256)
        bfp = inst.beam_fwhm_pix

        def one(i, sim=sim, bfp=bfp):
            return declustered_peaks(sim.make_map(seed=seed + 7919 * i), bfp)

        peaks = build_ensemble(run, f"beam_{int(beam * 10)}", one, n_maps)
        scan = run.cached(f"beamscan_{int(beam * 10)}", blocked_scan, peaks,
                          u, n_boot=100, min_exceed=80, seed=5,
                          max_pooled=5_000_000)
        out[beam] = dict(u=u, scan=scan, sigma_c=sig_c,
                         n_beam=bud["n_beam"])
        c = cmap(k / max(1, len(beams) - 1))
        g = np.isfinite(scan["xi"])
        lab = (fr"{beam:.0f}$''$  ($\sigma_c$={sig_c:.2f}, "
               fr"$N_b$={bud['n_beam']:.1f})")
        axes[0].plot(u[g], scan["xi"][g], "-o", color=c, lw=2.0, ms=3.6,
                     label=lab)
        axes[1].plot((u / sig_c)[g], scan["xi"][g], "-o", color=c, lw=2.0,
                     ms=3.6, label=lab)
        print(f"    {beam:5.1f}\"  sigma_c={sig_c:6.3f}  N_beam="
              f"{bud['n_beam']:6.2f}  {sum(p.size for p in peaks):7d} peaks")

    for ax, xlab in ((axes[0], r"threshold $u$ [mJy/beam]"),
                     (axes[1], r"$k = u/\sigma_c$")):
        ax.axhline(1.0 / (beta_ - 1.0), color="0.35", ls="--", lw=1.3)
        ax.axhline(0.0, color="0.6", lw=0.9)
        ax.set(xscale="log", xlabel=xlab, ylabel=r"$\hat\xi(u)$",
               ylim=(0.0, 0.78))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.8, title="FWHM", title_fontsize=8,
                  loc="upper right")
    axes[0].text(0.03, 1.0 / (beta_ - 1.0), r" $1/(\beta-1)=0.357$",
                 fontsize=8, color="0.3", va="bottom",
                 transform=axes[0].get_yaxis_transform())
    #  The bright-end asymptote is off-scale, and saying so IS the
    #  declustering statement: the peak convention compresses the dynamic
    #  range relative to the pixel P(D) the legacy figure plotted.
    axes[0].text(0.03, 0.955, r"asymptotic $1/(\alpha-1)=1.667$ lies "
                 "off-scale:\ndeclustering compresses the range",
                 fontsize=7.8, color="0.3", va="top",
                 transform=axes[0].transAxes)
    axes[0].set_title("(a) the beam decides whether "
                      r"$\hat\xi(u)$ falls or rises"
                      "\npeak-level, illustrative DPL, $\\sigma_N=0$",
                      fontsize=10.4)
    axes[1].set_title(r"(b) the same curves versus $u/\sigma_c$:"
                      "\nthe confusion core sets the scale",
                      fontsize=10.4)
    fig.tight_layout()
    path = _save(fig, "fig_beam_reshapes_xi")
    return dict(path=path, legs=out)


# %% Section 6 — main
if __name__ == "__main__":
    RES = {}
    if "9" in WHICH:
        RES["fig9"] = figure_herschel()
    if "4" in WHICH:
        RES["fig4"] = figure_dxi_validation()
    if "2" in WHICH:
        RES["fig2"] = figure_fingerprints()
    if "2b" in WHICH:
        RES["fig2b"] = figure_fingerprints_peak()
    if "3" in WHICH:
        RES["fig3"] = figure_beam_reshapes()
    print("\nDone. Figures in", FIG_DIR)
    print("They are NOT copied into 00_Paper_Draft/Figures/ — do that "
          "manually once the text is ready.")
