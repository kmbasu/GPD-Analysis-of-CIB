#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py -- build all figures of the paper at final physical size
===============================================================================

Reproduces Figs. 1-14, B.1 and E.1 of

    K. Basu, A. Guerrero & F. Bertoldi (2026), "Probing submillimeter number
    counts below the confusion limit: extreme-value statistics of the P(D)
    distribution and its modulation by gravitational lensing",
    arXiv:2609.19689

from the result files shipped with this repository.  No Planck or Herschel map
data and no Monte Carlo re-run is needed: every figure reads the `.npz`/`.json`
products that the analysis and simulation scripts write into their `results/`
folders.  The one exception is Fig. B.1, which re-runs a small, seeded,
self-contained simulation (about a minute) and caches it in
`paper/figures/.cache_figB1.npz`.

Figures are drawn at their A&A placement size (single column 90 mm = 3.543 in,
double column 184 mm = 7.244 in) with 7 pt lettering, so they are included in
the .tex at scale 1.0.

Inputs (all relative to the repository root)
--------------------------------------------
    num_count_fits/num_count_fit_results.json                         Fig. 1
    Simulations/results/paperfig_peaklevel_curves.npz        Figs. 2-5, 10
    Planck_analysis/results/planck_v2_unlensed_857_4.0e+20_gp40.npz Figs. 6-7
    Planck_analysis/results/planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz 8-9
    Herschel_analysis/results/herschel_unlensed_v3_350.npz           Fig. 10
    Herschel_analysis/results/herschel_peak_sims_350.npz             Fig. 11
    Herschel_analysis/results/herschel_lensed_350.npz                Fig. 12
    Simulations/results/ccat.npz                                  Figs. 13-14
    Simulations/results/herschel_null_seeds.json, herschel.npz      Fig. E.1

Output: `paper/figures/<name>.pdf` (the file names used in the .tex); set
`CIB_PAPER_FIGDIR` to write elsewhere.

Usage
-----
    python paper/make_paper_figures.py                        # all figures
    CIB_PAPER_FIGS=6,7 python paper/make_paper_figures.py     # a subset
    CIB_PREVIEW=1 python paper/make_paper_figures.py          # + PNG previews
    CIB_USETEX=1 python paper/make_paper_figures.py           # typeset with LaTeX

Requires numpy and matplotlib; Fig. B.1 additionally needs scipy and
`analysis_modules/`.  The per-figure docstrings keep the development notes on
why each figure has its present form.
"""

# %% Section 1 -- imports, paths, and the house style
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "Simulations"),
           os.path.join(_ROOT, "analysis_modules"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIG_DIR = os.environ.get("CIB_PAPER_FIGDIR", os.path.join(_HERE, "figures"))
ALT_DIR = os.path.join(FIG_DIR, "alternates")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ALT_DIR, exist_ok=True)

PLANCK_RES = os.path.join(_ROOT, "Planck_analysis", "results")
HERSCH_RES = os.path.join(_ROOT, "Herschel_analysis", "results")
SIM_RES = os.path.join(_ROOT, "Simulations", "results")

NPZ_PL_UNL = os.path.join(PLANCK_RES, "planck_v2_unlensed_857_4.0e+20_gp40.npz")
NPZ_PL_LEN = os.path.join(PLANCK_RES,
                          "planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz")
NPZ_H1C = os.path.join(HERSCH_RES, "herschel_peak_sims_350.npz")
JSON_NULL = os.path.join(SIM_RES, "herschel_null_seeds.json")
NPZ_H_SIM = os.path.join(SIM_RES, "herschel.npz")
NPZ_CCAT = os.path.join(SIM_RES, "ccat.npz")
NPZ_H_UNL = os.path.join(HERSCH_RES, "herschel_unlensed_v3_350.npz")
NPZ_H_LEN = os.path.join(HERSCH_RES, "herschel_lensed_350.npz")
JSON_COUNTS = os.path.join(_ROOT, "num_count_fits",
                           "num_count_fit_results.json")
#  Frozen peak-level model curves for Figs. 2, 3, 4, 5 and 10.  Produced by
#  `Simulations/export_peaklevel_curves.py`, which reads sim_core's Run cache
#  WITH CIB_CACHE=1 and writes the published realization here.  See that
#  script: without the cache flag sim_core recomputes and xi-hat moves.
NPZ_PKCURVES = os.path.join(SIM_RES, "paperfig_peaklevel_curves.npz")

WHICH = os.environ.get("CIB_PAPER_FIGS",
                       "1,2,3,4,5,6,7,8,9,10,11,12,13,14,B1,E1"
                       ).split(",")
WHICH = [w.strip() for w in WHICH if w.strip()]
PREVIEW = bool(int(os.environ.get("CIB_PREVIEW", "0")))
PREVIEW_DIR = os.path.join(FIG_DIR, "preview")

#  Set CIB_USETEX=1 to typeset every label with the local LaTeX installation,
#  which matches the paper's fonts exactly.  It needs a complete TeX (type1cm,
#  dvipng, ghostscript) and is roughly 5x slower, so the default is off and the
#  Computer Modern lookalike below is used instead.  All labels are written to
#  render identically either way; `PC` is the one character that differs.
USETEX = bool(int(os.environ.get("CIB_USETEX", "0")))
PC = r"\%" if USETEX else "%"

#  ---- final-size geometry ---------------------------------------------------
COL = 3.543          # in;  90 mm, A&A \hsize inside `figure`
DCOL = 7.244         # in; 184 mm, A&A \hsize inside `figure*`

#  ---- the house style -------------------------------------------------------
#  Fonts are set for 1:1 placement.  7 pt in the figure lands as 7 pt on the
#  page against 9 pt body text, which is what the journal asks for.  Line and
#  marker sizes are scaled down to match: a 1.8 pt line that looked right on a
#  13-inch canvas is a slab at 3.5 inches.
plt.rcParams.update({
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "errorbar.capsize": 1.5,
    "axes.formatter.use_mathtext": True,
    #  Computer Modern for both text and maths, so the figures sit inside an
    #  A&A page without a visible font change.  `axes.formatter.use_mathtext`
    #  above is what keeps cmr10's missing minus sign from biting.
    "text.usetex": USETEX,
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,          # embed TrueType, not Type 3 (journal rule)
    "ps.fonttype": 42,
})

#  Model colours: the canonical map from `counts_350um.MODEL_COLORS`, which
#  Figs. 2, 3, 10 and 13 already use.  The Planck module used a *different*
#  assignment (Schechter green / SPL red / DPL purple); adopting the canonical
#  one here makes the whole paper consistent.
MODEL_COLORS = {"Schechter": "C0", "SPL": "C2", "DPL": "C3"}
MODEL_LS = {"Schechter": "--", "SPL": "-", "DPL": ":"}
MODEL_ORDER = ["Schechter", "SPL", "DPL"]

GREY_CORE = "0.93"       # shading for the sub-threshold core region
GREEN_WIN = "C2"         # shading for the science window


def _save(fig, name, alt=False):
    """Write `name`.pdf (and optionally .png) at the figure's native size."""
    d = ALT_DIR if alt else FIG_DIR
    p = os.path.join(d, name + ".pdf")
    fig.savefig(p)
    if PREVIEW:
        #  Previews go to a SEPARATE directory.  Writing `name`.png beside the
        #  PDF would silently overwrite the legacy PNG of the same name -- the
        #  very file the previous draft still includes -- so a preview run
        #  would edit a figure the current .tex depends on.  It happened once
        #  (2026-09-01); the files were restored from `tar_archives/Figures/`.
        pv = os.path.join(PREVIEW_DIR, "alternates") if alt else PREVIEW_DIR
        os.makedirs(pv, exist_ok=True)
        fig.savefig(os.path.join(pv, name + ".png"), dpi=200)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  wrote {os.path.relpath(p, _HERE)}   "
          f"({w * 25.4:.0f} x {h * 25.4:.0f} mm)")
    return p


def _tidy(ax, grid=True):
    """Uniform axis furniture."""
    if grid:
        ax.grid(alpha=0.22, lw=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _logticks(ax, vals, axis="x"):
    """Label a log axis at `vals` in plain numerals.

    A log threshold axis spanning less than a decade and a half gets exactly
    one labelled major tick from Matplotlib's default locator -- "$10^1$" and
    nothing else -- which tells the reader almost nothing about where the
    features sit.  Fixing the ticks at round values and formatting them as
    plain integers is the whole fix.
    """
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(list(vals)))
    a.set_major_formatter(FixedFormatter([f"{v:g}" for v in vals]))
    a.set_minor_formatter(NullFormatter())


def _style_key(ax, entries, loc="upper right", **kw):
    """Draw a compact linestyle/marker key as a SECOND legend on `ax`.

    Several figures render each of the three count models twice -- simulated
    and asymptotic, or MC and analytic -- which makes a six-entry legend that
    cannot be set below about 4.5 pt inside a 90 mm column.  Naming each model
    once in a colour legend and each rendering once in this key says the same
    thing in three entries plus two, at a readable size.

    `entries` is a list of (style, label).  A style drawn from the four
    Matplotlib linestyles becomes a line; anything else is treated as a marker.
    """
    from matplotlib.lines import Line2D
    handles = []
    for st, lab in entries:
        if st in ("-", "--", "-.", ":"):
            handles.append(Line2D([], [], color="0.25", ls=st, lw=1.1,
                                  label=lab))
        else:
            handles.append(Line2D([], [], color="0.25", ls="none", marker=st,
                                  ms=2.4, label=lab))
    key = ax.legend(handles=handles, loc=loc, fontsize=5.6,
                    borderaxespad=0.35, **kw)
    ax.add_artist(key)
    return key


# %% Section 2 -- Figure 5: validation of the analytic Delta_xi relations
def fig05_dxi_validation(R=1.0, stacked=True):
    """
    Draft Fig. 5 -- the four routes to Delta_xi(u), and the pixel-vs-peak
    cancellation that licenses the differential design.

    BOTH PANELS ARE KEPT.  The right-hand panel is cited in Sect. 3.2 as a
    *test* of the convention cancellation ("tests that cancellation rather than
    assuming it"), and the paper's standing rule -- every model curve shown
    against data is simulated at peak level -- rests on its result.

    Data provenance
    ---------------
    All curves are read from `Simulations/results/paperfig_peaklevel_curves.npz`
    (keys `f5_*`), written by stage 5 of `Simulations/export_peaklevel_curves.py`
    from `paper_figures_peaklevel.figure_dxi_validation`: the analytic routes
    (linearized master relation, exact slope evaluation, exact mu-mixture
    P(D)) and the paired Monte Carlo Delta_xi of 800 simulated cutouts per arm
    at pixel and peak level.  (Until the release these were read from the
    simulation pickle cache; the exported values are identical to machine
    precision.)

    Parameters
    ----------
    R : float
        Aperture radius in units of theta_500 for the benchmark lens.
    stacked : bool
        Layout.  True (the DEFAULT, and what the .tex uses) stacks the two
        panels vertically, both at the full 90 mm column width; they share the
        threshold axis exactly, so one x-label serves both.  False lays them
        side by side inside the same 90 mm, which forces the legends down to
        4.6 pt and is written to figures/alternates/ as a fallback only.

        The stacked layout was adopted after compiling both: it is fully
        legible at 7 pt and, because the taller float still fits the page it
        lands on, it costs nothing -- the document is 28 pages either way.
    """
    print("\n[Fig 5] Delta_xi validation, pixel and peak conventions")
    K = np.load(NPZ_PKCURVES, allow_pickle=False)
    if float(K["f5_R"]) != R:
        raise ValueError(f"exported Fig. 5 curves are for R = {float(K['f5_R'])}"
                         f", not R = {R}; re-run export stage 5")
    u = K["f5_u"]
    dxi_lin, dxi_exact, dxi_mix = K["f5_lin"], K["f5_exact"], K["f5_mix"]
    out = {c: {"delta": K[f"f5_delta_{c}"], "sigma": K[f"f5_sigma_{c}"]}
           for c in ("pix", "pk")}
    for c in ("pix", "pk"):
        print(f"  MC {c:>3s}  Delta_xi: "
              + " ".join(f"{v:+.4f}" for v in out[c]["delta"]))

    #  ---- figure ------------------------------------------------------------
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(COL, 3.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1],
                                              "hspace": 0.08})
    else:
        #  Two panels inside 90 mm leaves ~38 mm of drawing area each, which
        #  will not carry 7 pt furniture.  This layout therefore drops to 6 pt
        #  -- still above the journal floor, but it is the reason the stacked
        #  variant in figures/alternates/ exists.
        fig, axes = plt.subplots(1, 2, figsize=(COL, 2.05),
                                 gridspec_kw={"width_ratios": [1.15, 1],
                                              "wspace": 0.48})
        for a in axes:
            a.tick_params(labelsize=5.4)
            a.xaxis.label.set_size(6.0)
            a.yaxis.label.set_size(6.0)

    ax = axes[0]
    ax.plot(u, dxi_lin, "-", color="C3", lw=1.1, label="linearized")
    ax.plot(u, dxi_exact, "--", color="C1", lw=1.1, label="exact slope")
    ax.plot(u, dxi_mix, ":", color="C0", lw=1.4, label=r"$\mu$-mixture $P(D)$")
    for conv, mk, c, lab in (("pix", "s", "0.45", "MC, pixel"),
                             ("pk", "o", "k", "MC, peak")):
        d = out[conv]
        g = np.isfinite(d["delta"])
        ax.errorbar(u[g] * (1 + 0.004 * (conv == "pk")), d["delta"][g],
                    yerr=d["sigma"][g], fmt=mk, ms=2.6, color=c,
                    mfc="w" if conv == "pix" else c, mew=0.7,
                    capsize=1.3, lw=0.8, label=lab, zorder=5)
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_ylabel(r"$\Delta\xi(u)$")
    ax.legend(loc="lower left", ncol=1, fontsize=5.4 if stacked else 4.6)
    _tidy(ax)

    ax = axes[1]
    xi_pix_c = K["f5_xi_pix_ctrl"]
    xi_pk_c = K["f5_xi_pk_ctrl"]
    ax.plot(u, xi_pix_c - xi_pk_c, "d-", color="C4", lw=1.0, ms=2.6,
            label=r"unlensed $\xi_{\rm pix}-\xi_{\rm pk}$")
    gg = np.isfinite(out["pix"]["delta"]) & np.isfinite(out["pk"]["delta"])
    ax.errorbar(u[gg], (out["pix"]["delta"] - out["pk"]["delta"])[gg],
                yerr=np.hypot(out["pix"]["sigma"], out["pk"]["sigma"])[gg],
                fmt="o", color="k", ms=2.6, capsize=1.3, lw=0.8,
                label=r"differential $\Delta\xi_{\rm pix}-\Delta\xi_{\rm pk}$")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_ylabel("pixel $-$ peak")
    ax.legend(loc="upper left", fontsize=5.4 if stacked else 4.6)
    _tidy(ax)

    for a in np.atleast_1d(axes).ravel():
        a.set_xlabel(r"threshold $u$ [mJy/beam]",
                     fontsize=7.0 if stacked else 6.0)
    if stacked:
        axes[0].set_xlabel("")

    return _save(fig, "dxi_validation" if stacked else "dxi_validation_sbs",
                 alt=not stacked)


# %% Section 3 -- Figure 6: the Planck GPD stability scan
def fig06_planck_stability():
    """
    Draft Fig. 6 -- the *Planck* 857 GHz threshold-stability scan, single panel.

    The mean-excess panel that used to sit on the right is not referenced
    anywhere in Sect. 4.3; every feature the text lists (floor climb, zero
    crossing, plateau onset, mask ceiling) belongs to this panel.  It is not
    discarded, though: it becomes panel (a) of Fig. 7, where it does specific
    work -- a nonparametric corroboration that the positive excursion is in
    the data rather than in the small-sample bias of the ML estimator.  That
    placement also honours Sect. 2.1, which introduces the mean excess as one
    of two diagnostics "applied throughout".

    Reads `planck_v2_unlensed_857_4.0e+20_gp40.npz`; no map data required.
    """
    print("\n[Fig 6] Planck GPD stability scan (single panel)")
    d = np.load(NPZ_PL_UNL, allow_pickle=True)
    k = d["k_grid"]
    xi, xierr = d["xi"], d["xi_err"]
    xi_peak = json.loads(str(d["xi_peak_json"]))
    n_cut = int(d["n_cutouts"])

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    ax.axvspan(k.min(), 3.0, color=GREY_CORE, zorder=0)
    ax.fill_between(k, d["null_lo"], d["null_hi"], color="0.55", alpha=0.30,
                    lw=0, zorder=1, label=f"simulated null, 95{PC}")
    ax.plot(k, d["null_med"], color="0.35", ls="-.", lw=0.8, zorder=2,
            label="null median")
    for nm in MODEL_ORDER:
        ax.plot(k, np.asarray(xi_peak[nm], float), MODEL_LS[nm],
                color=MODEL_COLORS[nm], lw=1.0, zorder=3, label=nm)
    ax.errorbar(k, xi, yerr=xierr, fmt="o", color="k", ms=2.6, capsize=1.3,
                lw=0.8, zorder=5, label=r"measured $\hat\xi(u)$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel=r"threshold $k=(u-\mu_{\rm core})/\sigma_{\rm core}$",
           ylabel=r"$\xi(u)$", xlim=(0.8, 6.3), ylim=(-0.62, 0.34))
    ax.legend(loc="lower right", ncol=2, fontsize=5.6, columnspacing=0.9)
    ax.text(0.025, 0.96, f"857 GHz, {n_cut} cutouts",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.0,
            color="0.3")
    _tidy(ax)
    return _save(fig, "planck_stability")


# %% Section 4 -- Figure 7: the bright population behind the excursion
def fig07_planck_bright():
    """
    Draft Fig. 7, REBUILT -- the two diagnostics that Sect. 4.3 cites, stacked
    vertically in a single column.

    The v5 file `Figures/planck_diagnostics.png` showed the declustering-window
    systematic, the threshold correlation matrix and the null sampling
    histogram; none of those is what the caption or the citing text describes.
    What Sect. 4.3 needs is a two-step argument about the positive excursion,
    and the two panels take one step each:

      (a) THE EXCESS IS REAL, NOT AN ESTIMATOR ARTEFACT.  Above a valid
          threshold the mean excess of a GPD tail obeys
          de/du = xi/(1-xi), so the turnover of e(u) locates the sign change
          of xi with no GPD fit anywhere in the calculation.  Measured on the
          same declustered peaks it turns at k = 2.23 and rises monotonically
          through the science window, against a fitted zero crossing at
          k ~ 1.85 -- two estimators of quite different character agreeing to
          dk ~ 0.4.  This answers the objection Appendix E itself raises, that
          the ML xi-hat carries a positive O(1/n) bias and a right-skewed
          sampling distribution at small exceedance counts.  A null Monte
          Carlo cannot answer it; a moment-based statistic can.

      (b) IT IS NOT THE FAINT COUNTS.  The measured xi-hat(u) against the
          truncated-counts null from which the significance is quoted, and
          against a bright-extended null whose simulated counts run to
          S = 1500 mJy.  The extension raises the null without lifting it to
          the measurement.

    Panel (a) is deliberately FIRST: the logic runs "it is real" then "here is
    what it is".  Panel (a) is relocated from what was Fig. E.1c; Fig. E.1
    returns to two panels.

    The bright-source census that used to be panel (b) is now reported in the
    text.  Two reasons it reads better there: the observed peak counts are
    compared with model SOURCE counts, so the panel over-promised (see the
    \\chk in Sect. 4.3), and the SPL curve tracked the observed points within
    a factor 3 at 650 mJy -- visually flattering the one model App. A.3
    falsifies outright through the 857 GHz monopole.

    Both panels are functions of the same threshold k, so they share one
    x-axis, and the axis range matches Fig. 6 exactly.  Reading down the
    figure, the two dotted verticals -- the mean-excess turnover in (a), the
    fitted zero crossing in (b) -- sit almost on top of one another; that near
    coincidence is the corroboration, and stacking is what makes it visible.

    Reads `planck_v2_unlensed_857_4.0e+20_gp40.npz`; no map data required.

    A NUMBER TO CHECK.  Sect. 4.3 states that the bright extension "absorbs
    roughly half of the excess".  On this cached run the extension moves the
    measured-minus-null offset by 14% at k = 3.0 and 30% at k = 3.5, and the
    significance from 7.8 to 6.7 sigma and 3.0 to 2.1 sigma respectively; at
    k = 4.0 the offset grows slightly.  The qualitative conclusion (the
    excursion survives the extension, so clustered CIB and cirrus remain) is
    unchanged, but "roughly half" is not what this npz contains.  The printout
    below reproduces the numbers.
    """
    print("\n[Fig 7] Planck excursion diagnostics (mean excess + both nulls)")
    d = np.load(NPZ_PL_UNL, allow_pickle=True)
    k = d["k_grid"]
    xi, xierr = d["xi"], d["xi_err"]
    nm_, nlo, nhi, nsd = d["null_med"], d["null_lo"], d["null_hi"], d["null_sd"]
    em, elo, ehi = d["ext_med"], d["ext_lo"], d["ext_hi"]
    s_ext = float(d["s_bright_ext"])
    mu_c, sg_c = float(d["mu_core"]), float(d["sigma_core"])
    s_psmask = 650.0                      # planck_unlensed_pd_v2.S_PSMASK_MJY
    k_psmask = (s_psmask - mu_c) / sg_c

    #  mean excess, and the two independent locators of the xi sign change
    kme = (d["me_grid"] - mu_c) / sg_c
    eme = d["mean_excess"] / sg_c
    good = np.isfinite(eme)
    kme, eme = kme[good], eme[good]
    k_emin = float(kme[np.argmin(eme)])
    #  linear interpolation of the fitted scan through xi = 0
    _i = int(np.nanargmax((xi > 0) & np.isfinite(xi)))
    k_zero = float(k[_i - 1] + (k[_i] - k[_i - 1])
                   * (-xi[_i - 1]) / (xi[_i] - xi[_i - 1]))
    print(f"  mean-excess turnover  k = {k_emin:.2f}")
    print(f"  fitted xi=0 crossing  k = {k_zero:.2f}   "
          f"(difference {abs(k_emin - k_zero):.2f})")
    print(f"  PS-mask limit {s_psmask:.0f} mJy = k {k_psmask:.2f}")
    print("     k      xi    null_med   ext_med   (xi-nm)/sd  (xi-em)/sd")
    for i in range(len(k)):
        if np.isfinite(nm_[i]):
            print("  %5.1f %7.4f  %8.4f  %8.4f   %8.2f   %8.2f"
                  % (k[i], xi[i], nm_[i], em[i],
                     (xi[i] - nm_[i]) / nsd[i], (xi[i] - em[i]) / nsd[i]))

    XLIM = (0.8, 6.3)          # identical to Fig. 6
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.55), sharex=True,
                             gridspec_kw={"height_ratios": [0.92, 1.15],
                                          "hspace": 0.07})

    def _bands(ax):
        """Shading and the mask line, shared with Fig. 6 and used in both."""
        ax.axvspan(XLIM[0], 3.0, color=GREY_CORE, zorder=0)
        ax.axvspan(3.0, 4.5, color=GREEN_WIN, alpha=0.09, zorder=0)
        ax.axvline(k_psmask, color="C3", ls="-.", lw=0.6, alpha=0.7, zorder=1)

    def _panel_label(ax, txt):
        """Panel labels go inside the axes: titles do not fit a stacked pair."""
        ax.text(0.015, 0.955, txt, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.6)

    #  ---- (a) the mean excess: a nonparametric read on the sign of xi -------
    ax = axes[0]
    _bands(ax)
    ax.plot(kme, eme, ".-", color="C0", ms=2.2, lw=0.9, zorder=4)
    ax.axvline(k_emin, color="0.35", ls=":", lw=0.8, zorder=3)
    ax.annotate(r"$e$ min, $k=%.1f$" % k_emin, xy=(k_emin, 0.40),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=5.8, color="0.35",
                va="center")
    ax.text(k_psmask - 0.07, 0.55, "PS-mask limit",
            transform=ax.get_xaxis_transform(), fontsize=5.6, color="C3",
            rotation=90, va="center", ha="right")
    #  one science-window label serves the pair, since the axis is shared
    ax.text(4.05, 0.82, "science window", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.0, color="0.35")
    ax.set_ylabel(r"mean excess $e(u)/\sigma_{\rm core}$")
    ax.set_ylim(0.68, 1.33)
    _panel_label(ax, r"(a) mean excess: rising $\Rightarrow \xi>0$")
    _tidy(ax)

    #  ---- (b) the excursion against both nulls ------------------------------
    ax = axes[1]
    _bands(ax)
    ax.fill_between(k, nlo, nhi, color="0.45", alpha=0.30, lw=0, zorder=1,
                    label=(r"truncated null, $S_{\rm cut}=100$ mJy "
                           f"(95{PC})"))
    ax.plot(k, nm_, color="0.30", ls="-.", lw=0.8, zorder=3)
    ax.fill_between(k, elo, ehi, facecolor="C3", alpha=0.16, lw=0, zorder=2,
                    label=(r"bright-extended, $S_{\rm cut}=%.0f$ mJy "
                           % s_ext + f"(95{PC})"))
    ax.plot(k, elo, color="C3", lw=0.5, zorder=2)
    ax.plot(k, ehi, color="C3", lw=0.5, zorder=2)
    ax.plot(k, em, color="C3", ls="--", lw=0.9, zorder=3)
    ax.errorbar(k, xi, yerr=xierr, fmt="o", color="k", ms=2.6, capsize=1.3,
                lw=0.8, zorder=6, label=r"measured $\hat\xi(u)$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(k_zero, color="0.35", ls=":", lw=0.8, zorder=3)
    ax.annotate(r"$\hat\xi=0$, $k=%.1f$" % k_zero, xy=(k_zero, 0.32),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=5.8, color="0.35",
                va="center")
    ax.set(xlabel=r"threshold $k=(u-\mu_{\rm core})/\sigma_{\rm core}$",
           ylabel=r"$\xi(u)$", xlim=XLIM, ylim=(-0.68, 0.34))
    ax.legend(loc="lower left", fontsize=5.6)
    _panel_label(ax, "(b) the excursion against both nulls")
    _tidy(ax)

    return _save(fig, "planck_bright")


# %% Section 5 -- Figure 8: the Planck differential measurement
def fig08_planck_dxi(ap="1.5", stacked=False):
    """
    Draft Fig. 8 -- the shape and amplitude channels of the *Planck*
    cluster/control differential.

    Both panels are kept: the left is cited in Sect. 4.4 for the null result,
    and the right is the only place the "two control constructions straddle
    unity" statement can be checked.

    HISTORY.  The production npz originally stored `ratio_{ap}` and
    `pred_ratio_{ap}` but not `ratio_err`, nor the offset-only control series
    (`MEAS_PLAIN`), so this function drew the shape panel only.  On 2026-09-14
    `Planck_analysis/augment_lensed_npz_from_cache.py` merged those arrays --
    plus the window-combined ratios `ratio_win_*` / `ratio_plain_win_*` with
    their 95% intervals -- into the npz straight from the module's stage cache
    (bit-identical to what the patched save block of `planck_lensed_pd_v2.py`
    now writes on a re-run).  Both panels are drawn whenever the keys exist.

    Amplitude panel conventions: thresholds at which either pool has zero
    exceedances give an undefined ratio (the cache stores 0 with error 0) and
    are not drawn; the open markers at mid-window are the window-combined
    ratios quoted in Sect. 4.4 (1.49 [0.98, 2.29] dust-matched, 0.89
    [0.63, 1.34] offset-only), with their 95% intervals as error bars.

    Parameters
    ----------
    ap : str
        Aperture tag, in units of theta_500, matching the npz key suffix.
    stacked : bool
        Vertical instead of side-by-side layout; writes to alternates/.
    """
    print("\n[Fig 8] Planck Delta_xi, shape and amplitude channels")
    d = np.load(NPZ_PL_LEN, allow_pickle=True)
    k = d["k_grid"]
    dxi = d[f"dxi_{ap}"]
    boot = d[f"dxi_boot_{ap}"]
    pred = d[f"pred_dxi_{ap}"]
    comb, sig = float(d[f"comb_{ap}"]), float(d[f"sig_paired_{ap}"])
    ul95 = float(d[f"ul95_{ap}"])
    kwin = d["k_window"]
    n_pairs = int(np.asarray(d["acc_idx"]).size)
    lo, hi = np.nanpercentile(boot, [16, 84], axis=0)

    have_amp = f"ratio_err_{ap}" in d.files
    if not have_amp:
        print("  NOTE: ratio_err / offset-only controls absent from the npz;"
              " drawing the shape panel only.")

    if not have_amp:
        if stacked:
            print("  (single panel: no separate stacked variant)")
            return None
        fig, ax0 = plt.subplots(figsize=(COL, 2.45))
        axes = [ax0]
    elif stacked:
        fig, axes = plt.subplots(2, 1, figsize=(COL, 4.0), sharex=True,
                                 gridspec_kw={"hspace": 0.08})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(COL, 1.62),
                                 gridspec_kw={"wspace": 0.42})

    #  ---- shape channel -----------------------------------------------------
    ax = axes[0]
    ax.axvspan(k.min(), 3.0, color=GREY_CORE, zorder=0)
    ax.axvspan(*kwin, color=GREEN_WIN, alpha=0.10, zorder=0)
    ax.errorbar(k, dxi, yerr=[dxi - lo, hi - dxi], fmt="o", color="C0",
                ms=2.6, capsize=1.3, lw=0.8, label=r"measured $\Delta\xi(u)$")
    ax.plot(k, pred, "-", color="C3", lw=1.0, label="NFW $\\mu$-mixture")
    if np.isfinite(comb):
        ax.errorbar([np.mean(kwin)], [comb], yerr=[sig], fmt="s", color="C2",
                    ms=4.5, mfc="none", mew=1.0, capsize=2.0,
                    label=(f"combined ${comb:+.3f}\\pm{sig:.3f}$\n"
                           f"95{PC} UL $|\\Delta\\xi|<{ul95:.2f}$"))
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel=r"$k=(u-\mu_{\rm core})/\sigma_{\rm core}$",
           ylabel=r"$\Delta\xi(u)$", xlim=(0.8, 5.0))
    ax.legend(loc="lower left", fontsize=5.4)
    ax.text(0.025, 0.96, f"{n_pairs} pairs, $r<{ap}\\,\\theta_{{500}}$",
            transform=ax.transAxes, va="top", fontsize=5.8, color="0.3")
    _tidy(ax)

    #  ---- amplitude channel -------------------------------------------------
    if have_amp:
        ax = axes[1]
        ax.axvspan(k.min(), 3.0, color=GREY_CORE, zorder=0)
        ax.axvspan(*kwin, color=GREEN_WIN, alpha=0.10, zorder=0)
        #  a ratio is defined only where both pools have exceedances; the
        #  cache stores 0 +- 0 otherwise, which must not be drawn as a point
        ncl, nct = d[f"n_exc_cl_{ap}"], d[f"n_exc_ct_{ap}"]
        ok = (ncl > 0) & (nct > 0) & np.isfinite(d[f"ratio_{ap}"])
        ax.errorbar(k[ok], d[f"ratio_{ap}"][ok], yerr=d[f"ratio_err_{ap}"][ok],
                    fmt="o", color="C0", ms=2.6, capsize=1.3, lw=0.8,
                    label="dust-matched")
        rp, rpe = d[f"ratio_plain_{ap}"], d[f"ratio_plain_err_{ap}"]
        okp = np.isfinite(rp) & (rpe > 0)
        ax.errorbar(k[okp] + 0.06, rp[okp], yerr=rpe[okp], fmt="^", color="C1",
                    ms=2.6, mfc="none", capsize=1.3, lw=0.8,
                    label="offset-only")
        ax.plot(k, d[f"pred_ratio_{ap}"], "-", color="C3", lw=1.0,
                label="predicted")
        #  window-combined ratios with 95% intervals (the Sect. 4.4 numbers)
        if f"ratio_win_{ap}" in d.files:
            km = float(np.mean(kwin))
            for nm, dx, mk, col in (("ratio_win", -0.10, "s", "C0"),
                                    ("ratio_plain_win", +0.10, "D", "C1")):
                r0 = float(d[f"{nm}_{ap}"])
                lo, hi = float(d[f"{nm}_lo_{ap}"]), float(d[f"{nm}_hi_{ap}"])
                ax.errorbar([km + dx], [r0], yerr=[[r0 - lo], [hi - r0]],
                            fmt=mk, color=col, ms=4.5, mfc="none", mew=1.0,
                            capsize=2.0, lw=1.0, zorder=4)
        ax.axhline(1, color="0.6", lw=0.6)
        ax.set(xlabel=r"$k=(u-\mu_{\rm core})/\sigma_{\rm core}$",
               ylabel=r"$\lambda_{\rm cl}/\lambda_{\rm ctrl}$",
               xlim=(0.8, 5.0), ylim=(-0.3, 3.4))
        ax.legend(loc="upper left", fontsize=5.4)
        _tidy(ax)
        for a_, lab in zip(axes, ("(a)", "(b)")):
            a_.text(0.97, 0.04, lab, transform=a_.transAxes, ha="right",
                    va="bottom", fontsize=6.5)
        if stacked:
            axes[0].set_xlabel("")

    return _save(fig, "planck_dxi" + ("_stacked" if stacked else ""),
                 alt=stacked)


# %% Section 6 -- Figure 9: aperture consistency and the dust closure
def fig09_planck_aperture_dust(ap="1.5"):
    """
    Draft Fig. 9 -- the aperture scan of the measured Delta_xi and the cluster
    dust closure test against Erler et al. (2018).

    THE LEFT PANEL OF THE v5 FIGURE IS DROPPED.  It showed the predicted
    |Delta_xi|, <ln mu> and a forecast SNR against aperture radius -- the
    "dilution versus statistics" trade.  It is not `\\ref`'d anywhere, the
    caption does not describe it, and the same statement is made three times
    already: analytically in Sect. 2.5 ("the aperture trade is scale-invariant,
    so it cannot be optimized away"), numerically in the last two columns of
    Table 5, and again for Planck specifically in Sect. 4.2.

    The two surviving panels are what the caption promises.  The dust-variant
    points are ADDED to the right panel: Sect. 4.5 says "the dust-corrected
    variants agree with the uncorrected one and with zero (Fig. 9)", and no
    panel of the v5 figure showed them, although the three numbers were in the
    npz all along (`dustvar_{none,stack,erler}_*`).

    Reads `planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz`; no map data required.
    """
    print("\n[Fig 9] Planck aperture consistency and dust closure")
    d = np.load(NPZ_PL_LEN, allow_pickle=True)
    aps = np.asarray(d["aperture_scan"], float)
    comb = np.array([float(d[f"comb_{f:.1f}"]) for f in aps])
    sigp = np.array([float(d[f"sig_paired_{f:.1f}"]) for f in aps])
    pred = np.array([float(d[f"pred_comb_{f:.1f}"]) for f in aps])

    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.95),
                             gridspec_kw={"wspace": 0.50})

    #  ---- (a) aperture consistency, with the dust-corrected variants --------
    #  `dustvar_none` is numerically the baseline point at the default
    #  aperture, so only the two CORRECTED variants are drawn as extras; they
    #  are what Sect. 4.5's "the dust-corrected variants agree with the
    #  uncorrected one and with zero" refers to.
    ax = axes[0]
    ax.errorbar(aps, comb, yerr=sigp, fmt="o", color="C0", ms=2.8,
                capsize=1.4, lw=0.8, zorder=4, label="measured")
    ax.plot(aps, pred, "-", color="C3", lw=1.0, zorder=3,
            label="NFW prediction")
    ap0 = float(ap)
    for dx, v, mk, lab in ((-0.13, "stack", "^", "dust: stack"),
                           (+0.13, "erler", "v", "dust: Erler+18")):
        ax.errorbar([ap0 + dx], [float(d[f"dustvar_{v}_comb"])],
                    yerr=[float(d[f"dustvar_{v}_sig"])], fmt=mk, color="C4",
                    ms=2.6, mfc="none", mew=0.8, capsize=1.2, lw=0.7,
                    alpha=0.85, zorder=5, label=lab)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel=r"aperture radius [$\theta_{500}$]",
           ylabel=r"window-combined $\Delta\xi$", xlim=(0.25, 2.75),
           ylim=(-0.58, 0.50))
    ax.set_xticks(aps)
    ax.legend(loc="upper left", fontsize=5.4, labelspacing=0.22)
    ax.set_title("(a) aperture scan", fontsize=6.6)
    _tidy(ax)

    #  ---- (b) the dust closure against the external stacked spectrum --------
    ax = axes[1]
    ax.plot(d["r_mid"], d["prof_meas"], "o-", color="C0", ms=2.0, lw=0.9,
            label=r"measured stack (cl$-$ct)")
    ax.plot(d["r_mid"], d["prof_erler"], "s--", color="C3", ms=2.0, lw=0.9,
            label=r"Erler+18, $A_{857}=%.2f$" % float(d["erler_a857"]))
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel=r"$r/\theta_{500}$", ylabel="mJy/beam", xlim=(0, 3.0),
           ylim=(-8, 92))
    ax.legend(loc="upper right", fontsize=5.0)
    ax.set_title("(b) cluster dust", fontsize=6.6)
    _tidy(ax)

    return _save(fig, "planck_aperture_dust")


# %% Section 7 -- Figure 11: the Herschel H1c decomposition
def fig11_h1c_decomposition():
    """
    Draft Fig. 11 -- the forward-simulation decomposition of the measured rise
    of xi-hat(u) in the GAMA-09 350 um map, single panel.

    The right panel of the v5 figure (the injected bright population, i.e. the
    Schechter dN/dS with a power-law tail joined at S_cut) is dropped: nothing
    in Sect. 5.3 depends on seeing it, and the injected slope is stated
    numerically in the text.  Dropping it lets the decomposition -- the panel
    that carries the result -- occupy the full column.

    Reads `herschel_peak_sims_350.npz`; no map data required.
    """
    print("\n[Fig 11] Herschel H1c decomposition (single panel)")
    d = np.load(NPZ_H1C, allow_pickle=True)
    u = d["u_grid"]
    err = d["xi_err"]

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    ax.fill_between(u, -err, err, color="0.86", zorder=0,
                    label=r"$\pm1\sigma$ data error")
    ax.plot(u, d["d_pixel_to_peak"], "o-", color="C1", ms=2.4, lw=0.9,
            label=r"B$-$A: pixel $\rightarrow$ peak")
    ax.plot(u, d["d_bright"], "s-", color="C3", ms=2.4, lw=0.9,
            label=r"C$-$B: bright tail")
    ax.plot(u, d["d_residual"], "^--", color="k", ms=2.4, lw=0.9,
            label=r"data$-$C: residual")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel=r"threshold $u$ [mJy/beam]", ylabel=r"$\Delta\xi$")
    ax.legend(loc="upper left", fontsize=5.8)
    _tidy(ax)
    return _save(fig, "herschel_h1c_decomposition")


# %% Section 8 -- Figure B.1: noise and declustering
def figB1_noise_declustering(force=False):
    """
    Appendix B, Fig. B.1 -- what white pixel noise does to declustering, and
    how a beam-correlated noise realization repairs it.  Two panels, promoted
    to a two-column `figure*` (appendix pages do not count against the limit).

    This is the one figure on the list with no cached product, so the small
    self-contained simulation (originally a development script,
    `fig3_noise.py`) is re-run here:
    2 count models x 2 noise modes x 6 noise levels x 4 maps of 1024^2.
    The result is cached to `figures/.cache_figB1.npz`, so subsequent calls
    re-plot in a second.  Pass force=True to recompute.

    Requires `analysis_modules` (counts, simulate_maps, gpd_tail).  No survey
    data.
    """
    print("\n[Fig B.1] noise and declustering")
    cache = os.path.join(FIG_DIR, ".cache_figB1.npz")
    F_N = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    keys = [(nm, md) for nm in ("Schechter", "DPL") for md in ("white", "beam")]

    if force or not os.path.exists(cache):
        from counts import Schechter, DoublePowerLaw
        from simulate_maps import CIBMapSimulator
        from gpd_tail import confusion_sigma, beam_declustered_peaks
        BEAM, PIX, NPIX = 20.0, 4.0, 1024
        S_MIN_, S_CUT_ = 1e-3, 100.0
        bfp, N_MAPS = BEAM / PIX, 4
        store = {}
        for nm, cnts in (("Schechter", Schechter(s_min=S_MIN_, s_max=S_CUT_)),
                         ("DPL", DoublePowerLaw.illustrative(s_min=S_MIN_,
                                                             s_max=S_CUT_))):
            cs = confusion_sigma(cnts, BEAM, s_cut=S_CUT_)
            for mode in ("white", "beam"):
                dens, rank = [], []
                for f in F_N:
                    sim = CIBMapSimulator(cnts, BEAM, PIX, npix=NPIX,
                                          sigma_noise=f * cs["sigma_c"],
                                          s_cut=S_CUT_, noise_mode=mode)
                    pix, pk = [], []
                    for i in range(N_MAPS):
                        m = sim.make_map(seed=4200 + i)
                        pix.append(m.ravel()[::5])
                        pk.append(beam_declustered_peaks(m, bfp))
                    pix = np.sort(np.concatenate(pix))
                    pk = np.concatenate(pk)
                    dens.append(pk.size / (NPIX ** 2 * N_MAPS))
                    rank.append(np.median(np.searchsorted(pix, pk) / pix.size))
                    print(f"    {nm:>10s} {mode:>6s} f_N={f:4.2f} "
                          f"dens={dens[-1]:.5f} rank={rank[-1]:.4f}", flush=True)
                store[f"{nm}_{mode}_dens"] = np.array(dens)
                store[f"{nm}_{mode}_rank"] = np.array(rank)
        np.savez(cache, F_N=np.array(F_N), **store)
        print(f"    cached -> {os.path.relpath(cache, _HERE)}")
    C = np.load(cache)

    style = {("Schechter", "white"): ("C3", "o-", "Schechter, white noise"),
             ("DPL", "white"): ("C1", "s-", "DPL, white noise"),
             ("Schechter", "beam"): ("C0", "o--", "Schechter, beam-correlated"),
             ("DPL", "beam"): ("C2", "s--", "DPL, beam-correlated")}

    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.55),
                           gridspec_kw={"wspace": 0.19})
    for kk in keys:
        c, ls, lab = style[kk]
        ax[0].plot(C["F_N"], C[f"{kk[0]}_{kk[1]}_dens"], ls, color=c, ms=2.6,
                   lw=1.0, label=lab)
        ax[1].plot(C["F_N"], C[f"{kk[0]}_{kk[1]}_rank"], ls, color=c, ms=2.6,
                   lw=1.0)
    ax[0].axhline(C["Schechter_beam_dens"][0], color="0.55", ls=":", lw=0.7)
    ax[0].text(1.97, C["Schechter_beam_dens"][0] * 1.10,
               r"$\sigma_N=0$: $\approx$ 1 peak per 4 beams",
               ha="right", va="bottom", fontsize=5.8, color="0.4")
    ax[0].set(xlabel=r"$f_N=\sigma_N/\sigma_c$",
              ylabel="declustered peaks per pixel", ylim=(0, 0.042))
    ax[0].set_title(r"(a) white noise inflates the peak count $4\times$;"
                    "\nbeam-correlated noise does not", fontsize=7.0)
    ax[0].legend(loc="upper left", fontsize=6.0)
    _tidy(ax[0])

    ax[1].axhline(0.913, color="k", lw=0.8)
    ax[1].text(0.04, 0.9165, r"Gaussian-field prediction, size-5 filter "
                             r"($q_*=0.913$)", fontsize=5.8, va="bottom")
    ax[1].annotate("noise spikes ride the flanks\nbefore they dominate",
                   xy=(0.27, 0.784), xytext=(0.72, 0.792), fontsize=5.8,
                   color="0.35", ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
    ax[1].set(xlabel=r"$f_N=\sigma_N/\sigma_c$",
              ylabel=r"pixel-CDF quantile of the median peak, $q_*$",
              ylim=(0.770, 0.968))
    ax[1].set_title("(b) where the declustered core sits\n"
                    "in the pixel distribution", fontsize=7.0)
    _tidy(ax[1])

    return _save(fig, "noise_declustering")


# %% Section 9 -- Figure E.1: null seed ensemble, plus the mean-excess panel
def figE1_null_seed_ensemble():
    """
    Appendix E, Fig. E.1 -- two panels, two-column `figure*`.

      (a) the null-arm pull across the seed ensemble, per aperture;
      (b) the arm-level xi at u = 25 mJy/beam, null against control, with the
          independent unlensed reference marked.

    A third panel carrying the *Planck* mean excess lived here briefly; it now
    sits in the main text as Fig. 7a, next to the excursion it diagnoses,
    which is a better-motivated home than "the canonical GPD diagnostic
    belongs in the GPD appendix".

    Reads `Simulations/results/herschel_null_seeds.json` and `herschel.npz`.
    No map data required.
    """
    print("\n[Fig E.1] null seed ensemble + mean excess")
    with open(JSON_NULL) as fh:
        STORE = json.load(fh)
    seeds = sorted(STORE, key=int)
    APS = [0.5, 1.0, 1.5, 2.0]
    U = np.asarray(STORE[seeds[0]]["u"], float)
    j0 = 0

    maxpull = {}
    for sd in seeds:
        rec, row = STORE[sd]["apertures"], []
        for R in APS:
            dd = np.asarray(rec[str(R)]["dxi_null"], float)
            ss = np.asarray(rec[str(R)]["dxi_null_sd"], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.abs(dd) / ss
            row.append(np.nanmax(p) if np.isfinite(p).any() else np.nan)
        maxpull[sd] = row
    per_ap = np.array([[maxpull[sd][i] for sd in seeds]
                       for i in range(len(APS))])
    print(f"  {len(seeds)} seeds; max-pull median "
          f"{np.nanmedian([np.nanmax(v) for v in maxpull.values()]):.2f}")

    ref = None
    if os.path.exists(NPZ_H_SIM):
        _r = np.load(NPZ_H_SIM, allow_pickle=True)
        if "unl_xi_Schechter" in _r:
            ref = np.asarray(_r["unl_xi_Schechter"], float)

    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.55),
                             gridspec_kw={"wspace": 0.20})

    #  ---- (a) null-arm pull -------------------------------------------------
    ax = axes[0]
    for i in range(len(APS)):
        ax.scatter([i + 0.11 * (kk - (len(seeds) - 1) / 2)
                    for kk in range(len(seeds))], per_ap[i], s=7,
                   c=np.arange(len(seeds)), cmap="viridis", zorder=3,
                   linewidths=0)
    ax.axhline(3.0, color="crimson", ls="--", lw=0.9, label="PASS/FAIL")
    ax.set_xticks(range(len(APS)))
    ax.set_xticklabels([f"{R}" for R in APS])
    ax.set(xlabel=r"aperture radius [$\theta_{500}$]", xlim=(-0.5, 3.5))
    ax.set_ylabel(r"max $|\Delta\xi_{\rm null}|/\sigma$")
    ax.set_title(f"(a) null-arm pull, {len(seeds)} seeds", fontsize=7.0)
    ax.legend(loc="lower right", fontsize=5.8)
    _tidy(ax)

    #  ---- (b) which arm moves ----------------------------------------------
    ax = axes[1]
    for i, R in enumerate(APS):
        v = np.array([STORE[sd]["apertures"][str(R)]["xi_ctrl"][j0]
                      for sd in seeds], float)
        n = np.array([STORE[sd]["apertures"][str(R)]["xi_null"][j0]
                      for sd in seeds], float)
        ax.scatter(v, n, s=9, linewidths=0, label=f"{R}")
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="0.5", lw=0.7, zorder=0)
    if ref is not None:
        ax.axvline(ref[j0], color="0.6", ls=":", lw=0.7)
        ax.axhline(ref[j0], color="0.6", ls=":", lw=0.7)
    ax.set(xlabel=r"$\xi_{\rm ctrl}$", ylabel=r"$\xi_{\rm null}$")
    ax.set_title(r"(b) arm-level $\xi$ at $u=%.0f\,$mJy/beam" % U[j0],
                 fontsize=7.0)
    ax.legend(loc="upper left", fontsize=5.6, ncol=2, columnspacing=0.7,
              title=r"$r/\theta_{500}$", title_fontsize=5.6)
    _tidy(ax)

    return _save(fig, "null_seed_ensemble")



# %% Section 9b -- Figure 1: the 350 um differential number counts
def fig01_counts():
    """
    Draft Fig. 1 -- the Bethermin et al. (2012) 350 um differential counts and
    the three fitted models, redrawn at final size.

    The descriptive title is gone: the caption now carries the data source, the
    'All' column and the model list.  Two substantive changes come with the
    redraw:

      * MODEL COLOURS are the canonical `counts_350um.MODEL_COLORS`
        (Schechter C0 / SPL C2 / DPL C3) rather than the crimson/darkgreen/navy
        this figure alone used.  Every other model-curve figure in the paper
        already uses the canonical map; this was the last hold-out.
      * The legend is split.  Model curves and data sets are separate keys, so
        six entries fit a 90 mm column without shrinking to 4 pt.

    Reads only `num_count_fits/num_count_fit_results.json` plus the published
    table below.  No fit is re-run: the parameters ARE the cached product.
    """
    print("\n[Fig 1] 350 um number counts and the three models")
    with open(JSON_COUNTS) as fh:
        J = json.load(fh)

    #  Bethermin et al. (2012), A&A 542, A58, Table 3, 350 um 'All' column.
    #  Y = (dN/dS) S^2.5 in Jy^1.5 sr^-1 with S in Jy.
    TAB = [
        (2.1, 4709., 1342., "GOODS-N"), (3.0, 6949., 1167., "GOODS-N"),
        (4.2, 9964., 1396., "GOODS-N"), (6.0, 21510., 3858., "COSMOS-s"),
        (8.4, 23820., 3174., "COSMOS-s"), (11.9, 24402., 2274., "COSMOS-s"),
        (16.8, 24229., 3158., "COSMOS-s"), (23.8, 18652., 1605., "COSMOS-r"),
        (33.6, 15285., 1448., "COSMOS-r"), (47.4, 9092., 1187., "COSMOS-r"),
        (67.0, 3487., 828., "COSMOS-r"), (94.6, 1163., 630., "COSMOS-r"),
        (133.7, 170., 273., "COSMOS-r"),
    ]
    #  Y [Jy^1.5 sr^-1] / S_Jy^2.5  ->  dN/dS [mJy^-1 deg^-2].  The conversion
    #  is exact and multiplicative: 1 Jy^-1 sr^-1 = 1e-3 mJy^-1 / (180/pi)^2
    #  deg^-2.  Cross-check against the astropy path in
    #  `num_count_fitting_v5fig.py`: S = 16.8 mJy, Y = 24229 gives 201.8.
    SR_PER_DEG2 = (180.0 / np.pi) ** 2
    S = np.array([r[0] for r in TAB])
    Y = np.array([r[1] for r in TAB])
    YE = np.array([r[2] for r in TAB])
    MTH = np.array([r[3] for r in TAB])
    dnds = Y / (S * 1e-3) ** 2.5 * 1e-3 / SR_PER_DEG2
    dnds_e = dnds * (YE / Y)
    #  Fitted: S < 100 mJy and not GOODS-N (the exclusion adopted 2026-07-25).
    used = (S < J["fit_flux_max_mJy"]) & (MTH != "GOODS-N")

    sc, sp, dp = J["schechter"], J["spl"], J["dpl"]
    x = np.geomspace(1.0, 200.0, 400)
    curves = {
        "Schechter": (sc["nstar_deg2"] / sc["sstar_mJy"])
        * (x / sc["sstar_mJy"]) ** sc["alpha"] * np.exp(-x / sc["sstar_mJy"]),
        "SPL": sp["N0"] * (x / sp["S0_mJy"]) ** sp["beta"],
        "DPL": (dp["phistar_deg2"] / dp["sstar_mJy"])
        / ((x / dp["sstar_mJy"]) ** dp["alpha"]
           + (x / dp["sstar_mJy"]) ** dp["beta"]),
    }

    fig, ax = plt.subplots(figsize=(COL, 2.95))
    mk = {"GOODS-N": ("d", "0.45", "stacked, GOODS-N"),
          "COSMOS-s": ("s", "0.15", "stacked, COSMOS"),
          "COSMOS-r": ("o", "0.15", "resolved, COSMOS")}
    dh, seen = [], set()
    for key, (m, c, lab) in mk.items():
        for sub, face in ((used, c), (~used, "none")):
            g = (MTH == key) & sub
            if not g.any():
                continue
            #  [2026-09-16] ms 2.2 -> 3.6 and mew 0.9: at 2.2 pt the open
            #  (excluded) symbols were indistinguishable from the filled ones.
            h = ax.errorbar(S[g], dnds[g], yerr=dnds_e[g], fmt=m, ms=3.6,
                            color=c, mfc=face, mec=c, mew=0.9, lw=0.7,
                            capsize=1.2, zorder=5)
            #  Label each DATA SET once, on whichever of its two handles comes
            #  first.  GOODS-N is excluded in its entirety, so keying only the
            #  filled handles would drop it from the legend altogether and
            #  leave the grey diamonds unidentified.
            if lab not in seen:
                h.set_label(lab)
                dh.append(h)
                seen.add(lab)
    #  One key for the open symbols.  Which points are excluded is a fact the
    #  caption asserts, so the figure has to show it: GOODS-N below 6 mJy and
    #  the single point above the 100 mJy cutoff are plotted but unfitted.
    ex = ax.errorbar([], [], fmt="o", ms=3.6, color="0.15", mfc="none",
                     mec="0.15", mew=0.9, lw=0.7,
                     label="open: excluded from fits")
    dh.append(ex)
    mh = []
    for nm in MODEL_ORDER:
        mh += ax.plot(x, curves[nm], MODEL_LS[nm], color=MODEL_COLORS[nm],
                      lw=1.1, zorder=3, label=nm)
    ax.axvline(J["fit_flux_max_mJy"], color="0.55", ls=":", lw=0.6)
    #  Left of the line: the 133.7 mJy point and its error bar occupy the
    #  space to the right of it, all the way down to the axis.
    ax.text(J["fit_flux_max_mJy"] * 0.92, 6e-3, "fit cutoff", fontsize=5.4,
            color="0.45", ha="right", va="bottom")
    ax.set(xscale="log", yscale="log", xlabel=r"$S$ [mJy]",
           ylabel=r"$dN/dS$ [mJy$^{-1}$ deg$^{-2}$]",
           xlim=(1.2, 210), ylim=(2e-3, 3e4))
    lg1 = ax.legend(handles=mh, loc="lower left", fontsize=6.0,
                    borderaxespad=0.35)
    ax.add_artist(lg1)
    ax.legend(handles=dh, loc="upper right", fontsize=5.4,
              borderaxespad=0.35, title="Bethermin+12", title_fontsize=5.4)
    _tidy(ax)
    return _save(fig, "counts_fits")


# %% Section 9c -- Figure 2: ideal flux-space fingerprints
def fig02_fingerprints():
    """
    Draft Fig. 2 -- the xi(u) fingerprints of the three count models in the
    ideal case (no beam, no noise, no declustering), redrawn at final size.

    Title dropped.  The solid/dotted distinction that the old two-line title
    spelled out is now a two-entry linestyle key inside the axes, which costs
    less space than repeating each model name twice in the legend: six entries
    become three plus a key.

    Reads `paperfig_peaklevel_curves.npz` (stage 2), which is a pure analytic
    projection -- no Monte Carlo.
    """
    print("\n[Fig 2] ideal flux-space fingerprints")
    d = np.load(NPZ_PKCURVES, allow_pickle=False)
    u = d["f2_u"]
    lo, hi = d["f2_fit_range"]

    fig, ax = plt.subplots(figsize=(COL, 2.65))
    ax.axvspan(lo, hi, color=GREY_CORE, zorder=0)
    ax.text(lo * 1.06, 0.047, "fitted range of the counts", fontsize=5.4,
            color="0.45")
    for nm in MODEL_ORDER:
        c = MODEL_COLORS[nm]
        ax.plot(u, d[f"f2_proj_{nm}"], "-", color=c, lw=1.2, label=nm)
        ax.plot(u, d[f"f2_asym_{nm}"], ":", color=c, lw=0.9)
    ax.axvline(19.01, color=MODEL_COLORS["Schechter"], ls="--", lw=0.6,
               alpha=0.7)
    ax.axvline(11.70, color=MODEL_COLORS["DPL"], ls="--", lw=0.6, alpha=0.7)
    #  y = 2.2 is the one band this figure leaves clear between u = 6 and 20:
    #  above it the DPL asymptote is still falling, below it the projections
    #  run.  The upper-right corner, where these labels sat before, is now the
    #  style key.
    ax.text(20.0, 2.2, r"$S_*^{\rm Sch}$", fontsize=5.8,
            color=MODEL_COLORS["Schechter"], va="center")
    ax.text(11.0, 2.2, r"$S_*^{\rm DPL}$", fontsize=5.8, ha="right",
            color=MODEL_COLORS["DPL"], va="center")
    ax.set(xscale="log", yscale="log", xlabel=r"threshold $u$ [mJy]",
           ylabel=r"$\xi(u)$", xlim=(0.3, 60), ylim=(0.04, 12))
    lg = ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.35)
    ax.add_artist(lg)
    _style_key(ax, [("-", "GPD projection"), (":", r"$1/(\eta(u)-1)$")],
               loc="upper right")
    _tidy(ax)
    return _save(fig, "xi_fingerprints")


# %% Section 9d -- Figure 3: peak-level fingerprints
def fig03_fingerprints_peak():
    """
    Draft Fig. 3 -- the same three models after the beam, the confusion core,
    the instrument noise and the declustering step, redrawn at final size.

    Title dropped; the beam and noise it quoted are in the caption, and the
    solid/dotted key replaces the second title line as in Fig. 2.  The
    bootstrap bands and the 3 sigma_c validity floor are unchanged.
    """
    print("\n[Fig 3] peak-level fingerprints")
    d = np.load(NPZ_PKCURVES, allow_pickle=False)
    u = d["f3_u"]
    sig_c = float(d["f3_sigma_c"])

    fig, ax = plt.subplots(figsize=(COL, 2.65))
    ax.axvspan(u.min() * 0.97, 3.0 * sig_c, color=GREY_CORE, zorder=0)
    for nm in MODEL_ORDER:
        c = MODEL_COLORS[nm]
        g = np.isfinite(d[f"f3_xi_{nm}"])
        ax.plot(u, d[f"f3_asym_{nm}"], ":", color=c, lw=0.9)
        ax.plot(u[g], d[f"f3_xi_{nm}"][g], "-", color=c, lw=1.2, label=nm)
        ax.fill_between(u[g], d[f"f3_lo_{nm}"][g], d[f"f3_hi_{nm}"][g],
                        color=c, alpha=0.15, lw=0)
    ax.axvline(3.0 * sig_c, color="crimson", ls="--", lw=0.7, zorder=1)
    ax.text(3.0 * sig_c * 1.04, 0.575, r"$u=3\sigma_c$", fontsize=5.6,
            color="crimson", va="top")
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]",
           ylabel=r"$\xi$", xlim=(u.min() * 0.97, u.max() * 1.03),
           ylim=(-0.24, 0.62))
    _logticks(ax, [15, 20, 30, 40, 60, 90])
    lg = ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.35)
    ax.add_artist(lg)
    _style_key(ax, [("-", "peak level, declustered"),
                    (":", r"asymptotic $1/(\eta-1)$")], loc="upper right")
    _tidy(ax, grid=False)
    ax.grid(alpha=0.22, lw=0.4, which="both")
    return _save(fig, "xi_fingerprints_peak")


# %% Section 9e -- Figure 4: what the beam does to xi-hat
def fig04_beam_reshapes(stacked=True):
    """
    Draft Fig. 4 -- the beam sweep on the illustrative double power law,
    demoted from a two-column `figure*` to a single stacked column.

    The stacking rule from the 2026-08-22 round applies cleanly here: the two
    panels share an ordinate and differ only in the abscissa variable, and
    side-by-side inside 90 mm would force the four-entry FWHM legend to ~4.5 pt.
    Stacked, both panels keep a 6 pt legend and the figure costs 99 column-mm
    against the 132 a `figure*` would take.

    Panel titles are gone; the (a)/(b) markers move inside the axes.  The
    off-scale-asymptote annotation is kept -- it is an argument, not a title.
    """
    print("\n[Fig 4] beam reshaping of xi-hat (stacked)" if stacked
          else "\n[Fig 4] beam reshaping of xi-hat (side by side)")
    d = np.load(NPZ_PKCURVES, allow_pickle=False)
    beams = d["f4_beams"]
    beta_ = float(d["f4_beta"])
    cmap = plt.get_cmap("viridis")

    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(COL, 4.35))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))

    for k, b in enumerate(beams):
        key = f"{int(b * 10)}"
        u = d[f"f4_u_{key}"]
        xi = d[f"f4_xi_{key}"]
        sc = float(d[f"f4_sigma_c_{key}"])
        nb = float(d[f"f4_n_beam_{key}"])
        g = np.isfinite(xi)
        c = cmap(k / max(1, len(beams) - 1))
        lab = fr"{b:.0f}$''$ ($\sigma_c$={sc:.2f}, $N_b$={nb:.0f})"
        axes[0].plot(u[g], xi[g], "-o", color=c, lw=0.9, ms=1.8, label=lab)
        axes[1].plot((u / sc)[g], xi[g], "-o", color=c, lw=0.9, ms=1.8)

    for ax, xlab in ((axes[0], r"threshold $u$ [mJy/beam]"),
                     (axes[1], r"$k = u/\sigma_c$")):
        ax.axhline(1.0 / (beta_ - 1.0), color="0.35", ls="--", lw=0.7)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set(xscale="log", xlabel=xlab, ylabel=r"$\hat\xi(u)$",
               ylim=(0.0, 0.80))
        _tidy(ax, grid=False)
        ax.grid(alpha=0.22, lw=0.4, which="both")
    axes[0].text(0.035, 1.0 / (beta_ - 1.0), r" $1/(\beta-1)=0.357$",
                 fontsize=5.6, color="0.3", va="bottom",
                 transform=axes[0].get_yaxis_transform())
    axes[0].text(0.035, 0.95, r"asymptotic $1/(\alpha-1)=1.667$ off scale:"
                 "\ndeclustering compresses the range", fontsize=5.6,
                 color="0.35", va="top", transform=axes[0].transAxes)
    #  Every curve lives between xi = 0.14 and 0.62, so the strip below 0.13
    #  is clear across the full width of both panels: a two-column legend drops
    #  into it without covering data, which "lower right" did not.
    axes[0].legend(loc="lower center", fontsize=5.4, title="FWHM",
                   title_fontsize=5.4, ncol=2, columnspacing=0.9,
                   borderaxespad=0.3)
    for ax, tag in zip(axes, "ab"):
        ax.text(0.975, 0.955, f"({tag})", transform=ax.transAxes,
                fontsize=7.0, ha="right", va="top")
    fig.tight_layout(pad=0.25, h_pad=0.7 if stacked else 0.0)
    return _save(fig, "beam_reshapes_xi", alt=not stacked)


# %% Section 9f -- Figure 10: the measured Herschel xi-hat(u)
def fig10_herschel_xi():
    """
    Draft Fig. 10 -- the GAMA-09 350 um measured xi-hat(u), masked and
    unmasked, against the three peak-level simulated count models.

    Title dropped (instrument, band and field are all in the caption).  The
    chi^2 values stay in the legend: they are the figure's quantitative
    content, not decoration, and each is three characters.

    Reads `herschel_unlensed_v3_350.npz` for the measurement and
    `paperfig_peaklevel_curves.npz` (stage 10) for the model curves and the
    covariance-correct chi^2.
    """
    print("\n[Fig 10] Herschel xi-hat(u) vs peak-level models")
    D = np.load(NPZ_H_UNL, allow_pickle=True)
    K = np.load(NPZ_PKCURVES, allow_pickle=False)
    u = D["u_grid"]
    xi, xie = D["xi_meas"], D["xi_err"]
    xim, xime = D["xi_masked"], D["xierr_masked"]
    umax = float(D["u_max_masked"])
    sel = (u <= umax) & np.isfinite(xim)
    uf = K["f10_u_fine"]

    fig, ax = plt.subplots(figsize=(COL, 2.75))
    ax.axvspan(umax, u.max() * 1.03, color=GREY_CORE, zorder=0)
    ax.text(umax * 1.02, 0.63, "masked variant\ncensored", fontsize=5.4,
            color="0.42", va="top", ha="left")
    for nm in MODEL_ORDER:
        ax.plot(uf, K[f"f10_curve_{nm}"], MODEL_LS[nm],
                color=MODEL_COLORS[nm], lw=1.1, zorder=2,
                label=fr"{nm}, $\chi^2$={K[f'f10_chi2_{nm}']:.0f}")
    ax.errorbar(u, xi, yerr=xie, fmt="s", ms=2.2, color="k", capsize=1.2,
                lw=0.7, zorder=4, label="measured, unmasked")
    ax.errorbar(u[sel], xim[sel], yerr=xime[sel], fmt="o", ms=2.4, mfc="w",
                color="k", capsize=1.2, lw=0.8, zorder=5,
                label="measured, bright-masked")
    ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)
    ax.set(xlabel=r"threshold $u$ [mJy/beam]", ylabel=r"$\hat\xi(u)$",
           xlim=(u.min() * 0.96, u.max() * 1.03), ylim=(-0.30, 0.70))
    ax.legend(loc="upper left", fontsize=5.8, borderaxespad=0.35)
    _tidy(ax)
    return _save(fig, "herschel_xi_vs_flux")


# %% Section 9g -- Figure 12: the Herschel Delta_xi result
def fig12_herschel_dxi():
    """
    Draft Fig. 12 -- the measured Delta_xi(u) over 110 eFEDS cluster/control
    pairs, four apertures, with the NFW predictions.  Stays a two-column
    `figure*`: four panels sharing an ordinate do need the width.

    The module-label suptitle ('H2: lensed ...') is gone.  The per-panel
    aperture titles are KEPT -- they identify which panel is which, which is
    exactly the case where a short label earns its place -- but the peak counts
    that used to ride along in those titles move to the caption: at 46 mm per
    panel there is no room for them at 7 pt, and they are supplementary rather
    than identifying.
    """
    print("\n[Fig 12] Herschel Delta_xi, four apertures")
    d = np.load(NPZ_H_LEN, allow_pickle=True)
    u = d["u_grid"]
    pred = json.loads(str(d["pred_json"]))
    aps = [f"{a:.1f}" for a in d["ap_fracs"]]

    fig, axes = plt.subplots(1, len(aps), figsize=(DCOL, 2.10), sharey=True)
    for j, (ax, ap) in enumerate(zip(axes, aps)):
        for nm in ("Schechter", "DPL"):
            key = f"{nm}_ap{ap}"
            if key in pred:
                ax.plot(u, pred[key], MODEL_LS[nm], color=MODEL_COLORS[nm],
                        lw=1.2, zorder=3, label=f"{nm} NFW prediction")
        ax.errorbar(u, d[f"ap{ap}_dxi"], yerr=d[f"ap{ap}_err"], fmt="o",
                    color="k", ms=2.0, capsize=1.2, lw=0.7,
                    label=r"measured $\Delta\xi$")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel(r"threshold $u$ [mJy/beam]")
        ax.set_title(fr"$r<{float(ap):.1f}\,\theta_{{500}}$", fontsize=7.0)
        ax.text(0.04, 0.05, f"({'abcd'[j]})", transform=ax.transAxes,
                fontsize=7.0, va="bottom")
        _tidy(ax)
    axes[0].set_ylabel(r"$\Delta\xi(u)=\xi_{\rm cl}-\xi_{\rm ctrl}$")
    #  Top raised from 0.46 so the three-entry legend clears the u = 32.5
    #  error bar, which reaches +0.29 in panel (a).
    axes[0].set_ylim(-0.52, 0.58)
    axes[0].legend(loc="upper left", fontsize=5.6, borderaxespad=0.35)
    fig.tight_layout(pad=0.25, w_pad=0.4)
    return _save(fig, "herschel_dxi")


# %% Section 9h -- Figure 13: CCAT/FYST unlensed forecast
def fig13_ccat_unlensed():
    """
    Draft Fig. 13 -- the simulated CCAT/FYST unlensed xi-hat(u) for the three
    count models on the extended threshold ladder, with the analytic
    pixel-level curves shown for comparison.

    Title dropped: beam, noise and area are all in the caption.  As in Figs. 2
    and 3 the six-entry legend (analytic + MC per model) collapses to three
    colour entries plus a two-entry style key.

    Reads `Simulations/results/ccat.npz`; no simulation is re-run.
    """
    print("\n[Fig 13] CCAT unlensed xi-hat(u)")
    d = np.load(NPZ_CCAT, allow_pickle=True)
    u = d["u_grid"]

    fig, ax = plt.subplots(figsize=(COL, 2.70))
    for i, nm in enumerate(MODEL_ORDER):
        c = MODEL_COLORS[nm]
        ax.plot(u, d[f"unl_theory_{nm}"], "-", color=c, lw=1.0, alpha=0.9,
                label=nm)
        xi = d[f"unl_xi_{nm}"]
        g = np.isfinite(xi)
        ax.errorbar(u[g] * (1 + 0.012 * i), xi[g],
                    yerr=[xi[g] - d[f"unl_lo_{nm}"][g],
                          d[f"unl_hi_{nm}"][g] - xi[g]],
                    fmt="o", ms=2.2, color=c, capsize=1.2, lw=0.7)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]",
           ylabel=r"$\xi(u)$")
    _logticks(ax, [6, 10, 20, 40])
    lg = ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.35)
    ax.add_artist(lg)
    _style_key(ax, [("-", "analytic, pixel level"), ("o", "peak-level MC")],
               loc="upper right")
    _tidy(ax)
    return _save(fig, "ccat_unlensed_xi")


# %% Section 9i -- Figure 14: CCAT analytic Delta_xi of the benchmark lens
def fig14_ccat_lensed_theory():
    """
    Draft Fig. 14 -- the exact analytic Delta_xi(u) of the benchmark lens at
    CCAT resolution, one panel per count model, curves by aperture.  Stays a
    `figure*`.

    The suptitle carried the benchmark mass, redshift and theta_500; all three
    move to the caption.  The per-panel model names are kept (they identify the
    panels), and the mean magnifications that used to sit in every legend entry
    move to the caption as well -- they are identical across the three panels,
    so printing them three times was pure repetition.
    """
    print("\n[Fig 14] CCAT analytic Delta_xi, benchmark lens")
    d = np.load(NPZ_CCAT, allow_pickle=True)
    u = d["u_grid"]
    aps = [f"{a:.1f}" for a in d["apertures"]]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.05), sharey=True)
    for j, (ax, nm) in enumerate(zip(axes, MODEL_ORDER)):
        for k, ap in enumerate(aps):
            ax.plot(u, d[f"dxi_theory_{ap}_{nm}"],
                    color=cmap(k / max(1, len(aps) - 1)), lw=1.0,
                    label=fr"$r<{float(ap):.1f}\,\theta_{{500}}$")
        ax.axhline(0.0, color="0.4", lw=0.6)
        ax.set(xscale="log", xlabel=r"threshold $u$ [mJy/beam]")
        _logticks(ax, [6, 10, 20, 40])
        ax.set_title(nm, fontsize=7.0)
        ax.text(0.04, 0.05, f"({'abc'[j]})", transform=ax.transAxes,
                fontsize=7.0, va="bottom")
        _tidy(ax)
    axes[0].set_ylabel(r"$\Delta\xi(u)$")
    axes[1].legend(loc="lower right", fontsize=5.6, borderaxespad=0.35)
    fig.tight_layout(pad=0.25, w_pad=0.4)
    return _save(fig, "ccat_lensed_theory")


# %% Section 10 -- driver
_REGISTRY = {
    "1": [fig01_counts],
    "2": [fig02_fingerprints],
    "3": [fig03_fingerprints_peak],
    "4": [lambda: fig04_beam_reshapes(),
          lambda: fig04_beam_reshapes(stacked=False)],
    "5": [lambda: fig05_dxi_validation(),
          lambda: fig05_dxi_validation(stacked=False)],
    "6": [fig06_planck_stability],
    "7": [fig07_planck_bright],
    "8": [lambda: fig08_planck_dxi(),
          lambda: fig08_planck_dxi(stacked=True)],
    "9": [fig09_planck_aperture_dust],
    "10": [fig10_herschel_xi],
    "11": [fig11_h1c_decomposition],
    "12": [fig12_herschel_dxi],
    "13": [fig13_ccat_unlensed],
    "14": [fig14_ccat_lensed_theory],
    "B1": [figB1_noise_declustering],
    "E1": [figE1_null_seed_ensemble],
}

if __name__ == "__main__":
    print("=" * 72)
    print("  make_paper_figures.py -- A&A final-size figures")
    print(f"  1 column = {COL * 25.4:.0f} mm, 2 columns = {DCOL * 25.4:.0f} mm")
    print("=" * 72)
    made = []
    for w in WHICH:
        if w not in _REGISTRY:
            print(f"  ?? unknown figure key '{w}'")
            continue
        for fn in _REGISTRY[w]:
            try:
                made.append(fn())
            except Exception as exc:                    # noqa: BLE001
                print(f"  !! {w} FAILED: {type(exc).__name__}: {exc}")
                raise
    print(f"\n  {len(made)} file(s) written to "
          f"{os.path.relpath(FIG_DIR, _HERE)}/")
