"""
Validation of the `clustering` mode of analysis_modules/simulate_maps.py,
for Secs. 10.5-10.6 of `Simulations/full_simulation_summary_v2.md`.

CONFIGURATION.  Everything below runs on the suite's HERSCHEL setup with
the FITTED 350 um Schechter counts of Sec. 2 -- alpha = -1.890,
n_* = 7014 deg^-2, S_* = 19.01 mJy -- NOT on the ALCS 1.2 mm placeholder
defaults of counts.Schechter.  That distinction matters: the placeholder
population has sigma_c = 0.45 mJy/beam at this beam, the fitted 350 um one
has 8.06, so a threshold ladder quoted in mJy means something completely
different between them.  S_min = 0.1 mJy is the suite's adopted faint
limit (memo Sec. 3.1); it captures 99.7% of q2 -- hence essentially all of
the tail and of sigma_c -- while keeping the injected population at 1.6e6
sources per map rather than 1e8.

Five checks, in order of what they would falsify:

  V1  BACKWARDS COMPATIBILITY.  clustering=None must reproduce the pre-v2
      module bit-for-bit (max|diff| == 0), so every cached ensemble and
      every figure already in the paper remains reproducible.

  V2  POWER SPECTRUM RECOVERY.  The realized C_l of the modulation field
      1+delta must match the input power law.  This tests the Fourier
      normalization of _gaussian_field_from_cl and the Coles & Jones
      log-transform together.

  V3  CORE VARIANCE.  The measured clustering:shot ratio of the
      beam-smoothed map variance must match the closed form
          R_core = cl_ratio * Gamma(1 + s/2) * (l_eq sigma_b)^(-s).

  V4  THE TAIL IS UNCHANGED.  xi-hat(u) from declustered peaks must agree
      between clustered and Poisson skies across the fitting window.  This
      is the prediction of the core-tail contrast argument (memo 10.3).

  V5  THE SCATTER.  Over-dispersion of declustered exceedance counts in
      cutout-sized cells, measured by the Fano factor F = Var(N)/E[N].

A CALIBRATION SUBTLETY WORTH KNOWING (memo Secs. 10.6, 11).  The module
sets C_l^dd = (q2/q1^2)(l/l_eq)^s from the injected population, which makes
the map's FLUX clustering power equal its own shot power at l_eq -- right
for MAP statistics (V2, V3, V4), which therefore need no correction.  But
delta itself, and hence any COUNT statistic (V5), depends on q1 separately,
and this counts model over-predicts q1 by 2.67x against the measured CIB
monopole (0.576 +/- 0.034 MJy/sr at 857 GHz, Odegard et al. 2019).  So V5
as run here UNDER-states the over-dispersion.  The corrected version, via
the module's cl_q1_ref argument, is in where_clustering_matters.py, and it
is the one to quote.

Run:  python3 verify_clustering_mode.py         (~30 s)
"""

# %% imports and configuration
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..",
                                                 "analysis_modules")))

from counts import Schechter                                     # noqa: E402
from simulate_maps import (CIBMapSimulator, gaussian_beam_convolve,   # noqa: E402
                           ARCSEC_TO_RAD)
from gpd_tail import beam_declustered_peaks, fit_gpd             # noqa: E402

#  Herschel row of the suite's instrument table (memo Sec. 3.4)
BEAM, PIX, NPIX = 25.0, 5.0, 1024
S_MIN, S_CUT = 0.1, 100.0
NMAPS = 24
SIGMA_C = 8.06                      # counts_350um.py value for this config
U_GRID = [5.0, 10.0, 15.0, 25.0, 35.0, 45.0]


def counts_350um():
    """The fitted 350 um Schechter of memo Sec. 2 (NOT the ALCS default)."""
    return Schechter(alpha=-1.890, log_sstar=np.log10(19.01),
                     log_phistar=np.log10(7014.0), s_min=S_MIN, s_max=1e3)


def make_sim(clustering=None, **kw):
    return CIBMapSimulator(counts_350um(), BEAM, PIX, npix=NPIX,
                           sigma_noise=0.0, s_cut=S_CUT,
                           clustering=clustering, **kw)


def banner(t):
    print("\n" + "=" * 76 + f"\n  {t}\n" + "=" * 76)


# %% V1 -- backwards compatibility of the Poisson default
def v1_backwards_compat():
    banner("V1  clustering=None reproduces the pre-v2 module")
    sim = make_sim(None)
    for seed in (0, 1, 7):
        rng = np.random.default_rng(seed)
        raw = np.zeros((NPIX, NPIX))
        lam = sim.cnts.dnds(sim.s_mid) * sim.ds * sim.om_pix * NPIX ** 2
        for s, l_tot in zip(sim.s_mid, lam):
            n = rng.poisson(l_tot)
            if n == 0:
                continue
            iy = rng.integers(0, NPIX, n)
            ix = rng.integers(0, NPIX, n)
            np.add.at(raw, (iy, ix), s)
        ref = gaussian_beam_convolve(raw, sim.beam_fwhm_pix)
        ref -= ref.mean()
        d = np.abs(ref - sim.make_map(seed=seed)).max()
        print(f"  seed {seed}:  max|diff| = {d:.3e}   "
              f"{'PASS' if d == 0.0 else 'FAIL'}")


# %% V2 -- does the modulation field have the requested power spectrum?
def radial_cl(field, pix_arcsec, nbin=24):
    """Measured C(l) of a field on a periodic square grid: P_k = |FFT|^2 A/N^4."""
    n = field.shape[0]
    box = n * pix_arcsec * ARCSEC_TO_RAD
    p = np.abs(np.fft.fft2(field - field.mean())) ** 2 * box ** 2 / n ** 4
    k = np.fft.fftfreq(n, d=1.0 / n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    ell = 2 * np.pi * np.sqrt(kx ** 2 + ky ** 2) / box
    m = ell > 0
    edges = np.geomspace(ell[m].min(), ell[m].max(), nbin + 1)
    idx = np.digitize(ell[m], edges) - 1
    out_l, out_c = [], []
    for b in range(nbin):
        sel = idx == b
        if sel.sum() > 8:
            out_l.append(ell[m][sel].mean())
            out_c.append(p[m][sel].mean())
    return np.array(out_l), np.array(out_c)


def v2_power_spectrum():
    banner("V2  realized C_l of (1+delta) vs the input power law")
    for mode in ("gaussian", "lognormal"):
        sim = make_sim(mode)
        info = sim.clustering_info()
        acc_l, acc_c = None, 0.0
        for i in range(8):
            d = sim._modulation(np.random.default_rng(1000 + i))
            acc_l, c = radial_cl(d, PIX)
            acc_c = acc_c + c / 8
        tgt = info["cl_amp_sr"] * (acc_l / info["ell_eq"]) ** info["cl_slope"]
        ok = (acc_l > 2e3) & (acc_l < 5e4)
        r = acc_c[ok] / tgt[ok]
        print(f"  {mode:10s} sigma_delta = {info['sigma_delta']:.3f}   "
              f"C_meas/C_target over l = 2e3-5e4:  median {np.median(r):.3f}"
              f"  range {r.min():.3f}-{r.max():.3f}")


# %% V3 -- the core variance ratio against the closed form
def v3_core_variance(nmaps=40):
    banner("V3  measured clustering:shot variance ratio vs closed form")
    sp, sc = make_sim(None), make_sim("lognormal")
    vp = np.array([sp.make_map(seed=i).var() for i in range(nmaps)])
    vc = np.array([sc.make_map(seed=i).var() for i in range(nmaps)])
    r = vc.mean() / vp.mean() - 1.0
    se = np.hypot(vc.std(ddof=1) / np.sqrt(nmaps) / vp.mean(),
                  vc.mean() * vp.std(ddof=1) / np.sqrt(nmaps) / vp.mean() ** 2)
    print(f"  Var(Poisson)   = {vp.mean():8.3f} (mJy/beam)^2   "
          f"-> sigma = {np.sqrt(vp.mean()):.2f}  [model sigma_c = {SIGMA_C}]")
    print(f"  Var(clustered) = {vc.mean():8.3f} (mJy/beam)^2")
    print(f"  measured R_core = {r:.3f} +/- {se:.3f}    closed form = "
          f"{sc.clustering_info()['R_core']:.3f}   ({nmaps} maps each)")


# %% V4/V5 -- the two predictions
def v4v5(nmaps=NMAPS, block=256, u_fano=25.0):
    banner("V4  tail shape unchanged   /   V5  cutout-scale over-dispersion")
    sp, sc = make_sim(None), make_sim("lognormal")
    nb = NPIX // block
    out = {}
    for tag, sim in (("Poisson", sp), ("clustered", sc)):
        peaks, cells, dvar = [], [], []
        for i in range(nmaps):
            m = sim.make_map(seed=500 + i)
            pk = beam_declustered_peaks(m, sim.beam_fwhm_pix)
            peaks.append(pk)
            loc = np.zeros_like(m, dtype=bool)
            from scipy.ndimage import maximum_filter
            size = max(1, int(round(sim.beam_fwhm_pix)))
            loc = (m == maximum_filter(m, size=size)) & (m > u_fano)
            cells.append(loc.reshape(nb, block, nb, block).sum(axis=(1, 3)))
            if sim.clustering is not None:
                d = sim._modulation(np.random.default_rng(500 + i))
                dvar.append(d.reshape(nb, block, nb, block).mean(axis=(1, 3)).var())
        out[tag] = (np.concatenate(peaks),
                    np.concatenate([c.ravel() for c in cells]).astype(float),
                    np.mean(dvar) if dvar else 0.0)
        print(f"  {tag}: {out[tag][0].size:,} peaks over {nmaps} maps")

    print(f"\n  V4  GPD shape from declustered peaks (sigma_c = {SIGMA_C} "
          f"mJy/beam)")
    print(f"{'u [mJy]':>9} {'u/sig_c':>8} {'xi Poisson':>18} "
          f"{'xi clustered':>18} {'Delta xi':>18} {'se':>6}")
    for u in U_GRID:
        a = fit_gpd(out["Poisson"][0][out["Poisson"][0] > u] - u)
        b = fit_gpd(out["clustered"][0][out["clustered"][0] > u] - u)
        d, se = b["xi"] - a["xi"], np.hypot(a["se_xi"], b["se_xi"])
        print(f"{u:>9.1f} {u/SIGMA_C:>8.1f} "
              f"{a['xi']:>+10.4f}+/-{a['se_xi']:.4f} "
              f"{b['xi']:>+10.4f}+/-{b['se_xi']:.4f} "
              f"{d:>+10.4f}+/-{se:.4f} {abs(d)/se:>6.2f}")

    print(f"\n  V5  Fano factor of exceedance counts above u = {u_fano} mJy, "
          f"{block*PIX/60:.1f}' cells ({nmaps*nb*nb} cells)")
    for tag in ("Poisson", "clustered"):
        c = out[tag][1]
        F = c.var(ddof=1) / c.mean()
        extra = (f"   naive prediction {1 + c.mean()*out[tag][2]:.3f}"
                 if out[tag][2] else "")
        print(f"    {tag:10s} Nbar = {c.mean():6.2f}   "
              f"F = {F:.3f} +/- {F*np.sqrt(2/(c.size-1)):.3f}{extra}")


if __name__ == "__main__":
    v1_backwards_compat()
    v2_power_spectrum()
    v3_core_variance()
    v4v5()
