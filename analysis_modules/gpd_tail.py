"""
gpd_tail.py — Module 5 of the CIB-lensing / GPD project
========================================================

Peaks-over-threshold (POT) fitting of the Generalized Pareto Distribution
(GPD) to the tail of P(D) maps, packaging the recipe of the project manual
(`GPD_tail_modelling_manual_v3.md`, Secs. 7, 7b, 8, 12) into reusable
functions, together with the ANALYTIC confusion-noise computation and a
scalable instrument-noise prescription.

Confusion noise and the scalable Gaussian component
---------------------------------------------------
Following `confusion_noise_reference.md` (Secs. 2.2, 3.2, 4.1), the
cumulants of the confusion field factorize into beam moments
Omega_n = Omega_beam/n (Gaussian beam) and flux moments of the counts,

    q_n = int_{S_min}^{S_cut} S^n (dN/dS) dS ,

giving the confusion variance  sigma_c^2 = Omega_2 q_2 = (Omega_beam/2) q_2
and the skewness diagnostic  gamma_1 = (2 sqrt2 / 3) q_3 q_2^{-3/2}
Omega_beam^{-1/2}.  For our Schechter / DPL models q_n is computed
NUMERICALLY (counts.BaseCounts.moment); the Schechter cutoff (or the DPL
bright slope beta > 3, or the mask S_cut) keeps q_2 finite, so the memo's
Sec. 3.4 divergence warning is respected by construction — `confusion_sigma`
reports gamma_1 and N_beam so the quality of the Gaussian-core
approximation is always visible.

The additional (instrument + other astrophysical) Gaussian noise is then
specified RELATIVE to the confusion noise through the configuration
constant NOISE_OVER_CONFUSION (= f_N):

    sigma_N = f_N * sigma_c ,      Sigma_total^2 = sigma_c^2 + sigma_N^2 .

f_N = 0: confusion-limited (Herschel-like);  f_N ~ 1: comparable;
f_N > 1: noise-dominated (Planck-like regime).

GPD machinery (manual Secs. 7-8, 12)
------------------------------------
* `beam_declustered_peaks` — collapse beam-correlated pixels to local
  maxima (~ one peak per source): the approximately independent extremes.
* `fit_gpd` — maximum-likelihood (xi, sigma) for exceedances above u,
  with the asymptotic error  se(xi) ~ (1+xi)/sqrt(N_exc).
* `mean_excess`, `xi_vs_threshold`, `suggest_threshold` — the Sec.-7b
  threshold diagnostics (mean-residual-life plot; parameter-stability
  scan with bootstrap bands; plateau + statistics-ceiling heuristic).
* `robust_core` — core centroid/width from the median and the LOWER
  16th-percentile distance (the heavy right tail never enters), used to
  anchor thresholds as  u = mu_core + k * sigma_core  (manual Sec. 7b).
* `block_bootstrap_xi` — spatial (tile) bootstrap for honest errors.
* `fit_evmm` — Route B of manual Sec. 12.3: Normal bulk + GPD tail with
  the threshold as a free parameter; the bulk absorbs the TOTAL Gaussian
  budget (confusion core + instrument noise) without needing the split.
* `xi_of_threshold_analytic` — the POPULATION value of xi(u): the KL
  projection of the exact (analytic, pixel-level) P(D) exceedance density
  onto the GPD family.  Deterministic and MC-noise-free; used for the
  xi(u; mu) science curves.  NOTE: it describes PIXEL exceedances, while
  the map pipeline fits DECLUSTERED-PEAK exceedances (~ per-source
  statistics); the two agree in trend but not exactly in value — always
  compare like with like across fields, never mix the two conventions.

Units: as elsewhere (mJy, mJy/beam, arcsec beam FWHM).
Requires numpy, scipy (+ project modules counts / pofd_analytic /
lens_model / simulate_maps for the demos).

Typical usage
-------------
>>> from counts import Schechter
>>> from gpd_tail import confusion_sigma, NOISE_OVER_CONFUSION
>>> cs = confusion_sigma(Schechter(s_min=1e-3), 14.9, s_cut=100.0)
>>> sigma_n = NOISE_OVER_CONFUSION * cs["sigma_c"]      # scalable noise
>>> # ... simulate maps with sigma_noise=sigma_n, then:
>>> from gpd_tail import beam_declustered_peaks, fit_gpd
>>> peaks = beam_declustered_peaks(map2d, beam_fwhm_pix)
>>> res = fit_gpd(peaks[peaks > u] - u)

Run the file directly for the demo suite (synthetic recovery test,
noise-budget printout, map-based stability scan lensed vs control, and
the analytic xi(u; mu) curves).  Full demo runtime ~2 minutes.
"""

# %% Imports
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

from counts import BaseCounts, FWHM_TO_SIGMA, DEG2_TO_ARCSEC2  # noqa: F401

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# %% Configuration
#  NOISE_OVER_CONFUSION (f_N): instrument/astrophysical Gaussian noise as
#  a multiple of the ANALYTIC confusion noise sigma_c of the same counts
#  model, beam and S_cut:  sigma_N = f_N * sigma_c.  Override per call or
#  via the environment variable CIB_NOISE_OVER_CONF.
NOISE_OVER_CONFUSION = 1.0
import os as _os                                   # noqa: E402
NOISE_OVER_CONFUSION = float(_os.environ.get("CIB_NOISE_OVER_CONF",
                                             NOISE_OVER_CONFUSION))

#  Plotting configuration (demo cells): default interactive; set True or
#  export CIB_SAVE_FIGURES=1 to write PNGs into FIGURE_DIR.
SAVE_FIGURES = bool(int(_os.environ.get("CIB_SAVE_FIGURES", "0")))
FIGURE_DIR = "."


def _finish_figure(fig, name):
    """Demo helper: show the figure (default) or save FIGURE_DIR/name.png."""
    if SAVE_FIGURES:
        path = _os.path.join(FIGURE_DIR, name + ".png")
        fig.savefig(path, dpi=140)
        print(f"Wrote {path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


# %% Analytic confusion noise (memo Secs. 2.2 / 3.2 / 4.1)
def confusion_sigma(cnts, beam_fwhm_arcsec, s_lo=None, s_cut=None):
    """
    Analytic confusion statistics for a Gaussian beam and a counts model.

    Returns a dict with
      sigma_c  : sqrt(Omega_2 q_2)                 [mJy/beam]
      q2, q3   : flux moments over [s_lo, s_cut]   [mJy^n deg^-2]
      gamma1   : compound-Poisson skewness (2sqrt2/3) q3 q2^-1.5 / sqrt(Om_b)
      n_beam   : sources per beam above s_lo (CLT-quality diagnostic)
    q_n is computed numerically, so any counts model (Schechter, DPL,
    lensed, mixture) is handled identically.
    """
    sig_b = beam_fwhm_arcsec * FWHM_TO_SIGMA                 # arcsec
    om_beam = 2.0 * np.pi * sig_b ** 2 / DEG2_TO_ARCSEC2     # deg^2
    q2 = cnts.moment(2, s_lo, s_cut)
    q3 = cnts.moment(3, s_lo, s_cut)
    n0 = cnts.moment(0, s_lo, s_cut)
    return dict(sigma_c=np.sqrt(0.5 * om_beam * q2),
                q2=q2, q3=q3,
                gamma1=(2.0 * np.sqrt(2.0) / 3.0) * q3 / q2 ** 1.5
                       / np.sqrt(om_beam),
                n_beam=om_beam * n0)


def sigma_noise_scaled(cnts, beam_fwhm_arcsec, factor=None,
                       s_lo=None, s_cut=None):
    """sigma_N = factor * sigma_c (factor defaults to NOISE_OVER_CONFUSION)."""
    f = NOISE_OVER_CONFUSION if factor is None else float(factor)
    return f * confusion_sigma(cnts, beam_fwhm_arcsec, s_lo, s_cut)["sigma_c"]


# %% Declustering and robust core (manual Secs. 7-8)
def beam_declustered_peaks(map2d, beam_fwhm_pix):
    """Local maxima separated by ~one beam FWHM: the (approximately)
    independent extremes of a beam-correlated map (manual Sec. 8).
    NaN pixels (masked) are excluded."""
    m = np.where(np.isfinite(map2d), map2d, -np.inf)
    size = max(1, int(round(beam_fwhm_pix)))
    is_max = (m == maximum_filter(m, size=size)) & np.isfinite(map2d)
    return map2d[is_max]


def robust_core(values):
    """Core centroid/width immune to the heavy right tail: centroid =
    median; width = median - 15.87th percentile (the lower 1-sigma
    distance, which the source tail barely touches).  Manual Sec. 7b."""
    v = values[np.isfinite(values)]
    mu = np.median(v)
    return mu, mu - np.percentile(v, 100.0 * 0.158655)


# %% GPD maximum likelihood (manual Sec. 7, step 3)
def gpd_nll(theta, y):
    """Negative log-likelihood; theta = (xi, log_sigma) so sigma > 0."""
    xi, log_sigma = theta
    sigma = np.exp(log_sigma)
    z = 1.0 + xi * y / sigma
    if np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-8:                          # exponential limit
        return y.size * log_sigma + y.sum() / sigma
    return y.size * log_sigma + (1.0 + 1.0 / xi) * np.log(z).sum()


def fit_gpd(y, x0=(0.1, None)):
    """
    MLE for GPD exceedances y > 0.  Returns dict(xi, sigma, n, se_xi, nll).
    se_xi is the asymptotic (Smith 1984) error (1+xi)/sqrt(n), valid for
    xi > -0.5; use block_bootstrap_xi for honest map-level errors.
    """
    y = np.asarray(y, float)
    y = y[y > 0]
    if y.size < 10:
        return dict(xi=np.nan, sigma=np.nan, n=y.size,
                    se_xi=np.nan, nll=np.nan)
    ls0 = np.log(y.std() + 1e-12) if x0[1] is None else x0[1]
    res = minimize(gpd_nll, x0=[x0[0], ls0], args=(y,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000})
    xi, sigma = res.x[0], np.exp(res.x[1])
    return dict(xi=xi, sigma=sigma, n=y.size,
                se_xi=(1.0 + xi) / np.sqrt(y.size), nll=res.fun)


# %% Threshold diagnostics (manual Sec. 7b)
def mean_excess(peaks, thresholds):
    """Mean-residual-life curve e(u) = <D - u | D > u> (linear for a GPD,
    slope xi/(1-xi); a kink marks a count break — see project notes)."""
    return np.array([(peaks[peaks > u] - u).mean()
                     if np.sum(peaks > u) > 5 else np.nan
                     for u in thresholds])


def xi_vs_threshold(peaks, thresholds, n_boot=200, min_exceed=40, seed=0):
    """
    Parameter-stability scan: MLE xi at each threshold plus a bootstrap
    16-84% band (i.i.d. resampling of the *declustered* peaks, which are
    approximately independent).  Returns dict of arrays:
    xi, xi_lo, xi_hi, sigma_star (= sigma - xi*u), n_exc.
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("xi", "xi_lo", "xi_hi", "sigma_star", "n_exc")}
    for u in thresholds:
        y = peaks[peaks > u] - u
        if y.size < min_exceed:
            for k in out:
                out[k].append(np.nan)
            continue
        f = fit_gpd(y)
        boots = []
        for _ in range(n_boot):
            fb = fit_gpd(rng.choice(y, y.size, replace=True),
                         x0=(f["xi"], np.log(f["sigma"])))
            boots.append(fb["xi"])
        lo, hi = np.nanpercentile(boots, [16, 84])
        out["xi"].append(f["xi"])
        out["xi_lo"].append(lo)
        out["xi_hi"].append(hi)
        out["sigma_star"].append(f["sigma"] - f["xi"] * u)
        out["n_exc"].append(y.size)
    return {k: np.asarray(v, float) for k, v in out.items()}


def suggest_threshold(thresholds, scan, delta_xi_target=0.1):
    """
    Manual Sec. 7b heuristic (ALWAYS confirm on the stability plot):
    lowest threshold that (a) sits on the plateau — its xi consistent
    with the highest-u estimate within the band — and (b) keeps
    N_exc >= (1+xi)^2 / delta_xi_target^2.  Returns u or None.
    """
    xi, lo, hi, nexc = (scan["xi"], scan["xi_lo"],
                        scan["xi_hi"], scan["n_exc"])
    valid = np.where(np.isfinite(xi))[0]
    if valid.size == 0:
        return None
    ref = xi[valid[-1]]
    need = (1.0 + np.nan_to_num(xi)) ** 2 / delta_xi_target ** 2
    ok = np.where((lo <= ref) & (ref <= hi) & (nexc >= need))[0]
    return thresholds[ok[0]] if ok.size else None


# %% Spatial block bootstrap (manual Sec. 7, step 4)
def block_bootstrap_xi(map2d, beam_fwhm_pix, u, tile=64, n_boot=200,
                       min_exceed=20, seed=0):
    """
    Honest error on xi-hat: decluster each `tile` x `tile` block
    separately, resample blocks with replacement, pool the peaks, refit.
    Tiles should span several beams (declustering edge effects are then
    negligible).  Returns (xi_16, xi_50, xi_84) percentiles.
    """
    rng = np.random.default_rng(seed)
    ny, nx = map2d.shape
    tiles = [map2d[i:i + tile, j:j + tile]
             for i in range(0, ny - tile + 1, tile)
             for j in range(0, nx - tile + 1, tile)]
    tile_peaks = [beam_declustered_peaks(t, beam_fwhm_pix) for t in tiles]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tile_peaks), len(tile_peaks))
        peaks = np.concatenate([tile_peaks[k] for k in pick])
        y = peaks[peaks > u] - u
        if y.size >= min_exceed:
            out.append(fit_gpd(y)["xi"])
    if len(out) < 10:
        return np.array([np.nan] * 3)
    return np.percentile(out, [16, 50, 84])


# %% Extreme-value mixture model, Route B (manual Sec. 12.3)
def evmm_normgpd_nll(theta, x):
    """Normal bulk (TOTAL Gaussian budget: confusion core + instrument
    noise) spliced to a GPD tail at a FITTED threshold u.
    theta = (mu, log_s, u, log_sigma, xi)."""
    from scipy.special import erfc
    mu, log_s, u, log_sigma, xi = theta
    s, sigma = np.exp(log_s), np.exp(log_sigma)
    z = (u - mu) / s
    phiu = 0.5 * erfc(z / np.sqrt(2.0))            # tail fraction 1-Phi(u)
    if phiu <= 0 or phiu >= 1:
        return np.inf
    below = x <= u
    ll = np.empty_like(x)
    ll[below] = (-0.5 * ((x[below] - mu) / s) ** 2
                 - np.log(s) - 0.5 * np.log(2 * np.pi))
    y = x[~below] - u
    zz = 1.0 + xi * y / sigma
    if np.any(zz <= 0):
        return np.inf
    ll[~below] = (np.log(phiu) - np.log(sigma)
                  - (1.0 + 1.0 / xi) * np.log(zz)) if abs(xi) > 1e-8 else \
        (np.log(phiu) - np.log(sigma) - y / sigma)
    return -np.sum(ll)


def fit_evmm(x):
    """MLE for the Normal+GPD mixture on (declustered) values spanning
    core and tail.  Returns dict(mu, sigma_total, u, sigma_gpd, xi, nll).
    Cross-check the fitted u against the Route-A stability plot (the EVMM
    threshold can chase tail fluctuations; manual Sec. 12.5)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    mu_r, s_r = robust_core(x)
    best = None
    #  multiple threshold starts: the (u, xi) likelihood surface is
    #  multimodal when bulk and tail overlap (manual Sec. 12.5)
    for q in (90.0, 95.0, 98.0):
        x0 = [mu_r, np.log(s_r + 1e-12),
              np.percentile(x, q), np.log(s_r + 1e-12), 0.1]
        res = minimize(evmm_normgpd_nll, x0, args=(x,),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8,
                                "maxiter": 20000})
        if best is None or res.fun < best.fun:
            best = res
    mu, log_s, u, log_sigma, xi = best.x
    return dict(mu=mu, sigma_total=np.exp(log_s), u=u,
                sigma_gpd=np.exp(log_sigma), xi=xi, nll=best.fun)


# %% Population xi(u) from the analytic P(D)  (KL projection)
def xi_of_threshold_analytic(d, p, u_grid, x0=(0.1, None), multistart=True):
    """
    For each threshold u: project the EXACT exceedance density of the
    analytic pixel P(D) (grid d, density p — e.g. PofD.d, PofD.p) onto
    the GPD family by minimizing the KL divergence; this is the value the
    (pixel-level) MLE xi-hat converges to in the infinite-data limit.
    Returns (xi_arr, sigma_arr).  Deterministic — no sampling noise.

    multistart : bool (default True)
        Run Nelder-Mead from several xi seeds at EVERY threshold and keep
        the best.  The original implementation warm-started each threshold
        from the previous one's solution, which is faster but can latch
        onto a bad local optimum and then propagate it up the whole grid
        (observed at a 10" beam with the illustrative DPL: a spurious,
        monotonically rising branch reaching xi ~ 1.07).  Set False to
        recover the old warm-start behaviour.

    NUMERICAL CAVEAT.  The KL integral runs over the whole tabulated grid
    above u.  `pofd_analytic._pofd_from_rate` clips its FFT output with
    np.maximum(p, 0), so a grid that extends far beyond the true support
    of P(D) carries a small positive floor of numerical noise, which the
    projection reads as an extremely heavy tail.  Keep PofD's `d_span`
    only modestly larger than the true support (a few times S_cut), or
    truncate (d, p) before calling this.
    """
    xi_out, sg_out = [], []
    x_prev = None
    for u in u_grid:
        m = d > u
        y, pe = d[m] - u, p[m]
        norm = _trapz(pe, y)
        if norm <= 0 or y.size < 50:
            xi_out.append(np.nan)
            sg_out.append(np.nan)
            continue
        pe = pe / norm

        def kl_nll(theta):
            xi, log_sigma = theta
            sigma = np.exp(log_sigma)
            z = 1.0 + xi * y / sigma
            if np.any(z <= 0):
                return np.inf
            if abs(xi) < 1e-8:
                lh = -log_sigma - y / sigma
            else:
                lh = -log_sigma - (1.0 + 1.0 / xi) * np.log(z)
            return -_trapz(pe * lh, y)

        ls0 = np.log(np.sqrt(_trapz(pe * y ** 2, y)) + 1e-12)
        seeds = [[s, ls0] for s in (0.05, 0.2, 0.5, 1.0)] if multistart \
            else [x_prev if x_prev is not None else [x0[0], ls0]]
        if multistart and x_prev is not None:
            seeds.append(list(x_prev))
        best = None
        for st in seeds:
            res = minimize(kl_nll, st, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-10,
                                    "maxiter": 5000})
            if best is None or res.fun < best.fun:
                best = res
        x_prev = list(best.x)
        xi_out.append(best.x[0])
        sg_out.append(np.exp(best.x[1]))
    return np.asarray(xi_out), np.asarray(sg_out)


# %% Demo / self-test suite
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from counts import Schechter
    from pofd_analytic import PofD
    from lens_model import UniformMu
    from simulate_maps import CIBMapSimulator

    rng = np.random.default_rng(42)

    # %% Demo A — synthetic recovery test (manual Sec. 9)
    xi_true, sigma_true = 0.35, 0.10
    from scipy.stats import genpareto
    tail = genpareto.rvs(c=xi_true, loc=0.0, scale=sigma_true,
                         size=4000, random_state=rng)
    core = rng.normal(0.0, 0.03, size=40000)
    sample = np.concatenate([core, tail])
    #  threshold must clear the Gaussian core (>~5 sigma_core here): a
    #  too-low u mixes core samples into the exceedances and *inflates*
    #  xi-hat badly (we measured xi ~ 0.9 with u at the 92nd percentile) —
    #  the quantitative version of the manual's Sec.-7b bias warning.
    u0 = np.percentile(sample, 97.5)
    fA = fit_gpd(sample[sample > u0] - u0)
    print("Demo A  synthetic GPD recovery:")
    print(f"  true xi = {xi_true:.3f} | fit xi = {fA['xi']:.3f} "
          f"+- {fA['se_xi']:.3f}  (n_exc = {fA['n']}, u = {u0:.3f})")

    # %% Demo B — analytic noise budget with the scalable factor
    sch = Schechter(s_min=1e-3)
    beam, pix, s_cut = 14.9, 4.0, 100.0
    cs = confusion_sigma(sch, beam, s_cut=s_cut)
    f_n = NOISE_OVER_CONFUSION
    sigma_n = f_n * cs["sigma_c"]
    sigma_tot = np.hypot(cs["sigma_c"], sigma_n)
    print("\nDemo B  noise budget (Schechter, 14.9\" beam, S_cut=100 mJy):")
    print(f"  sigma_c (analytic, q2)   = {cs['sigma_c']:.4f} mJy/beam")
    print(f"  q2 = {cs['q2']:.4g}  gamma1(skew) = {cs['gamma1']:.3f}  "
          f"N_beam = {cs['n_beam']:.1f}")
    print(f"  f_N = {f_n:.2f}  ->  sigma_N = {sigma_n:.4f}, "
          f"Sigma_total = {sigma_tot:.4f} mJy/beam")

    # %% Demo C — map-based stability scan: lensed (mu=3) vs control
    mu_lens = 3.0
    sim = CIBMapSimulator(sch, beam, pix, npix=1024,
                          sigma_noise=sigma_n, s_cut=s_cut)
    ctrl = [sim.make_map(seed=i) for i in range(2)]
    lens = [sim.make_map(lens=UniformMu(mu_lens), seed=100 + i)
            for i in range(2)]
    bfp = beam / pix
    pk_c = np.concatenate([beam_declustered_peaks(m, bfp) for m in ctrl])
    pk_l = np.concatenate([beam_declustered_peaks(m, bfp) for m in lens])

    mu0, sg0 = robust_core(np.concatenate([m.ravel() for m in ctrl]))
    kk = np.arange(1.5, 7.01, 0.5)
    uu = mu0 + kk * sg0                       # thresholds anchored to core
    scan_c = xi_vs_threshold(pk_c, uu, n_boot=100, seed=1)
    scan_l = xi_vs_threshold(pk_l, uu, n_boot=100, seed=2)

    u_star = suggest_threshold(uu, scan_c, delta_xi_target=0.1)
    print("\nDemo C  lensed vs control (declustered peaks):")
    print(f"  core: mu = {mu0:+.4f}, sigma = {sg0:.4f} mJy/beam "
          f"(analytic Sigma_total = {sigma_tot:.4f}; the robust core is "
          "narrower because the cumulant sigma_c includes bright-tail "
          "variance — memo Sec. 4.3)")
    print(f"  suggested u = {u_star:.3f} mJy/beam" if u_star is not None
          else "  no threshold satisfied the heuristic")
    if u_star is not None:
        fc = fit_gpd(pk_c[pk_c > u_star] - u_star)
        fl = fit_gpd(pk_l[pk_l > u_star] - u_star)
        bb_c = block_bootstrap_xi(ctrl[0], bfp, u_star, n_boot=100)
        bb_l = block_bootstrap_xi(lens[0], bfp, u_star, n_boot=100)
        dxi = fl["xi"] - fc["xi"]
        edxi = np.hypot(0.5 * (bb_c[2] - bb_c[0]),
                        0.5 * (bb_l[2] - bb_l[0]))
        print(f"  xi_control = {fc['xi']:+.3f} "
              f"(block-boot {bb_c[0]:+.3f}..{bb_c[2]:+.3f}, n={fc['n']})")
        print(f"  xi_lensed  = {fl['xi']:+.3f} "
              f"(block-boot {bb_l[0]:+.3f}..{bb_l[2]:+.3f}, n={fl['n']})")
        print(f"  Delta xi   = {dxi:+.3f} +- {edxi:.3f}")

    # figure: stability scan + mean excess
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for scan, c_, lab in [(scan_c, "C0", "control"),
                          (scan_l, "C3", f"lensed mu={mu_lens}")]:
        ax[0].fill_between(kk, scan["xi_lo"], scan["xi_hi"],
                           color=c_, alpha=0.25)
        ax[0].plot(kk, scan["xi"], c_ + "o-", ms=4, label=lab)
    if u_star is not None:
        ax[0].axvline((u_star - mu0) / sg0, color="0.5", ls=":",
                      label="chosen u")
    ax[0].axhline(0, color="0.8", lw=0.8)
    ax[0].set(xlabel=r"threshold  $k$  (u = $\mu_c + k\sigma_c$)",
              ylabel=r"$\hat\xi(u)$", title="parameter-stability scan")
    ax[0].legend(fontsize=8)
    me_c = mean_excess(pk_c, uu)
    me_l = mean_excess(pk_l, uu)
    ax[1].plot(uu, me_c, "C0o-", ms=4, label="control")
    ax[1].plot(uu, me_l, "C3o-", ms=4, label=f"lensed mu={mu_lens}")
    ax[1].set(xlabel="threshold u  [mJy/beam]",
              ylabel=r"$e(u) = \langle D-u\,|\,D>u\rangle$",
              title="mean-excess (linear for GPD)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, "demo_gpd_stability")

    # %% Demo D — population xi(u) and xi(mu): the science curves
    #  (pixel-level KL projection of the analytic P(D); deterministic)
    print("\nDemo D  analytic xi(u; mu) curves (pixel-level population "
          "values)...")
    mus = [1.0, 1.5, 2.0, 3.0, 5.0]
    u_grid = mu0 + np.arange(2.0, 8.01, 0.25) * sg0
    fig2, ax2 = plt.subplots(1, 2, figsize=(11.5, 4.4))
    xi_at_u4 = []
    for m_, c_ in zip(mus, ["k", "C0", "C1", "C2", "C3"]):
        pp = PofD(sch, beam, mu=m_, sigma_noise=sigma_n, s_cut=s_cut,
                  n_fft=2 ** 16)
        xi_u, _ = xi_of_threshold_analytic(pp.d, pp.p, u_grid)
        ax2[0].plot((u_grid - mu0) / sg0, xi_u, c_, label=f"mu = {m_}")
        xi_at_u4.append(np.interp(mu0 + 4 * sg0, u_grid, xi_u))
    ax2[0].axhline(0, color="0.8", lw=0.8)
    ax2[0].set(xlabel=r"threshold $k$", ylabel=r"$\xi_{\rm pop}(u)$",
               title="population GPD shape vs threshold")
    ax2[0].legend(fontsize=8)
    ax2[1].plot(mus, xi_at_u4, "ko-")
    ax2[1].set(xlabel=r"magnification $\mu$",
               ylabel=r"$\xi_{\rm pop}(u = \mu_c + 4\sigma_c)$",
               title=r"the $\xi$-$\mu$ lever (fixed threshold)")
    fig2.tight_layout()
    _finish_figure(fig2, "demo_gpd_xi_vs_mu")
    print("  xi(mu) at k=4:",
          ", ".join(f"mu={m}: {x:+.3f}" for m, x in zip(mus, xi_at_u4)))
