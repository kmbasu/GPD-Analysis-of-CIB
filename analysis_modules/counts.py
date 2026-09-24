"""
counts.py — Module 1 of the CIB-lensing / GPD project
======================================================

Differential number-count models dN/dS for the sub-mm/mm source population,
together with the exact gravitational-lensing transformation of the counts.

Purpose
-------
Every other module of the pipeline (analytic P(D), map simulation, GPD
forecasting) consumes a number-count model through the interface defined
here.  Two functional forms are provided, following Fujimoto et al. (2023),
ApJS 266, 10 (arXiv:2303.01658), their Eqs. (4)-(5) and Table 5:

  * Schechter:          dN/dS = (phi*/S*) (S/S*)^alpha exp(-S/S*)
                        with alpha < 0  (their best fit: alpha = -2.05)
  * Double power law:   dN/dS = (phi*/S*) [ (S/S*)^alpha + (S/S*)^beta ]^-1
                        with alpha, beta > 0 (faint slope -alpha, bright -beta)

IMPORTANT normalization convention
----------------------------------
Fujimoto et al. print phi* in deg^-2 (Table 5).  Their Eq. (4) as typeset
omits the 1/S* Jacobian, but only the convention

        dN/dS = (phi*/S*) * f(S/S*)          [mJy^-1 deg^-2]

reproduces their measured counts (their Table 4) — e.g. at S = 0.5 mJy the
Schechter fit gives log(dN/dS) = 4.04 vs the measured 3.99 (bin 0.47-0.63
mJy), whereas without the 1/S* factor the curve is high by ~0.6 dex
everywhere.  We therefore adopt the (phi*/S*) convention throughout.

Note on the published DPL fit: the best-fit knee log10(S*/mJy) = 2.97 lies
far outside the observed flux range (0.007-3 mJy), i.e. the MCMC ran into
the degenerate single-power-law limit.  For experiments that need a break
*inside* the P(D) window (the "break-crossing" lensing signal), use
`DoublePowerLaw.illustrative()`.

Lensing transformation (exact, per magnification value mu)
----------------------------------------------------------
Flux boost S -> mu*S and solid-angle dilution dOmega -> mu*dOmega give
(Lima, Jain & Devlin 2010, MNRAS 406, 2352, Eq. 31):

        n_mu(S) = mu^-2 * n_0(S/mu)

For a pure power law n_0 ~ S^-eta this changes only the amplitude
(by mu^(eta-2)), NOT the slope: the lensing observable in the GPD shape
parameter xi comes entirely from curvature/breaks in dN/dS and from
finite-threshold effects.  `LensedCounts` and `MixtureCounts` implement
the single-mu and the mixture-over-mu (aperture-averaged) cases.

Units
-----
Flux density S        : mJy
dN/dS                 : mJy^-1 deg^-2
Angular scales        : arcsec where needed (beam)

All functions are numpy-vectorized in S.  Only numpy/scipy/matplotlib
are required.

Typical usage
-------------
>>> from counts import Schechter, LensedCounts
>>> c0 = Schechter()                    # Fujimoto+2023 1.2 mm best fit
>>> c2 = LensedCounts(c0, mu=2.0)       # counts behind uniform mu = 2
>>> c0.dnds(1.0), c2.dnds(1.0)          # mJy^-1 deg^-2

Run the file top-to-bottom (or cell-by-cell in Spyder) for a demo.
"""

# %% Imports and constants
import numpy as np

# numpy 2.x renamed trapz -> trapezoid; keep both working
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

DEG2_TO_ARCSEC2 = 3600.0 ** 2          # 1 deg^2 in arcsec^2
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# %% Plotting configuration (used by the demo cells at the bottom)
#  Default: show figures interactively (plt.show()) — the right behaviour
#  inside an IDE like Spyder.  Set SAVE_FIGURES = True (or export
#  CIB_SAVE_FIGURES=1) to write PNGs into FIGURE_DIR instead, for batch
#  or headless runs.
import os as _os

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


# %% Base class
class BaseCounts:
    """
    Base class for differential number-count models.

    Subclasses must implement `dnds(S)` returning dN/dS in mJy^-1 deg^-2
    for S in mJy.  The base class supplies cumulative counts, moments,
    local slope diagnostics, and the asymptotic GPD shape parameter.
    """

    #  Subclasses should set a sensible faint/bright support (mJy); the
    #  moment integrals use these as default limits.  With faint slopes
    #  steeper than -2 the total flux diverges as S_min -> 0, so S_min
    #  is a *physical* input (it will matter for the P(D) mean, but not
    #  for the mean-subtracted P(D) shape, nor for the variance as long
    #  as the slope is shallower than -3).
    s_min = 1e-4
    s_max = 1e3

    def dnds(self, S):
        raise NotImplementedError

    # ---- integral quantities ------------------------------------------------
    def _log_grid(self, s_lo=None, s_hi=None, n=2048):
        s_lo = self.s_min if s_lo is None else s_lo
        s_hi = self.s_max if s_hi is None else s_hi
        return np.geomspace(s_lo, s_hi, n)

    def n_gtr(self, S, s_hi=None, n=4096):
        """Cumulative counts N(>S) [deg^-2] by log-space trapezoid rule."""
        S = np.atleast_1d(np.asarray(S, float))
        s_hi = self.s_max if s_hi is None else s_hi
        out = np.zeros_like(S)
        for i, s in enumerate(S):
            if s >= s_hi:
                continue
            g = np.geomspace(s, s_hi, n)
            out[i] = _trapz(self.dnds(g) * g, np.log(g))   # dS = S dlnS
        return out if out.size > 1 else out[0]

    def moment(self, k, s_lo=None, s_hi=None, n=4096):
        """
        Flux moments  M_k = int S^k dN/dS dS  over [s_lo, s_hi].
        Units: mJy^k deg^-2.
        k=0: source surface density; k=1: total flux (CIB intensity from
        this population); k=2: sets the confusion variance.
        """
        g = self._log_grid(s_lo, s_hi, n)
        return _trapz(self.dnds(g) * g ** (k + 1), np.log(g))

    # ---- local-slope diagnostics --------------------------------------------
    def eta(self, S, dlog=1e-3):
        """Local differential slope  eta(S) = -dln(dN/dS)/dlnS  (positive)."""
        S = np.asarray(S, float)
        lo, hi = S * (1 - dlog), S * (1 + dlog)
        return -(np.log(self.dnds(hi)) - np.log(self.dnds(lo))) \
            / (np.log(hi) - np.log(lo))

    def xi_asymptotic(self, S):
        """
        Asymptotic GPD shape parameter implied by the local count slope,
        xi(S) ~ 1/(eta(S) - 1)  (valid on a power-law stretch; see the
        GPD manual, Sec. 6).  Returns negative/large values where the
        slope is <1 (non-Pareto regime) — interpret with care.
        """
        return 1.0 / (self.eta(S) - 1.0)

    # ---- confusion statistics ------------------------------------------------
    def confusion_stats(self, beam_fwhm_arcsec, s_lo=None, s_hi=None):
        """
        Beam-folded shot-noise statistics for a Gaussian beam
        (peak-normalized, so map units are mJy/beam):

          mean  D_bar    = Om_eff * int S   dN/dS dS
          sigma_conf^2   = (Om_eff/2) * int S^2 dN/dS dS
          N_beam         = Om_b * N(>s_lo)   (sources per beam solid angle)

        with Om_eff = 2 pi sigma_b^2 (the integral of the beam) and
        Om_b = 2 pi sigma_b^2 as well for a Gaussian.  Returns a dict.
        The variance formula follows from int x^2 R(x) dx with
        R(x) = Om_eff N(>x)/x (see pofd_analytic.py).
        """
        sig_b = beam_fwhm_arcsec * FWHM_TO_SIGMA          # arcsec
        om_eff = 2.0 * np.pi * sig_b ** 2 / DEG2_TO_ARCSEC2   # deg^2
        m1 = self.moment(1, s_lo, s_hi)
        m2 = self.moment(2, s_lo, s_hi)
        n0 = self.moment(0, s_lo, s_hi)
        return dict(mean_mJy_per_beam=om_eff * m1,
                    sigma_conf_mJy_per_beam=np.sqrt(0.5 * om_eff * m2),
                    sources_per_beam=om_eff * n0)


# %% Concrete models (Fujimoto et al. 2023, Table 5)
class Schechter(BaseCounts):
    """
    Schechter counts, Fujimoto+2023 Eq. (4) with the (phi*/S*) convention:

        dN/dS = (phi*/S*) (S/S*)^alpha exp(-S/S*)

    Defaults: 1.2 mm ALCS best fit (their Table 5):
        alpha = -2.05, log10(S*/mJy) = 0.60, log10(phi*/deg^-2) = 2.85.
    These are PLACEHOLDERS for the project — to be replaced by 0.35 mm
    (350 um) parameters at the analysis stage.
    """

    def __init__(self, alpha=-2.05, log_sstar=0.60, log_phistar=2.85,
                 s_min=1e-4, s_max=None):
        self.alpha = float(alpha)
        self.sstar = 10.0 ** log_sstar          # mJy
        self.phistar = 10.0 ** log_phistar      # deg^-2
        self.s_min = s_min
        # the exponential makes the integrals converge; 30 S* is plenty
        self.s_max = 30.0 * self.sstar if s_max is None else s_max

    def dnds(self, S):
        S = np.asarray(S, float)
        x = S / self.sstar
        out = (self.phistar / self.sstar) * x ** self.alpha * np.exp(-x)
        return np.where(S > 0, out, 0.0)


class DoublePowerLaw(BaseCounts):
    """
    Double power law, Fujimoto+2023 Eq. (5) with the (phi*/S*) convention:

        dN/dS = (phi*/S*) [ (S/S*)^alpha + (S/S*)^beta ]^-1

    (faint end ~ S^-alpha, bright end ~ S^-beta, alpha < beta, both > 0).

    Defaults are the published Table 5 values (alpha=2.12, beta=3.81,
    log S* = 2.97, log phi* = 0.44).  CAUTION: that knee (~930 mJy) lies
    far outside the fitted flux range, i.e. the published fit is
    effectively a single power law over the P(D) window.  Use
    `DoublePowerLaw.illustrative()` for a knee inside the window.
    """

    def __init__(self, alpha=2.12, beta=3.81, log_sstar=2.97,
                 log_phistar=0.44, s_min=1e-4, s_max=1e4):
        self.alpha, self.beta = float(alpha), float(beta)
        self.sstar = 10.0 ** log_sstar
        self.phistar = 10.0 ** log_phistar
        self.s_min, self.s_max = s_min, s_max

    @classmethod
    def illustrative(cls, alpha=1.6, beta=3.8, sstar_mJy=4.0,
                     phistar_deg2=1.5e3, s_min=1e-4, s_max=1e4):
        """A DPL with the knee at a few mJy — inside the confusion P(D)
        window — for break-crossing (xi vs mu) experiments.  The
        amplitude is tuned to thread the Fujimoto+23 Table 4 points at
        0.1-0.5 mJy; these numbers are purely illustrative.  This is the
        DPL used in all downstream demos; the published Table-5 DPL
        (degenerate ~1 Jy knee) is kept only for the formalism-comparison
        figure."""
        return cls(alpha=alpha, beta=beta,
                   log_sstar=np.log10(sstar_mJy),
                   log_phistar=np.log10(phistar_deg2),
                   s_min=s_min, s_max=s_max)

    def dnds(self, S):
        S = np.asarray(S, float)
        x = S / self.sstar
        with np.errstate(divide="ignore", over="ignore"):
            out = (self.phistar / self.sstar) \
                / (x ** self.alpha + x ** self.beta)
        return np.where(S > 0, out, 0.0)


# %% Lensing transformations
class LensedCounts(BaseCounts):
    """
    Counts behind a *uniform* magnification mu (LJD 2010 Eq. 31):

        n_mu(S) = mu^-2 n_0(S/mu)

    Support scales with mu (s_min, s_max -> mu*s_min, mu*s_max).
    """

    def __init__(self, base, mu):
        if mu <= 0:
            raise ValueError("mu must be positive (use |mu| for P(D) work)")
        self.base, self.mu = base, float(mu)
        self.s_min = base.s_min * self.mu
        self.s_max = base.s_max * self.mu

    def dnds(self, S):
        return self.base.dnds(np.asarray(S, float) / self.mu) / self.mu ** 2


class MixtureCounts(BaseCounts):
    """
    Area-weighted mixture of lensed counts,

        n_eff(S) = sum_k w_k mu_k^-2 n_0(S/mu_k),   sum_k w_k = 1,

    describing the *average* image-plane counts over a region (aperture,
    annulus, stacked cutout) in which the magnification takes value mu_k
    over an area fraction w_k.  This is the object the analytic P(D) of a
    stacked cluster field needs (each sky position is an inhomogeneous
    Poisson process; the area-averaged P(D) uses area-averaged counts
    only when the field is analysed pixel-wise — see pofd_analytic.py
    for the distinction between the mixture-of-P(D)s and the
    P(D)-of-mixture; for one-point pixel histograms pooled over the
    region, the pooled histogram is the area-weighted mixture of the
    local P(D)s, and *that* is computed by passing each mu separately.
    MixtureCounts is provided for the R(x)-level shortcut, valid because
    the compound-Poisson CF is linear in the rate function.)
    """

    def __init__(self, base, mus, weights):
        mus = np.asarray(mus, float)
        w = np.asarray(weights, float)
        if np.any(mus <= 0):
            raise ValueError("all mu must be positive")
        self.base = base
        self.mus, self.w = mus, w / w.sum()
        self.s_min = base.s_min * mus.min()
        self.s_max = base.s_max * mus.max()

    def dnds(self, S):
        S = np.asarray(S, float)[..., None]
        return np.sum(self.w * self.base.dnds(S / self.mus) / self.mus ** 2,
                      axis=-1)


# %% Demo / self-test  (run this file directly, or cell-by-cell in Spyder)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sch = Schechter()
    dpl_pub = DoublePowerLaw()             # published Table-5 fit (knee ~1 Jy)
    dpl_ill = DoublePowerLaw.illustrative()

    # Fujimoto+2023 Table 4 (blind sample, MC-corrected), first 12 bins:
    # columns (S_lo, S_hi [mJy], log10 dN/dS [mJy^-1 deg^-2]).
    # Used here for VISUAL validation of the normalization convention only.
    t4 = np.array([
        [0.007, 0.012, 7.76], [0.012, 0.020, 7.51], [0.020, 0.029, 6.89],
        [0.029, 0.043, 6.29], [0.043, 0.063, 6.00], [0.063, 0.093, 5.47],
        [0.093, 0.136, 5.16], [0.140, 0.200, 5.06], [0.200, 0.270, 4.76],
        [0.270, 0.360, 4.54], [0.360, 0.470, 4.27], [0.470, 0.630, 3.99]])
    s_dat = np.sqrt(t4[:, 0] * t4[:, 1])          # geometric bin centres

    # --- console check against Table 4 (Schechter, adopted convention) -----
    checks = [(0.11, 5.16), (0.5, 3.99), (0.02, 7.1)]   # (S/mJy, log dN/dS)
    print("Schechter vs Fujimoto+23 Table 4 (log10 dN/dS):")
    for s, ref in checks:
        print(f"  S={s:6.3f} mJy : model {np.log10(sch.dnds(s)):5.2f}"
              f"  measured ~{ref:5.2f}")

    # --- confusion statistics for the project band -------------------------
    stats = sch.confusion_stats(beam_fwhm_arcsec=14.9, s_lo=1e-3)
    print("\nBeam-folded stats (14.9\" FWHM, S_min=1 uJy):")
    for k, v in stats.items():
        print(f"  {k:28s} = {v:.4g}")

    # %% Figure 1: functional-form conventions vs the measured counts
    #  * Schechter with the adopted (phi*/S*) Jacobian vs the literal
    #    reading of their Eq. 4 (no 1/S*): a constant 0.6 dex offset that
    #    misses the data everywhere;
    #  * DPL with the published (degenerate, ~930 mJy) knee vs the
    #    illustrative DPL with a knee at 4 mJy, inside the P(D) window.
    S = np.geomspace(5e-3, 2e3, 500)
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))
    ax1.loglog(S, sch.dnds(S), "C0-",
               label=r"Schechter, adopted $(\phi_*/S_*)$ convention")
    ax1.loglog(S, sch.dnds(S) * sch.sstar, "C0--",
               label=r"Schechter, Eq. 4 as typeset (no $1/S_*$)")
    ax1.loglog(S, dpl_pub.dnds(S), "C2-.",
               label=r"DPL, published knee ($S_*\!\approx\!930$ mJy)")
    ax1.loglog(S, dpl_ill.dnds(S), "C3-",
               label=r"DPL illustrative, knee at 4 mJy (fiducial DPL)")
    ax1.plot(s_dat, 10.0 ** t4[:, 2], "ko", ms=5, zorder=5,
             label="Fujimoto+23 Table 4 (blind, MC-corrected)")
    ax1.axvspan(0.007, 3.0, color="0.9", zorder=0)
    ax1.text(0.15, 3e8, "observed range", color="0.4", fontsize=8)
    ax1.set(xlabel="S  [mJy]", ylabel=r"dN/dS  [mJy$^{-1}$ deg$^{-2}$]",
            xlim=(5e-3, 2e3), ylim=(1e-4, 1e9),
            title="dN/dS formalism choices (1.2 mm placeholders)")
    ax1.legend(fontsize=8, loc="lower left")
    fig1.tight_layout()
    _finish_figure(fig1, "demo_counts_formalisms")

    # %% Figure 2: lensing transformation, local slope, implied GPD shape
    #  (the fiducial "DPL (illustrative)" is the only DPL used from here on)
    S = np.geomspace(1e-3, 60, 400)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    for mu, ls in [(1, "-"), (2, "--"), (5, ":")]:
        c = sch if mu == 1 else LensedCounts(sch, mu)
        ax[0].loglog(S, c.dnds(S), "C0", ls=ls, label=f"Schechter, mu={mu}")
        c2 = dpl_ill if mu == 1 else LensedCounts(dpl_ill, mu)
        ax[0].loglog(S, c2.dnds(S), "C3", ls=ls, label=f"DPL(ill.), mu={mu}")
    ax[0].set(xlabel="S [mJy]", ylabel="dN/dS [mJy$^{-1}$ deg$^{-2}$]",
              title="lensing: knee slides out, slopes invariant",
              ylim=(1e-3, 1e9))
    ax[0].legend(fontsize=7)

    ax[1].semilogx(S, sch.eta(S), "C0", label="Schechter")
    ax[1].semilogx(S, dpl_ill.eta(S), "C3", label="DPL (illustrative)")
    ax[1].set(xlabel="S [mJy]", ylabel=r"local slope $\eta(S)$",
              title="count curvature = where the signal lives")
    ax[1].legend(fontsize=8)

    m = (sch.eta(S) > 1.05)
    ax[2].semilogx(S[m], sch.xi_asymptotic(S[m]), "C0", label="Schechter")
    m = (dpl_ill.eta(S) > 1.05)
    ax[2].semilogx(S[m], dpl_ill.xi_asymptotic(S[m]), "C3",
                   label="DPL (illustrative)")
    ax[2].set(xlabel="S [mJy]", ylabel=r"$\xi_{\rm asym}(S)=1/(\eta-1)$",
              ylim=(-0.5, 3), title="implied GPD shape vs flux")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, "demo_counts")
