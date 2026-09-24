"""
pofd_analytic.py — Module 2 of the CIB-lensing / GPD project
=============================================================

Analytic (compound-Poisson) computation of the P(D) distribution — the
one-point PDF of pixel values in a confusion-limited map — with and
without gravitational lensing, via the characteristic-function / FFT
method (Scheuer 1957, 1974; Condon 1974; used in modern form by
Patanchon et al. 2009 and Vernstrom et al. 2014).

Theory
------
The deflection at a sky point is D = sum_i S_i b(theta_i), a compound
Poisson sum over sources folded through the beam b.  Its characteristic
function is exactly

    psi(w) = exp[ int R(x) (e^{iwx} - 1) dx ],

where R(x) dx is the mean number of *responses* with amplitude in
[x, x+dx]:   R(x) = int dOmega n(x / b(Omega)) / b(Omega).

For a GAUSSIAN beam b = exp(-r^2 / 2 sig_b^2) the substitution
r^2 = -2 sig_b^2 ln b collapses this to a closed form

    R(x) = Om_eff * N(> max(x, S_min)) / x,      Om_eff = 2 pi sig_b^2,

with N(>S) the cumulative counts: for Gaussian beams the response
spectrum is just the cumulative counts divided by the response amplitude.
Consistency checks that follow immediately:
    mean(D)  = int x  R dx = Om_eff int S   n(S) dS
    var(D)   = int x^2 R dx = (Om_eff/2) int S^2 n(S) dS,
matching counts.BaseCounts.confusion_stats.

Numerics
--------
* R is tabulated on the FFT flux grid x_k = k*dD; the contribution of
  responses below the first grid point (faint sources and far beam
  wings; R ~ 1/x there, integrable against (e^{iwx}-1)) is added
  analytically through its first two moments:  iw*m0 - w^2*v0/2.
* Instrument noise: multiply psi by exp(-sig_N^2 w^2 / 2) (exact
  convolution with a Gaussian).
* Bright-source masking: truncate the counts at S_cut (sources brighter
  than the mask limit are absent from the map).  NOTE this is the
  idealized, source-plane form of masking; the map-level mask (removing
  *pixels*) is handled in the simulation module.
* The output is MEAN-SUBTRACTED (psi *= e^{-iw D_bar}), matching how
  real maps are baselined; the divergence of the total flux for faint
  slopes steeper than -2 therefore never enters the *shape*.
* Everything is linear in R, so lensing enters simply by replacing the
  counts:  uniform mu -> counts.LensedCounts;  a magnification *profile*
  (aperture / annulus / stacked cutout, where the pooled pixel histogram
  is the area-weighted mixture of local P(D)s) -> the R functions add:
  R_eff(x) = sum_k w_k R_[mu_k](x), equivalent to counts.MixtureCounts.
  (Validity: mu must vary slowly across one beam FWHM.)

Aperture statistics (Hezaveh et al. 2013, Sec. 2.2)
---------------------------------------------------
`aperture_pofd` computes the PDF of the TOTAL flux inside a top-hat
aperture centred on a lens, decomposing the aperture into rings of
constant mu (their Eqs. 5-10 are the discrete-Poisson version of the
same characteristic function):  R_ap(x) = sum_j Om_j n_[mu_j](x).
This reproduces their Figure 1 construction with our machinery.

Units: fluxes in mJy; map values in mJy/beam (peak-normalized beam);
angles in arcsec (beam) / arcmin (aperture); counts objects from
counts.py (mJy^-1 deg^-2).

Typical usage
-------------
>>> from counts import Schechter
>>> from pofd_analytic import PofD
>>> p0 = PofD(Schechter(), beam_fwhm_arcsec=14.9, sigma_noise=0.3)
>>> p2 = PofD(Schechter(), beam_fwhm_arcsec=14.9, sigma_noise=0.3, mu=2.0)
>>> p0.sf(1.0), p2.sf(1.0)        # survival functions at D = 1 mJy/beam

Run the file directly for demos (pixel P(D) lensed vs unlensed, and the
Hezaveh-style aperture figure).
"""

# %% Imports
import numpy as np

# numpy 2.x renamed trapz -> trapezoid; keep both working
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from counts import (BaseCounts, LensedCounts, MixtureCounts,
                    DEG2_TO_ARCSEC2, FWHM_TO_SIGMA)

ARCMIN2_TO_DEG2 = 1.0 / 3600.0

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


# %% Cumulative counts on a reusable grid
def cumulative_counts(cnts, s_lo=None, s_hi=None, n=4096):
    """Tabulate N(>S) [deg^-2] on a log grid by cumulative trapezoid from
    the bright end.  Returns (s_grid, N_gtr)."""
    s_lo = cnts.s_min if s_lo is None else s_lo
    s_hi = cnts.s_max if s_hi is None else s_hi
    s = np.geomspace(s_lo, s_hi, n)
    f = cnts.dnds(s) * s                      # integrand vs ln S
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(np.log(s))
    ngtr = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return s, ngtr


class _NgtrInterp:
    """Fast log-log interpolator for N(>S), zero above s_max."""

    def __init__(self, cnts, n=4096):
        self.s, self.ngtr = cumulative_counts(cnts, n=n)
        self.s_min, self.s_max = self.s[0], self.s[-1]
        pos = self.ngtr > 0
        self._ls, self._ln = np.log(self.s[pos]), np.log(self.ngtr[pos])

    def __call__(self, s):
        s = np.asarray(s, float)
        out = np.exp(np.interp(np.log(np.maximum(s, self.s_min)),
                               self._ls, self._ln,
                               left=self._ln[0], right=-np.inf))
        return np.where(s >= self.s_max, 0.0, out)


# %% Core characteristic-function engine
def _pofd_from_rate(rate_fn, d_span, n_fft, sigma_noise=0.0,
                    x_faint_decades=8):
    """
    P(D) from a response-rate function R(x) (= rate_fn, vectorized,
    units mJy^-1 'per point').  Returns (d_centered, pdf).

    d_span : total flux span covered by the FFT grid (mJy);  the grid is
             d in [-d_span/2, d_span/2) after centering.
    """
    n = int(n_fft)
    dd = d_span / n
    x = np.arange(1, n) * dd                 # response grid (x > 0)
    R = rate_fn(x)

    # faint-end correction: moments of R below the first grid point
    x_f = np.geomspace(dd * 10 ** (-x_faint_decades), dd, 512)
    R_f = rate_fn(x_f)
    m0 = _trapz(R_f * x_f ** 2, np.log(x_f))          # int x R dx
    v0 = _trapz(R_f * x_f ** 3, np.log(x_f))          # int x^2 R dx

    d_bar = m0 + np.sum(x * R) * dd                     # total mean

    # sum_k R_k (e^{i w x_k} - 1) dD  via inverse FFT (positive exponent)
    Rpad = np.concatenate([[0.0], R])                   # index 0 <-> x = 0
    cf = n * np.fft.ifft(Rpad) * dd
    cf -= np.sum(R) * dd                                # the "-1" term
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=dd)
    cf += 1j * w * m0 - 0.5 * v0 * w ** 2               # faint part
    cf += -1j * w * d_bar                               # mean-subtract
    cf += -0.5 * (sigma_noise * w) ** 2                 # Gaussian noise
    p = np.real(np.fft.fft(np.exp(cf))) / (n * dd)

    d = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / (n * dd)))  # centered grid
    p = np.fft.fftshift(p)
    return d, np.maximum(p, 0.0)


# %% Pixel P(D) for a Gaussian beam
class PofD:
    """
    One-point P(D) of a beam-smoothed, mean-subtracted map.

    Parameters
    ----------
    cnts : counts.BaseCounts
        Unlensed source counts.
    beam_fwhm_arcsec : float
        Gaussian beam FWHM.  Map units are mJy/beam (peak-normalized).
    mu : float, or (mus, weights) tuple
        Uniform magnification (LensedCounts) or an area-weighted mixture
        (MixtureCounts) for pooled histograms over a mu(theta) region.
    sigma_noise : float
        Additive Gaussian noise, mJy/beam.
    s_cut : float or None
        Bright-source masking flux (counts truncated above it).
    d_span, n_fft : FFT grid controls (span in mJy/beam).

    Attributes: d (grid), p (pdf);  methods pdf(), sf(), moments().
    """

    def __init__(self, cnts, beam_fwhm_arcsec, mu=1.0, sigma_noise=0.0,
                 s_cut=None, d_span=200.0, n_fft=2 ** 17):
        if isinstance(mu, tuple):
            eff = MixtureCounts(cnts, *mu)
        elif mu != 1.0:
            eff = LensedCounts(cnts, mu)
        else:
            eff = cnts
        if s_cut is not None:                 # truncate counts at the mask
            eff = _TruncatedCounts(eff, s_cut)
        self.counts = eff
        sig_b = beam_fwhm_arcsec * FWHM_TO_SIGMA
        self.om_eff = 2.0 * np.pi * sig_b ** 2 / DEG2_TO_ARCSEC2   # deg^2
        ngtr = _NgtrInterp(eff)
        s_min = eff.s_min

        def rate(x):
            return self.om_eff * ngtr(np.maximum(x, s_min)) / x

        self.d, self.p = _pofd_from_rate(rate, d_span, n_fft, sigma_noise)
        self._cdf = np.concatenate([[0.0],
                                    np.cumsum(0.5 * (self.p[1:] + self.p[:-1])
                                              * np.diff(self.d))])

    def pdf(self, D):
        return np.interp(D, self.d, self.p)

    def sf(self, D):
        """Survival function P(>D)."""
        return 1.0 - np.interp(D, self.d, self._cdf / self._cdf[-1])

    def moments(self):
        m1 = _trapz(self.d * self.p, self.d)
        m2 = _trapz(self.d ** 2 * self.p, self.d)
        m3 = _trapz(self.d ** 3 * self.p, self.d)
        var = m2 - m1 ** 2
        return dict(mean=m1, sigma=np.sqrt(var),
                    skew=(m3 - 3 * m1 * var - m1 ** 3) / var ** 1.5)


class _TruncatedCounts(BaseCounts):
    """Counts set to zero above s_cut (source-plane bright masking)."""

    def __init__(self, base, s_cut):
        self.base, self.s_cut = base, float(s_cut)
        self.s_min, self.s_max = base.s_min, min(base.s_max, float(s_cut))

    def dnds(self, S):
        S = np.asarray(S, float)
        return np.where(S <= self.s_cut, self.base.dnds(S), 0.0)


# %% Aperture (top-hat) total-flux PDF with a mu(theta) profile
def aperture_pofd(cnts, lens=None, r_ap_arcmin=0.5, n_rings=64,
                  sigma_noise=0.0, s_cut=None, d_span=400.0, n_fft=2 ** 17,
                  r_inner_arcmin=1e-3):
    """
    PDF of the total flux inside a top-hat aperture of radius r_ap
    centred on the lens (no beam), decomposed into n_rings rings of
    locally-uniform magnification — the analytic construction of
    Hezaveh et al. (2013), Sec. 2.2, in characteristic-function form:

        psi(w) = exp[ sum_j Om_j int dS n_[mu_j](S) (e^{iwS} - 1) ]

    lens : lens_model.BaseLens or None (None -> unlensed)
    Returns (d_centered_mJy, pdf).
    """
    edges = np.linspace(r_inner_arcmin, r_ap_arcmin, n_rings + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    om_j = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * ARCMIN2_TO_DEG2
    mus = lens.mu(mid) if lens is not None else np.ones_like(mid)

    interps, s_mins = [], []
    for m in mus:
        eff = LensedCounts(cnts, m) if m != 1.0 else cnts
        if s_cut is not None:
            eff = _TruncatedCounts(eff, s_cut)
        interps.append(_NgtrInterp(eff))
        s_mins.append(eff.s_min)

    def rate(x):
        # R_ap(x) = sum_j Om_j n_j(x);  n(x) = -dN/dx recovered from the
        # tabulated dnds directly for accuracy:
        out = np.zeros_like(x)
        for m, om in zip(mus, om_j):
            if s_cut is not None:
                v = np.where(x <= s_cut,
                             cnts.dnds(x / m) / m ** 2, 0.0)
            else:
                v = cnts.dnds(x / m) / m ** 2
            out += om * v
        return out

    return _pofd_from_rate(rate, d_span, n_fft, sigma_noise)


# %% Demo / self-test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from counts import Schechter

    sch = Schechter(s_min=1e-3)
    beam = 14.9        # arcsec FWHM (project 350-um convention)

    # --- moment validation against closed-form confusion statistics --------
    p0 = PofD(sch, beam, sigma_noise=0.0)
    ref = sch.confusion_stats(beam)
    got = p0.moments()
    print("P(D) engine validation (unlensed, no noise):")
    print(f"  sigma_conf : FFT = {got['sigma']:.4f}  "
          f"closed-form = {ref['sigma_conf_mJy_per_beam']:.4f} mJy/beam")
    print(f"  mean (post-centering) = {got['mean']:.2e} (should be ~0)")

    # --- lensed vs unlensed pixel P(D) --------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mu, c_ in [(1.0, "k"), (2.0, "C0"), (5.0, "C3")]:
        pp = PofD(sch, beam, mu=mu, sigma_noise=0.3, s_cut=100.0)
        ax[0].semilogy(pp.d, pp.p, c_, label=f"mu = {mu}")
        ax[1].loglog(pp.d[pp.d > 0], pp.sf(pp.d[pp.d > 0]), c_,
                     label=f"mu = {mu}")
    ax[0].set(xlim=(-2, 8), ylim=(1e-6, 3),
              xlabel="D  [mJy/beam]", ylabel="P(D)",
              title="pixel P(D), 14.9\" beam, sig_N = 0.3 mJy")
    ax[1].set(xlabel="D  [mJy/beam]", ylabel="P(>D)",
              title="survival: lensing feeds the tail")
    for a in ax:
        a.legend()
    fig.tight_layout()
    _finish_figure(fig, "demo_pofd_pixel")

    # --- Hezaveh-style aperture figure (SIS lens, top-hat aperture) --------
    try:
        from lens_model import SISLens
        sis = SISLens(theta_e_arcmin=0.2, mu_max=50.0)
        d0, q0 = aperture_pofd(sch, lens=None, r_ap_arcmin=0.5)
        d1, q1 = aperture_pofd(sch, lens=sis, r_ap_arcmin=0.5)
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        ax2.semilogy(d0, q0, "k", label="unlensed")
        ax2.semilogy(d1, q1, "C3", label=r"SIS, $\theta_E=0.2'$")
        ax2.set(xlim=(-30, 60), ylim=(1e-7, 1),
                xlabel="aperture flux - mean  [mJy]", ylabel="PDF",
                title="Hezaveh+13-style aperture flux distribution")
        ax2.legend()
        fig2.tight_layout()
        _finish_figure(fig2, "demo_pofd_aperture")
    except ImportError:
        print("lens_model not importable; skipped aperture demo")
