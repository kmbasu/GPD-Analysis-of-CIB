"""
simulate_maps.py — Module 4 of the CIB-lensing / GPD project
=============================================================

Monte-Carlo generation of confusion-limited CIB maps, unlensed or lensed
by a cluster, by IMAGE-PLANE INHOMOGENEOUS-POISSON source injection.

Why this construction (and not ray tracing, and not "multiplying by mu")
------------------------------------------------------------------------
For unresolved point sources and one-point statistics, the entire effect
of lensing is the local flux boost S -> mu*S plus density dilution
n -> n/mu; a smooth remapping of a homogeneous Poisson process changes
its statistics only through the Jacobian of the map, which is exactly
1/mu.  Injecting sources directly in the image plane with the locally
lensed counts

        n(S, theta) = mu(theta)^-2 n_0( S / mu(theta) )

is therefore EQUIVALENT to full ray tracing for the P(D) (and for the
GPD tail built from it), at any lens geometry, with no deflection field
needed.  Two common shortcuts are wrong and are deliberately avoided:
  * multiplying an unlensed map (or its source fluxes) by mu(theta)
    boosts fluxes but skips the dilution -> overpopulates the tail;
  * lensing after beam convolution mixes image/source planes.
Caveats: multiple images of one strongly-lensed source are injected as
independent (correct in expectation; images land in the same beam at our
resolutions anyway); mu is clipped at the lens model's mu_max (finite
source size).  Source clustering is NO LONGER unmodelled -- see
`clustering` below -- but remains OFF by default.

Pipeline per realization
------------------------
  1. mu map from lens_model (or mu = 1);
  2. optional clustering modulation field 1 + delta (see `clustering`);
  3. per log-flux bin: Poisson draw per pixel with rate
     n(S_i, theta) dS_i Om_pix (1 + delta); deposit N * S_i;
  4. FFT convolution with a peak-normalized Gaussian beam
     -> map in mJy/beam (periodic boundaries: keep the lens away from
     edges or pad; for statistics maps periodicity is harmless);
  5. optional Gaussian instrument noise (see `noise_mode`: beam-correlated
     by default, or white-per-pixel), optional bright-PIXEL masking,
     mean subtraction.

Source clustering (`clustering`)
--------------------------------
Real CIB sources are clustered; the default injection is Poisson.  Whether
that matters depends entirely on WHERE in P(D) you look, and the contrast
is sharp (memo `full_simulation_summary_v2.md`, Sec. 10):

  * in the CORE, the clustering:shot ratio of the beam-smoothed variance is
        R_core = Gamma(0.4) * (l_eq sigma_b)^1.2,
    which is 0.11 for this project's 20" simulation beam, 0.15 for SPIRE
    350um, but 2.9 for Planck's 5' beam -- it scales as beam^1.2;
  * in the TAIL, the correction is suppressed by the number of sources per
    beam ABOVE THRESHOLD, R_tail ~ wbar * N_beam(>u), which falls from
    3e-2 to 4e-4 across the u = 25-75 mJy Herschel window.  Their ratio,
        R_core / R_tail = 2 q1^2 / (q2 N(>u)),
    is INDEPENDENT of the clustering amplitude and runs 850 to 6e4 across
    that window.

Measured consequences (memo Secs. 10.5-10.6), on the fitted 350um
Schechter at 25"/5":  xi-hat(u) is shifted by -0.02 at u ~ 1 sigma_c, the
shift DECAYS through zero by ~3 sigma_c, and is unresolvable above it --
so the tail is protected in the fitting window, as predicted.  The
cutout-to-cutout SCATTER is a different story and is the reason this mode
exists: see `cl_q1_ref` below before using it for an error budget.

Model: an isotropic power law in the source-overdensity power spectrum,
        C_l^dd = (q2 / q1^2) * cl_ratio * (l / l_eq)^cl_slope,
with defaults cl_slope = -1.2 and l_eq = 2000, cl_ratio = 1.  The slope is
the CIB value at 857 GHz (l^2 C_l ~ l^0.8, i.e. C_l ~ l^-1.2; Addison et
al. 2013, George et al. 2015, via the Guerrero memo Sec. 2.2), and equals
the flat-sky Fourier transform of w(theta) ~ theta^-0.8 as measured by
Wang et al. (2026) -- two independent routes to the same index.  The
normalization q2/q1^2 makes the CLUSTERING power equal the SHOT power at
l = l_eq, matching the WebSky/Planck 857 GHz spectrum of Stein et al.
(2020, Fig. 7); with the project's 350um counts this reproduces that
figure from l = 100 to 10^4 (memo Sec. 10.4).

The modulation is drawn as a lognormal field (Coles & Jones 1991), which
guarantees 1 + delta > 0 and is the standard approximation for a
non-linear density field; `clustering="gaussian"` gives the clipped
Gaussian instead, for comparison.  Sources of ALL fluxes see the SAME
delta: clustering is assumed luminosity-independent.  That is the same
assumption Wang et al. (2026) make, and it is known to be imperfect --
Viero et al. (2013) find the Poisson and 1-halo power depend on the
flux cut, i.e. brighter sources cluster more strongly.  Since the bright
end is what sets the GPD tail, treat this as the limiting assumption of
this mode rather than a detail.

Instrument-noise correlation (`noise_mode`)
-------------------------------------------
The one-point P(D) does not care whether the additive Gaussian noise is
white at the pixel scale or beam-correlated -- only its rms enters.  But
`gpd_tail.beam_declustered_peaks` cares enormously, because local maxima
are a small-scale statistic.  White pixel noise at f_N = 1 raises the
retained-peak density by a factor ~4 and pushes the declustered-peak core
from the ~0.88 to the ~0.93 quantile of the pixel distribution, so the MC
declustered-peak xi-hat(u) curve and the analytic pixel xi_pop(u) curve
stop agreeing below u ~ 4 sigma_c.  Beam-correlated noise (the default,
and the realistic description of a reduced map) restores agreement.  The
derivation and the measured numbers are in the project memo
`GPD_beam_and_declustering_v1.md`.

The injected-source bin centres are log-spaced; with >= 40 bins/decade
the P(D) discretization error is negligible compared to sample variance
(verify by doubling n_bins_per_decade).

Units: fluxes mJy, map mJy/beam, pixel scale arcsec.

Typical usage
-------------
>>> from counts import Schechter
>>> from lens_model import NFWLens
>>> from simulate_maps import CIBMapSimulator
>>> sim = CIBMapSimulator(Schechter(s_min=1e-3), beam_fwhm_arcsec=14.9,
...                       pix_arcsec=4.0, npix=1024, sigma_noise=0.3)
>>> m0 = sim.make_map(seed=1)                          # control field
>>> lens = NFWLens(1e15, z_l=0.3, z_s=2.0)
>>> m1 = sim.make_map(lens=lens, seed=1)               # lensed field

Run the file directly for the validation demo: MC pixel histogram vs the
analytic P(D) of pofd_analytic (the module-2/module-4 cross-check that
plays the role of Hezaveh et al. 2013, Fig. 1).
"""

# %% Imports
import numpy as np

from counts import DEG2_TO_ARCSEC2, FWHM_TO_SIGMA

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

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


# %% Beam convolution (FFT, periodic)
def gaussian_beam_convolve(img, beam_fwhm_pix):
    """Convolve with a PEAK-normalized Gaussian beam via FFT (periodic
    boundaries).  A point source of flux S then has peak S: the map is
    in mJy/beam."""
    n = img.shape[0]
    sig = beam_fwhm_pix * FWHM_TO_SIGMA
    k = np.fft.fftfreq(n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    # FT of exp(-r^2/2sig^2) (unnormalized peak=1 kernel): 2 pi sig^2 * G(k)
    bl = 2.0 * np.pi * sig ** 2 \
        * np.exp(-2.0 * (np.pi * sig) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * bl))


# %% Clustering: lognormal modulation field
def _gaussian_field_from_cl(cl_grid, box_rad, rng):
    """
    Draw a zero-mean Gaussian random field on a periodic npix x npix grid
    with a prescribed flat-sky angular power spectrum.

    `cl_grid` is C(l) evaluated on the FFT mode grid (same shape as the
    output, DC element ignored), `box_rad` is the box side in radians.

    Normalization: with numpy's ifft2 convention, drawing
        g = Re[ IFFT( FFT(white) * sqrt(C) ) ] * npix / sqrt(A)
    gives Var(g) = (1/A) sum_k C_k = int d2l/(2pi)^2 C(l), which is the
    continuum definition.  Verified numerically in
    `Simulations/validation/verify_clustering_mode.py` (recovered C_l vs input).
    """
    n = cl_grid.shape[0]
    w = rng.normal(size=(n, n))
    g = np.fft.ifft2(np.fft.fft2(w) * np.sqrt(np.maximum(cl_grid, 0.0)))
    return np.real(g) * n / box_rad          # box_rad = sqrt(A) for a square


def _cl_on_grid(npix, pix_arcsec, amp, ell_eq, slope, ell_min=None,
                ell_max=None):
    """C(l) = amp * (l/ell_eq)^slope on the FFT mode grid of the map.

    The DC mode is set to zero (the mean is carried by the counts, not by
    the modulation).  `ell_min`/`ell_max` optionally truncate the power
    law; `ell_max` matters because a slope shallower than -2 makes
    Var(delta) grow with resolution -- see the class docstring.
    """
    box_rad = npix * pix_arcsec * ARCSEC_TO_RAD
    k = np.fft.fftfreq(npix, d=1.0 / npix)               # integer modes
    ky, kx = np.meshgrid(k, k, indexing="ij")
    ell = 2.0 * np.pi * np.sqrt(kx ** 2 + ky ** 2) / box_rad
    cl = np.zeros_like(ell)
    good = ell > 0
    if ell_min is not None:
        good &= ell >= ell_min
    if ell_max is not None:
        good &= ell <= ell_max
    cl[good] = amp * (ell[good] / ell_eq) ** slope
    return cl, ell, box_rad


# %% Simulator
class CIBMapSimulator:
    """
    Confusion-map generator with optional lensing.

    Parameters
    ----------
    cnts : counts.BaseCounts     (UNLENSED counts; set s_min deliberately)
    beam_fwhm_arcsec, pix_arcsec, npix : map geometry
    sigma_noise : Gaussian instrument noise, mJy/beam.  In BOTH noise modes
            this is the resulting per-pixel rms of the noise map, so the
            analytic budget Sigma_total^2 = sigma_c^2 + sigma_noise^2 (and
            everything that depends on it, e.g. pofd_analytic.PofD's
            `sigma_noise` argument) is unchanged by the mode.
    noise_mode : {"beam", "white"}
            How the instrument noise is spatially correlated.

            "beam" (DEFAULT) — the noise is passed through the SAME
                Gaussian beam as the sources, i.e. it is correlated on the
                beam scale.  This is the realistic description of a
                reduced map in mJy/beam (Herschel/SPIRE Level-2, Planck
                frequency maps): the detector noise has already been
                through the optical chain and the map-maker, so it carries
                the instrument's own angular response.

            "white" — independent Gaussian deviate per pixel, added AFTER
                the beam convolution.  Physically this describes only a
                component introduced post-map-making (pixel-level
                zero-level / calibration jitter).  It gives the map far
                more small-scale power than the signal has, which matters
                a great deal downstream: `gpd_tail.beam_declustered_peaks`
                then retains ~4x more "peaks" (mostly noise spikes, not
                sources) and the declustered-peak marginal is displaced
                well above the pixel P(D).  See the memo
                `GPD_beam_and_declustering_v1.md`, Sec. 4, for the
                measured numbers.  Retained so the earlier
                `test_GPDshape_*_20asec_beam` figures remain reproducible.

            NOTE: the one-point P(D) is IDENTICAL in the two modes (a
            Gaussian of the same rms convolves in the same way); only the
            spatial correlation, and hence anything built from local
            maxima, differs.
    s_cut : source-plane bright cut (None = keep all);  map-level pixel
            masking is a separate, downstream operation (mask_bright)
    n_bins_per_decade : flux-bin resolution of the injection
    clustering : {None, "lognormal", "gaussian"}
            Spatial clustering of the injected sources.

            None (DEFAULT) — homogeneous Poisson placement, i.e. the
                behaviour of every version of this module before v2.  The
                random stream in this branch is unchanged, so all cached
                ensembles remain bit-for-bit reproducible.

            "lognormal" — modulate the injection rate by 1 + delta, with
                delta a lognormal field (Coles & Jones 1991) whose power
                spectrum is the power law described in the module
                docstring.  Guarantees 1 + delta > 0.

            "gaussian" — the same target C_l, but the field is Gaussian
                and clipped at 1 + delta >= 0.  Provided for comparison;
                the clipping breaks the target power spectrum when
                sigma_delta approaches 1, so prefer "lognormal".

    cl_ratio, ell_eq, cl_slope : clustering power-law parameters.
            C_l^dd = (q2/q1^2) * cl_ratio * (l/ell_eq)^cl_slope, so
            cl_ratio is the clustering:shot power ratio AT l = ell_eq.
            Defaults (1.0, 2000.0, -1.2) reproduce the 857 GHz spectrum of
            Stein et al. (2020, Fig. 7).
    cl_q1_ref : reference CIB monopole q1 [deg^-2 mJy], or None.
            The modulation amplitude is C_l^dd = cl_ratio (q2/q1^2)
            (l/l_eq)^cl_slope.  With `None` (default) q1 is the injected
            population's own first moment, which makes the map's FLUX
            clustering power equal its own shot power at l_eq -- correct
            for MAP statistics (variance, P(D), xi-hat) and self-consistent
            for any counts model.

            But delta itself, and therefore any COUNT statistic (scatter of
            source or peak counts per cutout), depends on q1 separately,
            and q1 is the one moment this project's counts model gets badly
            wrong: against a measured CIB monopole of 0.576 +/- 0.034
            MJy/sr at 857 GHz (Odegard et al. 2019), the fitted 350 um
            Schechter gives 1.54 MJy/sr from the sources actually injected
            at S_min = 0.1 mJy (2.67x high) and 2.42 MJy/sr integrated to
            1 uJy (4.2x).  Too many faint sources carry the background, so
            a given clustered flux is spread over too many objects and
            delta comes out too small -- by exactly the factor
            q1_injected/q1_measured in rms.

            Pass the MEASURED monopole here to fix that:
                cl_q1_ref = 1.755e5      # 0.576 MJy/sr in deg^-2 mJy
            This raises delta by ~2.7x in rms for the fiducial Herschel
            configuration and is the right setting for error-budget work.
            It leaves the flux clustering power (hence the map variance)
            correspondingly higher, so it is NOT the right setting for
            reproducing a target C_l -- the two cannot be satisfied at once
            while the counts model mis-predicts q1.  See memo Sec. 10.6.
    cl_ell_max : optional truncation of the clustering power law.
            A slope shallower than -2 makes Var(delta) grow without bound
            as the pixel scale shrinks (the integral int l^(1+slope) dl
            diverges at high l), so sigma_delta is formally
            resolution-dependent.  This is not a bug in the model but a
            statement that a single power law cannot hold to arbitrarily
            small scales; physically the 1-halo term turns over.  The
            beam-smoothed variance converges regardless (the beam supplies
            the cutoff), so leaving this at None is safe for anything
            computed from the smoothed map; set it if you want a
            resolution-independent sigma_delta.  `sigma_delta` is reported
            by `clustering_info()`.
    """

    def __init__(self, cnts, beam_fwhm_arcsec, pix_arcsec, npix=1024,
                 sigma_noise=0.0, s_cut=None, n_bins_per_decade=40,
                 noise_mode="beam", clustering=None, cl_ratio=1.0,
                 ell_eq=2000.0, cl_slope=-1.2, cl_ell_max=None,
                 cl_q1_ref=None):
        self.cnts, self.npix = cnts, int(npix)
        self.pix_arcsec = float(pix_arcsec)
        self.beam_fwhm_pix = beam_fwhm_arcsec / pix_arcsec
        if self.beam_fwhm_pix < 2.5:
            raise ValueError("beam under-sampled: need >= 2.5 pix/FWHM")
        self.om_pix = pix_arcsec ** 2 / DEG2_TO_ARCSEC2      # deg^2
        self.sigma_noise = float(sigma_noise)
        if noise_mode not in ("beam", "white"):
            raise ValueError("noise_mode must be 'beam' or 'white'")
        self.noise_mode = noise_mode
        #  Amplitude of the WHITE field that, after peak-normalized Gaussian
        #  beam convolution, has unit per-pixel variance.  For a
        #  peak-normalized beam b(r) = exp(-r^2 / 2 sig_b^2), convolving a
        #  white field of variance s_w^2 gives output variance
        #      s_w^2 * int b(r)^2 d^2r = s_w^2 * pi * sig_b^2      [pixels]
        #  so s_w = 1 / (sig_b sqrt(pi)) reproduces unit output rms.
        self._sig_b_pix = self.beam_fwhm_pix * FWHM_TO_SIGMA
        self._white_per_unit_beamed = 1.0 / (self._sig_b_pix * np.sqrt(np.pi))
        self.s_cut = s_cut
        # log-spaced flux bins spanning the (lensed) support generously
        s_lo, s_hi = cnts.s_min, (s_cut if s_cut else cnts.s_max)
        ndec = np.log10(s_hi / s_lo)
        self.s_edges = np.geomspace(s_lo, s_hi,
                                    int(ndec * n_bins_per_decade) + 1)
        self.s_mid = np.sqrt(self.s_edges[1:] * self.s_edges[:-1])
        self.ds = np.diff(self.s_edges)

        # ---- clustering set-up ---------------------------------------------
        if clustering not in (None, "none", "lognormal", "gaussian"):
            raise ValueError("clustering must be None, 'lognormal' or "
                             "'gaussian'")
        self.clustering = None if clustering in (None, "none") else clustering
        self.cl_ratio, self.ell_eq = float(cl_ratio), float(ell_eq)
        self.cl_slope, self.cl_ell_max = float(cl_slope), cl_ell_max
        self.cl_q1_ref = None if cl_q1_ref is None else float(cl_q1_ref)
        self._cl_grid = None
        if self.clustering is not None:
            #  amplitude: C_l^dd(l_eq) = cl_ratio * q2 / q1^2, so that the
            #  FLUX clustering power q1^2 C_l^dd equals the shot power q2 at
            #  l = l_eq.  q1, q2 are integrals of the SAME counts object that
            #  drives the injection, so the calibration is self-consistent
            #  whatever the counts model -- no external normalization enters.
            q1 = self._q_moment(1)
            q2 = self._q_moment(2)
            self._q1_injected = q1
            if self.cl_q1_ref is not None:
                q1 = self.cl_q1_ref          # measured monopole, see docstring
            self._q1, self._q2 = q1, q2
            amp = self.cl_ratio * q2 / q1 ** 2            # deg^2 -> see below
            #  q1 [deg^-2 mJy], q2 [deg^-2 mJy^2]  =>  q2/q1^2 has units of
            #  deg^2; convert to steradians so C_l^dd is in sr.
            amp *= DEG2_TO_ARCSEC2 * ARCSEC_TO_RAD ** 2
            self._cl_amp = amp
            self._cl_grid, self._ell_grid, self._box_rad = _cl_on_grid(
                self.npix, self.pix_arcsec, amp, self.ell_eq, self.cl_slope,
                ell_max=self.cl_ell_max)
            #  Var(delta) = (1/A) sum_k C_k
            self.sigma_delta = float(
                np.sqrt(self._cl_grid.sum()) / self._box_rad)
            #  For the lognormal we must NOT draw the Gaussian with C^dd and
            #  then exponentiate: exp() amplifies, so the realized delta
            #  power would exceed the target.  Coles & Jones (1991): if the
            #  Gaussian g has correlation xi_g and 1+delta = exp(g - s_g^2/2),
            #  then xi_delta = exp(xi_g) - 1.  So invert, xi_g = ln(1+xi_d),
            #  and draw g from the transformed spectrum.
            if self.clustering == "lognormal":
                A = self._box_rad ** 2
                xi_d = np.real(np.fft.ifft2(self._cl_grid)) * self.npix ** 2 / A
                cl_g = np.real(np.fft.fft2(np.log1p(xi_d))) * A / self.npix ** 2
                neg = cl_g < 0
                if neg.any():
                    #  small negative ringing from the DC removal / grid
                    #  truncation; clip and report how much power it moves
                    frac = -cl_g[neg].sum() / cl_g[~neg].sum()
                    if frac > 1e-3:
                        import warnings
                        warnings.warn(
                            f"lognormal C_l^gg had {frac:.1%} negative power "
                            "after the log transform; clipped. Reduce "
                            "cl_ratio or set cl_ell_max.", RuntimeWarning)
                    cl_g = np.maximum(cl_g, 0.0)
                cl_g[0, 0] = 0.0
                self._cl_gauss = cl_g
            else:
                self._cl_gauss = self._cl_grid

    def _q_moment(self, k):
        """q_k = int S^k dN/dS dS over the injected flux range
        [deg^-2 mJy^k].  Uses the same bin centres as the injection, so it
        is exactly the moment of the population actually simulated."""
        return float(np.sum(self.s_mid ** k
                            * self.cnts.dnds(self.s_mid) * self.ds))

    def clustering_info(self):
        """Diagnostics for the clustering mode (dict), or None if off.

        R_core is the predicted clustering:shot ratio of the beam-smoothed
        variance.  For C_l^clust = q2 cl_ratio (l/l_eq)^s it is

            R_core = cl_ratio * Gamma(1 + s/2) * (l_eq sigma_b)^(-s),

        obtained from int l^(1+s) exp(-l^2 sigma_b^2) dl divided by the
        same integral with s = 0.
        """
        if self.clustering is None:
            return None
        from math import gamma as _gamma
        sb_rad = self._sig_b_pix * self.pix_arcsec * ARCSEC_TO_RAD
        s = self.cl_slope
        r_core = self.cl_ratio * _gamma(1.0 + s / 2.0) \
            * (self.ell_eq * sb_rad) ** (-s)
        return dict(mode=self.clustering, cl_slope=s, ell_eq=self.ell_eq,
                    cl_ratio=self.cl_ratio, cl_amp_sr=self._cl_amp,
                    sigma_delta=self.sigma_delta, R_core=r_core,
                    sigma_b_arcsec=self._sig_b_pix * self.pix_arcsec,
                    q1=self._q1, q2=self._q2,
                    q1_injected=self._q1_injected,
                    q1_ref_used=self.cl_q1_ref is not None)

    def _modulation(self, rng):
        """Draw 1 + delta on the map grid, normalized to unit mean."""
        g = _gaussian_field_from_cl(self._cl_gauss, self._box_rad, rng)
        if self.clustering == "gaussian":
            d = np.maximum(1.0 + g, 0.0)
        else:                                   # lognormal
            d = np.exp(g - 0.5 * g.var())       # unit mean by construction
        return d / d.mean()

    # ---- core ---------------------------------------------------------------
    def make_map(self, lens=None, lens_center=None, seed=None,
                 mean_subtract=True, return_unsmoothed=False):
        """
        One map realization (npix x npix, mJy/beam).

        lens : lens_model.BaseLens or None.  The magnification map is
               evaluated at pixel centres; sources are injected with the
               locally lensed counts (image-plane inhomogeneous Poisson).
        """
        rng = np.random.default_rng(seed)
        raw = np.zeros((self.npix, self.npix))

        #  1 + delta, or None.  Drawn FIRST so that the source-placement
        #  stream is unaffected when clustering is off (backwards
        #  compatibility: the `clustering is None` branches below are
        #  byte-identical to the pre-v2 module).
        dmod = self._modulation(rng) if self.clustering is not None else None

        if lens is None and dmod is None:
            # homogeneous Poisson: one draw of the total per bin, then
            # uniform placement (fast path)
            lam = self.cnts.dnds(self.s_mid) * self.ds \
                * self.om_pix * self.npix ** 2
            for s, l_tot in zip(self.s_mid, lam):
                n_tot = rng.poisson(l_tot)
                if n_tot == 0:
                    continue
                iy = rng.integers(0, self.npix, n_tot)
                ix = rng.integers(0, self.npix, n_tot)
                np.add.at(raw, (iy, ix), s)
        elif lens is None:
            #  Clustered, unlensed.  An inhomogeneous Poisson process with
            #  rate lam(theta) = lam_bar (1+delta) can be drawn exactly as:
            #  N_tot ~ Poisson(int lam), then positions i.i.d. from the
            #  normalized rate.  That is one Poisson draw plus an
            #  inverse-CDF lookup per bin, rather than npix^2 Poisson draws
            #  per bin -- same distribution, ~100x faster.
            cdf = np.cumsum(dmod.ravel())
            cdf /= cdf[-1]
            lam = self.cnts.dnds(self.s_mid) * self.ds \
                * self.om_pix * self.npix ** 2
            flat = raw.ravel()
            for s, l_tot in zip(self.s_mid, lam):
                n_tot = rng.poisson(l_tot)
                if n_tot == 0:
                    continue
                idx = np.searchsorted(cdf, rng.random(n_tot))
                np.add.at(flat, np.minimum(idx, flat.size - 1), s)
            raw = flat.reshape(self.npix, self.npix)
        else:
            mu = lens.mu_map(self.npix, self.pix_arcsec, center=lens_center)
            inv_mu2 = mu ** -2
            for s, ds in zip(self.s_mid, self.ds):
                # rate map: mu^-2 n0(S/mu) dS Om_pix  (vectorized in pixels)
                rate = self.cnts.dnds(s / mu) * inv_mu2 * ds * self.om_pix
                if dmod is not None:
                    #  clustering multiplies the LOCAL rate; lensing and
                    #  clustering therefore commute here, which is the
                    #  statement that the two effects are independent
                    #  (background sources are not clustered with the
                    #  foreground lens -- see memo Sec. 11)
                    rate = rate * dmod
                if self.s_cut is not None:
                    rate[s > self.s_cut] = 0.0   # scalar s: no-op safeguard
                n_map = rng.poisson(rate)
                if n_map.any():
                    raw += n_map * s

        smooth = gaussian_beam_convolve(raw, self.beam_fwhm_pix)
        if self.sigma_noise > 0:
            if self.noise_mode == "white":
                #  independent per pixel, added after the beam
                smooth += rng.normal(0.0, self.sigma_noise, smooth.shape)
            else:
                #  beam-correlated: white noise through the SAME beam,
                #  amplitude set so the output per-pixel rms is sigma_noise
                w = rng.normal(0.0,
                               self.sigma_noise * self._white_per_unit_beamed,
                               smooth.shape)
                smooth += gaussian_beam_convolve(w, self.beam_fwhm_pix)
        if mean_subtract:
            smooth -= smooth.mean()
        return (smooth, raw) if return_unsmoothed else smooth

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def mask_bright(map2d, threshold, grow_pix=0):
        """Map-level masking: NaN out pixels above `threshold` (mJy/beam),
        optionally grown by grow_pix (crude square dilation).  Returns a
        masked copy; downstream histogramming should use finite pixels."""
        m = map2d.copy()
        bad = m > threshold
        if grow_pix > 0:
            from scipy.ndimage import binary_dilation
            bad = binary_dilation(bad, iterations=int(grow_pix))
        m[bad] = np.nan
        return m

    def pixel_histogram(self, maps, bins=200, d_range=None):
        """Pooled, mean-centred pixel histogram (density=True) over one
        map or a list of maps.  Returns (bin_centres, density)."""
        if not isinstance(maps, (list, tuple)):
            maps = [maps]
        v = np.concatenate([m[np.isfinite(m)].ravel() for m in maps])
        if d_range is None:
            d_range = (np.percentile(v, 0.01), np.percentile(v, 99.999))
        h, e = np.histogram(v, bins=bins, range=d_range, density=True)
        return 0.5 * (e[1:] + e[:-1]), h


# %% Demo / validation  (MC vs analytic P(D) — the module-2/4 cross-check)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from counts import Schechter
    from pofd_analytic import PofD

    sch = Schechter(s_min=1e-3)
    beam, pix = 14.9, 4.0
    sim = CIBMapSimulator(sch, beam, pix, npix=1024,
                          sigma_noise=0.3, s_cut=100.0)

    # ---- control field: MC histogram vs analytic P(D) ----------------------
    maps = [sim.make_map(seed=i) for i in range(4)]
    dc, hc = sim.pixel_histogram(maps, bins=240, d_range=(-2.5, 12.0))
    pa = PofD(sch, beam, sigma_noise=0.3, s_cut=100.0)

    # ---- lensed field (uniform mu patch test: cleanest validation) --------
    class _UniformMu:
        """Minimal lens stand-in: constant mu everywhere (validation only)."""
        def __init__(self, mu):
            self.mu_val = mu
        def mu_map(self, npix, pix_arcsec, center=None):
            return np.full((npix, npix), self.mu_val)

    mu0 = 3.0
    maps_l = [sim.make_map(lens=_UniformMu(mu0), seed=100 + i)
              for i in range(4)]
    dl, hl = sim.pixel_histogram(maps_l, bins=240, d_range=(-2.5, 12.0))
    pl = PofD(sch, beam, mu=mu0, sigma_noise=0.3, s_cut=100.0)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].semilogy(dc, hc, "C0.", ms=3, label="MC histogram (4 maps)")
    ax[0].semilogy(pa.d, pa.p, "k", lw=1, label="analytic P(D)")
    ax[0].set(xlim=(-2.5, 10), ylim=(1e-6, 3), xlabel="D [mJy/beam]",
              title="control field")
    ax[1].semilogy(dl, hl, "C3.", ms=3, label=f"MC, uniform mu={mu0}")
    ax[1].semilogy(pl.d, pl.p, "k", lw=1, label="analytic P(D)")
    ax[1].set(xlim=(-2.5, 10), ylim=(1e-6, 3), xlabel="D [mJy/beam]",
              title="lensed patch")
    for a in ax:
        a.legend()
    fig.tight_layout()
    _finish_figure(fig, "demo_simulate_validation")

    # ---- an NFW cluster field, for the eye ---------------------------------
    try:
        from lens_model import NFWLens
        lens = NFWLens(1e15, z_l=0.3, z_s=2.0)
        mlens = sim.make_map(lens=lens, seed=7)
        fig2, ax2 = plt.subplots(figsize=(5.5, 5))
        im = ax2.imshow(mlens, vmin=-1, vmax=4, origin="lower",
                        cmap="magma")
        fig2.colorbar(im, label="mJy/beam")
        ax2.set(title=r"lensed CIB map, $10^{15}M_\odot$ at centre")
        fig2.tight_layout()
        _finish_figure(fig2, "demo_simulate_lensedmap")
    except ImportError:
        print("lens_model not importable; skipped cluster-map demo")
