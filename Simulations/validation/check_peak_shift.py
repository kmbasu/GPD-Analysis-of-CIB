"""
check_peak_shift.py -- numerical verification of the analytic peak-selection
shift for beam-declustered maps (project memo, Q2).

Tests, in order:
  T1. Continuum 2D Gaussian random field peak theory (Rice/BBKS/Bond-Efstathiou):
        n_pk(nu) = (sigma_2^2 / (4 pi sigma_1^2)) * phi(nu) * G2(gamma, nu)
        G2(gamma,nu) = E_{J|nu}[ (J^2 - 1 + e^{-J^2}) 1_{J>0} ],  J|nu ~ N(gamma nu, 1-gamma^2)
      Check the total maxima density against a direct simulation, and the
      predicted mode / median / mean of the peak-height distribution against
      the histogram of `beam_declustered_peaks` output.
  T2. Beam-only (no white noise) case: gamma = 1/sqrt(2) exactly for a
      Gaussian-beam-smoothed Poisson/white field. Scan beam FWHM.
  T3. White instrument noise added AFTER beam convolution (as in
      simulate_maps.CIBMapSimulator): recompute gamma from the composite
      power spectrum and re-predict the shift.  Compare to simulation.
  T4. Apply to the actual (non-Gaussian) confusion field.
"""
# %% imports
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
rng = np.random.default_rng(7)


# %% analytic 2D peak-height distribution
def G2(gamma, nu):
    """E_{J|nu}[ (J^2 - 1 + e^{-J^2}) 1_{J>0} ] with J|nu ~ N(gamma*nu, 1-gamma^2).

    Closed form: with m = gamma*nu, s^2 = 1-gamma^2,
      E[J^2 1_{J>0}] = (m^2+s^2) Phi(m/s) + m s phi(m/s)
      E[1_{J>0}]     = Phi(m/s)
      E[e^{-J^2} 1]  = (1/sqrt(1+2 s^2)) exp(-m^2/(1+2 s^2)) * Phi( m/(s sqrt(1+2 s^2)) )
    """
    m = gamma * np.asarray(nu, float)
    s2 = 1.0 - gamma ** 2
    s = np.sqrt(s2)
    z = m / s
    P, p = norm.cdf(z), norm.pdf(z)
    e_j2 = (m ** 2 + s2) * P + m * s * p
    e_1 = P
    a = 1.0 + 2.0 * s2
    e_exp = np.exp(-m ** 2 / a) / np.sqrt(a) * norm.cdf(m / (s * np.sqrt(a)))
    return e_j2 - e_1 + e_exp


def _check_G2_closed_form(gamma=0.7071, nu=1.3):
    """Brute-force numeric check of the closed form above."""
    m, s = gamma * nu, np.sqrt(1 - gamma ** 2)
    f = lambda J: (J ** 2 - 1 + np.exp(-J ** 2)) * norm.pdf(J, m, s)
    num = quad(f, 0, 40)[0]
    return num, G2(gamma, nu)


def peak_pdf(nu, gamma):
    """Normalised pdf of declustered-peak heights nu = (D-mu)/sigma_0."""
    w = norm.pdf(nu) * G2(gamma, nu)
    return w


def peak_stats(gamma):
    """Mode, median, mean of the 2D-GRF peak-height distribution."""
    grid = np.linspace(-4, 8, 24001)
    w = peak_pdf(grid, gamma)
    w = np.maximum(w, 0)
    norm_ = np.trapezoid(w, grid)
    pdf = w / norm_
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    mode = grid[np.argmax(pdf)]
    # refine mode
    r = minimize_scalar(lambda x: -peak_pdf(x, gamma),
                        bracket=(mode - 0.2, mode, mode + 0.2))
    mode = r.x
    median = np.interp(0.5, cdf, grid)
    mean = np.trapezoid(grid * pdf, grid)
    return dict(mode=mode, median=median, mean=mean, grid=grid, pdf=pdf)


def n_peaks_per_area(sig1, sig2):
    """Total density of maxima per unit area (continuum, 2D isotropic GRF)."""
    return sig2 ** 2 / (8.0 * np.sqrt(3.0) * np.pi * sig1 ** 2)


# %% "max of n_eff iid" comparison model
def iid_mode(n):
    """Mode of the max of n iid N(0,1): solves z = (n-1) phi(z)/Phi(z)."""
    f = lambda z: z - (n - 1) * norm.pdf(z) / norm.cdf(z)
    return brentq(f, 1e-6, 12)


def iid_median(n):
    """Median of max of n iid N(0,1): Phi(z)^n = 1/2."""
    return norm.ppf(0.5 ** (1.0 / n))


# %% map machinery (mirrors analysis_modules)
def gaussian_beam_convolve(img, beam_fwhm_pix):
    n = img.shape[0]
    sig = beam_fwhm_pix * FWHM_TO_SIGMA
    k = np.fft.fftfreq(n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    bl = 2.0 * np.pi * sig ** 2 * np.exp(-2.0 * (np.pi * sig) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * bl))


def declustered_peaks(m, beam_fwhm_pix):
    size = max(1, int(round(beam_fwhm_pix)))
    is_max = (m == maximum_filter(m, size=size))
    return m[is_max]


def spectral_moments(m):
    """sigma_0,1,2 from the discrete 2D power spectrum, in pixel units."""
    n = m.shape[0]
    F = np.fft.fft2(m - m.mean())
    P = np.abs(F) ** 2 / n ** 4
    k = 2 * np.pi * np.fft.fftfreq(n)          # rad / pixel
    ky, kx = np.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    s0 = P.sum()
    s1 = (k2 * P).sum()
    s2 = (k2 ** 2 * P).sum()
    return np.sqrt(s0), np.sqrt(s1), np.sqrt(s2)


# %% ---------------- T1/T2: beam-smoothed white field ----------------------
if __name__ == "__main__":
    print("=" * 78)
    print("T0. closed-form G2 check (gamma=0.7071, nu=1.3):")
    a, b = _check_G2_closed_form()
    print(f"    quad = {a:.10f}   closed form = {b:.10f}   diff = {a-b:.2e}")

    print("\n" + "=" * 78)
    print("T1/T2. Beam-smoothed white (Poisson-like) field, NO instrument noise")
    print("       continuum prediction: gamma = 1/sqrt(2) = 0.70711 exactly")
    st = peak_stats(1 / np.sqrt(2))
    print(f"       -> predicted peak-height  mode  = {st['mode']:.4f} sigma")
    print(f"                                 median= {st['median']:.4f} sigma")
    print(f"                                 mean  = {st['mean']:.4f} sigma")
    print()
    hdr = (f"{'FWHM[pix]':>10s} {'gamma_sim':>10s} {'mode_pred':>10s} "
           f"{'mode_sim':>9s} {'med_pred':>9s} {'med_sim':>8s} "
           f"{'npk_pred':>9s} {'npk_sim':>8s}")
    print(hdr)
    NPIX = 2048
    for fwhm_pix in [3.725, 5.0, 7.5, 10.0]:
        white = rng.normal(size=(NPIX, NPIX))
        m = gaussian_beam_convolve(white, fwhm_pix)
        s0, s1, s2 = spectral_moments(m)
        g_sim = s1 ** 2 / (s0 * s2)
        pk = declustered_peaks(m, fwhm_pix)
        z = (pk - np.median(m)) / m.std()
        # empirical mode via KDE-free histogram peak on a fine grid
        h, e = np.histogram(z, bins=200, range=(-2, 6), density=True)
        c = 0.5 * (e[1:] + e[:-1])
        # smooth the histogram lightly before taking argmax
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); k /= k.sum()
        mode_sim = c[np.argmax(np.convolve(h, k, mode="same"))]
        stp = peak_stats(g_sim)
        npk_pred = n_peaks_per_area(s1, s2)             # per pixel^2
        npk_sim = pk.size / m.size
        print(f"{fwhm_pix:10.3f} {g_sim:10.4f} {stp['mode']:10.4f} "
              f"{mode_sim:9.4f} {stp['median']:9.4f} {np.median(z):8.4f} "
              f"{npk_pred:9.5f} {npk_sim:8.5f}")

    # %% ---------------- T3: white noise added AFTER the beam ---------------
    print("\n" + "=" * 78)
    print("T3. White instrument noise ADDED AFTER beam convolution")
    print("    (this is what simulate_maps.CIBMapSimulator does)")
    print(f"{'f_N':>6s} {'gamma_sim':>10s} {'mode_pred':>10s} {'mode_sim':>9s} "
          f"{'med_pred':>9s} {'med_sim':>8s} {'npk_pred':>9s} {'npk_sim':>8s}")
    fwhm_pix = 5.0
    white = rng.normal(size=(NPIX, NPIX))
    base = gaussian_beam_convolve(white, fwhm_pix)
    base /= base.std()                                # unit "confusion" sigma
    for f_N in [0.0, 0.25, 0.5, 1.0, 2.0]:
        m = base + f_N * rng.normal(size=base.shape)
        s0, s1, s2 = spectral_moments(m)
        g_sim = s1 ** 2 / (s0 * s2)
        pk = declustered_peaks(m, fwhm_pix)
        z = (pk - np.median(m)) / m.std()
        h, e = np.histogram(z, bins=200, range=(-2, 6), density=True)
        c = 0.5 * (e[1:] + e[:-1])
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); k /= k.sum()
        mode_sim = c[np.argmax(np.convolve(h, k, mode="same"))]
        stp = peak_stats(g_sim)
        print(f"{f_N:6.2f} {g_sim:10.4f} {stp['mode']:10.4f} {mode_sim:9.4f} "
              f"{stp['median']:9.4f} {np.median(z):8.4f} "
              f"{n_peaks_per_area(s1,s2):9.5f} {pk.size/m.size:8.5f}")

    # %% -------- iid comparison model, for reference -------------------------
    print("\n" + "=" * 78)
    print("Reference: 'max of n_eff iid Gaussians' model")
    for n in [5, 10, 15, 20, 25, 30]:
        print(f"   n_eff = {n:3d}:  mode = {iid_mode(n):.3f}, "
              f"median = {iid_median(n):.3f}")
