"""
lens_model.py — Module 3 of the CIB-lensing / GPD project
==========================================================

Analytic magnification profiles mu(theta) for cluster-scale halos.

Purpose
-------
Provides convergence kappa(theta), shear gamma(theta) and magnification

        mu(theta) = 1 / | (1 - kappa)^2 - gamma^2 |

for three circularly-symmetric lens models:

  * NFWLens  — truncated NFW halo, following Lima, Jain & Devlin (2010),
               MNRAS 406, 2352, Eqs. (10)-(17), i.e. the Takada & Jain
               (2003a,b) closed forms for a profile truncated at r_vir.
  * GNFWLens — generalized NFW (free inner slope gamma_in), Sigma(theta)
               by direct 1-D numerical integration (no closed form).
  * SISLens  — singular isothermal sphere, mu = |theta| / (|theta|-thetaE)
               (Hezaveh et al. 2013, Eq. 6); used to reproduce their
               analytic aperture-P(D) test (their Figure 1).

Although the project analysis uses only the magnification (shear does
nothing observable to unresolved point sources, and deflection enters
one-point statistics only through its Jacobian = 1/mu), computing mu
requires BOTH kappa and gamma — hence the full profiles are implemented.

Conventions and assumptions
---------------------------
* Circular symmetry.  LJD's elliptical-potential generalization changes
  the critical-curve geometry but barely changes the area of high-mu
  regions (their Figs. 3, 7); ellipticity is deferred to a later module
  version and noted as a systematic to test.
* Physical (proper) distances and the standard weak-lensing critical
  density  Sigma_cr = c^2 D_s / (4 pi G D_l D_ls)  with angular-diameter
  distances from astropy (default cosmology: Planck18).  LJD write the
  equivalent expressions in comoving units; the resulting kappa, gamma
  are identical.
* Virial definition: Bryan & Norman (1998) Delta_c relative to the
  *critical* density (LJD Eq. 6).  Mass input is M_vir in Msun.
* Concentration: Duffy et al. (2008) 'full' virial relation by default,
      c_vir = 7.85 (M_vir / 2e12 h^-1 Msun)^-0.081 (1+z)^-0.71,
  or the Bullock-style relation used by LJD (their Eq. 2), or a fixed
  user value.
* mu is clipped at mu_max (default 100): near the critical curves the
  point-source magnification formally diverges and is physically
  regulated by finite source size (cf. Hezaveh et al. 2012, 2013).
  Downstream results must be shown to be insensitive to mu_max.

Units
-----
Masses in Msun, angles in ARCMIN (project standard for cluster work),
internal distances in proper Mpc.

Requires: numpy, scipy, astropy.

Typical usage
-------------
>>> from lens_model import NFWLens
>>> lens = NFWLens(m_vir=1e15, z_l=0.3, z_s=2.0)
>>> lens.mu(np.array([0.5, 1.0, 5.0]))     # magnification at 0.5', 1', 5'
>>> mu_map = lens.mu_map(npix=512, pix_arcsec=6.0)

Run the file directly for a demo reproducing the qualitative behaviour of
LJD Figs. 1-2.
"""

# %% Imports
import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18
from astropy import constants as const
from astropy import units as u

ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)

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


# %% Cosmology helpers
def sigma_crit_msun_mpc2(z_l, z_s, cosmo=Planck18):
    """Critical surface density Sigma_cr = c^2 D_s / (4 pi G D_l D_ls)
    in Msun / Mpc^2 (proper), with angular-diameter distances."""
    d_l = cosmo.angular_diameter_distance(z_l)
    d_s = cosmo.angular_diameter_distance(z_s)
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    sc = (const.c ** 2 / (4 * np.pi * const.G)) * d_s / (d_l * d_ls)
    return sc.to(u.Msun / u.Mpc ** 2).value


def bryan_norman_delta_c(z, cosmo=Planck18):
    """Virial overdensity wrt critical density (Bryan & Norman 1998,
    flat cosmologies; LJD Eq. 6):  Delta_c = 18 pi^2 + 82 x - 39 x^2,
    x = Omega_m(z) - 1."""
    x = cosmo.Om(z) - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2


def duffy_cvir(m_vir, z, cosmo=Planck18):
    """Duffy et al. (2008) virial concentration ('full' sample)."""
    m_piv = 2e12 / cosmo.h                    # Msun
    return 7.85 * (m_vir / m_piv) ** (-0.081) * (1.0 + z) ** (-0.71)


def bullock_cvir(m_vir, z, m_star=1.5e13):
    """Bullock-style relation as adopted by LJD (their Eq. 2):
    c = 9/(1+z) (M/M*)^-0.13.  M* default is indicative only."""
    return 9.0 / (1.0 + z) * (m_vir / m_star) ** (-0.13)


# %% Truncated-NFW closed forms (LJD Eqs. 11 and 16; Takada & Jain 2003a,b)
def _tnfw_F(x, c):
    """Convergence shape function F(x), x = theta/theta_s = r_perp/r_s.
    Support: F=0 for x > c.  Vectorized, with series-free handling of
    the x=1 point by evaluation at 1 +/- eps."""
    scalar_in = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float)).copy()
    out = np.zeros_like(x)
    eps = 1e-6
    x[np.abs(x - 1.0) < eps] = 1.0 + eps      # dodge the removable point
    m_lo = (x < 1.0)
    m_hi = (x >= 1.0) & (x < c)
    xl = x[m_lo]
    if xl.size:
        s = np.sqrt(c ** 2 - xl ** 2)
        out[m_lo] = (-s / ((1 - xl ** 2) * (1 + c))
                     + np.arccosh((xl ** 2 + c) / (xl * (1 + c)))
                     / (1 - xl ** 2) ** 1.5)
    xh = x[m_hi]
    if xh.size:
        s = np.sqrt(c ** 2 - xh ** 2)
        out[m_hi] = (-s / ((1 - xh ** 2) * (1 + c))
                     - np.arccos((xh ** 2 + c) / (xh * (1 + c)))
                     / (xh ** 2 - 1) ** 1.5)
    return out[0] if scalar_in else out


def _tnfw_G(x, c, f_inv):
    """Shear shape function G(x) (LJD Eq. 16).  For x > c the halo acts
    as a point mass: G = 2 f^-1 / x^2, with f^-1 = ln(1+c) - c/(1+c)."""
    scalar_in = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float)).copy()
    out = np.zeros_like(x)
    eps = 1e-6
    x[np.abs(x - 1.0) < eps] = 1.0 + eps
    m_lo = (x < 1.0)
    m_hi = (x >= 1.0) & (x < c)
    m_out = (x >= c)

    def common(xx):
        s = np.sqrt(c ** 2 - xx ** 2)
        t1 = ((2 - xx ** 2) * s / (1 - xx ** 2) - 2 * c) / (xx ** 2 * (1 + c))
        t2 = (2.0 / xx ** 2) * np.log(xx * (1 + c) / (c + s))
        return s, t1 + t2

    xl = x[m_lo]
    if xl.size:
        s, t12 = common(xl)
        out[m_lo] = t12 + ((2 - 3 * xl ** 2)
                           * np.arccosh((xl ** 2 + c) / (xl * (1 + c)))
                           / (xl ** 2 * (1 - xl ** 2) ** 1.5))
    xh = x[m_hi]
    if xh.size:
        s, t12 = common(xh)
        out[m_hi] = t12 - ((2 - 3 * xh ** 2)
                           * np.arccos((xh ** 2 + c) / (xh * (1 + c)))
                           / (xh ** 2 * (xh ** 2 - 1) ** 1.5))
    out[m_out] = 2.0 * f_inv / x[m_out] ** 2
    return out[0] if scalar_in else out


# %% Lens classes
class BaseLens:
    """Common interface: kappa(theta), gamma(theta), mu(theta) with theta
    in arcmin; mu_map() renders a pixel grid of |mu|."""

    mu_max = 100.0

    def kappa(self, theta_arcmin):
        raise NotImplementedError

    def gamma(self, theta_arcmin):
        raise NotImplementedError

    def mu(self, theta_arcmin, mu_max=None, signed=False):
        """Magnification.  By default returns |mu| clipped at mu_max
        (point-source work; the sign only flags image parity)."""
        mm = self.mu_max if mu_max is None else mu_max
        k = self.kappa(theta_arcmin)
        g = self.gamma(theta_arcmin)
        det = (1.0 - k) ** 2 - g ** 2
        with np.errstate(divide="ignore"):
            m = 1.0 / det
        if not signed:
            m = np.abs(m)
            m = np.where(np.isfinite(m), np.minimum(m, mm), mm)
        return m

    def mu_map(self, npix, pix_arcsec, center=None, mu_max=None):
        """|mu| on an npix x npix grid (pixel scale pix_arcsec).  The lens
        sits at `center` (pixels; default map center)."""
        cy = cx = (npix - 1) / 2.0
        if center is not None:
            cy, cx = center
        yy, xx = np.indices((npix, npix), dtype=float)
        theta = np.hypot(yy - cy, xx - cx) * pix_arcsec / 60.0   # arcmin
        theta = np.maximum(theta, 1e-4)                          # avoid r=0
        return self.mu(theta, mu_max=mu_max)


class NFWLens(BaseLens):
    """Truncated NFW halo (LJD 2010 Sec. 3.1)."""

    def __init__(self, m_vir, z_l, z_s, cosmo=Planck18,
                 conc="duffy", mu_max=100.0):
        self.m_vir, self.z_l, self.z_s, self.cosmo = m_vir, z_l, z_s, cosmo
        self.mu_max = mu_max
        if conc == "duffy":
            self.c = duffy_cvir(m_vir, z_l, cosmo)
        elif conc == "bullock":
            self.c = bullock_cvir(m_vir, z_l)
        else:
            self.c = float(conc)
        # virial radius (proper Mpc) from Delta_c x rho_crit(z)
        dc = bryan_norman_delta_c(z_l, cosmo)
        rho_c = cosmo.critical_density(z_l).to(u.Msun / u.Mpc ** 3).value
        self.r_vir = (3.0 * m_vir / (4.0 * np.pi * dc * rho_c)) ** (1.0 / 3.0)
        self.f_inv = np.log(1.0 + self.c) - self.c / (1.0 + self.c)
        d_l = cosmo.angular_diameter_distance(z_l).to(u.Mpc).value
        self.theta_vir = (self.r_vir / d_l) / ARCMIN_TO_RAD   # arcmin
        self.theta_s = self.theta_vir / self.c
        self.sigma_cr = sigma_crit_msun_mpc2(z_l, z_s, cosmo)
        # prefactor of Sigma(theta): M f c^2 / (2 pi r_vir^2)   [Msun/Mpc^2]
        self.sig0 = (m_vir / self.f_inv) * self.c ** 2 \
            / (2.0 * np.pi * self.r_vir ** 2)

    def kappa(self, theta_arcmin):
        x = np.asarray(theta_arcmin, float) / self.theta_s
        return self.sig0 * _tnfw_F(x, self.c) / self.sigma_cr

    def gamma(self, theta_arcmin):
        x = np.asarray(theta_arcmin, float) / self.theta_s
        return self.sig0 * _tnfw_G(x, self.c, self.f_inv) / self.sigma_cr


class GNFWLens(BaseLens):
    """
    Generalized NFW:  rho(r) = rho_s / [(r/r_s)^g_in (1+r/r_s)^(3-g_in)]
    truncated at r_vir, with Sigma(theta) and mean convergence computed
    numerically (1-D quadratures; gamma = kappa_bar - kappa for circular
    symmetry).  g_in = 1 reproduces NFWLens (numerically).
    Slower than NFWLens; intended for profile-systematics tests.
    """

    def __init__(self, m_vir, z_l, z_s, g_in=1.0, cosmo=Planck18,
                 conc="duffy", mu_max=100.0, n_grid=512):
        self.g_in, self.mu_max = float(g_in), mu_max
        if conc == "duffy":
            self.c = duffy_cvir(m_vir, z_l, cosmo)
        elif conc == "bullock":
            self.c = bullock_cvir(m_vir, z_l)
        else:
            self.c = float(conc)
        dc = bryan_norman_delta_c(z_l, cosmo)
        rho_c = cosmo.critical_density(z_l).to(u.Msun / u.Mpc ** 3).value
        self.r_vir = (3.0 * m_vir / (4.0 * np.pi * dc * rho_c)) ** (1.0 / 3.0)
        self.r_s = self.r_vir / self.c
        # rho_s from M_vir = 4 pi int_0^rvir rho r^2 dr
        integ = quad(lambda r: r ** 2 / ((r / self.r_s) ** self.g_in
                     * (1 + r / self.r_s) ** (3 - self.g_in)),
                     0.0, self.r_vir, limit=200)[0]
        self.rho_s = m_vir / (4.0 * np.pi * integ)
        d_l = cosmo.angular_diameter_distance(z_l).to(u.Mpc).value
        self.theta_vir = (self.r_vir / d_l) / ARCMIN_TO_RAD
        self.sigma_cr = sigma_crit_msun_mpc2(z_l, z_s, cosmo)
        # tabulate kappa(theta) and kappa_bar(theta) once on a log grid
        th = np.geomspace(self.theta_vir * 1e-4, self.theta_vir, n_grid)
        rp = th * ARCMIN_TO_RAD * d_l                       # proper Mpc
        sig = np.array([self._sigma_proj(r) for r in rp])
        self._th, self._k = th, sig / self.sigma_cr
        # kappa_bar(<theta) = 2/theta^2 int_0^theta k(t) t dt  (log-grid trapz)
        integrand = self._k * th ** 2                       # k t * t dlnt
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                        * np.diff(np.log(th)))])
        # add the inner-disk contribution assuming k ~ theta^(1-g_in) below grid
        p = 1.0 - self.g_in
        inner = self._k[0] * th[0] ** 2 / (2.0 + p) if (2.0 + p) > 0 else 0.0
        self._kbar = 2.0 * (cum + inner) / th ** 2

    def _sigma_proj(self, r_perp):
        """Projected surface density at proper radius r_perp (Mpc),
        integrating along the LOS to the truncation sphere r = r_vir."""
        if r_perp >= self.r_vir:
            return 0.0
        zmax = np.sqrt(self.r_vir ** 2 - r_perp ** 2)

        def rho(zz):
            r = np.hypot(r_perp, zz)
            return self.rho_s / ((r / self.r_s) ** self.g_in
                                 * (1 + r / self.r_s) ** (3 - self.g_in))
        return 2.0 * quad(rho, 0.0, zmax, limit=200)[0]

    def kappa(self, theta_arcmin):
        th = np.asarray(theta_arcmin, float)
        return np.interp(th, self._th, self._k, left=self._k[0], right=0.0)

    def gamma(self, theta_arcmin):
        th = np.asarray(theta_arcmin, float)
        kbar_in = np.interp(th, self._th, self._kbar)
        # outside r_vir the enclosed mass is fixed: kbar ~ theta^-2
        out = self._kbar[-1] * (self._th[-1] / np.maximum(th, 1e-12)) ** 2
        kbar = np.where(th <= self._th[-1], kbar_in, out)
        return kbar - self.kappa(th)


class UniformMu(BaseLens):
    """Constant magnification over the whole field.  Not a physical lens:
    a stand-in for validation runs (uniform-mu patches have exactly the
    analytic P(D) of counts.LensedCounts, so simulator and analytic
    pipelines can be compared with no profile-mixing ambiguity)."""

    def __init__(self, mu, mu_max=100.0):
        self.mu_val, self.mu_max = float(mu), mu_max

    def kappa(self, theta_arcmin):
        raise NotImplementedError("UniformMu defines mu directly")

    gamma = kappa

    def mu(self, theta_arcmin, mu_max=None, signed=False):
        return np.full(np.shape(np.asarray(theta_arcmin, float)) or (),
                       self.mu_val)


class SISLens(BaseLens):
    """Singular isothermal sphere.  kappa = gamma = thetaE/(2 theta), so
    mu = |theta/(theta - thetaE)| (Hezaveh+13 Eq. 6).  Construct either
    from theta_E directly (arcmin) or from sigma_v (km/s) + redshifts."""

    def __init__(self, theta_e_arcmin=None, sigma_v_kms=None,
                 z_l=None, z_s=None, cosmo=Planck18, mu_max=100.0):
        self.mu_max = mu_max
        if theta_e_arcmin is not None:
            self.theta_e = float(theta_e_arcmin)
        else:
            d_s = cosmo.angular_diameter_distance(z_s)
            d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
            th = 4.0 * np.pi * (sigma_v_kms * u.km / u.s / const.c) ** 2 \
                * (d_ls / d_s)
            self.theta_e = float(th.decompose().value) / ARCMIN_TO_RAD

    def kappa(self, theta_arcmin):
        return self.theta_e / (2.0 * np.asarray(theta_arcmin, float))

    gamma = kappa


# %% Demo / self-test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # LJD Fig. 1 anchor: M = 1e14/h, z_l = 0.2, z_s = 1.0 (Bullock-like c)
    h = Planck18.h
    lens = NFWLens(1e14 / h, 0.2, 1.0, conc="bullock")
    th = np.geomspace(1e-2, 30, 400)
    print(f"c = {lens.c:.2f}, r_vir = {lens.r_vir:.2f} Mpc, "
          f"theta_vir = {lens.theta_vir:.2f}'")
    for t in [0.01, 0.1, 1.0]:
        print(f"theta = {t:5.2f}' : kappa = {lens.kappa(t):.3f}, "
              f"gamma = {lens.gamma(t):.3f}, mu = {lens.mu(t):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].loglog(th, lens.kappa(th), label=r"$\kappa$")
    ax[0].loglog(th, np.abs(lens.gamma(th)), label=r"$|\gamma|$")
    ax[0].loglog(th, lens.mu(th) - 1, label=r"$\mu-1$")
    ax[0].set(xlabel=r"$\theta$ [arcmin]", ylim=(1e-4, 30),
              title=r"tNFW, $10^{14}h^{-1}M_\odot$, $z_l$=0.2, $z_s$=1")
    ax[0].legend()

    # mass scaling of mu at fixed angle (the xi(M) lever arm)
    masses = np.geomspace(1e14, 2e15, 12)
    for zl, mk in [(0.2, "o-"), (0.5, "s--")]:
        mus = [NFWLens(m, zl, 2.0).mu(0.5) for m in masses]
        ax[1].semilogx(masses, mus, mk, label=f"z_l={zl}")
    ax[1].set(xlabel=r"$M_{\rm vir}$ [$M_\odot$]",
              ylabel=r"$\mu(0.5')$", title="mass scaling at 0.5'")
    ax[1].legend()

    # GNFW inner-slope comparison and a mu map
    for g_in, c_ in [(0.5, "C0"), (1.0, "C1"), (1.5, "C3")]:
        gl = GNFWLens(1e15, 0.3, 2.0, g_in=g_in)
        ax[2].loglog(th, gl.mu(th) - 1, c_, label=fr"GNFW $\gamma$={g_in}")
    nl = NFWLens(1e15, 0.3, 2.0)
    ax[2].loglog(th, nl.mu(th) - 1, "k:", label="tNFW closed form")
    ax[2].set(xlabel=r"$\theta$ [arcmin]", ylabel=r"$\mu-1$",
              title=r"$10^{15}M_\odot$, $z_l$=0.3, $z_s$=2")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, "demo_lens_model")
