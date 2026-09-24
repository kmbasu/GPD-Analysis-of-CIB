#!/usr/bin/env python3
"""
herschel_xid_flux_response.py -- module H4: is the matched filter's 1.043
point-source flux response real?

PURPOSE
-------
Appendix D of the paper reports that filtering an IDEALISED Gaussian PSF
(FWHM 25.15") with the HELP `Matchedfilter` kernel returns a peak of 1.0426
relative to unit normalisation (herschel_unlensed_v3.py, cell 2).  If the
real SPIRE point-source response of the self-filtered map were 1.043 rather
than 1.000, the xi(u) flux axis of Sect. 5 would be mis-scaled by 4.3 %
(worth Delta-xi ~ 0.013 at the measured d xi / d ln u).  The alternative is
that the 4.3 % is an artefact of the Gaussian stand-in: the kernel is built
for the true SPIRE PSF (with wings the Gaussian lacks), and HELP normalise
MFILT so that a true point source has unit response.

This module settles it empirically with the HELP XID+ catalogue (dmu26),
whose 350 um fluxes are PSF-fitted on the UNFILTERED image at prior
positions and are therefore independent of the filter.  For bright, isolated
XID+ sources we measure

    R_sf  = SF_IMAGE peak / F_XID+     (self-filtered IMAGE, our operator)
    R_img = IMAGE peak    / F_XID+     (raw map: unit response by construction
                                        of the Jy/beam convention -- control)
    R_mf  = HELP MFILT peak / F_XID+   (HELP's own filtered product -- second
                                        control, must equal R_sf up to the
                                        NEBFILT-vs-IMAGE input difference)

and the direct filter gain  G = SF_IMAGE peak / IMAGE peak  at the same
positions, which is the quantity the appendix actually asks about.  Peaks are
read at the catalogue position with a local 3x3 quadratic refinement (removes
the up-to-7 % pixel-centring loss at 8" pixels) after subtracting a local
median baseline from an annulus of 1.5'-4'.

Selection: F_XID+ in [F_MIN, F_MAX] mJy (default 60-400: bright enough that
confusion (sigma_c ~ 7 mJy) and instrument noise (~9 mJy) are a <~15 %
perturbation, faint enough to avoid the handful of saturated/extended
objects), no XID+ neighbour with flux > F_NEIGH_FRAC x F within R_ISO
arcsec (default 0.2 x F within 45" = 1.8 FWHM), fully valid pixels in the
padded filtering cutout.  Results are reported as the median ratio with a
bootstrap error, the error-weighted slope of peak vs flux through the origin,
and the same in three flux bins so a flux-dependent (confusion-boost) trend
is visible.  Two robustness legs: (i) the raw pixel value at the nearest
pixel instead of the quadratic peak, (ii) the isolation radius doubled.

INPUTS
------
data/GAMA-09_SPIRE350_v1.0.fits      (HDU 1 IMAGE, 3 ERROR,
                                                         5 MASK, 6 MFILT,
                                                         8 Matchedfilter)
data/dmu26_XID+SPIRE_GAMA-09_20180508.fits  (dmu26 table)

OUTPUTS  (results/)
-------
herschel_xid_flux_response_350.npz   per-source table + summary statistics
herschel_h4_flux_response.png        peak-vs-flux and ratio-vs-flux panels

ENVIRONMENT
-----------
CIB_XID_FILE, CIB_HERSCHEL_DIR, CIB_FMIN, CIB_FMAX, CIB_RISO (arcsec),
CIB_NMAX (cap on the number of sources, default 2000).
"""
#%% ---------------------------------------------------------- imports/config
import os, sys, time, json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.signal import oaconvolve
from scipy.ndimage import maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CIB_HERSCHEL_DIR", os.path.join(HERE, "data"))
MAP_FILE = os.path.join(DATA_DIR, "GAMA-09_SPIRE350_v1.0.fits")
XID_FILE = os.environ.get("CIB_XID_FILE", os.path.join(
    DATA_DIR, "dmu26_XID+SPIRE_GAMA-09_20180508.fits"))
OUT_DIR = os.path.join(HERE, "results")
F_MIN = float(os.environ.get("CIB_FMIN", "60"))
F_MAX = float(os.environ.get("CIB_FMAX", "400"))
R_ISO = float(os.environ.get("CIB_RISO", "45"))          # arcsec
F_NEIGH_FRAC = 0.2
N_MAX = int(os.environ.get("CIB_NMAX", "2000"))
JY2MJY = 1.0e3
PIX = 8.0                                                # arcsec
FWHM = 25.15
HALF = 80          # cutout half-size in pixels (161 px = 21.5'); kernel half-width is 50
ANN_IN, ANN_OUT = 1.5 * 60 / PIX, 4.0 * 60 / PIX         # baseline annulus, pixels
T0 = time.time()
os.makedirs(OUT_DIR, exist_ok=True)


def apply_matched_filter(d, sigma, valid, K, K2):
    """Chapin et al. (2011) inverse-variance matched filter, identical to
    herschel_unlensed_v3.apply_matched_filter (copied so this module has no
    import-time side effects from the production script)."""
    w = np.where(valid, 1.0 / np.maximum(sigma, 1e-30) ** 2, 0.0)
    num = oaconvolve(np.where(valid, d, 0.0) * w, K, mode="same")
    den = oaconvolve(w, K2, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def quad_peak(a, y, x):
    """Value of the local quadratic surface fitted to the 3x3 neighbourhood of
    (y, x), evaluated at its stationary point if that lies within +-1 px,
    else the maximum of the 3x3 block.  Removes the pixel-centring loss."""
    z = a[y - 1:y + 2, x - 1:x + 2]
    if not np.all(np.isfinite(z)):
        return np.nan, np.nan, np.nan
    # 2-D quadratic: f = c0 + cx x + cy y + cxx x^2 + cyy y^2 + cxy xy
    yy, xx = np.mgrid[-1:2, -1:2]
    A = np.column_stack([np.ones(9), xx.ravel(), yy.ravel(), xx.ravel() ** 2,
                         yy.ravel() ** 2, (xx * yy).ravel()])
    c, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    if np.linalg.det(H) > 0 and c[3] < 0:                    # a maximum
        dx, dy = np.linalg.solve(H, -c[1:3])
        if abs(dx) <= 1 and abs(dy) <= 1:
            val = (c[0] + c[1] * dx + c[2] * dy + c[3] * dx ** 2
                   + c[4] * dy ** 2 + c[5] * dx * dy)
            return float(val), float(dx), float(dy)
    return float(z.max()), np.nan, np.nan


#%% ------------------------------------------------------------ load inputs
print("=" * 79); print("  H4: XID+ flux-response cross-match, GAMA-09 350 um"); print("=" * 79)
hdul = fits.open(MAP_FILE, memmap=True)
wcs = WCS(hdul[1].header)
K = hdul[8].data.astype(np.float64); K2 = K ** 2
NY, NX = hdul[1].data.shape
print(f"  map {NX}x{NY} @ {PIX}\" ; kernel {K.shape}, sum K = {K.sum():.4f}, sum K^2 = {K2.sum():.4f}")

# Gaussian-PSF response, reproduced here for the record (App. D's 1.0426)
n_s, c_s = 401, 200
yy, xx = np.mgrid[:n_s, :n_s]
sig_b = (FWHM / PIX) / (2 * np.sqrt(2 * np.log(2)))
PSF_G = np.exp(-((yy - c_s) ** 2 + (xx - c_s) ** 2) / (2 * sig_b ** 2))
FLUX_RESP_GAUSS = float((oaconvolve(PSF_G, K, mode="same") / K2.sum()).max())
print(f"  Gaussian-PSF response of the filter (App. D)   = {FLUX_RESP_GAUSS:.4f}")

if not os.path.exists(XID_FILE):
    sys.exit(f"XID+ catalogue not found: {XID_FILE}")
cat = Table.read(XID_FILE)
cols = {c.lower(): c for c in cat.colnames}
def col(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    raise KeyError(f"none of {names} in catalogue; columns: {cat.colnames[:40]}")
C_RA, C_DEC = col("RA", "ra_help", "RA_HELP"), col("Dec", "DEC", "dec_help", "DEC_HELP")
C_F = col("F_SPIRE_350", "f_spire_350", "F_MED_350", "F_350")
C_FE_LO = cols.get("ferr_spire_350_l") or cols.get("f_spire_350_l") or cols.get("ferr_spire_350_lo")
C_FE_HI = cols.get("ferr_spire_350_u") or cols.get("f_spire_350_u") or cols.get("ferr_spire_350_hi")
print(f"  catalogue {len(cat)} rows; using RA={C_RA}, Dec={C_DEC}, flux={C_F} "
      f"(err cols {C_FE_LO}, {C_FE_HI})")
ra = np.asarray(cat[C_RA], float); dec = np.asarray(cat[C_DEC], float)
F = np.asarray(cat[C_F], float)
F = F * (JY2MJY if np.nanmedian(F[F > 0]) < 1.0 else 1.0)       # Jy -> mJy if needed
ok = np.isfinite(F) & np.isfinite(ra) & np.isfinite(dec)
ra, dec, F = ra[ok], dec[ok], F[ok]
xpix, ypix = wcs.all_world2pix(ra, dec, 0)

#%% ------------------------------------------------- select bright, isolated
bright = (F >= F_MIN) & (F <= F_MAX)
idx_b = np.where(bright)[0]
print(f"  {idx_b.size} XID+ sources with {F_MIN} <= F350 <= {F_MAX} mJy")
# isolation against ALL catalogue sources (any flux), using the pixel grid
from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([xpix, ypix]))
r_iso_pix = R_ISO / PIX
keep = []
for i in idx_b:
    nb = tree.query_ball_point([xpix[i], ypix[i]], r_iso_pix)
    nb = [j for j in nb if j != i]
    if not any(F[j] > F_NEIGH_FRAC * F[i] for j in nb):
        keep.append(i)
keep = np.array(keep)[:N_MAX]
print(f"  {keep.size} isolated (no neighbour > {F_NEIGH_FRAC:.1f} F within {R_ISO:.0f}\")")

#%% ------------------------------------------------------ measure per source
rows = []
mask_hdu = hdul[5].data
for n, i in enumerate(keep):
    xc, yc = int(round(xpix[i])), int(round(ypix[i]))
    if not (HALF < xc < NX - HALF - 1 and HALF < yc < NY - HALF - 1):
        continue
    B = (slice(yc - HALF, yc + HALF + 1), slice(xc - HALF, xc + HALF + 1))
    img = hdul[1].data[B].astype(np.float64) * JY2MJY
    err = hdul[3].data[B].astype(np.float64) * JY2MJY
    hmf = hdul[6].data[B].astype(np.float64) * JY2MJY
    val = np.isfinite(img) & np.isfinite(err) & (err > 0) & (mask_hdu[B] == 0)
    if val.mean() < 0.999:                                  # require full coverage
        continue
    sf = apply_matched_filter(img, err, val, K, K2)
    # local baselines: median in an annulus, evaluated per map
    ry, rx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    rr = np.hypot(ry, rx)
    ann = (rr >= ANN_IN) & (rr <= ANN_OUT)
    out = dict(idx=int(i), F=float(F[i]), ra=float(ra[i]), dec=float(dec[i]),
               err_c=float(err[HALF, HALF]))
    for tag, m in (("img", img), ("sf", sf), ("mf", hmf)):
        base = float(np.nanmedian(m[ann]))
        pk, dx, dy = quad_peak(m - base, HALF, HALF)
        out[f"pk_{tag}"] = pk
        out[f"pix_{tag}"] = float(m[HALF, HALF] - base)           # robustness leg (i)
        out[f"dx_{tag}"], out[f"dy_{tag}"] = dx, dy
        # local maximum within 1 FWHM (peak-selection convention of the pipeline)
        sub = (m - base)[HALF - 3:HALF + 4, HALF - 3:HALF + 4]
        out[f"max_{tag}"] = float(np.nanmax(sub))
    rows.append(out)
    if (n + 1) % 100 == 0:
        print(f"    ... {n + 1}/{keep.size} ({time.time() - T0:.0f} s)")
tab = Table(rows=rows)
print(f"  {len(tab)} sources measured")

#%% ------------------------------------------------------------- statistics
rng = np.random.default_rng(1234)
def ratio_stats(num, den, w=None, nboot=2000):
    r = num / den
    good = np.isfinite(r)
    r, num, den = r[good], num[good], den[good]
    med = np.median(r)
    boots = np.array([np.median(rng.choice(r, r.size)) for _ in range(nboot)])
    # slope through the origin, inverse-variance weights from the map noise
    slope = np.sum(num * den) / np.sum(den ** 2)
    sb = np.array([(lambda k: np.sum(num[k] * den[k]) / np.sum(den[k] ** 2))(
        rng.choice(r.size, r.size)) for _ in range(nboot)])
    return dict(n=int(r.size), median=float(med), median_err=float(boots.std()),
                slope=float(slope), slope_err=float(sb.std()),
                p16=float(np.percentile(r, 16)), p84=float(np.percentile(r, 84)))

Fv = np.asarray(tab["F"])
S = {}
print("\n  %-28s %5s %9s %9s %s" % ("ratio", "N", "median", "slope", "(16-84 %)"))
for name, num, den in (("SF_IMAGE / XID+", tab["pk_sf"], Fv),
                       ("IMAGE / XID+", tab["pk_img"], Fv),
                       ("HELP MFILT / XID+", tab["pk_mf"], Fv),
                       ("SF_IMAGE / IMAGE  (gain G)", tab["pk_sf"], tab["pk_img"]),
                       ("HELP MFILT / SF_IMAGE", tab["pk_mf"], tab["pk_sf"]),
                       ("[pixel] SF_IMAGE / XID+", tab["pix_sf"], Fv),
                       ("[pixel] SF_IMAGE / IMAGE", tab["pix_sf"], tab["pix_img"]),
                       ("[max1FWHM] SF_IMAGE / XID+", tab["max_sf"], Fv),
                       ("[max1FWHM] IMAGE / XID+", tab["max_img"], Fv)):
    s = ratio_stats(np.asarray(num, float), np.asarray(den, float))
    S[name] = s
    print("  %-28s %5d %6.3f+-%.3f %6.3f+-%.3f (%.3f-%.3f)"
          % (name, s["n"], s["median"], s["median_err"], s["slope"], s["slope_err"], s["p16"], s["p84"]))

print("\n  flux bins, SF_IMAGE/XID+ and gain G = SF/IMAGE:")
edges = [F_MIN, 90, 140, F_MAX]
BINS = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Fv >= lo) & (Fv < hi)
    if m.sum() < 5: continue
    a = ratio_stats(np.asarray(tab["pk_sf"])[m], Fv[m])
    g = ratio_stats(np.asarray(tab["pk_sf"])[m], np.asarray(tab["pk_img"])[m])
    c = ratio_stats(np.asarray(tab["pk_img"])[m], Fv[m])
    BINS.append(dict(lo=lo, hi=hi, n=int(m.sum()), sf_xid=a["median"], sf_xid_err=a["median_err"],
                     img_xid=c["median"], img_xid_err=c["median_err"], gain=g["median"], gain_err=g["median_err"]))
    print("    %3.0f-%3.0f mJy  N=%3d  SF/XID %.3f+-%.3f  IMG/XID %.3f+-%.3f  G %.3f+-%.3f"
          % (lo, hi, m.sum(), a["median"], a["median_err"], c["median"], c["median_err"], g["median"], g["median_err"]))

print(f"""
  READING.  G = SF_IMAGE/IMAGE is the empirical point-source gain of our
  operator on the real PSF.  Compare with the Gaussian-stand-in value
  {FLUX_RESP_GAUSS:.4f} of App. D: G ~ 1.00 means the 4.3 % is an artefact of
  the Gaussian; G ~ 1.04 means the flux axis of Sect. 5 is mis-scaled by
  that amount.  SF/XID+ and IMAGE/XID+ carry, in addition, the common
  confusion boost of a flux-limited selection (positive, decreasing with F),
  which is why the bins are shown and why G is the cleaner number.""")

np.savez_compressed(os.path.join(OUT_DIR, "herschel_xid_flux_response_350.npz"),
                    table=tab.as_array(), summary_json=json.dumps(S), bins_json=json.dumps(BINS),
                    flux_resp_gauss=FLUX_RESP_GAUSS, f_min=F_MIN, f_max=F_MAX, r_iso=R_ISO,
                    xid_file=os.path.basename(XID_FILE))
print(f"  wrote results/herschel_xid_flux_response_350.npz  ({time.time() - T0:.0f} s)")

#%% ----------------------------------------------------------------- figure
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(Fv, tab["pk_img"], "o", ms=3, alpha=.5, label="IMAGE peak")
    ax[0].plot(Fv, tab["pk_sf"], "s", ms=3, alpha=.5, label="self-filtered peak")
    ax[0].plot([F_MIN, F_MAX], [F_MIN, F_MAX], "k-", lw=.8)
    ax[0].plot([F_MIN, F_MAX], [FLUX_RESP_GAUSS * F_MIN, FLUX_RESP_GAUSS * F_MAX], "r--", lw=.8, label=f"x{FLUX_RESP_GAUSS:.3f}")
    ax[0].set(xlabel="XID+ $F_{350}$ [mJy]", ylabel="map peak [mJy/beam]"); ax[0].legend(fontsize=8)
    ax[1].plot(Fv, tab["pk_sf"] / tab["pk_img"], "o", ms=3, alpha=.5)
    ax[1].axhline(1, color="k", lw=.8); ax[1].axhline(FLUX_RESP_GAUSS, color="r", ls="--", lw=.8)
    g = S["SF_IMAGE / IMAGE  (gain G)"]
    ax[1].axhline(g["median"], color="C2", lw=1.2, label=f"median G = {g['median']:.3f}$\\pm${g['median_err']:.3f}")
    ax[1].set(xlabel="XID+ $F_{350}$ [mJy]", ylabel="gain $G$ = filtered / raw peak", ylim=(0.7, 1.4)); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "herschel_h4_flux_response.png"), dpi=130)
    print("  wrote results/herschel_h4_flux_response.png")
except Exception as exc:
    print("  (figure skipped: %s)" % exc)
