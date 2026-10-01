"""
herschel_model_curves.py -- module H5: the Fig. 10 model curves, forward-modeled
through the SAME operations as the bright-masked data
=====================================================================

PURPOSE
-------
Sect. 5.2 of the paper compares the bright-masked GAMA-09 xi_hat(u) with the three
350 um count models.  Until arXiv v1 the model curves were simulated with the
counts TRUNCATED at S_cut = 100 mJy and with sky and instrument noise both
smoothed by the same Gaussian beam.  Two audits (2026-09-25/27) showed that
neither matches what is done to the data:

  1. The data are MASKED in the map: pixels above 100 mJy/beam (after baseline
     removal) are removed with a 3-pixel (one FWHM) growth.  A map-level mask
     imposes a hard endpoint in MAP units, whereas truncating the injected
     counts leaves a soft edge smeared by confusion and noise, and sources of
     100-130 mJy whose peaks fall below 100 mJy/beam survive the mask.  The
     masked-simulation curve sits below the truncated-counts curve by 0.01 at
     25 mJy/beam growing to 0.09 at 50, and reproduces the steep fall of the
     masked DATA above 55 mJy/beam that the truncated curve cannot.
  2. The filtered map's sky response is PSF * K and its noise is white * K, with
     K the HELP matched-filter kernel (negative sidelobes): both are LESS
     correlated than a Gaussian-beam field (lag-1 autocorrelation 0.72 vs 0.87),
     which raises the declustered-peak density from 0.255 to 0.36-0.37 per beam
     (data: 0.374) and shifts xi_hat by +0.01 to +0.04 in the window.

This module therefore simulates, for each count model, the intrinsic counts
below S_cut PLUS the observed bright population above it (the power-law tail
calibrated in H1c to the observed bright peak counts, identical for every
model), passes the map through the matched-filter forward model
(CIBMapSimulator(post_filter=K)), and applies the data's baseline removal,
mask, declustering, edge exclusion and core-centroid anchoring byte-for-byte
(the H1v2/H1c conventions).  It writes the curves on the fine 2.5 mJy/beam
ladder, the covariance-correct chi^2 against the masked data with both Hartlap
conventions (400 replicates / 103 cutouts), the u/1.04 flux-scale check of
App. D, the curves' own map-bootstrap errors, and the diagnostic quantities
(peak density, lag-1 autocorrelation, pixel rms, masked area fraction).  For
the record it also writes the three legacy conventions (beam noise and/or
truncated counts), which App. D quotes.

USAGE
-----
    python herschel_model_curves.py            # ~3 min for 400 maps x 3 models

Needs only the shipped files below, no survey data.  The per-(model, variant)
peak lists are cached under results/h5_cache/ (not shipped).

ENVIRONMENT
-----------
CIB_N_SIM_MAPS   maps per model (default 400; each 225 px = 0.25 deg^2)
CIB_SIM_SEED     base seed (default 20260927)
CIB_SIM_FILTER   "1" (default) matched-filter forward model; "0" beam noise

INPUTS   results/herschel_unlensed_v3_350.npz   (measured curves, bootstrap)
         results/herschel_peak_sims_350.npz     (bright-tail calibration)
         results/matchedfilter_kernel_350.npy   (HDU 8 of the HELP map)
OUTPUT   results/herschel_model_curves_350.npz  (read by make_paper_figures fig10)

Author: CIB-lensing / GPD project, Herschel task, September 2026.
"""
#%% ------------------------------------------------------------------ setup --
import os, sys, time, json
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_modules"))
sys.path.insert(0, os.path.join(ROOT, "Simulations"))
os.environ.setdefault("CIB_FIGS", "none")
from counts import BaseCounts                                   # noqa: E402
from simulate_maps import CIBMapSimulator                       # noqa: E402
from gpd_tail import fit_gpd                                    # noqa: E402
from counts_350um import make_models, MODEL_ORDER               # noqa: E402

RES_DIR = os.path.join(HERE, "results")
N_SIM = int(os.environ.get("CIB_N_SIM_MAPS", "400"))
SEED = int(os.environ.get("CIB_SIM_SEED", "20260927"))
SIM_FILTER = os.environ.get("CIB_SIM_FILTER", "1") == "1"
S_CUT, S_MIN, S_MASK, MASK_GROW = 100.0, 1.0, 100.0, 3
U_FINE = np.arange(25.0, 76.0, 2.5)

v3 = np.load(os.path.join(RES_DIR, "herschel_unlensed_v3_350.npz"), allow_pickle=True)
h1c = np.load(os.path.join(RES_DIR, "herschel_peak_sims_350.npz"), allow_pickle=True)
FWHM, PIX, SIDE = float(v3["fwhm"]), float(v3["pix"]), int(v3["side_pix"])
SIGMA_INST, MU_CORE = float(v3["sigma_inst"]), float(v3["mu_core"])
EDGE = int(np.ceil(FWHM / PIX)); DECL = int(round(FWHM / PIX)) | 1
BEAM_PIX = 1.133 * (FWHM / PIX) ** 2
BRIGHT_A, BRIGHT_NB = float(h1c["bright_a"]), float(h1c["bright_nb"])
KERNEL = np.load(os.path.join(RES_DIR, "matchedfilter_kernel_350.npy"))
T0 = time.time()
print("H5  model curves through the data's operations: %d maps x %d px per model, "
      "filter=%s" % (N_SIM, SIDE, SIM_FILTER))
print("    bright tail above %.0f mJy: N(>S) = %.3f (S/100)^-%.3f deg^-2 (H1c calibration)"
      % (S_CUT, BRIGHT_NB, BRIGHT_A))

#%% ------------------------------------------------- counts and simulators --
class PlusObservedBrightTail(BaseCounts):
    """Intrinsic model below S_b, observed power-law population above it."""
    def __init__(self, base, s_b, n_b, a):
        self.base, self.s_b, self.n_b, self.a = base, float(s_b), float(n_b), float(a)
        self.s_min, self.s_max = base.s_min, 1e4
    def dnds(self, S):
        S = np.asarray(S, float)
        faint = np.where(S <= self.s_b, self.base.dnds(S), 0.0)
        bright = np.where(S > self.s_b, self.a * self.n_b
                          * (S / self.s_b) ** (-(self.a + 1.0)) / self.s_b, 0.0)
        return faint + bright

def simulator(cnts, s_cut, filt):
    if filt:
        s0 = CIBMapSimulator(cnts, FWHM, PIX, npix=SIDE, sigma_noise=0.0,
                             s_cut=s_cut, noise_mode="white", post_filter=KERNEL)
        return CIBMapSimulator(cnts, FWHM, PIX, npix=SIDE,
                               sigma_noise=s0.white_sigma_for(SIGMA_INST),
                               s_cut=s_cut, noise_mode="white", post_filter=KERNEL)
    return CIBMapSimulator(cnts, FWHM, PIX, npix=SIDE, sigma_noise=SIGMA_INST,
                           s_cut=s_cut, noise_mode="beam")

#%% ------------------------------------------ the data's operations, exactly --
def poly_baseline(cut, order=1):
    """First-order polynomial baseline, as herschel_unlensed_v2/v3."""
    ny, nx = cut.shape
    y, x = np.mgrid[:ny, :nx]
    xs = (x - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
    ys = (y - (ny - 1) / 2.0) / ((ny - 1) / 2.0)
    terms = [(xs ** i) * (ys ** (t - i)) for t in range(order + 1) for i in range(t + 1)]
    A = np.stack([t.ravel() for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, cut.ravel(), rcond=None)
    return cut - sum(c * t for c, t in zip(coef, terms))

INTERIOR = np.zeros((SIDE, SIDE), bool)
INTERIOR[EDGE:SIDE - EDGE, EDGE:SIDE - EDGE] = True

def peaks_of(cut, mask):
    """Baseline, optional bright mask (grown by MASK_GROW px), declustering
    over DECL px, edge exclusion -- herschel_unlensed_v3 cell 1d/mask branch."""
    res = poly_baseline(cut, 1)
    valid = np.ones_like(res, bool)
    if mask:
        hot = res > S_MASK
        if MASK_GROW > 0 and hot.any():
            hot = maximum_filter(hot, size=2 * MASK_GROW + 1, mode="constant", cval=False)
        valid &= ~hot
    filled = np.where(valid, res, -np.inf)
    pk = valid & INTERIOR & (filled == maximum_filter(filled, size=DECL,
                                                      mode="constant", cval=-np.inf))
    stats = dict(rms=float(res.std()), acf1=float(np.mean((res[:, :-1] - res.mean())
                 * (res[:, 1:] - res.mean())) / res.var()), masked=float((~valid).mean()),
                 npk=int(pk.sum()), npix=int(INTERIOR.sum()))
    return res[pk].astype(np.float32), stats

def xi_curve(units, u_grid, min_exc=50):
    pooled = np.concatenate(units)
    pooled = pooled + (MU_CORE - np.median(pooled))        # core-centroid anchoring
    xi = np.full(len(u_grid), np.nan); nex = np.zeros(len(u_grid), int)
    for i, u in enumerate(u_grid):
        y = pooled[pooled > u] - u; nex[i] = y.size
        if y.size >= min_exc:
            xi[i] = fit_gpd(y)["xi"]
    return xi, nex

#%% ------------------------------------------------------- run the variants --
#  fiducial = "FM": matched-filter forward model, full counts, masked like the
#  data.  Legacy conventions kept for App. D: "BT" beam noise + truncated counts
#  (arXiv v1), "BM" beam noise + mask, "FT" filter + truncated counts.
VARIANTS = {"FM": (True, True), "FT": (True, False), "BM": (False, True), "BT": (False, False)}
if not SIM_FILTER:
    VARIANTS = {k: v for k, v in VARIANTS.items() if not v[0]}
models_trunc = make_models(s_min=S_MIN, s_max=S_CUT)
OUT = {}
for nm in MODEL_ORDER:
    full = PlusObservedBrightTail(models_trunc[nm], S_CUT, BRIGHT_NB, BRIGHT_A)
    sims = {}
    for key, (filt, masked) in VARIANTS.items():
        cnts = full if masked else models_trunc[nm]
        sims[key] = simulator(cnts, 1e4 if masked else S_CUT, filt)
    peaks = {k: [] for k in VARIANTS}; stats = {k: [] for k in VARIANTS}
    #  Per-(model, variant) cache of the peak lists, so a long run can be split
    #  across sessions (CIB_H5_ONLY="Schechter:FM,DPL:FM" restricts a call).
    only = os.environ.get("CIB_H5_ONLY", "")
    only = [tuple(x.split(":")) for x in only.split(",") if x] if only else None
    cache_dir = os.path.join(RES_DIR, "h5_cache"); os.makedirs(cache_dir, exist_ok=True)
    for key, (filt, masked) in VARIANTS.items():
        cf = os.path.join(cache_dir, "%s_%s_n%d_s%d.npz" % (nm, key, N_SIM, SEED))
        if os.path.exists(cf):
            z = np.load(cf, allow_pickle=True)
            peaks[key] = [z["pk_%d" % j] for j in range(N_SIM)]
            stats[key] = list(json.loads(str(z["stats"])))
            continue
        if only is not None and (nm, key) not in only:
            continue
        for j in range(N_SIM):
            m = sims[key].make_map(seed=SEED + j)
            pk, st = peaks_of(m, masked)
            peaks[key].append(pk); stats[key].append(st)
        np.savez(cf, stats=json.dumps(stats[key]), **{"pk_%d" % j: a for j, a in enumerate(peaks[key])})
    for key in VARIANTS:
        if not peaks[key]:
            continue
        xi, nex = xi_curve(peaks[key], U_FINE)
        rng = np.random.default_rng(SEED + 7)
        #  map-bootstrap error of the curve: fiducial variant only (the legacy
        #  conventions are quoted without errors)
        if key == ("FM" if SIM_FILTER else "BT"):
            reps = np.array([xi_curve([peaks[key][i] for i in rng.integers(0, N_SIM, N_SIM)],
                                      U_FINE)[0] for _ in range(100)])
        else:
            reps = np.full((2, len(U_FINE)), np.nan)
        st = {k: float(np.mean([s[k] for s in stats[key]])) for k in ("rms", "acf1", "masked")}
        st["pk_per_beam"] = float(sum(s["npk"] for s in stats[key])
                                  / sum(s["npix"] for s in stats[key]) * BEAM_PIX)
        OUT[(nm, key)] = dict(xi=xi, nex=nex, sig=np.nanstd(reps, axis=0), **st)
        print("  %-9s %s  xi(25/50/75) = %+.4f %+.4f %+.4f  pk/beam %.3f  acf1 %.3f  rms %.2f"
              "  masked %.2f%%  [%.0f s]" % (nm, key, xi[0], xi[10], xi[20], st["pk_per_beam"],
              st["acf1"], st["rms"], 100 * st["masked"], time.time() - T0))

#%% ------------------------------------------- chi^2 against the masked data --
u = v3["u_grid"]; xim = v3["xi_masked"]; boot = v3["boot_masked"]
umax = float(v3["u_max_masked"])
sel = (u <= umax) & np.isfinite(xim); p = int(sel.sum())
Bm = boot[:, sel]; good = np.isfinite(Bm).all(axis=1)
C = np.cov(Bm[good], rowvar=False); n_boot = int(good.sum()); n_cut = int(v3["n_cutouts"])
hart_boot = (n_boot - p - 2) / (n_boot - 1); hart_cut = (n_cut - p - 2) / (n_cut - 1)
Cinv = np.linalg.inv(C) * hart_boot
def chi2(curve, scale=1.0):
    r = xim[sel] - np.interp(u[sel] / scale, U_FINE, curve)
    return float(r @ Cinv @ r)
print("\n  covariance-correct chi^2 vs the bright-masked data, %d thresholds, Hartlap %.4f "
      "(400 replicates) / %.4f (103 cutouts)" % (p, hart_boot, hart_cut))
print("  %-5s %10s %10s %10s   | at u/1.04" % ("var", *MODEL_ORDER))
for key in VARIANTS:
    if any((nm, key) not in OUT for nm in MODEL_ORDER):
        print("  %-5s (incomplete -- run the remaining CIB_H5_ONLY pieces)" % key); continue
    c = [chi2(OUT[(nm, key)]["xi"]) for nm in MODEL_ORDER]
    c4 = [chi2(OUT[(nm, key)]["xi"], 1.04) for nm in MODEL_ORDER]
    print("  %-5s %10.1f %10.1f %10.1f   | %6.1f %6.1f %6.1f" % (key, *c, *c4))
    for nm, cc, cc4 in zip(MODEL_ORDER, c, c4):
        OUT[(nm, key)].update(chi2=cc, chi2_cut=cc * hart_cut / hart_boot, chi2_u104=cc4)

#%% ------------------------------------------------------------------ save --
save = dict(u_fine=U_FINE, hartlap_boot=hart_boot, hartlap_cut=hart_cut, p=p,
            n_sim=N_SIM, seed=SEED, sim_filter=SIM_FILTER, fiducial="FM" if SIM_FILTER else "BT",
            bright_a=BRIGHT_A, bright_nb=BRIGHT_NB, s_cut=S_CUT, s_mask=S_MASK, mask_grow=MASK_GROW,
            variants=json.dumps({k: {"filter": v[0], "masked": v[1]} for k, v in VARIANTS.items()}))
for (nm, key), d in OUT.items():
    for k, v in d.items():
        save["%s_%s_%s" % (key, nm, k)] = v
out = os.path.join(RES_DIR, "herschel_model_curves_350.npz")
np.savez(out, **save)
print("\nwrote", out, "(%.0f s)" % (time.time() - T0))
