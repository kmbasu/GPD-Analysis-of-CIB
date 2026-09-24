"""verify_memo_numbers.py -- single reproducible source for every number
quoted in GPD_beam_and_declustering_v1.md that is not already produced by
fig1_compute / fig2_peaks / fig3_noise.

Run:  python verify_memo_numbers.py
"""
import sys, os
import numpy as np
from scipy.stats import norm
from scipy.ndimage import maximum_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis_modules"))
sys.path.insert(0, _HERE)
from counts import Schechter, DoublePowerLaw
from simulate_maps import CIBMapSimulator, gaussian_beam_convolve
from gpd_tail import confusion_sigma, beam_declustered_peaks, robust_core, fit_gpd
from check_peak_shift import peak_stats, n_peaks_per_area, _check_G2_closed_form

BEAM, PIX, NPIX = 20.0, 4.0, 1024
S_MIN, S_CUT = 1e-3, 100.0
bfp = BEAM / PIX
N_MAPS = 25
SEED0 = 31337

sch = Schechter(s_min=S_MIN, s_max=S_CUT)
dpl = DoublePowerLaw.illustrative(s_min=S_MIN, s_max=S_CUT)

print("=" * 84)
print("[1] closed-form G2 vs quadrature")
a, b = _check_G2_closed_form()
print(f"    quad={a:.12f}  closed={b:.12f}  diff={a-b:.2e}   -> memo Sec 3.3")

print("\n[2] 2D-GRF peak statistics for gamma = 1/sqrt(2)")
st = peak_stats(1 / np.sqrt(2))
print(f"    mode  = {st['mode']:.4f} sigma   (memo 1.2722)")
print(f"    median= {st['median']:.4f} sigma   (memo 1.2928)  "
      f"q* = {norm.cdf(st['median']):.4f} (memo 0.9020)")
print(f"    mean  = {st['mean']:.4f} sigma   (memo 1.3029)")
sb = 1.0 / (2 * np.sqrt(2 * np.log(2)))            # sigma_b per unit FWHM
nmax_per_fwhm2 = 1.0 / (4 * np.sqrt(3) * np.pi * sb ** 2)
om_beam_per_fwhm2 = 2 * np.pi * sb ** 2
print(f"    n_max = 1 per {1/nmax_per_fwhm2:.3f} FWHM^2 (memo 3.925) "
      f"= 1 per {1/nmax_per_fwhm2/om_beam_per_fwhm2:.3f} Omega_beam (memo 3.47)")
print(f"    mean peak separation = {np.sqrt(1/nmax_per_fwhm2):.3f} FWHM "
      f"(memo ~2 FWHM = 10 pix at FWHM=5 pix)")

print("\n[3] Noise budget and the usable-window rule R = S_cut/(50 Sigma_tot)")
print(f"    {'beam':>6s} {'sigma_c(Sch)':>13s} {'N_b(Sch)':>9s} "
      f"{'sigma_c(DPL)':>13s} {'N_b(DPL)':>9s} {'R(DPL)':>8s}")
for bm in [14.9, 20.0, 30.0, 40.0]:
    cS = confusion_sigma(sch, bm, s_cut=S_CUT)
    cD = confusion_sigma(dpl, bm, s_cut=S_CUT)
    print(f"    {bm:6.1f} {cS['sigma_c']:13.4f} {cS['n_beam']:9.1f} "
          f"{cD['sigma_c']:13.4f} {cD['n_beam']:9.1f} "
          f"{S_CUT/(50*cD['sigma_c']):8.2f}")
print(f"    Schechter S_* = {sch.sstar:.3f} mJy (memo 3.98)")
print("    DPL xi_asym: " + ", ".join(
    f"S={S:g}: {dpl.xi_asymptotic(S):.4f}" for S in [8., 15., 30.]) +
    "  (memo 0.416 / 0.372 / 0.360)")

print("\n[4] Real confusion field, sigma_N = 0: quantile vs sigma normalisation")
print(f"    {'model':>10s} {'N_beam':>8s} {'sigma_c':>8s} {'sig_rob':>8s} "
      f"{'q*':>7s} {'d/sig_c':>8s} {'d/sig_rob':>10s}")
store = {}
for nm, cnts in [("Schechter", sch), ("DPL", dpl)]:
    cs = confusion_sigma(cnts, BEAM, s_cut=S_CUT)
    sim = CIBMapSimulator(cnts, BEAM, PIX, npix=NPIX, sigma_noise=0.0,
                          s_cut=S_CUT)
    pix, pk = [], []
    for i in range(N_MAPS):
        m = sim.make_map(seed=SEED0 + i)
        pix.append(m.ravel()[::5]); pk.append(beam_declustered_peaks(m, bfp))
    pix = np.concatenate(pix); pk = np.concatenate(pk)
    spix = np.sort(pix)
    mu_r, sg_r = robust_core(pix)
    d = np.median(pk) - np.median(pix)
    q = np.median(np.searchsorted(spix, pk) / spix.size)
    store[nm] = (pix, pk)
    print(f"    {nm:>10s} {cs['n_beam']:8.1f} {cs['sigma_c']:8.4f} "
          f"{sg_r:8.4f} {q:7.4f} {d/cs['sigma_c']:8.2f} {d/sg_r:10.2f}")

print("\n[5] Gaussian reference at the SAME geometry (size-5 filter)")
rng = np.random.default_rng(SEED0)
g = gaussian_beam_convolve(rng.normal(size=(2048, 2048)), bfp)
pkg = g[g == maximum_filter(g, size=5)]
zg = (np.median(pkg) - np.median(g)) / g.std()
print(f"    z_med = {zg:.4f} sigma  ->  q* = {norm.cdf(zg):.4f}   "
      f"(memo: 1.361 sigma, q* = 0.9132)")
print(f"    non-Gaussian offset in q*: "
      f"{np.median([0]) if False else ''}"
      f"{0.8825 - norm.cdf(zg):+.4f} (approx; memo -0.031)")

print("\n[6] Tail invariance: xi(pixels) vs xi(declustered peaks)")
for nm, cnts in [("Schechter", sch), ("DPL", dpl)]:
    cs = confusion_sigma(cnts, BEAM, s_cut=S_CUT)
    for fN, mode in [(0.0, "beam"), (1.0, "white")]:
        sim = CIBMapSimulator(cnts, BEAM, PIX, npix=NPIX,
                              sigma_noise=fN * cs["sigma_c"], s_cut=S_CUT,
                              noise_mode=mode)
        pix, pk = [], []
        for i in range(N_MAPS):
            m = sim.make_map(seed=9000 + i)
            pix.append(m.ravel()[::3]); pk.append(beam_declustered_peaks(m, bfp))
        pix = np.concatenate(pix); pk = np.concatenate(pk)
        print(f"    {nm}, f_N={fN} ({mode}):")
        for u in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]:
            yp, yk = pix[pix > u] - u, pk[pk > u] - u
            if yk.size < 40:
                continue
            print(f"      u={u:4.1f}  xi_pix={fit_gpd(yp)['xi']:+.4f}  "
                  f"xi_pk={fit_gpd(yk)['xi']:+.4f}  "
                  f"frac pk/pix = {yk.size/yp.size:.3f}")

print("\n[7] noise_mode normalisation")
cs = confusion_sigma(sch, BEAM, s_cut=S_CUT)
sn = cs["sigma_c"]
for mode in ("white", "beam"):
    sim = CIBMapSimulator(sch, BEAM, PIX, npix=NPIX, sigma_noise=sn,
                          s_cut=S_CUT, noise_mode=mode)
    r = np.random.default_rng(5)
    if mode == "white":
        n = r.normal(0.0, sn, (NPIX, NPIX))
    else:
        n = gaussian_beam_convolve(
            r.normal(0.0, sn * sim._white_per_unit_beamed, (NPIX, NPIX)), bfp)
    mm = np.concatenate([sim.make_map(seed=600 + i).ravel()[::5]
                         for i in range(6)])
    print(f"    mode={mode:6s} noise-only rms={n.std():.5f} "
          f"(target {sn:.5f}, {n.std()/sn-1:+.3%})   "
          f"map rms={mm.std():.4f} (Sigma_tot={np.hypot(cs['sigma_c'],sn):.4f})")
print("\ndone.")
