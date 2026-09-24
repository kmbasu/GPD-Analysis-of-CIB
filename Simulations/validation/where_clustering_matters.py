"""
Where does source clustering start to matter?   Companion to
verify_clustering_mode.py, for Secs. 10.5-10.6 of
`Simulations/full_simulation_summary_v2.md`.

Same configuration as verify_clustering_mode.py: the suite's Herschel row
(25" beam, 5" pixels, 1024^2, S_cut = 100 mJy, sigma_N = 0) with the FITTED
350 um Schechter, S_min = 0.1 mJy, sigma_c = 8.06 mJy/beam.

Three measurements:

  (A) Delta xi(u) = xi_clustered - xi_Poisson on a fine threshold ladder,
      with enough maps to resolve it at the top of the ladder.  The
      question this answers is not "is it zero" but "does it DECAY with u,
      as the core-tail argument predicts, or is it a constant offset".

  (B) The Fano factor F = Var(N)/E[N] of declustered exceedance counts in
      cutout-sized cells, vs threshold.  F = 1 is Poisson; F > 1 means a
      cutout-level bootstrap sees more scatter than Poisson simulations
      would predict, i.e. simulation-derived error bars are optimistic.

  (C) The same, with the clustering amplitude corrected for the q1
      truncation (see below), which is the physically calibrated version
      for COUNT statistics.

WHY (B) AND (C) DIFFER: THE q1 PROBLEM.  simulate_maps sets
C_l^dd = (q2/q1^2)(l/l_eq)^s from the INJECTED population, which makes the
map's FLUX clustering power equal its own shot power at l_eq -- the right
target for MAP statistics, and self-consistent for any counts model.  But
delta itself, and hence any COUNT statistic, depends on q1 separately, and
q1 is the moment this project's counts model gets badly wrong.  Against a
MEASURED CIB monopole of 0.576 +/- 0.034 MJy/sr at 857 GHz (Odegard et al.
2019), the fitted 350 um Schechter gives 1.54 MJy/sr from the injected
population alone (S_min = 0.1 mJy, a 2.67x over-prediction) and 2.42
MJy/sr integrated to S_min = 1 uJy (4.2x).  Too many faint sources are
carrying the background, so the same clustered flux is spread over too
many objects and delta comes out too SMALL.

Passing the measured monopole as cl_q1_ref (1.755e5 deg^-2 mJy) fixes it,
raising sigma_delta from 0.284 to 0.758 -- a factor 2.67 in rms, 7.1 in
variance.  So:

    (B) cl_q1_ref = None  ->  map statistics right, counts UNDER-dispersed
    (C) cl_q1_ref = meas. ->  counts right, flux power correspondingly high

(C) is the number to quote for error-bar inflation; (A), (B) and the map
statistics of verify_clustering_mode.py need no correction.  The two
cannot both be satisfied while the counts model mis-predicts q1, and that
mis-prediction is itself a result -- see memo Sec. 10.4.

Two features of the output are worth not misreading:
  * F < 1 for the Poisson sky.  Declustering imposes a minimum separation
    between retained peaks, so the retained process is hard-core and
    genuinely sub-Poissonian.  The comparison that matters is
    clustered-vs-Poisson at the same u, never F against 1.
  * The thresholds are NESTED and therefore strongly correlated (Effect C
    of GPD_beam_and_declustering_v4.md Sec. 6).  The rows of (A) are not
    independent tests; treat the ladder as ~2 independent numbers.

Run:  python3 where_clustering_matters.py       (~90 s)
"""

# %% imports
import numpy as np
from scipy.ndimage import maximum_filter

import verify_clustering_mode as V
from gpd_tail import beam_declustered_peaks, fit_gpd

NM = 96                       # maps per arm; 4x verify's, to resolve high u
BLK = 256                     # 21.3' cells
U_LADDER = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]


#  Measured CIB monopole at 857 GHz: 0.576 +/- 0.034 MJy/sr
#  (Odegard et al. 2019, ApJ 877, 40), converted to deg^-2 mJy.
Q1_MEASURED = 0.576e6 * (np.pi / 180.0) ** 2 * 1e3     # Jy/sr -> mJy/deg^2


def harvest(sim, nmaps, block, u_list):
    """Peaks (pooled) and per-cell exceedance counts at each threshold."""
    nb = V.NPIX // block
    peaks, cells, dvar = [], {u: [] for u in u_list}, []
    size = max(1, int(round(sim.beam_fwhm_pix)))
    for i in range(nmaps):
        m = sim.make_map(seed=500 + i)
        peaks.append(beam_declustered_peaks(m, sim.beam_fwhm_pix))
        loc = (m == maximum_filter(m, size=size))
        for u in u_list:
            cells[u].append((loc & (m > u)).reshape(
                nb, block, nb, block).sum(axis=(1, 3)))
        if sim.clustering is not None:
            d = sim._modulation(np.random.default_rng(500 + i))
            dvar.append(d.reshape(nb, block, nb, block).mean(axis=(1, 3)).var())
    return (np.concatenate(peaks),
            {u: np.concatenate([c.ravel() for c in v]).astype(float)
             for u, v in cells.items()},
            float(np.mean(dvar)) if dvar else 0.0)


def main():
    sp = V.make_sim(None)
    sc = V.make_sim("lognormal")
    sx = V.make_sim("lognormal", cl_q1_ref=Q1_MEASURED)
    ic, ix = sc.clustering_info(), sx.clustering_info()
    print(f"q1 injected = {ic['q1']:.4e} deg^-2 mJy "
          f"({ic['q1'] / ((np.pi/180)**2 * 1e3) / 1e6:.2f} MJy/sr)")
    print(f"q1 measured = {Q1_MEASURED:.4e} deg^-2 mJy (0.576 MJy/sr, "
          f"Odegard+2019)  -> model over-predicts by "
          f"{ic['q1']/Q1_MEASURED:.2f}x")
    print(f"sigma_delta: default -> {ic['sigma_delta']:.3f}   "
          f"with measured q1 -> {ix['sigma_delta']:.3f}   "
          f"(x{ix['sigma_delta']/ic['sigma_delta']:.2f} in rms)")

    P = harvest(sp, NM, BLK, U_LADDER)
    C = harvest(sc, NM, BLK, U_LADDER)
    X = harvest(sx, NM, BLK, U_LADDER)

    print(f"\n(A)  Delta xi(u) = clustered - Poisson, {NM} maps per arm, "
          f"cl_ratio = 1")
    print(f"{'u [mJy]':>8} {'u/sig_c':>8} {'n_pk':>8} {'xi_P':>9} {'xi_C':>9}"
          f" {'Delta xi':>10} {'se':>7} {'sig':>6}")
    for u in U_LADDER:
        a = fit_gpd(P[0][P[0] > u] - u)
        b = fit_gpd(C[0][C[0] > u] - u)
        d, se = b["xi"] - a["xi"], np.hypot(a["se_xi"], b["se_xi"])
        print(f"{u:>8.1f} {u/V.SIGMA_C:>8.1f} {a['n']:>8d} {a['xi']:>+9.4f}"
              f" {b['xi']:>+9.4f} {d:>+10.4f} {se:>7.4f} {abs(d)/se:>6.1f}")

    nb = V.NPIX // BLK
    print(f"\n(B)/(C)  Fano factor of exceedance counts, {BLK*V.PIX/60:.1f}' "
          f"cells ({NM*nb*nb} cells)")
    print(f"{'u [mJy]':>8} {'u/sig_c':>8} {'Nbar':>8} {'F Pois':>8}"
          f" {'F default':>10} {'F phys':>8} {'excess':>9} {'err-bar x':>10}")
    for u in U_LADDER:
        cp, cc, cx = P[1][u], C[1][u], X[1][u]
        FP = cp.var(ddof=1)/cp.mean()
        FC = cc.var(ddof=1)/cc.mean()
        FX = cx.var(ddof=1)/cx.mean()
        print(f"{u:>8.1f} {u/V.SIGMA_C:>8.1f} {cp.mean():>8.1f} {FP:>8.3f}"
              f" {FC:>9.3f} {FX:>8.3f} {FX-FP:>+9.3f}"
              f" {np.sqrt(max(FX/FP, 0)):>10.2f}")
    print(f"\nVar(cell-mean delta) at {BLK*V.PIX/60:.1f}': "
          f"default {C[2]:.2e}   physical {X[2]:.2e}"
          f"  -> sigma_delta,cell = {np.sqrt(X[2]):.4f}")


if __name__ == "__main__":
    main()
