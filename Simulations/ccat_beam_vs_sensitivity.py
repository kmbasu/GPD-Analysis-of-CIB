"""
ccat_beam_vs_sensitivity.py -- the beam x sensitivity factorial of Sect. 6.3
(`sec:ccat:factorial`): which of the two CCAT/FYST gains over Herschel/SPIRE --
the smaller beam or the lower instrument noise -- buys the Delta_xi(u)
detection.  Four corners (Herschel or CCAT beam x Herschel or CCAT noise) on the
common benchmark lens, aperture r < 1.0 theta_500.

Run from anywhere:  python Simulations/ccat_beam_vs_sensitivity.py
Output: Simulations/results/q2_factorial_v2.{npz,log} (run tag kept from the
development name of this script, `q2_factorial_v2.py`).

Development notes follow.
Q2 (v2): beam vs sensitivity, with ladders matched in units of each corner's own
Sigma_tot.

WHY v2.  A single COMMON ABSOLUTE ladder (v1) is not a controlled comparison:
Sigma_tot differs by 1.7x across the corners, so the same u sits at k = 1.0 for
one corner and k = 2.6 for another, and the forecast optimum then wanders into
the Gaussian core where Delta_xi is a core effect and not a counts effect --
exactly the trap `Revised_Planck_analysis.md` Sec. 7.5 warns about.  Matching in
k = u/Sigma_tot puts every corner at the same place in its own P(D), which is the
statistically meaningful control, and lets the ABSOLUTE flux fall where the
instrument puts it.  That is the point: a smaller beam has a lower confusion
floor (sigma_c is exactly linear in FWHM), so at the same k it reads dN/dS at a
lower flux.  That resolution gain is real and the k-matched ladder captures it.

Restricted to k >= 2.5 so the answer is not carried by floor-dominated bins.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_core as sc
from sim_core import (Instrument, Benchmark, Run, S_MIN, make_models, budget,
                      stage_lensed_mc, stage_lensed_theory,
                      clusters_for_detection, peak_budget)

sc.N_BOOT_DEFAULT = 250
N_CL, APER = 200, 1.0
K = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
K_FLOOR = 2.5

BENCH = Benchmark()
run = Run("q2_factorial_v2", config_key=("q2v2", N_CL, tuple(K), APER))
run.rule("Q2 v2 — beam x sensitivity, k-matched ladders, r < 1.0 theta_500")
run.say(BENCH.summary())
run.say(f"  aperture r = {APER*BENCH.theta500:.3f}', N = {N_CL} clusters, "
        f"k = u/Sigma_tot = " + ", ".join(f"{x:.1f}" for x in K)
        + f";  optimum taken over k >= {K_FLOOR}")

CORNERS = [("HH  Herschel beam / Herschel noise", 25.15, 8.0, 5.402),
           ("HC  Herschel beam / CCAT noise",     25.15, 8.0, 3.100),
           ("CH  CCAT beam     / Herschel noise", 15.00, 3.0, 5.402),
           ("CC  CCAT beam     / CCAT noise",     15.00, 3.0, 3.100)]

res = {}
for lab, beam, pix, sig_n in CORNERS:
    probe = Instrument("X", beam, pix, sig_n, 100.0, [1.0])
    bud0 = budget(make_models(s_min=S_MIN, s_max=100.0)["Schechter"], probe)
    U = K * bud0["sigma_tot"]
    inst = Instrument("X", beam, pix, sig_n, 100.0, U,
                      apertures_theta500=(APER,), n_clusters=N_CL,
                      npix_unlensed=256, n_maps_unlensed=4, note=lab)
    mods = make_models(s_min=S_MIN, s_max=inst.s_cut)
    bud = budget(mods["Schechter"], inst)
    th = stage_lensed_theory(run, inst, mods, {"Schechter": bud}, BENCH)
    mc = stage_lensed_mc(run, inst, mods["Schechter"], bud, BENCH, n_clusters=N_CL)
    dth = th[(APER, "Schechter")]["delta"]
    sig = mc[APER]["delta"]["sigma"]
    mask = K >= K_FLOOR
    fc = clusters_for_detection(np.where(mask, dth, np.nan), sig, N_CL)
    _, pk = peak_budget(inst, APER * BENCH.theta500)
    res[lab] = dict(bud=bud, U=U, dth=dth, sig=sig, fc=fc, peaks=pk)

run.rule("SUMMARY", ch="=")
run.say(f"{'corner':>36s} {'sig_c':>7s} {'sig_N':>7s} {'Sig_tot':>8s} "
        f"{'pk/cl':>7s} {'u@opt':>7s} {'Dxi@opt':>9s} {'sig_1':>8s} {'N(3sig)':>10s}")
for lab, *_ in CORNERS:
    r = res[lab]; j = r["fc"]["j_best"]
    run.say(f"{lab:>36s} {r['bud']['sigma_c']:7.2f} {r['bud']['sigma_n']:7.2f} "
            f"{r['bud']['sigma_tot']:8.2f} {r['peaks']:7.1f} {r['U'][j]:7.1f} "
            f"{r['dth'][j]:+9.4f} {r['fc']['sigma_1'][j]:8.4f} "
            f"{r['fc']['n_needed'][j]:10.3e}")

n = {c[0]: res[c[0]]["fc"]["n_needed"][res[c[0]]["fc"]["j_best"]] for c in CORNERS}
HH, HC, CH, CC = (n[c[0]] for c in CORNERS)
run.say("")
run.say("  Factorial decomposition, N(3sigma) ratios:")
run.say(f"    TOTAL Herschel -> CCAT      N(HH)/N(CC) = {HH/CC:7.2f}")
run.say(f"    beam,  at Herschel noise    N(HH)/N(CH) = {HH/CH:7.2f}")
run.say(f"    beam,  at CCAT noise        N(HC)/N(CC) = {HC/CC:7.2f}")
run.say(f"    noise, at Herschel beam     N(HH)/N(HC) = {HH/HC:7.2f}")
run.say(f"    noise, at CCAT beam         N(CH)/N(CC) = {CH/CC:7.2f}")
run.say("")
run.say(f"  Reference scalings: beam solid angle (25.15/15)^2 = "
        f"{(25.15/15.)**2:.2f} (peaks per cluster);")
run.say(f"  sigma_c ratio 25.15/15 = {25.15/15.:.3f} (sigma_c is exactly linear "
        f"in FWHM at fixed counts).")
run.save(k=K, **{f"n_{i}": res[c[0]]["fc"]["n_needed"] for i, c in enumerate(CORNERS)})
