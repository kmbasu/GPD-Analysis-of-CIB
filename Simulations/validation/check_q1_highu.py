"""
check_q1_highu.py -- what drives the HIGH-u decline of xi_pop(u)?

Candidates: (a) the DPL's own bright-end slope steepening toward beta = 3.8,
(b) the S_cut = 100 mJy bright-source mask, which right-truncates the tail.

Also documents a numerical trap: xi_of_threshold_analytic integrates the
tabulated P(D) all the way to d_span/2.  Because _pofd_from_rate clips the
FFT output with np.maximum(p, 0), the far grid (where P(D) is numerically
zero) carries a small positive floor that the KL projection reads as a very
heavy tail.  Keep d_span only modestly larger than the true support.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis_modules")))
from counts import DoublePowerLaw
from pofd_analytic import PofD
from gpd_tail import xi_of_threshold_analytic, confusion_sigma

S_MIN = 1e-3
BEAM = 20.0
U = np.array([2., 4., 6., 8., 10., 12., 15., 20., 25., 30.])

print("Counts-only expectation: xi_asym(S) = 1/(eta(S)-1) for the DPL")
c = DoublePowerLaw.illustrative(s_min=S_MIN, s_max=1e4)
for S in [2., 4., 8., 15., 30., 100.]:
    print(f"   S = {S:6.1f} mJy   eta = {c.eta(S):.3f}   "
          f"xi_asym = {c.xi_asymptotic(S):.4f}")

print()
print("=" * 78)
print("xi_pop(u), 20\" beam, sigma_N = 0, varying the bright mask S_cut.")
print("Grid held FIXED (n_fft = 2^19, d_span = 900 -> grid to +-450 mJy)")
print(f"\n{'S_cut':>8s} {'sigma_c':>8s} " + "".join(f"{u:>8.0f}" for u in U))
for sc in [30.0, 60.0, 100.0, 200.0, 400.0]:
    cn = DoublePowerLaw.illustrative(s_min=S_MIN, s_max=sc)
    cs = confusion_sigma(cn, BEAM, s_cut=sc)
    pp = PofD(cn, BEAM, mu=1.0, sigma_noise=0.0, s_cut=sc,
              n_fft=2 ** 19, d_span=900.0)
    # restrict the KL integration to a sensible upper limit: 3 * S_cut
    keep = pp.d < 3.0 * sc
    xk, _ = xi_of_threshold_analytic(pp.d[keep], pp.p[keep], U)
    print(f"{sc:8.0f} {cs['sigma_c']:8.4f} " + "".join(f"{v:8.3f}" for v in xk))

print()
print("Same, but with NO upper-limit restriction on the KL integration")
print("(illustrating the numerical trap):")
print(f"\n{'S_cut':>8s} " + "".join(f"{u:>8.0f}" for u in U))
for sc in [100.0, 400.0]:
    cn = DoublePowerLaw.illustrative(s_min=S_MIN, s_max=sc)
    pp = PofD(cn, BEAM, mu=1.0, sigma_noise=0.0, s_cut=sc,
              n_fft=2 ** 19, d_span=900.0)
    xk, _ = xi_of_threshold_analytic(pp.d, pp.p, U)
    print(f"{sc:8.0f} " + "".join(f"{v:8.3f}" for v in xk))
