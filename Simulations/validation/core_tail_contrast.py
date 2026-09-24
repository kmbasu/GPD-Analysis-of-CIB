"""
core_tail_contrast.py -- Table F.1: the amplitude-free core-tail contrast
=========================================================================

Evaluates Eq. (F.1) of the paper,

    R_core / R_tail = 2 q_1^2 / ( q_2 N(>u) ),     q_k = int S^k n_0(S) dS,

on the adopted 350 um Schechter counts (s_min = 1 mJy, as everywhere in the
paper; not truncated at S_cut, i.e. integrated to the make_models default of
10^4 mJy) over the Herschel fitting window, together with
N(>u) [deg^-2] and the number of sources per beam above u,
N_beam(>u) = N(>u) * Omega_beam, for the 25.15" SPIRE 350 um beam
(Omega_beam = 1.133 FWHM^2).

The ratio is independent of the clustering amplitude, which is the point of
the table.  This script reproduces the tabulated N(>u) and N_beam(>u) exactly
and R_core/R_tail to <= 0.2% (123 / 540 / 1288 / 9155 against the printed
123 / 541 / 1290 / 9170, which were evaluated on a different quadrature grid).
Truncating the counts at S_cut = 100 mJy instead changes the ratios by
+1% to +21% from u = 25 to 75 mJy (pass s_max=100).  (With s_min = 1 microJy instead, q_1 is dominated by the faint-end
extrapolation and the ratios grow by a factor ~7; see App. F.)

Run:  python Simulations/validation/core_tail_contrast.py      (< 1 s)
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "analysis_modules")):
    sys.path.insert(0, os.path.normpath(_p))

from counts_350um import make_models                              # noqa: E402

S_MIN, S_MAX = 1.0, 1.0e4          # mJy (no bright-end truncation)
FWHM = 25.15                       # arcsec, SPIRE 350 um effective beam
OMEGA_DEG2 = 1.133 * FWHM ** 2 / 3600.0 ** 2
U = np.array([25.0, 40.0, 50.0, 75.0])


def table_f1(s_min=S_MIN, s_max=S_MAX, u=U):
    """Return N(>u) [deg^-2], N_beam(>u) and R_core/R_tail (Schechter)."""
    cnt = make_models(s_min=s_min, s_max=s_max)["Schechter"]
    q1, q2 = cnt.moment(1), cnt.moment(2)            # mJy deg^-2, mJy^2 deg^-2
    n_gt = np.array([cnt.moment(0, s_lo=x) for x in u])
    return n_gt, n_gt * OMEGA_DEG2, 2.0 * q1 ** 2 / (q2 * n_gt)


if __name__ == "__main__":
    n_gt, nb, ratio = table_f1()
    print("Table F.1 -- Schechter, s_min = %.0f mJy, s_max = %.0e mJy, FWHM %.2f\""
          % (S_MIN, S_MAX, FWHM))
    print("  u [mJy]   N(>u) [deg^-2]   N_beam(>u)   R_core/R_tail")
    for x, a, b, c in zip(U, n_gt, nb, ratio):
        print("  %6.0f   %14.1f   %10.2e   %13.0f" % (x, a, b, c))
