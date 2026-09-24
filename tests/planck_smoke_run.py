"""Smoke-test harness for the Planck v2 modules.

Replaces healpy.read_map with a synthetic Nside=1024 sky (beam-like
small-scale structure + a point-source population + a Galactic-latitude
dust gradient) so every code path in planck_unlensed_pd_v2.py and
planck_lensed_pd_v2.py can be exercised in seconds without touching the
400 MB Lenz products.  Numbers produced here are meaningless; only the
absence of exceptions and the sanity of the printed structure matter.

Usage (from the repository root):
    python tests/planck_smoke_run.py unlensed
    python tests/planck_smoke_run.py lensed      # needs the unlensed run first

Outputs go to tests/_smoke/ (git-ignored), never to Planck_analysis/results/.
Needs healpy (and scipy, astropy), but no Planck data.
"""
import os
import shutil
import sys
import numpy as np
import healpy as hp

NSIDE = 1024
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "analysis_modules"))

os.environ.setdefault("CIB_MASK_VARIANT", "SMOKE")
os.environ.setdefault("CIB_QUICK_TEST", "1")
os.environ.setdefault("CIB_USE_CACHE", "1")
os.environ.setdefault("CIB_SAVE_FIGURES", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

_rng = np.random.default_rng(1234)
_npix = hp.nside2npix(NSIDE)


def _synth_sky():
    """Confusion-like sky: smoothed white noise + a Poisson bright tail."""
    base = _rng.normal(0, 1.0, _npix).astype(np.float32)
    sky = hp.smoothing(base, fwhm=np.radians(5.0 / 60.0), verbose=False) \
        if "verbose" in hp.smoothing.__code__.co_varnames else \
        hp.smoothing(base, fwhm=np.radians(5.0 / 60.0))
    sky = np.asarray(sky, np.float32)
    sky *= 0.06 / max(sky.std(), 1e-12)          # ~0.06 MJy/sr core
    # bright sources
    src = np.zeros(_npix, np.float32)
    idx = _rng.integers(0, _npix, 40000)
    src[idx] += (_rng.pareto(2.0, idx.size) * 0.02).astype(np.float32)
    src = np.asarray(hp.smoothing(src, fwhm=np.radians(5.0 / 60.0)),
                     np.float32)
    return sky + src


_SKY = _synth_sky()
_NOISE_O = (_rng.normal(0, 0.012, _npix)).astype(np.float32)
_NOISE_E = (_rng.normal(0, 0.012, _npix)).astype(np.float32)
_theta, _phi = hp.pix2ang(NSIDE, np.arange(_npix))
_b = 90.0 - np.degrees(_theta)
_MASK = (np.abs(_b) > 25.0).astype(np.float32)
_DUST = (0.3 + 2.0 * np.exp(-np.abs(_b) / 25.0)).astype(np.float32)

_real_read_map = hp.read_map


def _fake_read_map(path, field=0, dtype=np.float32, **kw):
    name = os.path.basename(str(path))
    if "mask_bool" in name:
        return _MASK.astype(dtype)
    if "dust_model" in name:
        return _DUST.astype(dtype)
    if "oddring" in name:
        return (_SKY + _NOISE_O).astype(dtype)
    if "evenring" in name:
        return (_SKY + _NOISE_E).astype(dtype)
    if "fullmission" in name:
        return (_SKY + 0.5 * (_NOISE_O + _NOISE_E)).astype(dtype)
    raise FileNotFoundError(path)


hp.read_map = _fake_read_map

which = sys.argv[1] if len(sys.argv) > 1 else "unlensed"
target = os.path.join(_ROOT, "Planck_analysis", f"planck_{which}_pd_v2.py")
#  Run the module as if it lived in a scratch copy of Planck_analysis/, so its
#  results/ and cache folders are created there and not in the repository.
work = os.environ.get("CIB_SMOKE_DIR",
                      os.path.join(_HERE, "_smoke", "Planck_analysis"))
os.makedirs(work, exist_ok=True)
shutil.copy2(os.path.join(_ROOT, "Planck_analysis",
                          "Planck_772_cluster_sample.txt"), work)
print(f"=== SMOKE RUN: {target} (synthetic Nside {NSIDE} sky) ===")
print(f"=== scratch directory: {work} ===\n")
src = open(target).read()
g = {"__name__": "__main__", "__file__": os.path.join(work, os.path.basename(target))}
exec(compile(src, target, "exec"), g)
print("\n=== SMOKE RUN COMPLETED WITHOUT EXCEPTION ===")
