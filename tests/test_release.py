"""
Release checks for the GPD/P(D) code release (pytest).

    python -m pytest tests -q

What is checked
---------------
* every shipped result file is present and readable;
* the curated Bethermin et al. (2012) CSV agrees with the table embedded in
  num_count_fits/num_count_fitting.py, and re-running the fit reproduces the
  shipped JSON;
* all paper figures build from the shipped results (Fig. B.1, which runs a
  small simulation, is included);
* Table F.1 is reproduced by Simulations/validation/core_tail_contrast.py;
* the simulation engine uses session-independent seeds;
* the Planck modules run end to end on a synthetic sky (skipped when healpy
  is not installed).

Nothing is written inside the repository: all outputs go to pytest's tmp_path.
Run time ~1-2 minutes.
"""
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(args, env_extra=None, cwd=None, timeout=900):
    env = dict(os.environ, MPLBACKEND="Agg", **(env_extra or {}))
    r = subprocess.run([sys.executable] + args, cwd=cwd or ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, (r.stdout[-3000:] + "\n" + r.stderr[-3000:])
    return r.stdout


# --------------------------------------------------------------------------
SHIPPED = [
    "num_count_fits/num_count_fit_results.json",
    "Simulations/results/paperfig_peaklevel_curves.npz",
    "Simulations/results/planck.npz", "Simulations/results/herschel.npz",
    "Simulations/results/ccat.npz", "Simulations/results/q2_factorial_v2.npz",
    "Simulations/results/herschel_null_seeds.json",
    "Planck_analysis/results/planck_v2_unlensed_857_4.0e+20_gp40.npz",
    "Planck_analysis/results/planck_v2_lensed_857_4.0e+20_gp40.npz",
    "Planck_analysis/results/planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz",
    "Herschel_analysis/results/herschel_noise_analysis_350.npz",
    "Herschel_analysis/results/herschel_unlensed_v2_350.npz",
    "Herschel_analysis/results/herschel_unlensed_v3_350.npz",
    "Herschel_analysis/results/herschel_peak_sims_350.npz",
    "Herschel_analysis/results/herschel_lensed_350.npz",
    "Herschel_analysis/results/herschel_xid_flux_response_350.npz",
]


@pytest.mark.parametrize("rel", SHIPPED)
def test_shipped_result_readable(rel):
    p = os.path.join(ROOT, rel)
    assert os.path.exists(p), rel
    if p.endswith(".npz"):
        with np.load(p, allow_pickle=True) as d:
            assert len(d.files) > 0
    else:
        with open(p) as fh:
            assert json.load(fh)


def test_peaklevel_export_has_all_stages():
    with np.load(os.path.join(
            ROOT, "Simulations/results/paperfig_peaklevel_curves.npz")) as d:
        for prefix in ("f2_", "f3_", "f4_", "f5_", "f10_"):
            assert any(k.startswith(prefix) for k in d.files), prefix


# --------------------------------------------------------------------------
def _embedded_bethermin_table():
    src = open(os.path.join(ROOT, "num_count_fits/num_count_fitting.py")).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "BETHERMIN2012_350UM"
                for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("BETHERMIN2012_350UM not found")


def test_bethermin_csv_matches_script():
    rows = []
    with open(os.path.join(ROOT, "num_count_fits/bethermin2012_350um_counts.csv")) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            rows.append((float(r["S_mJy"]), float(r["Y"]), float(r["sigma_Y"]),
                         r["method"], int(r["used_in_fit"])))
    table = _embedded_bethermin_table()
    assert len(rows) == len(table)
    for (s, y, e, m, used), (s2, y2, e2, m2) in zip(rows, table):
        assert (s, y, e, m) == (s2, y2, e2, m2)
        assert used == int(6.0 <= s < 100.0 and m != "stack_GOODSN")


def test_count_fits_reproduce_json(tmp_path):
    work = tmp_path / "num_count_fits"
    work.mkdir()
    shutil.copy2(os.path.join(ROOT, "num_count_fits/num_count_fitting.py"), work)
    shutil.copytree(os.path.join(ROOT, "analysis_modules"),
                    tmp_path / "analysis_modules")
    _run([str(work / "num_count_fitting.py")], cwd=str(work))
    new = json.load(open(work / "num_count_fit_results.json"))
    old = json.load(open(os.path.join(ROOT, "num_count_fits/num_count_fit_results.json")))
    for model, keys in (("schechter", ("alpha", "nstar_deg2", "sstar_mJy", "redchi2")),
                        ("spl", ("N0", "beta", "redchi2")),
                        ("dpl", ("alpha", "beta", "sstar_mJy", "phistar_deg2"))):
        for k in keys:
            assert np.isclose(new[model][k], old[model][k], rtol=1e-6), (model, k)


# --------------------------------------------------------------------------
PAPER_PDFS = ["counts_fits", "xi_fingerprints", "xi_fingerprints_peak",
              "beam_reshapes_xi", "dxi_validation", "planck_stability",
              "planck_bright", "planck_dxi", "planck_aperture_dust",
              "herschel_xi_vs_flux", "herschel_h1c_decomposition",
              "herschel_dxi", "ccat_unlensed_xi", "ccat_lensed_theory",
              "noise_declustering", "null_seed_ensemble"]


def test_paper_figures_build(tmp_path):
    _run(["paper/make_paper_figures.py"],
         env_extra={"CIB_PAPER_FIGDIR": str(tmp_path)})
    for n in PAPER_PDFS:
        p = tmp_path / f"{n}.pdf"
        assert p.exists() and p.stat().st_size > 5000, n


# --------------------------------------------------------------------------
def test_core_tail_table_f1():
    sys.path.insert(0, os.path.join(ROOT, "Simulations", "validation"))
    import core_tail_contrast as ctc
    n_gt, nb, ratio = ctc.table_f1()
    assert np.allclose(n_gt, [537, 122, 51, 7.2], rtol=0.02)
    assert np.allclose(nb, [3.0e-2, 6.7e-3, 2.8e-3, 4.0e-4], rtol=0.02)
    assert np.allclose(ratio, [123, 541, 1290, 9170], rtol=0.005)


def test_seeds_are_session_independent():
    sim = os.path.join(ROOT, "Simulations")
    for f in os.listdir(sim):
        if f.endswith(".py"):
            code = [l for l in open(os.path.join(sim, f))
                    if not l.lstrip().startswith("#")]
            assert not any(re.search(r"seed\s*=\s*hash\(", l) for l in code), f
    out = _run(["-c", "import sys; sys.path[:0] = ['Simulations', 'analysis_modules'];"
                      "import sim_core; print(sim_core.stable_seed('Schechter'))"])
    assert int(out.split()[-1]) == zlib.crc32(b"Schechter") % 9973


# --------------------------------------------------------------------------
def test_planck_smoke(tmp_path):
    pytest.importorskip("healpy")
    env = {"CIB_SMOKE_DIR": str(tmp_path / "Planck_analysis")}
    out = _run(["tests/planck_smoke_run.py", "unlensed"], env_extra=env)
    assert "COMPLETED WITHOUT EXCEPTION" in out
    out = _run(["tests/planck_smoke_run.py", "lensed"], env_extra=env)
    assert "COMPLETED WITHOUT EXCEPTION" in out
