# Simulations and validation

Everything in this folder runs on simulated skies; no survey data are needed. The scripts import the library in `../analysis_modules/` and the shared configuration in `counts_350um.py` (the three adopted 350 µm count models: Schechter, single and double power law), so they cannot disagree on the counts.

## The simulation engine and the three-instrument ladder (Sect. 3, Sect. 6)

| script | paper | output in `results/` | run time* |
|---|---|---|---|
| `sim_core.py` | shared engine: benchmark lens (M₅₀₀ = 10¹⁵ M☉, z_l = 0.5, z_s = 2), instrument/noise bookkeeping, declustered-peak estimators, blocked bootstrap, forecasts. `python sim_core.py` prints the benchmark lens and its aperture statistics (⟨μ⟩, ⟨ln μ⟩). | — | — |
| `simulate_planck.py` | Table 3 (*Planck* column), Table 4 row | `planck.{npz,log}` | several min |
| `simulate_herschel.py` | Table 3 (*Herschel* column), Table 4 row | `herschel.{npz,log}` | several min |
| `simulate_ccat.py` | Table 2, Table 3 (CCAT column), Table 4 row, Figs. 13, 14 | `ccat.{npz,log}` | ~8 min |
| `ccat_beam_vs_sensitivity.py` | Sect. 6.3: beam × sensitivity factorial (N₃σ ratio 7.45 from *Herschel* to CCAT) | `q2_factorial_v2.{npz,log}` | ~2 min |
| `herschel_null_reseed.py` | App. E, Fig. E.1: null-seed ensemble (seeds 5000–5007; seed 5000 is the fiducial run) | `herschel_null_seeds.json`, `.log` | per seed ≈ one `simulate_herschel.py` run |

\* Single core of a laptop-class Linux machine. The `.log` files are the full printed output of the fiducial runs and contain the tabulated numbers of the paper.

```bash
python Simulations/simulate_ccat.py                             # likewise _planck, _herschel
python Simulations/ccat_beam_vs_sensitivity.py
CIB_SEEDS=5000,5001,5002,5003 python Simulations/herschel_null_reseed.py   # batches of seeds
CIB_SEEDS=5004,5005,5006,5007 python Simulations/herschel_null_reseed.py
CIB_AGGREGATE_ONLY=1 python Simulations/herschel_null_reseed.py
```

`herschel_null_reseed.py` reads `results/herschel.npz`, so `simulate_herschel.py` comes first.

Useful switches (read by `sim_core`): `CIB_QUICK_TEST=1` (reduced ensembles, not for science), `CIB_CACHE=1` (cache ensembles and scans under `Simulations/tmp/`, so an interrupted run resumes), `CIB_RESULT_DIR` / `CIB_FIGURE_DIR` (write elsewhere, for example to compare a re-run with the shipped files), `CIB_SHOW=1` (display the diagnostic figures instead of writing them).

## The peak-level model curves (Figs. 2–5, 10)

Every model curve that the paper shows against data or Monte Carlo is a simulated *peak-level* curve, because the declustered-peak distribution differs from the pixel P(D) at intermediate thresholds (App. B).

* `paper_figures_peaklevel.py` builds those curves: the peak-level fingerprints (Fig. 3), the beam sweep (Fig. 4), the Δξ validation at pixel and peak level (Fig. 5) and the *Herschel* model curves with the covariance-correct model ranking (Fig. 10).
* `export_peaklevel_curves.py` calls those builders and writes the curves, together with the analytic flux-space fingerprints of Fig. 2, into `results/paperfig_peaklevel_curves.npz`, which `paper/make_paper_figures.py` reads. Stages: `CIB_EXPORT=2,3,4,5,10` (the default).

A fresh run of these Monte Carlo curves takes hours. The shipped `paperfig_peaklevel_curves.npz` holds the realization published in the paper. The top-level README ("Seeds and determinism") explains why a fresh run is statistically equivalent but not bit-identical to it.

## Validation scripts (`validation/`, Appendices B, C, F)

Stand-alone checks. Each prints its results and writes nothing.

| script | paper | run time |
|---|---|---|
| `check_q1_highu.py` | Table B.1: ξ(u) at a 20″ beam for S_cut = 30–400 mJy; the truncation bias is the depression relative to the 400 mJy row | 30 s |
| `verify_rate_thinning.py` | App. B.3: above the core, declustering thins the exceedance rate by a constant C and leaves (ξ, σ) unchanged | 15 s |
| `check_peak_shift.py` | App. C: 2D Gaussian-random-field peak statistics against simulated beam-smoothed fields | 5 s |
| `verify_memo_numbers.py` | App. C: q* = 0.913 for the one-FWHM (size-5) window, 0.90 for all maxima; the G₂ closed form | 25 s |
| `verify_clustering_mode.py` | Sect. 3.3, App. F: validation of the clustering mode of `simulate_maps.py` (power spectrum, R_core, tail shape, over-dispersion) | 30 s |
| `where_clustering_matters.py` | Tables F.2, F.3: clustered-minus-Poisson Δξ̂(u) and the Fano factors of the exceedance counts (96 maps per arm) | 90 s |
| `core_tail_contrast.py` | Table F.1: the amplitude-free core–tail contrast R_core/R_tail = 2q₁²/(q₂N(>u)) | < 1 s |

```bash
python Simulations/validation/where_clustering_matters.py
```

In the release check, `where_clustering_matters.py` reproduced Tables F.2 and F.3 digit for digit, `verify_memo_numbers.py` gave q* = 0.9128, and `core_tail_contrast.py` reproduced Table F.1 to within 0.2 %.
