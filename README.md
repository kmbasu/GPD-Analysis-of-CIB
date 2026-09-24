# Extreme-value statistics of the P(D) distribution — code release

Code, curated inputs and result files for

> **K. Basu, A. Guerrero & F. Bertoldi (2026), *Probing submillimeter number counts below the confusion limit: extreme-value statistics of the P(D) distribution and its modulation by gravitational lensing*.** [arXiv:2609.19689](https://arxiv.org/abs/2609.19689)

The paper fits a generalized Pareto distribution (GPD) to the upper tail of the one-point distribution P(D) of confusion-limited submillimeter maps. The shape parameter ξ(u) above a threshold u reads the local logarithmic slope of the source counts dN/dS near S ≈ u, including below the confusion limit. Gravitational lensing by a foreground cluster modulates the tail, Δξ(u). The method is validated on simulations and applied to *Planck* 857 GHz and *Herschel*/SPIRE 350 µm data, with a forecast for CCAT/FYST. This repository contains:

1. **the analysis library** (`analysis_modules/`): source-count models, the analytic P(D), NFW lensing, the map simulator (with an optional clustering mode), and the GPD/peaks-over-threshold machinery including beam declustering;
2. **the simulation suite** (`Simulations/`): the three-instrument ladder (*Planck*, *Herschel*, CCAT/FYST), the peak-level model curves, the null-seed ensemble, the CCAT beam-versus-sensitivity factorial, and the validation scripts behind Appendices B, C and F;
3. **the *Planck* 857 GHz analysis** (`Planck_analysis/`), unlensed ξ(u) and lensed Δξ(u) behind PSZ2 clusters;
4. **the *Herschel*/SPIRE 350 µm analysis** (`Herschel_analysis/`): the matched-filter treatment and noise budget, the unlensed ξ(u), the forward simulation of its rise, the lensed Δξ(u) behind eFEDS clusters, and the flux-response check against XID+;
5. **the 350 µm number-count fits** of Appendix A (`num_count_fits/`), with the curated Béthermin et al. (2012) data;
6. **the figure script** (`paper/make_paper_figures.py`), which redraws every figure of the paper from the shipped result files in seconds.

No survey data are included. Where the data come from and where to put them is described in `Planck_analysis/README.md` and `Herschel_analysis/README.md`.

---

## Contents

```
.
├── README.md  LICENSE (MIT)  CITATION.cff  pyproject.toml
│
├── analysis_modules/        the library (imported by everything else)
│   ├── counts.py                dN/dS models (Schechter, double power law, lensed/mixture counts)
│   ├── pofd_analytic.py         analytic P(D) by FFT of the Poisson characteristic function
│   ├── lens_model.py            NFW / SIS lenses, magnification maps, mass-concentration relations
│   ├── simulate_maps.py         Poisson (optionally clustered) confusion maps, beam, noise, lensing
│   └── gpd_tail.py              GPD fits, xi(u) scans, beam declustering, confusion widths
│
├── Simulations/             simulations (no survey data needed; see Simulations/README.md)
│   ├── counts_350um.py          the three adopted 350 um count models and instrument specs
│   ├── sim_core.py              shared engine: benchmark lens, estimators, bootstrap, forecasts
│   ├── simulate_planck.py  simulate_herschel.py  simulate_ccat.py     the three-instrument ladder
│   ├── ccat_beam_vs_sensitivity.py   beam x sensitivity factorial (Sect. 6.3)
│   ├── herschel_null_reseed.py       null-seed ensemble (App. E, Fig. E.1)
│   ├── paper_figures_peaklevel.py    peak-level model curves (Figs. 2-5, 10)
│   ├── export_peaklevel_curves.py    freezes those curves into results/paperfig_peaklevel_curves.npz
│   ├── validation/                   declustering (App. B, C) and clustering (App. F) checks
│   └── results/                      shipped outputs (.npz, .json, .log)
│
├── Planck_analysis/         Planck 857 GHz (see Planck_analysis/README.md)
│   ├── planck_unlensed_pd_v2.py  planck_lensed_pd_v2.py
│   ├── Planck_772_cluster_sample.txt     the PSZ2 sample of Erler et al. (2018)
│   └── results/                          shipped outputs
│
├── Herschel_analysis/       Herschel/SPIRE 350 um, GAMA-09 (see Herschel_analysis/README.md)
│   ├── noise_analysis.py  herschel_unlensed_v3.py  herschel_peak_sims.py
│   ├── herschel_lensed.py  herschel_xid_flux_response.py
│   └── results/                          shipped outputs
│
├── num_count_fits/          App. A: num_count_fitting.py, bethermin2012_350um_counts.csv, results JSON
├── paper/                   make_paper_figures.py -> paper/figures/*.pdf
└── tests/                   release checks (pytest) and the Planck smoke test
```

Every analysis and simulation script writes into the `results/` folder next to it, and the shipped result files are the ones those scripts produced for the paper. Re-running a script therefore replaces the shipped file. The version suffixes in some file names (`_v2`, `_v3`) are the development versions that became final and are kept so that the result-file names match.

---

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate      # or a conda environment
pip install -e .                 # numpy, scipy, matplotlib, astropy
pip install -e ".[planck,test]"  # + healpy (Planck analysis) and pytest
```

This installs only the dependencies. The scripts find `analysis_modules/` and `Simulations/` through relative paths, so they can be run from any working directory and also without installation, in Spyder or from the command line. Most scripts are divided into `#%%` cells. No uncommon libraries are used: `healpy` is needed only by the *Planck* scripts.

The release was checked on Linux with Python 3.10, NumPy 2.2, SciPy 1.15, Matplotlib 3.10, Astropy 6.1 and healpy 1.20.

---

## Quick start

```bash
python paper/make_paper_figures.py        # all 16 figures from the shipped results into paper/figures/ (~15 s)
python -m pytest tests -q                 # release checks (~2 min; the Planck smoke test is skipped without healpy)
```

---

## Paper figures and tables → scripts

Figure and table numbers refer to arXiv:2609.19689v1. The figures are drawn by `paper/make_paper_figures.py` from the file listed under "data"; the script that writes that file is listed under "computed by". Development docstrings sometimes use earlier ("draft") figure numbers; this table is the correspondence.

| paper | content | computed by | data |
|---|---|---|---|
| Fig. 1, Table A.1 | 350 µm counts and the three fitted models | `num_count_fits/num_count_fitting.py` | `num_count_fit_results.json` |
| Fig. 2 | ideal ξ(u) fingerprints of the count models | `Simulations/export_peaklevel_curves.py` (stage 2) | `paperfig_peaklevel_curves.npz` |
| Fig. 3 | peak-level fingerprints at the SPIRE beam | `paper_figures_peaklevel.py` → export (stage 3) | same |
| Fig. 4 | beam sweep on the illustrative DPL | `paper_figures_peaklevel.py` → export (stage 4) | same |
| Fig. 5 | Δξ validation, pixel vs peak | `paper_figures_peaklevel.py` → export (stage 5) | same |
| Table 2 | why the lensing modulation is small | `Simulations/simulate_ccat.py` | `ccat.log` |
| Table 3 | the three-instrument ladder | `simulate_{planck,herschel,ccat}.py` | `{planck,herschel,ccat}.log` |
| Sect. 3.3, Tables F.2, F.3 | source clustering | `Simulations/validation/where_clustering_matters.py`, `verify_clustering_mode.py` | printed |
| Figs. 6, 7 | *Planck* unlensed ξ̂(u), mean excess, nulls | `Planck_analysis/planck_unlensed_pd_v2.py` | `planck_v2_unlensed_857_4.0e+20_gp40.npz` |
| Figs. 8, 9 | *Planck* lensed Δξ, amplitude channel, apertures, dust | `Planck_analysis/planck_lensed_pd_v2.py` (`CIB_MIN_VALID_AP=1.0`) | `planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz` |
| Sects. 4.4–4.5 (378-pair sample) | looser 97 % coverage sample | `planck_lensed_pd_v2.py` (default) | `planck_v2_lensed_857_4.0e+20_gp40.npz` |
| Fig. 10 | *Herschel* ξ̂(u) with peak-level model curves | `Herschel_analysis/herschel_unlensed_v3.py`; curves: export stage 10 | `herschel_unlensed_v3_350.npz`, `paperfig_peaklevel_curves.npz` |
| Fig. 11 | why ξ̂(u) rises (forward simulation) | `Herschel_analysis/herschel_peak_sims.py` | `herschel_peak_sims_350.npz` |
| Fig. 12 | *Herschel* lensed Δξ, eFEDS clusters | `Herschel_analysis/herschel_lensed.py` | `herschel_lensed_350.npz` |
| Figs. 13, 14, Table 4 | CCAT/FYST forecast; N₃σ (all three instruments) | `simulate_ccat.py` (+ `simulate_planck.py`, `simulate_herschel.py` for Table 4) | `ccat.npz`, `*.log` |
| Sect. 6.3 | beam versus sensitivity | `Simulations/ccat_beam_vs_sensitivity.py` | `q2_factorial_v2.{npz,log}` |
| Table B.1 | truncation bias vs S_cut | `Simulations/validation/check_q1_highu.py` (S_cut sweep) | printed |
| App. B.3 | declustering as rate thinning | `Simulations/validation/verify_rate_thinning.py` | printed |
| Fig. B.1 | noise and declustering | `paper/make_paper_figures.py` (small seeded simulation, ~10 s) | computed on the fly |
| App. C | 2D Gaussian-random-field peak statistics (q* = 0.913) | `Simulations/validation/verify_memo_numbers.py`, `check_peak_shift.py` | printed |
| App. D | matched filter, noise budget, flux response | `Herschel_analysis/noise_analysis.py`, `herschel_xid_flux_response.py` | `herschel_noise_analysis_350.npz`, `herschel_xid_flux_response_350.npz` |
| App. E, Fig. E.1 | threshold covariance; null-seed ensemble | `herschel_unlensed_v3.py`, `planck_unlensed_pd_v2.py`; `Simulations/herschel_null_reseed.py` | `herschel_null_seeds.json` |
| Table F.1 | core–tail contrast | `Simulations/validation/core_tail_contrast.py` | printed |

---

## Reproducing the results: three levels

**(a) Figures from the shipped results (seconds, no data).** `python paper/make_paper_figures.py`. In the release check, 14 of the 16 figures were pixel-identical to the PDFs in the arXiv submission. In the other two (Figs. 5 and 11) the plotted data are identical and only the typesetting of the axis labels differs.

**(b) Simulations and validation (minutes to hours, no data).** The scripts in `Simulations/` and `Simulations/validation/` need no survey data. See `Simulations/README.md` for the run order and run times. The validation scripts reproduce the numbers quoted in Appendices B, C and F. Tables F.2 and F.3 are reproduced digit for digit.

**(c) The data analyses (needs the public *Planck* and *Herschel* products).** Download the data as described in the two analysis READMEs and run the scripts in the order given there. Every expensive stage is cached on disk under `results/`, so an interrupted run resumes where it stopped.

### Seeds and determinism

All map simulations use explicit seeds. One change was made for the release. The bootstrap and pool-thinning seed of the simulation engine was `hash(model_name) % 9973`, and Python randomizes string hashes per interpreter session. A fresh run of `sim_core`-based scripts was therefore a statistically equivalent but not bit-identical realization of the published one. The release uses a stable CRC32 seed (`sim_core.stable_seed`), so fresh runs are now deterministic. They do not reproduce the published realization bit for bit, because its seeds cannot be recovered. The map ensembles themselves, and all lensed and theory quantities (Tables 2 and 4, Figs. 5 and 14), are unaffected. For example, a fresh `simulate_ccat.py` reproduces `ccat.npz` exactly except for the unlensed ξ̂(u) curves, which change at the level of the pool-thinning noise, mostly within the plotted 68 % bootstrap band. The published peak-level curves are shipped in `Simulations/results/paperfig_peaklevel_curves.npz`, and the figures are drawn from them.

`sim_core` can also cache its ensembles and scans (`CIB_CACHE=1`, under `Simulations/tmp/`). The caches are not shipped: they are large, and only the exported results are needed.

---

## Data used in the paper (not included)

| data | used by | source |
|---|---|---|
| *Planck* 857 GHz CIB maps of Lenz, Doré & Lagache (2019), mask variant `4.0e+20_gp40` (full mission, odd/even ring, boolean mask, H I dust model) | `Planck_analysis/` | [LAMBDA](https://lambda.gsfc.nasa.gov/product/planck/curr/planck_tp_lenz_get.html), [Harvard Dataverse](https://doi.org/10.7910/DVN/8A1SR3) |
| PSZ2 cluster sample (772 clusters, Erler et al. 2018) | `Planck_analysis/` | **included**: `Planck_772_cluster_sample.txt` |
| HELP SPIRE 350 µm map of GAMA-09, `GAMA-09_SPIRE350_v1.0.fits` (dmu19) | `Herschel_analysis/` | [HeDaM, HELP data products](https://hedam.lam.fr/HELP/dataproducts/) (Shirley et al. 2019, 2021) |
| HELP XID+ SPIRE catalog of GAMA-09 (dmu26) | `herschel_xid_flux_response.py` | same (Hurley et al. 2017) |
| eFEDS cluster catalog, 542 clusters (Liu et al. 2022, A&A 661, A2), file `eFEDS_clusters_V3.2.fits` | `Herschel_analysis/` | [eROSITA-DE Early Data Release](https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/) |
| eFEDS weak-lensing masses, Table C1 of Chiu et al. (2022, A&A 661, A11), `tablec1.dat` | `herschel_lensed.py` | [CDS J/A+A/661/A11](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/661/A11) |
| 350 µm number counts, Béthermin et al. (2012, A&A 542, A58), Table 3 | `num_count_fits/` | **included**: `bethermin2012_350um_counts.csv` |

---

## Notes

* **Comments in the code** refer at places to internal working documents (for example `GPD_beam_and_declustering_v4.md`, `Full_Analysis_Summary_Herschel.md`, `Revised_Planck_analysis.md`, `full_simulation_summary_v2.md`) and to development module names (H0, H1c, M4a-v2, …). These documents are not part of the release; the paper and its appendices are the public description of every method. Dated comments record why a piece of code has its present form and are kept on purpose.
* **Environment switches.** The scripts are configured through `CIB_*` environment variables (data locations, bootstrap sizes, quick-test mode, caching). Each script lists its switches in its docstring; the defaults are the settings used in the paper.
* **Quick-test mode.** `CIB_QUICK_TEST=1` runs most scripts with reduced ensembles for a fast end-to-end check. The numbers from such a run are not science results.

---

## Citation

If you use this code, please cite the paper. To pin the exact code version, also cite the archived release (the DOI badge will appear here once the release is archived on Zenodo):

```bibtex
@article{Basu2026GPD,
  author        = {Basu, Kaustuv and Guerrero, Andrea and Bertoldi, Frank},
  title         = {Probing submillimeter number counts below the confusion limit:
                   extreme-value statistics of the {P(D)} distribution and its
                   modulation by gravitational lensing},
  year          = {2026},
  eprint        = {2609.19689},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO},
  doi           = {10.48550/arXiv.2609.19689}
}
```

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the software citation.

## License

MIT; see [`LICENSE`](LICENSE).

## Acknowledgments

The codebase was written with the help of Claude (Anthropic), as stated in the paper; the scientific content, the methods and their validation are the authors' responsibility. This work uses data from *Planck* (ESA), the *Herschel* Extragalactic Legacy Project (HELP), and the eROSITA Final Equatorial-Depth Survey (eFEDS).

## Contact

Kaustuv Basu ([ORCID 0000-0001-5276-8730](https://orcid.org/0000-0001-5276-8730)), Argelander-Institut für Astronomie, Universität Bonn — kbasu@uni-bonn.de. Questions and bug reports are welcome as GitHub issues.
