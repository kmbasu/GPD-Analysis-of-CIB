# *Herschel*/SPIRE 350 µm analysis (GAMA-09)

Scripts behind Sect. 5 and Appendix D of the paper: the matched-filter treatment of the SPIRE map, the unlensed ξ̂(u) with the count-model comparison, the forward simulation that explains its rise, the lensed Δξ(u) behind eFEDS clusters, and the flux-response check against the XID+ catalog.

## Data (not included)

Place the four files below in `Herschel_analysis/data/`, or point `CIB_HERSCHEL_DIR` to the folder that holds them. (`CIB_XID_FILE` can override the path of the XID+ catalog alone.)

| file | content | source |
|---|---|---|
| `GAMA-09_SPIRE350_v1.0.fits` | HELP homogenized SPIRE 350 µm map of GAMA-09 (dmu19), 8″ pixels. HDUs: 1 IMAGE, 2 NEBFILT, 3 ERROR, 4 EXPOSURE, 5 MASK, 6 MFILT, 7 MFILT_ERROR, 8 Matchedfilter (101 × 101 kernel). **Use v1.0**: the v0.9 file lacks HDUs 6–8. | [HELP data products](https://hedam.lam.fr/HELP/dataproducts/), product `dmu19_HELP-SPIRE-maps` (Shirley et al. 2019, MNRAS 490, 634; 2021, MNRAS 507, 129) |
| `dmu26_XID+SPIRE_GAMA-09_20180508.fits` | HELP XID+ SPIRE catalog of GAMA-09 (dmu26) | same, product `dmu26_XID+SPIRE_GAMA-09` (Hurley et al. 2017, MNRAS 464, 885) |
| `eFEDS_clusters_V3.2.fits` | eFEDS cluster catalog, 542 clusters: positions, redshifts, X-ray properties | [eROSITA-DE Early Data Release catalogs](https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/) (Liu et al. 2022, A&A 661, A2) |
| `tablec1.dat` | weak-lensing-calibrated masses (`logM500R0.5B`, in h⁻¹ M☉) of the eFEDS clusters, Table C1 of Chiu et al. (2022) | [CDS J/A+A/661/A11](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/661/A11) (Chiu et al. 2022, A&A 661, A11) |

The unlensed analysis uses the eFEDS catalog to keep its cutouts at least 5′ from any cluster. Without the catalog the script still runs, but it prints a warning and its cutouts, and hence all its numbers, differ from the paper.

## Scripts, in run order

| script | paper | output in `results/` | run time* |
|---|---|---|---|
| `noise_analysis.py` | App. D: reproduces the HELP matched-filter operator exactly (MFILT = [(d/σ²)∗K]/[(1/σ²)∗K²], correlation 1.0000000), the noise budget, the lag-one autocorrelation, the point-source response | `herschel_noise_analysis_350.npz` | 15 s |
| `herschel_unlensed_v3.py` | Sect. 5.2, Fig. 10 (data), App. E: self-filtered cutouts, ξ̂(u), three count models, bright-source masked variant, threshold covariance | `herschel_unlensed_v3_350.npz` | 1 min |
| `herschel_peak_sims.py` | Sect. 5.3, Fig. 11: forward simulation separating the pixel-to-peak effect from the bright lensed population | `herschel_peak_sims_350.npz` | 1 min |
| `herschel_lensed.py` | Sect. 5.4, Fig. 12: Δξ(u) over 110 eFEDS cluster/control pairs, null tests, upper limits | `herschel_lensed_350.npz` | 1–2 min |
| `herschel_xid_flux_response.py` | App. D: peak flux in the self-filtered map against XID+ fluxes of 132 isolated sources (ratio 1.04) | `herschel_xid_flux_response_350.npz` | 5 s |

\* On a laptop-class Linux machine, with the map on local disk. `CIB_SAVE_FIGURES=1` (the default) additionally writes diagnostic PNGs into `results/`.

In the release check, each of the five scripts reproduced its shipped result file exactly (to floating-point rounding, ≤ 2 × 10⁻⁷).

**Input of `herschel_peak_sims.py`.** This script reads the measured ξ̂(u), its errors and the cutout geometry from `results/herschel_unlensed_v2_350.npz`, the output of the development version (v2) of `herschel_unlensed_v3.py`. That file is shipped as an input. Its ξ̂(u) and geometry are identical to those of v3. Its per-threshold bootstrap errors come from a smaller bootstrap and differ from the v3 errors by up to 0.012. They set the ±1σ band of Fig. 11 and the weights of the slope decomposition quoted in Sect. 5.3.

## Configuration

Every script lists its `CIB_*` environment switches in its docstring; the defaults are the settings of the paper. The most useful switches:

* `CIB_HERSCHEL_DIR`: data folder (default `Herschel_analysis/data/`);
* `CIB_QUICK_TEST=1`: reduced bootstraps and simulations for a fast end-to-end check (not for science);
* `CIB_USE_CACHE=0`: recompute every cached stage (caches live in `results/v3_cache/` and `results/h2_cache/`);
* `CIB_SAVE_FIGURES=0`: no diagnostic PNGs.

The paper figures themselves are drawn by `paper/make_paper_figures.py` from the result files.
