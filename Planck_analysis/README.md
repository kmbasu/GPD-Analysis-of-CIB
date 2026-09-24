# *Planck* 857 GHz analysis

Scripts behind Sect. 4 of the paper: the unlensed ξ̂(u) of the 857 GHz CIB, with its simulated null and threshold covariance, and the lensed Δξ(u) behind PSZ2 clusters, with the amplitude channel and the dust closure against Erler et al. (2018).

## Data

**Included.** `Planck_772_cluster_sample.txt` contains the 772 PSZ2 clusters (RA, Dec, z, R₅₀₀, T_X) of Erler, Basu, Chluba & Bertoldi (2018, MNRAS 476, 3360), the sample used in the paper.

**Not included.** The *Planck* CIB maps of Lenz, Doré & Lagache (2019, ApJ 883, 75), available from [LAMBDA](https://lambda.gsfc.nasa.gov/product/planck/curr/planck_tp_lenz_get.html) and the [Harvard Dataverse](https://doi.org/10.7910/DVN/8A1SR3). The release is organized by frequency and mask variant. The paper uses the 857 GHz maps with the mask variant `4.0e+20_gp40` (N_HI < 4 × 10²⁰ cm⁻², 40 % Galactic-plane mask). HEALPix N_side = 2048, MJy/sr, 5′ FWHM. The scripts expect

```
<CIB_DATA_DIR>/4.0e+20_gp40/cib_fullmission.hpx.fits     # unlensed and lensed
                            cib_oddring.hpx.fits         # unlensed (noise from odd-even)
                            cib_evenring.hpx.fits        # unlensed
                            mask_bool.hpx.fits           # unlensed and lensed
                            dust_model.hpx.fits          # lensed (dust-matched controls)
```

where `CIB_DATA_DIR` is the release's `857/` folder (default: `Planck_analysis/data_857/`, which can be a symbolic link). If the downloaded files are named slightly differently, rename them or link them to these names.

## Scripts, in run order

```bash
cd Planck_analysis
export CIB_DATA_DIR=/path/to/PlanckCIB/857

python planck_unlensed_pd_v2.py                      # Sect. 4.3, Figs. 6, 7
python planck_lensed_pd_v2.py                        # 97 % aperture coverage: 378 pairs (Sects. 4.4-4.5)
CIB_MIN_VALID_AP=1.0 python planck_lensed_pd_v2.py   # primary sample, 100 % coverage: 363 pairs (Figs. 8, 9)
```

| run | output in `results/` |
|---|---|
| unlensed | `planck_v2_unlensed_857_4.0e+20_gp40.npz` |
| lensed, default | `planck_v2_lensed_857_4.0e+20_gp40.npz` |
| lensed, `CIB_MIN_VALID_AP=1.0` | `planck_v2_lensed_857_4.0e+20_gp40_ap1.00.npz` |

The lensed script reads the unlensed result (threshold window and measured Gaussian budget), so the unlensed script must run first. Every expensive stage is cached under `results/v2_cache/` and `results/v2_cache_lensed/`, keyed by a hash of the configuration, so interrupted runs resume. `CIB_USE_CACHE=0` recomputes everything. `CIB_SAVE_FIGURES=1` writes diagnostic PNGs.

**Note on the shipped lensed result.** The amplitude-channel arrays of Fig. 8 (`ratio_err_*`, `n_exc_cl_*`, `n_exc_ct_*`, `ratio_plain_*`, `ratio_win_*`) were added to the save block after the original run. They were merged into the shipped `_ap1.00` file from that run's stage cache, not recomputed. The current script writes them directly, and a cached re-run gives identical values.

## Smoke test without data

`tests/planck_smoke_run.py` replaces `healpy.read_map` with a synthetic N_side = 1024 sky and runs every code path of both scripts in about a minute:

```bash
python tests/planck_smoke_run.py unlensed
python tests/planck_smoke_run.py lensed
```

Its numbers mean nothing. It shows that the installation and the pipeline work end to end before the maps are downloaded. Output goes to `tests/_smoke/`.

## Configuration

Both scripts list their `CIB_*` switches in a table in their header; the defaults are the paper's settings (mask variant, 1600 cutouts, 400 bootstrap replicates, four controls per cluster, S_min = 1 mJy, S_cut = 100 mJy, z_s = 2). The cluster-dust variants of Fig. 9 are computed within the same lensed run (keys `dustvar_*`). `CIB_QUICK_TEST=1` gives a reduced, non-science run.
