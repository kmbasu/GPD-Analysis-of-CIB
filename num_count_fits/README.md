# 350 µm number-count fits (Appendix A)

`num_count_fitting.py` fits the three dN/dS models of the paper to the 350 µm differential counts of Béthermin et al. (2012, A&A 542, A58, Table 3, "All" column):

* a Schechter function,
* a single power law (Patanchon et al. 2009 form, S₀ = 2.2 mJy fixed),
* a double power law (break regularized by a loose prior on log₁₀ S*).

All three are fitted over 6 ≤ S < 100 mJy. The three stacked GOODS-N points below 6 mJy and the 133.7 mJy point are excluded; they are still plotted, with open symbols.

| file | content |
|---|---|
| `bethermin2012_350um_counts.csv` | the curated input table: flux, S²·⁵ dN/dS, error, method, and whether the point enters the fit |
| `num_count_fitting.py` | unit conversion (with `astropy.units`), weighted least-squares fits, parameter errors, χ² |
| `num_count_fit_results.json` | best-fit parameters, 1σ errors and χ²/ν (Table A.1), plus confusion widths per beam; read by `paper/make_paper_figures.py` for Fig. 1 |

```bash
python num_count_fits/num_count_fitting.py      # ~1 s; rewrites the JSON and a diagnostic PNG
```

The data table is also embedded in the script (`BETHERMIN2012_350UM`), so the script runs on its own; `tests/test_release.py` checks that the CSV and the embedded table agree. In the release check the script reproduced the shipped JSON exactly.

Elsewhere the fitted models enter through `Simulations/counts_350um.py`, which holds the adopted best-fit parameters. The fiducial integration limits of the paper, S_min = 1 mJy and S_cut = 100 mJy, are set by the calling scripts (`sim_core.S_MIN`, `CIB_SMIN_MJY`, `CIB_SCUT_MJY`).
