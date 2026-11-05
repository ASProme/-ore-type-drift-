# Calibrated ore-type mapping and tonnage forecasting under operational drift

Code for the paper "Calibrated Machine Learning Model for Ore-Type Mapping and Tonnage Forecasting under Operational Drift" by Abdus Sukkur Prome, Sajjad Hosen Bijoy and Shofiqul Islam (submitted to AIUB Journal of Science and Engineering).

`analysis.py` produces every number in the paper in a single run with fixed random seeds. `make_figs.py` then draws Figures 3–6 from its outputs.

## Data

Both data files are in the `data/` folder:

- `data/ore_type_data.csv`: the Kaggle dataset "Ore Type Classification" (https://www.kaggle.com/datasets/alfredofardo/ore-type-classification). It has 15,038 samples from 1 January 2023 to 30 December 2024.
- `data/simulated_tonnage.csv`: a simulated daily tonnage series with 752 days, from 1 January 2023 to 21 January 2025.

## Running in Google Colab

Open `colab_run.ipynb` in Colab and run the cells from top to bottom. The notebook clones this repository, installs the package versions used for the paper, runs both scripts, shows the main tables and figures, and checks that the new results match the paper. The full run takes roughly 3–20 minutes depending on the machine.

## Running on your own computer

```
pip install -r requirements.txt
COLUMN_FIGS=1 python analysis.py
python make_figs.py
```

Python 3.11 or newer is needed. The paper results were produced with Python 3.13 and the versions in `requirements.txt`. With other versions, small differences in the last decimal places are possible.

## Outputs

`analysis.py` writes:

- `out/tab/*.csv`: all tables
- `out/results.json`: headline numbers and the package versions used
- `out/fig_col/`: figures at single-column width

`make_figs.py` writes Figures 3–6 to `fig_paper/`.

The outputs from the run used in the paper are kept in `results/` for comparison. `results/all_tables.xlsx` holds the same tables in one Excel file, with one sheet per table and sheets named after the paper's tables.

| Paper | File |
|---|---|
| Table 1 (variables) | `T_data_variables.csv` |
| Table 2 (class counts by year and pit) | `T_class_counts_year_pit.csv` |
| Table 3 (classification models) | `T1_classification_models.csv` |
| Table 4 (calibration metrics) | `T2_calibration.csv` |
| Table 5 (selective prediction) | `T3_selective_prediction.csv`, `T4_classwise_selective.csv` |
| Table 6 (monthly results, 2024) | `T5_monthly_M-cal.csv` |
| Table 7 (March 2024 confusion matrix) | `results.json`, key `march_cm` |
| Table 8 (pit hold-out) | `T_pit_holdout.csv` |
| Table 9 (retraining policies) | `T6_policies.csv` (rows with label delay 1) |
| Table 10 (tonnage forecasting) | `T7_forecasting.csv` (Table 10 shows five of the rows) |
| Feature importance (Section 3.8) | `T8_forecast_importance.csv` |
| Calendar and missingness ablation (Section 3.5) | `T_ablation.csv` |
| Supplementary Table S1 | `T_class_counts_month_pit.csv` |
| Supplementary Table S2 | `T6_policies.csv` (rows with label delay 0) |
| Figures 1–2 | `out/fig_col/Fig1_reliability_with_counts.png`, `out/fig_col/Fig2_risk_coverage.png` |
| Figures 3–6 | `fig_paper/fig3.png` to `fig_paper/fig6.png` |

`T_provenance.csv` lists which model (M-full, M-base or M-cal) is used in each table and figure.

Some outputs are extra checks that the paper's tables do not show:

- **Temperature scaling (M-ts):** in `T2_calibration.csv`; mentioned briefly in Section 4.6.
- **XGBoost:** in `T1_classification_models.csv`.
- **The predicted-class PSI trigger (P3):** in `T6_policies.csv`.
- **SHAP values:** in `T8_forecast_importance.csv`.
- **Further forecasting baselines:** seasonal naïve, 7-day rolling mean, ETS, Ridge, LightGBM without ore features, and a residual model, all in `T7_forecasting.csv`. Table 10 shows five of these rows, and Section 3.8 also mentions the 7-day mean.

## Models in brief

- **M-full:** LightGBM trained on all of 2023, uncalibrated.
- **M-base:** LightGBM trained on January–September 2023, uncalibrated, then kept fixed.
- **M-cal:** M-base with one-vs-rest isotonic calibration fitted on October–December 2023.

LightGBM settings: 700 trees, learning rate 0.03, 63 leaves, subsample 0.85 (subsample_freq 1), colsample_bytree 0.85, random_state 42.

All models are tested on 2024. Confidence intervals come from a bootstrap over test days (2,000 replicates).
