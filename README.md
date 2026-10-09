# pvt-nn-from-scratch

Feedforward neural network, trained with a hand-written PyTorch loop, that predicts
**bubble point pressure (Pb)** and **oil formation volume factor (Bo)** from solution GOR,
gas gravity, API gravity and temperature. Benchmarked against Standing, Vasquez-Beggs,
Glaso, Al-Marhoun and Petrosky on a source-grouped test split.

## Status
- [x] Data cleaning (`src/pvt/clean_data.py`)
- [x] Source-grouped train/val/test split (`splits.py`)
- [x] Classical baselines: Standing, Vasquez-Beggs, Glaso, Al-Marhoun, Petrosky-Farshad (`baselines.py`)
- [x] Network and hand-written training loop (`model.py`, `train.py`)
- [x] Evaluation: repeated seeds, validation-selected hyperparameters, figures (`evaluate.py`)
- [ ] Optional: joint Pb+Bo network, feature ablation, FastAPI service

## Data
783 saturated-oil PVT records after cleaning (from 864 raw rows: summary rows, one synthetic row,
blank rows, 5 impossible values and 59 near-duplicate oils removed). The `pb` column is the
saturation pressure, i.e. at or below the bubble point, with Rs the solution GOR at that pressure.
The raw sheet is not committed; `data/processed/pvt_clean.csv` is the cleaned output.

**Split (70/15/15, seed 42).** Rows with a named source label are split by source, so no source
appears in two splits. Rows whose ID is only a number (258 rows) carry no usable source label, so
each is its own group and that part of the split is random. As a result the test set has 117 rows,
of which only 32 (from 9 sources) are source-held-out. Results are reported for both.

## Method
- Inputs: log10(Rs), gas gravity, API gravity, temperature. Standardised with training-split statistics only.
- Targets: log10(Pb) and Bo, one network per target (64-64 SiLU MLP, Adam, MSE, early stopping on validation loss).
- Training loop written by hand in `src/pvt/train.py` (no Trainer wrappers).
- Hyperparameters (layer sizes, dropout, weight decay; 12 settings x 5 seeds) selected on validation AARE only.
- Final numbers: selected setting retrained with 10 seeds.
- Vasquez-Beggs is used without separator-gas correction because separator conditions are not in the data.

## Results (test split, AARE %, lower is better)

**Bubble point pressure (Pb)**

| Model | All test (n=117) | Named sources only (n=32) |
|---|---|---|
| Neural net (mean of 10 seeds) | 26.9 +/- 1.0 | 16.9 +/- 1.2 |
| Neural net (10-seed ensemble) | 26.2 | 16.2 |
| Standing | 28.2 | 28.5 |
| Vasquez-Beggs | 30.8 | 30.3 |
| Glaso | 37.5 | 29.6 |
| Al-Marhoun | 44.1 | 50.6 |
| Petrosky-Farshad | 88.3 | 80.9 |

**Oil formation volume factor (Bo)**

| Model | All test (n=117) | Named sources only (n=32) |
|---|---|---|
| Neural net (mean of 10 seeds) | 2.81 +/- 0.18 | 2.64 +/- 0.23 |
| Neural net (10-seed ensemble) | 2.65 | 2.54 |
| Standing | 3.89 | 4.15 |
| Vasquez-Beggs | 3.24 | 3.43 |
| Glaso | 3.68 | 3.96 |
| Al-Marhoun | 3.47 | 3.80 |
| Petrosky-Farshad | 4.55 | 4.78 |

Full tables: `results/metrics.csv` (models) and `results/ablations.csv` (hyperparameter grid).
Figures: `results/figures/` (parity plots, error by GOR range, training curves).

### What the results do and do not show
- **Bo:** the network beats all five correlations on both subsets, by about 0.4 AARE points over the best
  correlation (roughly two seed-to-seed standard deviations). A real but modest gain.
- **Pb, full test set:** only a small edge over Standing (26.9 vs 28.2, about one standard deviation).
- **Pb, named sources:** a large gap (16.9 vs 28.5), but this rests on 32 rows from 9 sources and should be
  read as encouraging, not conclusive.
- **Where the network loses:** for Pb at low GOR (Rs < 200 scf/STB) it is worse than Standing
  (about 42% vs 37%); its advantage is concentrated at Rs > 1000 (about 14% vs 27%).
- **Hyperparameters:** validation AARE varied only from 19.6 to 22.8 (Pb) across the whole grid, so the
  architecture is not sensitive to these settings and most of that spread is noise.
- **Petrosky-Farshad** returns zero or negative Pb for 111 low-GOR rows (outside its fitted range); its
  predictions are reported as they are, not clipped.
- Pb correlation coefficients were not independently checked against worked examples from the original papers.