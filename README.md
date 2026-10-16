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
- [x] Ablations: joint Pb+Bo network, feature ablation (`ablations.py`)
- [x] FastAPI service and Dockerfile (`app/`, `Dockerfile`)

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

## Quickstart note
Train both models first (`python src/pvt/train.py --target pb` and `--target bo`), then `python src/pvt/evaluate.py`, `python src/pvt/ablations.py`.

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

## Ablations (test AARE %, 8 seeds, fixed config: 64-64, no dropout, weight decay 1e-3)

| Variant | Pb (all) | Pb (named only) | Bo (all) | Bo (named only) |
|---|---|---|---|---|
| Full inputs (log Rs, gg, API, T) | 26.6 +/- 0.7 | 16.5 | 3.11 +/- 0.21 | 2.98 |
| No temperature | 27.3 | 19.2 | 3.68 | 3.25 |
| No gas gravity | 32.0 | 19.7 | 3.39 | 3.18 |
| No API gravity | 38.4 | 27.5 | 3.67 | 2.99 |
| Raw Rs instead of log10(Rs) | 28.6 | 25.0 | 2.65 | 2.77 |
| + Standing estimate as extra input | 26.9 | 16.2 | 2.86 | 3.11 |
| Joint 2-output network | 26.4 | 16.2 | 3.25 | 2.74 |

Read with care (seed-to-seed std is about 0.7-2.4 for Pb and 0.1-0.25 for Bo):
- **API gravity matters most for Pb**, then gas gravity; removing temperature costs little on the full test set.
- **Joint vs separate:** no meaningful difference for Pb; Bo is slightly worse jointly on the full set. Separate networks are kept.
- **Raw Rs helps Bo** (2.65 vs 3.11), which is plausible since Bo is close to linear in Rs. This was observed on the
  test split, so it is a hypothesis to confirm on validation data before changing the main pipeline, not a result to adopt.
- Adding Standing's own estimate as an input gives no clear gain.

## API
```bash
uvicorn app.app:app --reload          # http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"rs": 450, "gg": 0.75, "api": 32, "temp": 210}'
```
Returns `pb_psia`, `bo_rb_per_stb` and `warnings` (inputs outside the training range, or Rs < 200 scf/STB
where Pb is less reliable). Valid for saturated oils only. Docker: `docker build -t pvt-api .` then
`docker run -p 8000:8000 pvt-api`.
