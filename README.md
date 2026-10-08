# pvt-nn-from-scratch

Feedforward neural network, trained with a hand-written PyTorch loop, that predicts
**bubble point pressure (Pb)** and **oil formation volume factor (Bo)** from solution GOR,
gas gravity, API gravity and temperature. Benchmarked against Standing, Vasquez-Beggs,
Glaso, Al-Marhoun and Petrosky on a source-grouped test split.

## Status
- [x] Data cleaning (`src/pvt/clean_data.py`)
- [ ] Group split (`splits.py`)
- [ ] Classical baselines (`baselines.py`)
- [ ] Network + training loop (`model.py`, `train.py`)
- [ ] Evaluation and ablations (`evaluate.py`)
- [ ] Results table + figures in this README

## Data
Compiled saturated-oil PVT records (literature sources). The `pb` column is the saturation
pressure (saturated oil, so pressure is at or below bubble point; Rs is the GOR at that pressure).
The raw sheet is not committed; `data/processed/pvt_clean.csv` is the cleaned output.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/pvt/clean_data.py data/raw/Data_Sheet.xlsx data/processed/pvt_clean.csv
pytest
```

## Results
_To be filled in once evaluation is done._
