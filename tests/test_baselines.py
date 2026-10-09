import numpy as np
import pandas as pd

from pvt.baselines import BO_MODELS, PB_MODELS, predict_all
from pvt.metrics import aare

DATA = "data/processed/pvt_clean.csv"


def _df():
    return pd.read_csv(DATA)


def test_all_models_return_finite_values():
    df = _df()
    for target in ("pb", "bo"):
        for name, pred in predict_all(df, target).items():
            assert np.isfinite(pred).all(), f"{target}/{name} produced nan/inf"


def test_bo_baselines_are_in_a_sane_range():
    # Published Bo correlations typically score ~2-6% AARE on saturated oils.
    # A typo in a coefficient shows up as a much larger error.
    df = _df()
    for name, pred in predict_all(df, "bo").items():
        assert aare(df["bo"], pred) < 8, name


def test_pb_baselines_are_in_a_sane_range():
    # Pb is harder on a mixed dataset; anything beyond this signals a coding error.
    df = _df()
    for name in ("Standing", "Vasquez-Beggs", "Glaso", "Al-Marhoun"):
        assert aare(df["pb"], predict_all(df, "pb")[name]) < 60, name


def test_every_pb_model_has_a_bo_model():
    assert set(PB_MODELS) == set(BO_MODELS)