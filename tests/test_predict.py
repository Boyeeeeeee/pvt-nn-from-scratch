import math

from pvt.predict import Predictor


def test_predictions_are_finite_and_positive():
    out = Predictor().predict(rs=450, gg=0.75, api=32, temp=210)
    assert math.isfinite(out["pb_psia"]) and out["pb_psia"] > 0
    assert 1.0 < out["bo_rb_per_stb"] < 3.0
    assert out["warnings"] == []


def test_close_to_standing_for_a_typical_oil():
    # Standing gives about 2246 psia and 1.277 rb/STB for this oil; the network should be in the same region.
    out = Predictor().predict(rs=450, gg=0.75, api=32, temp=210)
    assert abs(out["pb_psia"] - 2246) / 2246 < 0.35
    assert abs(out["bo_rb_per_stb"] - 1.277) / 1.277 < 0.08


def test_warns_outside_training_range_and_low_gor():
    out = Predictor().predict(rs=100, gg=0.7, api=20, temp=90)
    text = " ".join(out["warnings"])
    assert "temp" in text and "rs <" in text
