import pandas as pd
from pvt.clean_data import clean, RAW_COLS


def _raw():
    rows = [
        # sn, pb, temp, bo, sg_oil, api, rho, gg, rs, mu_dead, mu_sat
        ["A1", 3000, 200, 1.5, 0.85, 35, 0.7, 0.8, 600, 1.0, 0.5],
        ["A2", 2000, 180, 1.3, 0.86, 33, 0.7, 0.0, 400, 1.2, 0.6],              # gg == 0 -> dropped
        ["count", 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],                                 # summary row -> dropped
        ["Generated1", 1800, 215, 1.3, 0.85, 34, 0.75, 0.8, 400, None, None],    # synthetic -> dropped
        ["B1", 3000, 200, 1.5, 0.85, 35, 0.7, 0.8, 610, 1.0, 0.5],               # near-duplicate of A1 -> dropped
        [None] * 11,                                                              # blank -> dropped
    ]
    return pd.DataFrame(rows, columns=RAW_COLS)


def test_non_data_rows_removed():
    out, _ = clean(_raw())
    assert list(out["sn"]) == ["A1"]


def test_no_impossible_values():
    out, _ = clean(_raw())
    assert (out["gg"] > 0).all() and (out["rs"] > 0).all() and (out["pb"] > 0).all()


def test_source_column_present():
    out, _ = clean(_raw())
    assert "source" in out.columns
