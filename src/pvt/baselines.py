"""Classical saturated-oil correlations for Pb (psia) and Bo (rb/STB).

Inputs (numpy arrays or pandas Series):
    rs   solution GOR, scf/STB
    gg   gas specific gravity (air = 1)
    api  oil API gravity
    temp reservoir temperature, deg F

Notes
- Vasquez-Beggs corrects gas gravity to a 100 psig separator. Separator conditions are not in
  this dataset, so gg is used uncorrected (same choice for every row; state it in the README).
- Correlations are applied outside some of their original ranges; that is part of the benchmark.
"""
import numpy as np


def _gamma_o(api):
    return 141.5 / (np.asarray(api, dtype=float) + 131.5)


# ---------------------------------------------------------------- Standing (1947)
def standing_pb(rs, gg, api, temp):
    return 18.2 * ((rs / gg) ** 0.83 * 10 ** (0.00091 * temp - 0.0125 * api) - 1.4)


def standing_bo(rs, gg, api, temp):
    f = rs * (gg / _gamma_o(api)) ** 0.5 + 1.25 * temp
    return 0.9759 + 0.000120 * f ** 1.2


# ---------------------------------------------------------------- Vasquez-Beggs (1980)
def vasquez_beggs_pb(rs, gg, api, temp):
    api = np.asarray(api, dtype=float)
    c1 = np.where(api <= 30, 0.0362, 0.0178)
    c2 = np.where(api <= 30, 1.0937, 1.1870)
    c3 = np.where(api <= 30, 25.7240, 23.931)
    return (rs / (c1 * gg * np.exp(c3 * api / (temp + 460.0)))) ** (1.0 / c2)


def vasquez_beggs_bo(rs, gg, api, temp):
    api = np.asarray(api, dtype=float)
    c1 = np.where(api <= 30, 4.677e-4, 4.670e-4)
    c2 = np.where(api <= 30, 1.751e-5, 1.100e-5)
    c3 = np.where(api <= 30, -1.811e-8, 1.337e-9)
    return 1.0 + c1 * rs + (temp - 60.0) * (api / gg) * (c2 + c3 * rs)


# ---------------------------------------------------------------- Glaso (1980, North Sea)
def glaso_pb(rs, gg, api, temp):
    f = (rs / gg) ** 0.816 * temp ** 0.172 / api ** 0.989
    logpb = 1.7669 + 1.7447 * np.log10(f) - 0.30218 * np.log10(f) ** 2
    return 10 ** logpb


def glaso_bo(rs, gg, api, temp):
    bob_star = rs * (gg / _gamma_o(api)) ** 0.526 + 0.968 * temp
    lb = np.log10(bob_star)
    return 1.0 + 10 ** (-6.58511 + 2.91329 * lb - 0.27683 * lb ** 2)


# ---------------------------------------------------------------- Al-Marhoun (1988, Middle East)
def almarhoun_pb(rs, gg, api, temp):
    go = _gamma_o(api)
    tr = temp + 459.67
    return 5.38088e-3 * rs ** 0.715082 * gg ** -1.87784 * go ** 3.1437 * tr ** 1.32657


def almarhoun_bo(rs, gg, api, temp):
    go = _gamma_o(api)
    tr = temp + 459.67
    f = rs ** 0.742390 * gg ** 0.323294 * go ** -1.202040
    return 0.497069 + 0.862963e-3 * tr + 0.182594e-2 * f + 0.318099e-5 * f ** 2


# ---------------------------------------------------------------- Petrosky-Farshad (1993, Gulf of Mexico)
def petrosky_pb(rs, gg, api, temp):
    x = 7.916e-4 * api ** 1.5410 - 4.561e-5 * temp ** 1.3911
    return 112.727 * rs ** 0.577421 / (gg ** 0.8439 * 10 ** x) - 1391.051


def petrosky_bo(rs, gg, api, temp):
    go = _gamma_o(api)
    inner = rs ** 0.3738 * (gg ** 0.2914 / go ** 0.6265) + 0.24626 * temp ** 0.5371
    return 1.0113 + 7.2046e-5 * inner ** 3.0936


PB_MODELS = {
    "Standing": standing_pb,
    "Vasquez-Beggs": vasquez_beggs_pb,
    "Glaso": glaso_pb,
    "Al-Marhoun": almarhoun_pb,
    "Petrosky-Farshad": petrosky_pb,
}
BO_MODELS = {
    "Standing": standing_bo,
    "Vasquez-Beggs": vasquez_beggs_bo,
    "Glaso": glaso_bo,
    "Al-Marhoun": almarhoun_bo,
    "Petrosky-Farshad": petrosky_bo,
}


def predict_all(df, target: str) -> dict:
    """Run every baseline for `target` ('pb' or 'bo') on a dataframe with rs, gg, api, temp."""
    models = PB_MODELS if target == "pb" else BO_MODELS
    args = (df["rs"].to_numpy(float), df["gg"].to_numpy(float), df["api"].to_numpy(float), df["temp"].to_numpy(float))
    with np.errstate(all="ignore"):
        return {name: fn(*args) for name, fn in models.items()}
