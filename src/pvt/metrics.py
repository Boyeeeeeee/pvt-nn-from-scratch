"""Error metrics used to compare models. Relative errors are in percent."""
import numpy as np


def are(y, yhat):
    """Average relative error (signed bias), %."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.nanmean((yhat - y) / y) * 100)


def aare(y, yhat):
    """Average absolute relative error, %."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.nanmean(np.abs((yhat - y) / y)) * 100)


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.nanmean((yhat - y) ** 2)))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[m], yhat[m]
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def summary(y, yhat) -> dict:
    return {"AARE_%": aare(y, yhat), "ARE_%": are(y, yhat), "RMSE": rmse(y, yhat), "R2": r2(y, yhat)}
