import numpy as np
import torch

from pvt.features import Standardizer, from_target, to_target
from pvt.model import PVTNet
from pvt.train import fit


def test_output_shape():
    out = PVTNet()(torch.randn(7, 4))
    assert out.shape == (7,)


def test_target_transform_round_trip():
    y = np.array([500.0, 3000.0])
    assert np.allclose(from_target(to_target(y, "pb"), "pb"), y)
    assert np.allclose(from_target(to_target(y, "bo"), "bo"), y)


def test_standardizer_round_trip():
    a = np.random.RandomState(0).rand(20, 3) * 10
    s = Standardizer().fit(a)
    assert np.allclose(s.inverse(s.transform(a)), a)


def test_training_reduces_loss_on_toy_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 4, generator=g)
    y = x[:, 0] * 2 - x[:, 1] + 0.1 * torch.randn(200, generator=g)
    model = PVTNet(hidden=(16,))
    hist = fit(model, (x[:150], y[:150]), (x[150:], y[150:]), max_epochs=60, patience=60)
    assert hist["train_loss"][-1] < 0.5 * hist["train_loss"][0]
