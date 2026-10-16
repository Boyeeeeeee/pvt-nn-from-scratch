from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)
GOOD = {"rs": 450, "gg": 0.75, "api": 32, "temp": 210}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["pb_psia"] > 0 and 1.0 < body["bo_rb_per_stb"] < 3.0
    assert isinstance(body["warnings"], list)


def test_predict_rejects_invalid_input():
    assert client.post("/predict", json={**GOOD, "rs": -5}).status_code == 422
    assert client.post("/predict", json={**GOOD, "api": 200}).status_code == 422
    assert client.post("/predict", json={"rs": 450}).status_code == 422
