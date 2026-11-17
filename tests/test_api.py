"""Testes independentes do CSV e do modelo pesado; exercitam API real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.main import app, _load_bundle
from src.analysis import FEATURES


@pytest.fixture
def ready_client(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    sample = pd.DataFrame(rng.normal(size=(160, len(FEATURES))), columns=FEATURES)
    sample["Time"] = np.arange(160) + 1
    sample["Amount"] = rng.uniform(1, 100, size=160)
    y = (sample["V1"] + 0.7 * sample["V2"] > 0).astype(int)
    model = Pipeline([("scaler", StandardScaler()),
                      ("model", LogisticRegression(max_iter=200, random_state=42))])
    model.fit(sample, y)
    file = tmp_path / "test_fraud.joblib"
    joblib.dump({"pipeline": model, "threshold": 0.5,
                 "features": FEATURES, "model_name": "Logistic Regression"}, file)
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(file))
    _load_bundle.cache_clear()
    with TestClient(app) as client:
        yield client
    _load_bundle.cache_clear()


def request_json():
    return {"Time": 1000.0, "Amount": 10.0, **{f"V{i}": 0.0 for i in range(1, 29)}}


def test_predict_api(ready_client):
    response = ready_client.post("/predict", json=request_json())
    assert response.status_code == 200
    doc = response.json()
    assert 0 <= doc["risk_score"] <= 1
    assert doc["predicted_class"] == int(doc["risk_score"] >= doc["threshold"])
    assert doc["selected_model"] == "Logistic Regression"
    assert ready_client.get("/ready").json()["ready"] is True


def test_explain_api(ready_client):
    response = ready_client.post("/explain", json=request_json())
    assert response.status_code == 200
    assert len(response.json()["top_contributions"]) == 8
    assert response.json()["top_contributions"][0]["feature"] in FEATURES


def test_input_validation(ready_client):
    payload = request_json()
    payload.pop("V28")
    assert ready_client.post("/predict", json=payload).status_code == 422
    payload = request_json()
    payload["AnotherField"] = 999
    assert ready_client.post("/predict", json=payload).status_code == 422
    payload = request_json()
    payload["Amount"] = -10
    assert ready_client.post("/predict", json=payload).status_code == 422


def test_model_missing_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "absent.joblib"))
    _load_bundle.cache_clear()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json=request_json()).status_code == 503
