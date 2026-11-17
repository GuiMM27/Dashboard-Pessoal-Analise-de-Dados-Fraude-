"""FastAPI: previsão a partir de um modelo local confiável (sem dados brutos)."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.analysis import FEATURES

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    Time: float = Field(ge=0, description="Segundos desde a primeira transação do conjunto de origem")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Valor sem unidade monetária informada")


class Prediction(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    predicted_class: int
    selected_model: str
    warning: str


class FeatureEffect(BaseModel):
    feature: str
    contribution_to_logit: float


class Explanation(BaseModel):
    model: str
    intercept: float
    top_contributions: list[FeatureEffect]
    warning: str


app = FastAPI(
    title="Fraud Detection — ML Engineering API",
    version="2.0.0",
    description="API experimental. Scores não calibrados; não utilizar para bloquear transações reais.",
)


def _model_path() -> Path:
    return Path(os.environ.get("FRAUD_MODEL_PATH", str(ROOT / "models" / "fraud_model.joblib"))).expanduser().resolve()


@lru_cache(maxsize=4)
def _load_bundle(path: str) -> dict:
    # Joblib é baseado em pickle; NÃO carregue arquivos recebidos de usuários.
    bundle = joblib.load(path)
    import sklearn
    if bundle.get("sklearn_version") and bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError("Versão do scikit-learn diferente do ambiente de treino")
    if bundle.get("features") != FEATURES:
        raise ValueError("Esquema do modelo incompatível")
    if not 0 <= float(bundle["threshold"]) <= 1:
        raise ValueError("Limiar inválido")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ValueError("O estimador não suporta predict_proba")
    return bundle


def bundle_ready() -> dict:
    path = _model_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Modelo não encontrado. Execute python -m src.train localmente.")
    try:
        return _load_bundle(str(path))
    except (ValueError, KeyError, OSError, ImportError, AttributeError, TypeError):
        log.exception("Falha ao abrir artefato do modelo")
        raise HTTPException(status_code=503, detail="Modelo incompatível ou indisponível.") from None


def score(payload: Transaction):
    bundle = bundle_ready()
    x = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    result = float(bundle["pipeline"].predict_proba(x)[0, 1])
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise HTTPException(status_code=503, detail="Modelo gerou score inválido")
    return bundle, x, result


@app.get("/health")
def health():
    return {"status": "online", "detail": "Servidor ativo; verifique /ready para disponibilidade do modelo."}


@app.get("/ready")
def ready():
    bundle = bundle_ready()
    return {"ready": True, "features_count": len(bundle["features"]),
            "model": bundle.get("model_name", "Selecionado na validação")}


@app.post("/predict", response_model=Prediction)
def predict(payload: Transaction):
    bundle, _, value = score(payload)
    return Prediction(
        risk_score=value, threshold=float(bundle["threshold"]),
        predicted_class=int(value >= bundle["threshold"]),
        selected_model=bundle.get("model_name", "Selecionado na validação"),
        warning="Score experimental NÃO calibrado. Não constitui decisão financeira nem bloqueio automático.",
    )


@app.post("/explain", response_model=Explanation)
def explain(payload: Transaction):
    bundle, x, _ = score(payload)
    pipe = bundle["pipeline"]
    if not hasattr(pipe, "named_steps") or "scaler" not in pipe.named_steps or "model" not in pipe.named_steps:
        raise HTTPException(status_code=501, detail="Explicação local disponível somente para a Logistic Regression padronizada.")
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise HTTPException(status_code=501, detail="Modelo sem coeficientes lineares.")
    scaled = pipe.named_steps["scaler"].transform(x)[0]
    contribution = scaled * model.coef_[0]
    positions = np.argsort(-np.abs(contribution))[:8]
    return Explanation(
        model=bundle.get("model_name", "Logistic Regression"),
        intercept=float(model.intercept_[0]),
        top_contributions=[FeatureEffect(feature=FEATURES[int(i)], contribution_to_logit=float(contribution[i]))
                           for i in positions],
        warning="Contribuições ao logit do modelo para este exemplo; NÃO explicam causas reais nem probabilidades calibradas.",
    )
