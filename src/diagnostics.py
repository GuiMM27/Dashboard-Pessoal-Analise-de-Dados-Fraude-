"""Auditoria de erros e explicabilidade do modelo SELECIONADO anteriormente.

O teste é utilizado apenas para descrever erros da decisão já fixada.
NÃO alterar modelo, variáveis, limiar ou hiperparâmetros com base no teste.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .analysis import FEATURES, AMOUNT_BINS, AMOUNT_LABELS
from .train import ROOT, chronological_split, evaluate


def diagnostics(data_path: Path | str = ROOT / "data" / "creditcard.csv",
                model_path: Path | str = ROOT / "models" / "fraud_model.joblib") -> tuple[dict, dict]:
    data_path, model_path = Path(data_path), Path(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset indisponível: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError("Modelo local não encontrado; execute python -m src.train antes.")

    # Joblib/Pickle pode executar código arbitrário: abrir somente arquivos que você treinou e confia.
    bundle = joblib.load(model_path)
    if bundle.get("features") != FEATURES:
        raise ValueError("Esquema do modelo incompatível com as 30 variáveis esperadas")
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Limiar inválido no artefato")

    df = pd.read_csv(data_path)
    _, validation, test = chronological_split(df)
    x_val, y_val = validation[FEATURES], validation["Class"].to_numpy()
    x_test, y_test = test[FEATURES], test["Class"].to_numpy()
    val_score = pipeline.predict_proba(x_val)[:, 1]
    test_score = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_score >= threshold).astype(int)
    metrics = evaluate(y_test, test_score, threshold)

    report_path = ROOT / "reports" / "model_metrics.json"
    if not report_path.is_file():
        raise FileNotFoundError("Falta reports/model_metrics.json; execute o treinamento.")
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    selected_name = previous["selected_model"]
    previous_metrics = next((m for m in previous["models"] if m["model"] == selected_name), None)
    if previous_metrics is None:
        raise ValueError("Modelo selecionado ausente do relatório")
    for key in ("tp", "fp", "fn", "tn"):
        if int(previous_metrics[key]) != int(metrics[key]):
            raise ValueError("Contagens diferentes das do relatório original! Verifique dataset, versão e modelo.")

    outcome = np.select(
        [(y_test == 1) & (test_pred == 1),
         (y_test == 0) & (test_pred == 1),
         (y_test == 1) & (test_pred == 0)],
        ["TP", "FP", "FN"], default="TN"
    )
    bucket = pd.cut(x_test["Amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS, include_lowest=True)
    counts = (pd.DataFrame({"faixa_amount": bucket, "resultado": outcome})
              .groupby(["faixa_amount", "resultado"], observed=False).size())
    amounts = [
        {"faixa": label, **{k: int(counts.get((label, k), 0)) for k in ("TP", "FP", "FN", "TN")}}
        for label in AMOUNT_LABELS
    ]

    # As alternativas abaixo são um estudo de SENSIBILIDADE da VALIDAÇÃO, não novos parâmetros para o teste.
    thresholds = [threshold, 0.5, float(np.quantile(val_score, 0.995)),
                  float(np.quantile(val_score, 0.999))]
    thresholds = sorted(set(float(t) for t in thresholds))
    val_alternatives = [
        {"threshold": t, **evaluate(y_val, val_score, t)} for t in thresholds
    ]

    # Histogramas AGREGADOS: nunca inclui transações brutas ou identificadores.
    bins = np.linspace(0, 1, 11)
    hist = []
    for lab, name in ((0, "legitima"), (1, "fraude")):
        c, _ = np.histogram(test_score[y_test == lab], bins=bins)
        hist.extend({"classe": name, "faixa": f"{bins[i]:.1f}–{bins[i+1]:.1f}",
                     "quantidade": int(v)} for i, v in enumerate(c))

    errors = {
        "selected_model": selected_name,
        "threshold_fixed_on_validation": threshold,
        "test_metrics": metrics,
        "validation_candidates_only": val_alternatives,
        "amount_error_buckets_test": amounts,
        "score_histogram_test": hist,
        "uncalibrated_score_brier_test": float(brier_score_loss(y_test, test_score)),
        "methodology": "Análise descritiva de erros no teste já congelado; alternativas de limiar calculadas APENAS na validação. Scores não calibrados.",
        "limitations": [
            "O teste não foi usado para alterar modelo ou limiar.",
            "Score do modelo com class_weight não corresponde necessariamente à probabilidade real de fraude.",
            "Não é possível explicar causas de negócio a partir das V1–V28 anonimizadas.",
            "Intervalos de confiança e teste prospectivo ainda não foram realizados.",
        ],
    }

    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(model, "coef_") and hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
        # Coeficientes aplicados sobre variáveis padronizadas.
        scores = model.coef_[0].astype(float)
        kind = "logistic_standardized_coefficients"
        explanation = (
            "Coeficientes no espaço padronizado: sinal indica contribuição ao logit do modelo; "
            "magnitude NÃO é causalidade nem valor monetário. Correlações podem alterar a interpretação."
        )
    elif hasattr(model, "feature_importances_"):
        scores = model.feature_importances_.astype(float)
        kind = "tree_impurity_importance"
        explanation = (
            "Importância por redução de impureza (árvores): pode favorecer variáveis contínuas "
            "ou correlacionadas; não fornece causas comerciais."
        )
    else:
        scores = np.zeros(len(FEATURES))
        kind = "unavailable"
        explanation = "Modelo não possui importâncias nativas; relatório sem atribuição."
    if len(scores) != len(FEATURES):
        raise ValueError("Dimensão da explicabilidade incompatível")
    order = np.argsort(-np.abs(scores))
    explain = {
        "selected_model": selected_name, "method": kind, "interpretation": explanation,
        "feature_importances": [
            {"feature": FEATURES[i], "importance": float(scores[i]),
             "absolute_importance": float(abs(scores[i]))} for i in order
        ],
        "note": "Explicação global do comportamento do estimador; não atribui responsabilidade individual ou causalidade.",
    }

    destination = ROOT / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "error_analysis.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / "explainability.json").write_text(json.dumps(explain, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Auditoria salva em reports/error_analysis.json e reports/explainability.json")
    print(f"Erros no teste: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print("Top 5 variáveis (não-causais):", [r["feature"] for r in explain["feature_importances"][:5]])
    return errors, explain


if __name__ == "__main__":
    diagnostics()
