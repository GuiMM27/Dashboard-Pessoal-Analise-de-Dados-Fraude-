"""Train a reproducible fraud classifier, select threshold on validation only.

Train/validation/test are chronological (60/20/20), based on relative Time.
No resampling is performed before split; duplicates are audited, not silently
removed. All processing is fit only on the training split.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import FEATURES, validate_data

ROOT = Path(__file__).resolve().parents[1]


def chronological_split(df: pd.DataFrame):
    validate_data(df)
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut_train, cut_val = int(len(ordered) * .60), int(len(ordered) * .80)
    parts = (ordered.iloc[:cut_train], ordered.iloc[cut_train:cut_val], ordered.iloc[cut_val:])
    for name, part in zip(("train", "validation", "test"), parts):
        if part["Class"].nunique() < 2:
            raise ValueError(f"A partição {name} contém apenas uma classe; revise o particionamento")
    return parts


def threshold_from_validation(y_true, prob) -> float:
    """Choose max-F1 threshold using validation set, not test set."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    f1s = (2 * precision[:-1] * recall[:-1] /
           np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps))
    return float(thresholds[np.argmax(f1s)])


def evaluate(y_true, prob, threshold: float) -> dict:
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(average_precision_score(y_true, prob)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def train(data_path: Path | str = ROOT / "data" / "creditcard.csv"):
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    train_df, val_df, test_df = chronological_split(df)
    x_train, y_train = train_df[FEATURES], train_df["Class"]
    x_val, y_val = val_df[FEATURES], val_df["Class"]
    x_test, y_test = test_df[FEATURES], test_df["Class"]

    estimators = {
        "Dummy (prevalência)": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=24, max_depth=12, min_samples_leaf=3, max_samples=0.30,
            class_weight="balanced_subsample", n_jobs=2, random_state=42
        ),
    }

    rows = []
    results_dir = ROOT / "reports"
    models_dir = ROOT / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    candidates = {}
    for name, estimator in estimators.items():
        print(f"Treinando {name}...")
        estimator.fit(x_train, y_train)
        val_prob = estimator.predict_proba(x_val)[:, 1]
        threshold = 0.5 if name == "Dummy (prevalência)" else threshold_from_validation(y_val, val_prob)
        val_ap = float(average_precision_score(y_val, val_prob))
        candidates[name] = (val_ap, estimator, threshold)
        # All candidates evaluated on final holdout for transparent comparison,
        # but the final deployed model is selected by *validation* PR-AUC.
    best_name = max((name for name in candidates if name != "Dummy (prevalência)"),
                    key=lambda name: candidates[name][0])
    for name, (_, estimator, threshold) in candidates.items():
        metrics = evaluate(y_test, estimator.predict_proba(x_test)[:, 1], threshold)
        rows.append({"model": name, "threshold": round(threshold, 6), **metrics})
    import sklearn
    joblib.dump({
        "pipeline": candidates[best_name][1],
        "threshold": candidates[best_name][2],
        "features": FEATURES,
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
    }, models_dir / "fraud_model.joblib")
    report = {
        "methodology": "Ordenação por Time; treino 60%, validação 20%, teste final 20%. "
                       "O melhor modelo foi selecionado por PR-AUC de validação; "
                       "limiares foram ajustados somente na validação. "
                       "Observações duplicadas auditadas, não eliminadas automaticamente. "
                       "Métricas pertencem a dados históricos e não representam desempenho em produção.",
        "selected_model": best_name,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "train_fraud_count": int(y_train.sum()), "validation_fraud_count": int(y_val.sum()),
        "test_fraud_count": int(y_test.sum()),
        "models": rows,
    }
    (results_dir / "model_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo selecionado (validação PR-AUC): {best_name}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Modelo salvo em models/fraud_model.joblib (arquivo local ignorado pelo Git)")
    return report


if __name__ == "__main__":
    train()
