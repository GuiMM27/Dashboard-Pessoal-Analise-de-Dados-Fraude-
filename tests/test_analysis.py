import pandas as pd
import pytest
from src.analysis import FEATURES, summarize_data, validate_data
from src.train import chronological_split, evaluate, threshold_from_validation


def sample(n=30):
    rows = []
    for i in range(n):
        row = {key: 0.0 for key in FEATURES}
        row.update({"Time": float(i), "Amount": float(i + 1), "Class": i % 5 == 0})
        rows.append(row)
    return pd.DataFrame(rows)


def test_summary_counts():
    result = summarize_data(sample())
    assert result["n_rows"] == 30
    assert result["fraud_count"] == 6
    assert result["missing_values"] == 0
    assert sum(x["count"] for x in result["amount_groups"]) == 30


def test_required_columns():
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        validate_data(pd.DataFrame({"Class": [0, 1]}))


def test_time_split_no_shuffle():
    d = sample(100).sample(frac=1, random_state=42)
    a, b, c = chronological_split(d)
    assert len(a) == 60 and len(b) == 20 and len(c) == 20
    assert a.Time.max() <= b.Time.min() <= c.Time.min()


def test_evaluate_fraud_confusion():
    out = evaluate([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9], 0.7)
    assert out["tp"] == 2 and out["fp"] == 0
    assert out["fn"] == 0 and out["tn"] == 2
    assert threshold_from_validation([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) > 0.2
