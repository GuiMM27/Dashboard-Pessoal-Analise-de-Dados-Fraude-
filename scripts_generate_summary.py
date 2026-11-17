"""Regenerate public aggregate report from a local CSV (the CSV is never committed)."""
import json
from pathlib import Path
import pandas as pd
from src.analysis import summarize_data

ROOT = Path(__file__).resolve().parent
source = ROOT / "data" / "creditcard.csv"
if not source.exists():
    raise SystemExit(f"Arquivo ausente: {source}. Coloque seu CSV nessa pasta.")
report = summarize_data(pd.read_csv(source))
out = ROOT / "reports" / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Resumo salvo em {out}")
