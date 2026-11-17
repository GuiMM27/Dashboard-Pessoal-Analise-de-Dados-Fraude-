"""Exemplo de requisição local à API. O JSON contém valores SINTÉTICOS."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]
payload = (root / 'examples' / 'transaction.example.json').read_bytes()
request = Request('http://127.0.0.1:8000/predict', data=payload,
                  headers={'Content-Type': 'application/json'}, method='POST')
with urlopen(request, timeout=30) as resp:
    print(json.dumps(json.load(resp), indent=2, ensure_ascii=False))
