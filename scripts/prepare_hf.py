"""Prepara localmente um pacote CURADO para o Hugging Face, sem upload automático."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import sklearn
import numpy
import scipy
import pandas
import joblib as joblib_lib

from src.analysis import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def prepare() -> Path:
    model_path = ROOT / 'models' / 'fraud_model.joblib'
    metrics_path = ROOT / 'reports' / 'model_metrics.json'
    if not model_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError('Execute python -m src.train antes de preparar a publicação.')

    # Carregue SOMENTE modelo local que você treinou e confia.
    bundle = joblib.load(model_path)
    if bundle.get('features') != FEATURES:
        raise ValueError('Esquema incompatível')
    if bundle.get('sklearn_version') and bundle['sklearn_version'] != sklearn.__version__:
        raise ValueError('scikit-learn local diferente do que treinou o modelo; reexecute o treinamento')
    report = json.loads(metrics_path.read_text(encoding='utf-8'))
    name = report['selected_model']
    if bundle.get('model_name') and bundle['model_name'] != name:
        raise ValueError('O modelo do disco difere do relatório')
    metric = next(m for m in report['models'] if m['model'] == name)

    output = ROOT / 'hf_export'
    output.mkdir(exist_ok=True)
    shutil.copy2(model_path, output / 'fraud_model.joblib')
    shutil.copy2(ROOT / 'examples' / 'transaction.example.json', output / 'example_input.json')
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        'model': name,
        'threshold_selected_on_validation': float(bundle['threshold']),
        'feature_names': FEATURES,
        'sklearn_version_trained': bundle.get('sklearn_version', 'não registrado na primeira versão'),
        'sklearn_version_preparing_package': sklearn.__version__,
        'test_metrics': metric,
        'dataset_source': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
        'sha256': sha256,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'requirements-model.txt').write_text(
        f'scikit-learn=={sklearn.__version__}\n'
        f'numpy=={numpy.__version__}\n'
        f'scipy=={scipy.__version__}\n'
        f'pandas=={pandas.__version__}\n'
        f'joblib=={joblib_lib.__version__}\n', encoding='utf-8')

    card = f'''---
language: pt
tags:
- scikit-learn
- tabular-classification
- fraud-detection
- educational
---

# Credit Card Fraud Detection — {name}

Este modelo foi desenvolvido para estudo de **Machine Learning Engineering** a partir do
[dataset público ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
NÃO é um sistema financeiro validado para uso real.

## Finalidade e limitações

- **Uso previsto:** benchmark e demonstração educacional de identificação de eventos raros.
- **Não indicado:** bloqueio de cartões, decisões financeiras ou inferência em dados de outra instituição.
- `V1` a `V28` são componentes anonimizadas; `Time` mede tempo relativo (segundos); `Amount` não informa moeda.
- O `risk_score` retornado por `predict_proba` **não é probabilidade de fraude calibrada**.
- O classificador foi treinado em dados históricos e não foi validado prospectivamente.
- Não redistribuímos o CSV original. Consulte as condições de uso da fonte para uso ou redistribuição de derivados.

## Método

- Divisão cronológica aproximada 60% treino / 20% validação / 20% teste.
- Seleção do modelo por PR-AUC na validação; limiar escolhido por F1 na validação.
- Conjunto de teste utilizado apenas para a avaliação final.
- Duplicatas auditadas, mas não removidas automaticamente.

## Avaliação no teste (modelo selecionado)

| Métrica | Resultado |
|---|---:|
| PR-AUC (Average Precision) | {metric['pr_auc']:.4f} |
| Precision | {metric['precision']:.4f} |
| Recall | {metric['recall']:.4f} |
| F1 | {metric['f1']:.4f} |
| Verdadeiros positivos (TP) | {metric['tp']} |
| Falsos positivos (FP) | {metric['fp']} |
| Falsos negativos (FN) | {metric['fn']} |
| Verdadeiros negativos (TN) | {metric['tn']} |
| Limiar fixado na validação | {float(bundle['threshold']):.6f} |

A contagem de fraudes na base de teste foi **{report['test_fraud_count']}**.
Os valores refletem somente este benchmark. Não há garantias de generalização.

## Entrada, saída e compatibilidade

A entrada é um registro com **30 variáveis** numéricas na ordem:
`Time, V1, V2, ..., V28, Amount`. Um JSON sintético está em `example_input.json`.
O artefato é um dicionário com `pipeline`, `threshold` e `features`.
Saída: score entre 0 e 1 e classe 1 caso `score >= threshold`.

**Segurança:** `fraud_model.joblib` usa serialização Python/pickle; carregue apenas artefatos confiáveis.
Evite abrir modelos desconhecidos. Reproduza as versões de dependências do ambiente de treino;
`scikit-learn` registrado: **{bundle.get('sklearn_version', 'não registrado')}**.

## Reprodutibilidade

Código-fonte: [GitHub](https://github.com/GuiMM27/Dashboard-Pessoal-Analise-de-Dados-Fraude-).
Para reproduzir a análise, obtenha o dataset diretamente no Kaggle e execute os scripts
`python -m src.train` e `python -m src.diagnostics` no repositório.

Checksum SHA-256 do modelo: `{sha256}`
'''
    (output / 'README.md').write_text(card, encoding='utf-8')
    print(f'Pacote pronto: {output}')
    print('Conteúdo: README.md, fraud_model.joblib, example_input.json, metadata.json, requirements-model.txt')
    print('Nenhum upload foi realizado. Confira o Model Card antes de publicar.')
    return output


if __name__ == '__main__':
    prepare()
