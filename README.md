# Credit Card Fraud Detection — ML Engineering

[![Python checks](https://github.com/GuiMM27/Dashboard-Pessoal-Analise-de-Dados-Fraude-/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiMM27/Dashboard-Pessoal-Analise-de-Dados-Fraude-/actions/workflows/ci.yml)

Projeto de engenharia de Machine Learning focado em **fraudes raras**. Evoluiu de um dashboard acadêmico para um pipeline reproduzível, análise de erros, explicabilidade global e local, API FastAPI e preparação de modelo publicável no Hugging Face.

- **Dashboard público:** https://fraud-detection-gm.streamlit.app/
- **Modelo no Hugging Face:** https://huggingface.co/GuiMM27/fraud-detection-ml
- **Dados:** [Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Natureza:** estudo educacional; não é sistema financeiro real nem ferramenta de bloqueio de transações.

## Problema

O dataset possui **284.807 transações** com **492 fraudes** (0,1727%). Uma previsão constante de classe legítima teria mais de 99,8% de acurácia e recall de fraude zero. Por isso, avaliamos **PR-AUC, precision, recall, F1 e matriz de confusão**, em vez de depender somente da acurácia.

## Arquitetura

```text
Dataset ULB/Kaggle (local, NÃO versionado)
        |
        +--> src/analysis.py --> reports/summary.json --> app.py (Streamlit)
        |
        +--> src/train.py --> models/fraud_model.joblib (local, ignorado pelo Git)
                   |                      |
                   +--> reports/model_metrics.json
                   +--> src/diagnostics.py --> reports/error_analysis.json
                   |                       --> reports/explainability.json
                   |
                   +--> api/main.py (FastAPI /predict e /explain)
                   |
                   +--> scripts/prepare_hf.py --> hf_export/ (local, ignorado pelo Git)
                                            --> scripts/publish_to_hf.py (upload explícito)
```

O dashboard público usa **apenas relatórios agregados**, sem necessidade de servir o CSV de 144 MB ou o artefato do modelo. A API **não é hospedada pelo Streamlit**: execute-a localmente com seu modelo ou publique-a posteriormente em um serviço de backend separado.

## Resultados do teste: seleção fixa feita na validação

Separação cronológica por `Time`: treino 60%, validação 20%, teste 20%. Modelo selecionado por PR-AUC de validação, limiar selecionado por F1 na validação. **Não houve escolha do modelo com base no teste.**

| Modelo | PR-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,7436 | 0,9434 | 0,6667 | 0,7812 | 50 | 3 | 25 |
| Random Forest | 0,7874 | 0,8730 | 0,7333 | 0,7971 | 55 | 8 | 20 |

**Modelo selecionado pela validação:** Logistic Regression. Uma pontuação de `predict_proba` nesta configuração com `class_weight="balanced"` **não é necessariamente uma probabilidade calibrada de fraude**.

## Análise de erros e explicabilidade

A aba **Diagnóstico de ML** exibe matriz de confusão, erros por faixas de `Amount`, comparação exploratória de limiares **somente na validação** e pesos globais das variáveis. `src/diagnostics.py` gera os relatórios sem incluir linhas individuais ou dados brutos no GitHub.

Os coeficientes da Logistic Regression são medidos no espaço padronizado e ilustram associações **do modelo**, não causas de fraude. `V1`–`V28` são componentes anonimizadas; `Time` não é hora local; `Amount` não declara moeda. A API `/explain` pode fornecer contribuições locais ao logit para a Logistic Regression, com as mesmas limitações.

## Executar localmente

Recomendação: **Python 3.12**, com terminal Git Bash no Windows.

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Baixe o CSV original (144 MB) da fonte acima e coloque-o como `data/creditcard.csv` (nome exato). Ele é ignorado pelo Git.

```bash
python scripts_generate_summary.py
python -m src.train
python -m src.diagnostics
python -m streamlit run app.py
```

O Streamlit mostra **seis abas**, incluindo Diagnóstico de ML. `reports/summary.json`, `reports/model_metrics.json`, `reports/error_analysis.json` e `reports/explainability.json` são agregados versionáveis; não são substituídos automaticamente por um CSV enviado na interface.

### API FastAPI

Em outro terminal, com `.venv` ativado:

```bash
python -m uvicorn api.main:app --reload
```

- Swagger: http://127.0.0.1:8000/docs
- `GET /health`: servidor online
- `GET /ready`: só retorna 200 se o modelo local estiver disponível
- `POST /predict`: score entre 0 e 1, limiar e classe 0/1
- `POST /explain`: contribuições locais para Logistic Regression padronizada

Experimente pelo Swagger usando `examples/transaction.example.json`, ou em outro terminal:

```bash
python -m scripts.request_example
```

O JSON de exemplo é **sintético** e não representa transação autêntica. A API exige 30 variáveis numéricas: `Time`, `V1` a `V28`, `Amount`. Um esquema inválido retorna HTTP 422; se o modelo não existir, retorna HTTP 503.

### Hugging Face

**Modelo publicado:** [GuiMM27/fraud-detection-ml](https://huggingface.co/GuiMM27/fraud-detection-ml). O repositório no Hugging Face contém o artefato treinado, Model Card, metadados, exemplo sintético e versões de dependências. O Hub **não hospeda automaticamente** nossa API FastAPI.

Para reproduzir ou atualizar o pacote localmente (após treinar e avaliar o modelo), execute os comandos a seguir. Não envie o CSV nem tokens para o repositório.

```bash
python -m pip install -r requirements-hf.txt
python -m scripts.prepare_hf
# revisar hf_export/README.md e hf_export/metadata.json
hf auth login
python -m scripts.publish_to_hf --repo-id GuiMM27/fraud-detection-ml
```

O Hub armazena os arquivos do modelo e o **Model Card**; isto não cria automaticamente uma API de inferência online. `hf_export/` é ignorada pelo Git e só inclui arquivos curados. `joblib` usa pickle; abra apenas modelos confiáveis e respeite a versão do `scikit-learn` registrada no artefato. Confira as condições de uso dos dados de origem e seus derivados antes da publicação.

## Estrutura

```text
api/                    # FastAPI e validação Pydantic
src/analysis.py         # Validação e agregados
src/train.py            # Treinamento + seleção pela validação
src/diagnostics.py      # Erros de teste e importância global
scripts/prepare_hf.py   # Prepara pacote publicável local
scripts/publish_to_hf.py # Upload explícito do pacote curado
scripts/request_example.py
examples/transaction.example.json
reports/                 # JSON agregados e auditáveis
models/                  # Artefatos locais, ignorados pelo Git
data/                    # CSV original local, ignorado pelo Git
tests/                   # Testes de análise e API
app.py                   # Streamlit
```

## Limitações, segurança e próximos passos

- Dados históricos anonimizados; não representam transações do iFood ou de uma instituição específica.
- Há 1.081 registros duplicados que ainda requerem investigação; foram preservados na avaliação.
- `Time` mede segundos desde o primeiro registro, não dia nem fuso; `Amount` tem moeda não especificada.
- Há apenas 75 fraudes no teste final, logo existe incerteza estatística relevante.
- Scores não calibrados; decisões de fraude precisam de avaliação de custos, privacidade, justiça, segurança e testes prospectivos.
- Nenhum endpoint de inferência deve ser aberto ao público sem autenticação, rate limiting, proteção de dados e observabilidade.
- Versionamento de dependências é essencial: `joblib` pode executar código ao carregar arquivos não confiáveis.

Para execução operacional real, faltam teste de generalização fora deste dataset, calibração, avaliação financeira, monitoramento de drift e política de revisão humana.

**Licenças e fonte:** o CSV bruto não é redistribuído. Verifique diretamente as condições de uso do dataset ULB/Kaggle antes de reutilizá-lo ou distribuir derivados em outro contexto.
