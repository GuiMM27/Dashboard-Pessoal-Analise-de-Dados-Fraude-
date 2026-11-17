# Credit Card Fraud Detection — Data Analysis & ML Engineering

Projeto independente de Engenharia de IA, com análise exploratória, treinamento reproduzível de modelos de Machine Learning e visualização de resultados. É uma evolução de um projeto acadêmico anterior, agora focada exclusivamente no problema de detecção de fraudes.

## Problema

O dataset possui 284.807 transações e apenas 492 rotuladas como fraude (aprox. 0,173%). Um classificador que sempre responde "legítima" alcançaria mais de 99,8% de acurácia e recall de 0%. Por isso, a avaliação utiliza PR-AUC, precision, recall, F1, ROC-AUC e matriz de confusão.

## Perguntas investigadas

1. Qual a proporção de transações fraudulentas?
2. Como a taxa de fraude varia ao longo do tempo **relativo** presente no arquivo?
3. Como média, mediana e distribuição de `Amount` diferem entre classes?
4. Quais modelos de ML conseguem identificar fraudes raras melhor do que um baseline?
5. Qual trade-off entre falsos positivos e fraudes não detectadas, para diferentes limiares?

## Limitações importantes

- `V1`–`V28` são componentes anonimizados. Não permitem atribuir uma causa comercial direta (cidade, lojista etc.).
- `Time` representa **segundos decorridos desde a primeira transação**, e não fornece dia, fuso horário ou hora real local.
- `Amount` não explicita moeda. Nenhum gráfico atribui valores em reais.
- Há linhas duplicadas, que foram preservadas nesta versão até investigação do processo que as gerou.
- Resultados são históricos e demonstrativos, não um detector de fraude pronto para uso financeiro.
- Dados históricos disponíveis publicamente não representam distribuição de transações de uma empresa específica.

## Estrutura

```text
app.py                        # Interface Streamlit, sem perfil pessoal
src/analysis.py               # Validação, análise e agregações
src/train.py                  # Baseline + Logistic Regression + Random Forest
scripts_generate_summary.py   # Regenera estatísticas públicas
reports/summary.json          # Agregados reais para demonstração no cloud
reports/model_metrics.json    # Gerado após executar treinamento
models/fraud_model.joblib      # Gerado localmente, ignorado pelo Git
data/creditcard.csv            # Adicionado localmente; ignorado pelo Git
tests/test_analysis.py         # Testes automatizados
```

## Como executar

Recomendação: Python 3.11 ou 3.12.

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

O dashboard funciona com o relatório agregado disponível em `reports/summary.json` mesmo sem o CSV completo. Para recalcular a análise ou treinar, baixe o dataset original e salve em `data/creditcard.csv`. Se o seu arquivo tiver outro nome, renomeie uma cópia para esse caminho. O CSV **não deve ser commitado**, pois tem aproximadamente 144 MB.

```bash
python scripts_generate_summary.py
python -m src.train
```

Treinamento: divisão cronológica 60/20/20 (treino/validação/teste) com `Time`. O melhor modelo é escolhido por PR-AUC em validação. O limiar é escolhido usando F1 na validação; depois as métricas são calculadas no teste. O teste não influencia seleção de modelo nem limiar. O método é experimental e deverá ser comparado com uma divisão estratificada e auditoria de duplicatas posteriormente.

## Resultados iniciais no conjunto de teste

A execução inicial com divisão cronológica produziu estes resultados; escolha do modelo baseada exclusivamente na validação:

| Modelo | PR-AUC (teste) | Precision | Recall | F1 | Falsos positivos | Fraudes não detectadas |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,7436 | 0,9434 | 0,6667 | 0,7813 | 3 | 25 |
| Random Forest (24 árvores) | 0,7874 | 0,8730 | 0,7333 | 0,7971 | 8 | 20 |

O modelo selecionado pela PR-AUC de validação foi a Logistic Regression, apesar de a Random Forest ter superado esse resultado específico no teste. Não se deve reescolher o modelo após observar o teste. Estas métricas são experimentais e requerem auditoria adicional sobre a independência das observações, generalização e calibração.

## Próximas entregas

- Análise de falsos positivos/falsos negativos e incerteza estatística.
- Explicação das previsões em termos das componentes anonimizadas (não de causas comerciais).
- API de inferência com FastAPI, validação de esquema e testes de contrato.
- Publicação de modelo e Model Card no Hugging Face Hub.
- CI com GitHub Actions e documentação de arquitetura.

## Fonte dos dados

Dataset público [Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), contendo transações anonimizadas; os arquivos de transações completos não são redistribuídos neste repositório. Consulte os termos da fonte antes de republicar dados ou modelos.
