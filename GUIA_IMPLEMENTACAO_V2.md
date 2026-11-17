# Guia de implementação — Fraud Detection V2 (Windows, VS Code, Git Bash)

**Pré-condição:** aplicação V1 já publicada, branch `main` integrada e ambiente Python 3.12 em funcionamento.

## Parte 1 — Criar uma branch SEM afetar o site atual

Abra o VS Code na pasta `Dashboard-Pessoal-Analise-de-Dados-Fraude-`.

```bash
git status
git switch main
git pull origin main
git switch -c feature/ml-diagnostics-api-hf
```

> Se `git status` mostrar alterações locais que você não reconhece, pare e revise antes de trocar de branch. Não apague `.git`, `data/creditcard.csv`, `models/` ou `.venv/`.

## Parte 2 — Copiar os arquivos

Extraia `fraud-detection-v2-overlay.zip`. Entre na pasta `fraud-detection-v2-overlay` e copie **seu conteúdo** para a raiz do repositório clonado, substituindo `app.py`, `README.md`, `requirements-dev.txt`, `.gitignore` e `src/train.py`. Inclua as pastas `api`, `src`, `scripts`, `tests`, `reports`, `examples` e os novos `requirements-api.txt` e `requirements-hf.txt`.

**Não copie a pasta extraída como uma subpasta do repositório.** Não mexa no `.git` original. Não sobrescreva ou elimine `models/fraud_model.joblib` local.

## Parte 3 — Instalar e testar

```bash
source .venv/Scripts/activate
python --version
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

A versão enviada foi validada com **8 testes aprovados** (4 antigos + 4 novos). Se houver erro, não faça push ainda.

## Parte 4 — Reexecutar treinamento e gerar análises

Confirme `data/creditcard.csv` com 144 MB:

```bash
ls -lh data/creditcard.csv
python -m src.train
python -m src.diagnostics
```

O primeiro comando gera o modelo local e `reports/model_metrics.json`.
O segundo analisa erros do teste com limiar já fixado e produz:

- `reports/error_analysis.json`: matriz de confusão, erros por faixas de valores, estudo de limiares na validação.
- `reports/explainability.json`: coeficientes padronizados da Logistic Regression, **não causais**.

Não ajuste o limiar usando o teste. Não coloque dados individuais ou seu CSV em `reports/`.

## Parte 5 — Verificar a sexta aba

```bash
python -m streamlit run app.py
```

Acesse a aplicação local e abra a aba **🔬 Diagnóstico de ML**. Confira matriz de confusão, erros por faixa e influência das variáveis. As outras cinco abas devem permanecer funcionando.

## Parte 6 — Executar e testar a API

Em outro terminal Git Bash na mesma pasta, ative `.venv` novamente:

```bash
source .venv/Scripts/activate
python -m uvicorn api.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` no navegador. Teste:

1. `GET /health`: deve responder 200.
2. `GET /ready`: deve responder 200 se `models/fraud_model.joblib` estiver disponível.
3. `POST /predict`: clique em **Try it out**, cole o conteúdo de `examples/transaction.example.json` e clique em **Execute**.
4. `POST /explain`: utilize o mesmo exemplo para ver as oito maiores contribuições no logit.

Se preferir, com o servidor ainda ligado, execute em um terceiro terminal:

```bash
python -m scripts.request_example
```

**Não exponha `localhost` como URL do portfólio:** a API está apenas no seu computador. O dashboard no Streamlit continuará sendo publicado separadamente.

## Parte 7 — Preparar Hugging Face (revisão antes de publicar)

```bash
python -m pip install -r requirements-hf.txt
python -m scripts.prepare_hf
```

Isso criará `hf_export/` (ignorado pelo Git) com:

```text
hf_export/
├── README.md                # Model Card
├── fraud_model.joblib       # Modelo treinado
├── metadata.json            # Versão, threshold, métricas, SHA-256
├── example_input.json       # Exemplo sintético
└── requirements-model.txt   # Dependências do ambiente de criação
```

Leia o `hf_export/README.md`. Confirme que os dados e métricas estão corretos e que a licença/condições de publicação do dataset permitem o uso pretendido.

**Publicar é uma decisão explícita sua.** Se quiser publicar agora:

1. Crie uma conta em https://huggingface.co/join.
2. Gere um token com permissão de escrita no Hugging Face, se solicitado no login.
3. Faça login **no seu terminal**, sem colocar o token no GitHub nem enviá-lo para o chat:

```bash
hf auth login
```

4. Execute (trocando `SEU_USUARIO` pelo seu usuário real do Hugging Face):

```bash
python -m scripts.publish_to_hf --repo-id SEU_USUARIO/fraud-detection-ml
```

5. Abra `https://huggingface.co/SEU_USUARIO/fraud-detection-ml` e confira a página e os arquivos.

Não envie `creditcard.csv` nem qualquer credencial. A publicação do modelo **não** hospeda automaticamente a API FastAPI.

## Parte 8 — GitHub e deploy Streamlit

Interrompa o Streamlit (`Ctrl+C`). Confira antes de commit:

```bash
git status
git check-ignore -v data/creditcard.csv
git check-ignore -v models/fraud_model.joblib
git check-ignore -v hf_export/fraud_model.joblib
git add -A
git diff --cached --stat
git diff --cached --check
```

Confira que arquivos `data/*.csv`, `models/*.joblib`, `.venv/`, `hf_export/` e tokens não serão enviados. Os arquivos agregados `reports/*.json` **podem** ir ao Git.

```bash
git commit -m "feat: add fraud diagnostics explainability FastAPI and HF export"
git push -u origin feature/ml-diagnostics-api-hf
```

No GitHub, abra um Pull Request com `base: main` e `compare: feature/ml-diagnostics-api-hf`. Confirme testes do GitHub Actions; faça merge somente se aprovados.

Depois do merge, atualize a branch local:

```bash
git switch main
git pull origin main
```

Seu Streamlit Community Cloud ligado à `main` poderá atualizar automaticamente. Verifique `https://fraud-detection-gm.streamlit.app/`, especialmente a **sexta aba**. A API **não** será automaticamente disponibilizada nesse domínio.

## Checklist de qualidade para currículo

- [ ] 8 testes aprovados localmente e no GitHub Actions
- [ ] Nova aba de diagnóstico carregando relatórios agregados
- [ ] `GET /ready` com modelo local e `POST /predict` funcionando
- [ ] `POST /explain` mostrando contribuições sem afirmar causalidade
- [ ] Modelo selecionado por validação, não pelo teste
- [ ] Model Card revisado e publicação HF concluída **se autorizada**
- [ ] Nenhum CSV bruto, modelo binário ou token commitado no GitHub
- [ ] README e demonstrações atualizados
