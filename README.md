# Dashboard Profissional + Análise de Dados (Detecção de Fraudes)

Projeto Streamlit multi-páginas para a CP1.

## Estrutura
```
creditcard-fraud-dashboard/
├─ Home.py
├─ pages/
│  ├─ 1_📚_Formacao_e_Experiencia.py
│  ├─ 2_🛠️_Skills.py
│  └─ 3_📊_Analise_de_Dados.py
├─ utils/
│  └─ analysis_utils.py
├─ data/
│  └─ creditcard.csv              
├─ requirements.txt
├─ README.md
```
> Dica: você pode usar **upload de arquivo** diretamente pela aba *Análise de Dados* caso não queira salvar o CSV em `data/`.

## Passo a passo para rodar localmente
1. Instale as dependências (de preferência em um ambiente virtual):
   ```bash
   pip install -r requirements.txt
   ```
2. Baixe o dataset do Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e extraia o arquivo `creditcard.csv`.
3. Coloque o `creditcard.csv` em `data/` **ou** suba o arquivo pela interface na aba *Análise de Dados*.
4. Rode o dashboard:
   ```bash
   streamlit run Home.py
   ```

## Deploy (Streamlit Community Cloud)
1. Suba este projeto para um repositório no GitHub.
2. Acesse https://share.streamlit.io e conecte ao seu repositório.
3. Selecione `Home.py` como entrypoint. O `requirements.txt` já está pronto.
4. Clique em **Deploy** e compartilhe o link (vale ponto extra).

## O que contém
- **Home**: apresentação pessoal e objetivo profissional.
- **Formação e Experiência**: cursos, certificações e experiências.
- **Skills**: tecnologias, ferramentas e soft skills.
- **Análise de Dados (obrigatória)**: 
  - Apresentação do conjunto de dados e tipos de variáveis
  - Medidas centrais, dispersão, distribuição, correlações
  - Intervalos de confiança (IC) e Testes de hipótese com visualizações e interpretações

## Como personalizar
- Edite os textos das abas *Home*, *Formação e Experiência* e *Skills* para refletir seu perfil.
- Na *Análise de Dados*, você pode ajustar perguntas, filtros e amostragens.
