# 📊 Dashboard Profissional & Análise de Fraudes

Este projeto é uma aplicação interativa construída em **Streamlit**. Ele funciona como um portfólio dinâmico (apresentando perfil, experiências e habilidades) e integra uma robusta Análise Exploratória de Dados (EDA) focada em **Detecção de Fraudes em Cartões de Crédito**.

## 🚀 Funcionalidades da Aplicação

O Dashboard é dividido em 4 páginas principais:
- **🏠 Home:** Apresentação pessoal e objetivo profissional.
- **📚 Formação e Experiência:** Histórico acadêmico e trajetória profissional.
- **🛠️ Skills:** Hard e soft skills, tecnologias e ferramentas de domínio.
- **📈 Análise de Dados:** Exploração de um dataset real de transações financeiras, contendo:
  - Análise de distribuições e variáveis.
  - Correlações e medidas de dispersão.
  - Testes de Hipótese e cálculo de Intervalos de Confiança (IC).

## 📂 Estrutura do Projeto

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
│  └─ creditcard.csv (Necessário download manual)             
├─ requirements.txt
├─ README.md
```

## ⚙️ Como executar localmente
Siga os passos abaixo para rodar a aplicação em sua máquina:

Clone o repositório:
```
git clone [https://github.com/SeuUsuario/creditcard-fraud-dashboard.git](https://github.com/SeuUsuario/creditcard-fraud-dashboard.git)
cd creditcard-fraud-dashboard
```
Instale as dependências (recomenda-se o uso de um ambiente virtual - venv):

```
pip install -r requirements.txt
```
Baixe os dados:

**Acesse o Dataset no Kaggle.**

**Extraia o arquivo creditcard.csv.**

**Coloque o arquivo dentro da pasta data/ do projeto (ou faça o upload do arquivo diretamente pela interface da aplicação na aba de Análise).**

Inicie o servidor local do Streamlit:

```
streamlit run Home.py
```
## 🌐 Deploy

O projeto está configurado para fácil deploy na nuvem via Streamlit Community Cloud. Basta conectar este repositório, selecionar Home.py como entrypoint e o ambiente instalará automaticamente as bibliotecas contidas no requirements.txt.
