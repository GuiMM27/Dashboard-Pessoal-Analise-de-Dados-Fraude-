# -------------------- Análise de Dados --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

st.title("Análise de Fraudes em Transações Bancárias")

# -------------------- Loader --------------------
with st.spinner("Carregando dados..."):
    url = "https://drive.google.com/uc?id=1NdnQDFrrb30ILs0i8wvnht-jqoSAlmTC"
    git add .df = pd.read_csv(url)

st.success(f"Dataset carregado com {len(df):,} linhas e {df.shape[1]} colunas.")

# -------------------- Apresentação dos Dados --------------------
st.markdown("""
### ℹ️ Sobre o Dataset
- **Fonte:** Conjunto de dados de fraudes em cartões de crédito (Kaggle).
- **Variáveis principais:**
  - `Time`: segundos desde a primeira transação registrada.
  - `Amount`: valor da transação.
  - `Class`: 0 = normal, 1 = fraude.
  - `V1–V28`: variáveis resultantes de redução PCA para anonimização.
- **Perguntas que vamos responder:**
  1. Em quais horários do dia ocorrem mais fraudes?
  2. Qual a faixa típica de valores normais das transações?
  3. Existe diferença estatística entre os valores de transações normais e fraudulentas?
""")

# Mostrar tipos de variáveis
st.subheader("📋 Estrutura dos Dados")
st.write(df.dtypes)

# -------------------- Estatísticas Descritivas --------------------
st.subheader("📊 Estatísticas Iniciais")
st.write(df[['Time', 'Amount', 'Class']].describe())

# -------------------- Gráfico 1: Fraudes por hora --------------------
st.subheader("1️⃣ Quantidade de Fraudes por Hora do Dia")

# Converter segundos em horas do dia
df['Hour'] = (df['Time'] // 3600) % 24
fraude_por_hora = df.groupby('Hour')['Class'].sum()
total_por_hora = df.groupby('Hour')['Class'].count()
taxa_fraude = fraude_por_hora / total_por_hora * 100  # percentual

# Definir cores baseado em faixas
colors = []
for v in taxa_fraude.values:
    if v < 0.15:
        colors.append('green')
    elif 0.15 <= v <= 0.25:
        colors.append('yellow')
    else:
        colors.append('red')

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(taxa_fraude.index, taxa_fraude.values, color=colors)
ax.set_xticks(range(0,24))
ax.set_xlabel("Hora do dia")
ax.set_ylabel("Taxa de fraude (%)")
ax.set_title("Taxa de Fraude por Hora do Dia")
ax.axhline(0.15, color='black', linestyle='--', linewidth=1.5, label='Limite inferior aceitável')
ax.axhline(0.25, color='black', linestyle='--', linewidth=1.5, label='Limite superior aceitável')
ax.legend()
st.pyplot(fig)

st.write(
    """
    💡 **Interpretação:**  
    - Verde: risco baixo de fraude.  
    - Amarelo: faixa de atenção.  
    - Vermelho: risco crítico que merece monitoramento reforçado.
    """
)

# -------------------- Gráfico 2: Distribuição dos valores --------------------
st.subheader("2️⃣ Faixa Típica de Valores das Transações")

# Estatísticas principais
mean_amount = df['Amount'].mean()
std_amount = df['Amount'].std()
median_amount = df['Amount'].median()
limit_value = mean_amount + 2*std_amount  # limite superior aceitável

st.write(f"Média: {mean_amount:.2f}, Mediana: {median_amount:.2f}, Desvio padrão: {std_amount:.2f}")

# Histograma com escala logarítmica no eixo Y
fig2, ax2 = plt.subplots(figsize=(10,4))
bins = np.linspace(0, df['Amount'].max(), 100)
ax2.hist(df['Amount'], bins=bins, color='skyblue', edgecolor='black')
ax2.axvline(limit_value, color='red', linestyle='--', linewidth=2, label=f'Limite aceitável: {limit_value:.2f}')
ax2.set_xlabel("Valor da transação")
ax2.set_ylabel("Quantidade (escala log)")
ax2.set_yscale("log")
ax2.set_title("Distribuição dos Valores das Transações (Escala Logarítmica)")
ax2.legend()
st.pyplot(fig2)

# -------------------- Correlação --------------------
st.subheader("3️⃣ Correlação entre Variáveis")
corr = df[['Time','Amount','Class']].corr()
fig_corr, ax_corr = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
ax_corr.set_title("Matriz de Correlação")
st.pyplot(fig_corr)

st.write("""
💡 **Interpretação da Correlação:**  
- Valores próximos de **1** = relação positiva forte.  
- Valores próximos de **-1** = relação negativa forte.  
- Valores próximos de **0** = pouca ou nenhuma relação.  
""")

# -------------------- Intervalo de Confiança --------------------
st.subheader("4️⃣ Intervalo de Confiança")

normal_amount = df[df['Class'] == 0]['Amount']
conf_int = stats.t.interval(
    0.95, len(normal_amount)-1, loc=np.mean(normal_amount), scale=stats.sem(normal_amount)
)
st.write(
    f"🔹 Intervalo de Confiança 95% para a média das transações normais: "
    f"{conf_int[0]:.2f} – {conf_int[1]:.2f}"
)

# -------------------- Teste de Hipótese --------------------
st.subheader("5️⃣ Teste de Hipótese: Diferença entre Valores Normais e Fraudulentos")

fraud_amount = df[df['Class']==1]['Amount']

t_stat, p_val = stats.ttest_ind(normal_amount, fraud_amount, equal_var=False)

st.write(f"T-stat: {t_stat:.2f}, p-valor: {p_val:.5f}")
if p_val < 0.05:
    st.success("✅ Existe diferença estatisticamente significativa entre os valores de transações normais e fraudulentas.")
else:
    st.warning("⚠️ Não foi encontrada diferença significativa.")

# -------------------- Respostas & Conclusões --------------------
st.markdown("---")
st.subheader("✅ Respostas às Perguntas & Conclusões")

# horas críticas e de atenção
horas_criticas = taxa_fraude[taxa_fraude > 0.25].index.tolist()
horas_atencao = taxa_fraude[(taxa_fraude >= 0.15) & (taxa_fraude <= 0.25)].index.tolist()
top3_horas = taxa_fraude.sort_values(ascending=False).head(3)

# limiares por valor
p99_normal = np.percentile(normal_amount, 99)  # bem conservador
limite_estatistico = float(limit_value)

st.markdown(f"""
**1) Em quais horários do dia ocorrem mais fraudes?**  
- **Top 3 horas por taxa de fraude (%):** {', '.join([f'{int(h)}h ({top3_horas[h]:.3f}%)' for h in top3_horas.index])}.  
- **Horas críticas (>{25}%):** {', '.join([f'{int(h)}h' for h in horas_criticas]) if horas_criticas else 'Nenhuma acima de 25% na amostra.'}  
- **Faixa de atenção (15%–25%):** {', '.join([f'{int(h)}h' for h in horas_atencao]) if horas_atencao else 'Sem horas na faixa de atenção.'}

**2) Qual é a faixa típica de valores normais?**  
- A maior parte das transações concentra-se em valores baixos.  
- Um **limite estatístico** calculado por **média + 2·desv. padrão** resulta em **~{limite_estatistico:.2f}**.  
- Para ser ainda mais conservador, o **percentil 99** das transações **normais** é **~{p99_normal:.2f}**.

**3) Existe diferença estatística entre valores de transações normais e fraudulentas?**  
- **Resultado do teste t:** *p-valor* = **{p_val:.5f}** ⇒ {"há **diferença significativa**" if p_val < 0.05 else "não há diferença significativa"} entre as médias.  
{"- O sinal do T-stat é negativo, indicando que, em média, transações **fraudulentas tendem a ter valores menores** do que as normais." if t_stat < 0 else ""}

### 🎯 Recomendações para o Banco
- **Monitoramento por horário:** intensificar regras nas horas de maior taxa (top 3 acima).  
- **Confirmação extra por valor:** exigir **validação dupla** para transações **acima de R$ {max(limite_estatistico, p99_normal):.2f}**  
  (limiar definido como o **máximo** entre *média + 2·dp* e *P99* das transações normais).  
- **Política adaptativa:** ajustar automaticamente os limiares por agência/região conforme novas amostras forem chegando.  
- **Observação:** como `Amount` e `Time` têm **baixa correlação** com `Class`, a detecção de fraude realista demanda **modelos multivariados** (usando `V1–V28`) e regras combinadas (valor + horário + perfil do cliente).
""")