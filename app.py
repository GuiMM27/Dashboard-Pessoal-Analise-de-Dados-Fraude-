"""Streamlit: fraud analysis only, with real precomputed aggregates.

The public application works without the 144 MB CSV: reports/summary.json was
created from the actual dataset. When the complete CSV is available locally or
uploaded, the same aggregates can be recomputed on demand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import summarize_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Fraud Detection | ML Engineering", page_icon="🔎", layout="wide")
st.title("🔎 Credit Card Fraud — Análise e Machine Learning")
st.caption("Estudo exploratório e experimental com transações anonimizadas. Não é um sistema financeiro de produção.")

@st.cache_data(show_spinner="Analisando transações...")
def load_summary_file() -> dict:
    return json.loads((ROOT / "reports" / "summary.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner="Calculando métricas...")
def summary_from_bytes(content: bytes) -> dict:
    import io
    return summarize_data(pd.read_csv(io.BytesIO(content)))

with st.sidebar:
    st.header("Dados")
    st.write("O painel já contém um resumo agregado calculado sobre os dados reais.")
    upload = st.file_uploader("Opcional: analisar seu CSV completo", type=["csv"])
    st.caption("O CSV é processado nesta sessão. Não publique dados brutos no repositório.")

try:
    if upload is not None:
        summary = summary_from_bytes(upload.getvalue())
        origin = "CSV enviado nesta sessão"
    elif (ROOT / "data" / "creditcard.csv").exists():
        @st.cache_data(show_spinner="Lendo CSV local...")
        def local_summary(path_str: str, mtime: float) -> dict:
            return summarize_data(pd.read_csv(path_str))
        path = ROOT / "data" / "creditcard.csv"
        summary = local_summary(str(path), path.stat().st_mtime)
        origin = "CSV local"
    else:
        summary = load_summary_file()
        origin = "Resumo agregado real (gerado offline)"
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"Erro ao processar os dados: {exc}")
    st.stop()

st.info(f"Fonte atual: **{origin}**. A moeda dos valores não é informada e o tempo é relativo à primeira transação.")

overview, time_tab, values_tab, model_tab, conclusions_tab, diagnostics_tab = st.tabs([
    "📌 Visão geral", "⏱️ Distribuição temporal", "💳 Valores das transações",
    "🧠 Modelos de ML", "📋 Conclusões", "🔬 Diagnóstico de ML"
])

with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de transações", f"{summary['n_rows']:,}")
    c2.metric("Transações fraudulentas", f"{summary['fraud_count']:,}")
    c3.metric("Taxa de fraude", f"{summary['fraud_rate_pct']:.4f}%")
    c4.metric("Registros duplicados", f"{summary['duplicate_rows']:,}")
    classes = pd.DataFrame([
        {"Classe": "Legítimas", "Transações": summary["legitimate_count"]},
        {"Classe": "Fraudes", "Transações": summary["fraud_count"]},
    ])
    st.plotly_chart(px.bar(classes, x="Classe", y="Transações", text="Transações", title="Transações por classe"), use_container_width=True)
    st.write("**Pergunta: qual é o principal desafio analítico?** A classe de fraude é rara. Acurácia isolada pode mascarar modelos que não detectam fraudes.")
    st.caption("Duplicatas são auditadas, não removidas automaticamente: podem representar repetições legítimas do registro.")

with time_tab:
    hour_df = pd.DataFrame(summary["hourly"])
    st.plotly_chart(px.bar(hour_df, x="hour", y="fraud_rate_pct", hover_data=["count", "fraud_count"],
        labels={"hour": "Hora relativa (Time % 24h)", "fraud_rate_pct": "Fraudes (%)"},
        title="Taxa de fraude por hora relativa à primeira observação"), use_container_width=True)
    st.plotly_chart(px.bar(hour_df, x="hour", y=["count", "fraud_count"], barmode="group",
        labels={"hour": "Hora relativa"}, title="Transações totais e fraudes por janela"), use_container_width=True)
    st.warning("Time é medido em segundos desde a primeira transação. Os agrupamentos NÃO representam horários reais de relógio em uma região ou fuso específico.")
    st.caption("Taxas maiores em alguns intervalos não estabelecem, por si só, limiares de risco operacional.")

with values_tab:
    st.subheader("Qual é o perfil dos valores das transações?")
    amt = pd.DataFrame(summary["amount_groups"])
    left, right = st.columns(2)
    left.plotly_chart(px.bar(amt, x="range", y="count", title="Volume de transações por faixa",
        labels={"range": "Amount (unidade não informada)", "count": "Transações"}), use_container_width=True)
    right.plotly_chart(px.bar(amt, x="range", y="fraud_rate_pct", title="Taxa de fraude por faixa",
        hover_data=["fraud_count", "count"], labels={"range": "Amount (unidade não informada)", "fraud_rate_pct": "Fraudes (%)"}), use_container_width=True)
    grp = summary["by_class"]
    st.dataframe(pd.DataFrame([
        {"Classe": "Legítima", "Média": grp["0"]["amount_mean"], "Mediana": grp["0"]["amount_median"], "P90": grp["0"]["amount_p90"]},
        {"Classe": "Fraude", "Média": grp["1"]["amount_mean"], "Mediana": grp["1"]["amount_median"], "P90": grp["1"]["amount_p90"]},
    ]), hide_index=True, use_container_width=True)
    st.write("**Interpretação:** a média e a mediana contam histórias diferentes. Não é correto estabelecer bloqueio automático com base apenas no valor da transação.")

with model_tab:
    st.subheader("Predição de fraude: experimentos reproduzíveis")
    path = ROOT / "reports" / "model_metrics.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = pd.DataFrame(result["models"])
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.caption(result["methodology"])
    else:
        st.write("Nenhum modelo foi treinado nesta cópia. Execute `python -m src.train` com o CSV completo para gerar as métricas.")
    st.info("Os modelos foram avaliados usando PR-AUC, precision, recall, F1 e limiares escolhidos na validação.")
    st.markdown("O conjunto de teste será utilizado apenas ao final; não vamos escolher o limiar com base nele.")


with conclusions_tab:
    st.subheader("Conclusões do estudo e próximos passos")
    st.caption(
        "Síntese dos dados exploratórios exibidos nesta sessão e, quando disponível, "
        "do relatório de modelos salvo após o treinamento."
    )

    st.markdown("### 1. A raridade das fraudes é o desafio central")
    legitimate_rate = 100 - summary["fraud_rate_pct"]
    ratio = summary["n_rows"] / max(summary["fraud_count"], 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Transações analisadas", f"{summary['n_rows']:,}")
    c2.metric("Fraudes identificadas no dataset", f"{summary['fraud_count']:,}")
    c3.metric("Prevalência de fraude", f"{summary['fraud_rate_pct']:.4f}%")
    st.markdown(
        f"Aproximadamente **1 em cada {ratio:,.0f} transações** é fraudulenta. "
        f"Um classificador que previsse sempre a classe legítima teria "
        f"**{legitimate_rate:.2f}% de acurácia**, mas **recall de fraude igual a zero**. "
        "Portanto, acurácia isolada não é adequada para julgar este problema."
    )

    st.markdown("### 2. O valor da transação não basta para identificar uma fraude")
    class_stats = summary["by_class"]
    legit = class_stats["0"]
    fraud = class_stats["1"]
    comparison = pd.DataFrame([
        {
            "Classe": "Legítimas",
            "Média (Amount)": round(legit["amount_mean"], 2),
            "Mediana (Amount)": round(legit["amount_median"], 2),
        },
        {
            "Classe": "Fraudes",
            "Média (Amount)": round(fraud["amount_mean"], 2),
            "Mediana (Amount)": round(fraud["amount_median"], 2),
        },
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.markdown(
        f"Nos dados analisados, a média de `Amount` para fraudes é "
        f"{fraud['amount_mean']:.2f}, contra {legit['amount_mean']:.2f} para "
        f"transações legítimas; as medianas são {fraud['amount_median']:.2f} "
        f"e {legit['amount_median']:.2f}, respectivamente. "
        "Essas estatísticas, isoladamente, não fundamentam bloquear uma "
        "transação apenas por seu valor."
    )
    st.caption(
        "`Amount` não possui moeda especificada. `Time` é contado em segundos "
        "desde a primeira observação e não identifica a hora local do dia."
    )

    st.markdown("### 3. Resultados dos modelos: detecção e erros")
    report_path = ROOT / "reports" / "model_metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        models_by_name = {row["model"]: row for row in report.get("models", [])}
        chosen_name = report.get("selected_model")
        chosen = models_by_name.get(chosen_name)
        if chosen:
            test_frauds = int(chosen["tp"] + chosen["fn"])
            st.markdown(
                f"O modelo escolhido por **PR-AUC no conjunto de validação** foi "
                f"**{chosen_name}**. No conjunto final de teste, com "
                f"**{int(report['n_test']):,} transações e {test_frauds} fraudes**:"
            )
            a, b, c, d = st.columns(4)
            a.metric("Fraudes detectadas (TP)", str(chosen["tp"]))
            b.metric("Fraudes perdidas (FN)", str(chosen["fn"]))
            c.metric("Alertas falsos (FP)", str(chosen["fp"]))
            d.metric("PR-AUC (teste)", f"{chosen['pr_auc']:.3f}")
            st.markdown(
                f"O modelo alcançou **recall de {chosen['recall']:.1%}** e "
                f"**precision de {chosen['precision']:.1%}** no teste. "
                "Isso evidencia um compromisso entre detectar fraudes e gerar falsos alertas."
            )
            forest = models_by_name.get("Random Forest")
            if forest and chosen_name != "Random Forest":
                st.info(
                    f"Na comparação exploratória do teste, o Random Forest detectou "
                    f"{forest['tp']} fraudes, deixou passar {forest['fn']} e gerou "
                    f"{forest['fp']} falsos alertas. Apesar desse resultado, "
                    "não devemos mudar para o Random Forest somente depois "
                    "de examinar o conjunto de teste: a escolha deve permanecer "
                    "baseada na validação."
                )
            st.caption(
                "As métricas acima pertencem ao treinamento registrado em "
                "`reports/model_metrics.json`. Elas NÃO são recalculadas quando "
                "outro CSV é enviado pelo painel."
            )
        else:
            st.warning("O relatório existe, mas não contém o modelo selecionado esperado.")
    else:
        st.info(
            "Ainda não existe relatório de treinamento. Rode `python -m src.train` "
            "para gerar `reports/model_metrics.json` e visualizar as conclusões dos modelos."
        )

    st.markdown("### 4. Limitações e próximos passos")
    st.markdown(
        f"**Limitações:** dados históricos anonimizados; ausência de moeda e "
        f"hora local; {summary['duplicate_rows']:,} registros duplicados nesta análise; "
        "classe muito rara; e resultados que ainda não foram "
        "validados em operação real. Os scores não devem ser tratados como "
        "probabilidades calibradas sem uma avaliação adicional."
    )
    st.markdown(
        "**Evolução:** a análise de erros e a explicabilidade global estão documentadas "
        "na aba Diagnóstico de ML; a API FastAPI pode ser executada localmente. "
        "Próximas validações: calibração, custos de falso positivo/negativo, testes prospectivos "
        "e publicação de um modelo documentado no Hugging Face."
    )
    st.success(
        "Conclusão: o dataset mostra por que detectar eventos raros exige "
        "métricas apropriadas, validação rigorosa e decisões transparentes. "
        "Este projeto é um estudo de ML Engineering, não um mecanismo pronto "
        "para bloquear transações reais."
    )


with diagnostics_tab:
    st.subheader("Análise de erros e explicabilidade do modelo")
    st.caption("Dados agregados do treinamento local. Não carregamos observações individuais na nuvem.")
    error_path = ROOT / "reports" / "error_analysis.json"
    explain_path = ROOT / "reports" / "explainability.json"
    if error_path.exists() and explain_path.exists():
        error = json.loads(error_path.read_text(encoding="utf-8"))
        explanation = json.loads(explain_path.read_text(encoding="utf-8"))
        m = error["test_metrics"]
        st.markdown(f"**Modelo auditado:** {error['selected_model']} | limiar fixado na validação: `{error['threshold_fixed_on_validation']:.6f}`")
        x1, x2, x3 = st.columns(3)
        x1.metric("Fraudes detectadas — TP", m["tp"])
        x2.metric("Fraudes não detectadas — FN", m["fn"])
        x3.metric("Alertas incorretos — FP", m["fp"])
        st.markdown("### Matriz de confusão no teste — decisão já congelada")
        import plotly.graph_objects as go
       
        # Matriz de confusão com cores por tipo de resultado
        matriz = go.Figure(
            data=go.Heatmap(
                z=[
                    [0, 2],
                    [1, 3]
                ],
                x=["Prevista legítima", "Prevista fraude"],
                y=["Real legítima", "Real fraude"],
                text=[
                    [str(m["tn"]), str(m["fp"])],
                    [str(m["fn"]), str(m["tp"])]
                ],
                texttemplate="%{text}",
                textfont=dict(
                    color="white",
                    size=18
                ),
                colorscale=[
                    [0.0, "#17365D"],
                    [0.1666, "#17365D"],
                    [0.1667, "#B53B4A"],
                    [0.4999, "#B53B4A"],
                    [0.5, "#9A5D12"],
                    [0.8333, "#9A5D12"],
                    [0.8334, "#167347"],
                    [1.0, "#167347"]
                ],
                zmin=0,
                zmax=3,
                showscale=False,
                xgap=4,
                ygap=4,
                hovertemplate=(
                    "%{y}<br>"
                    "%{x}<br>"
                    "Quantidade: %{text}"
                    "<extra></extra>"
                )
            )
        )

        matriz.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(
            matriz,
            use_container_width=True
        )

        st.caption(
            "🔵 Azul: legítimas identificadas corretamente | "
            "🟡 Amarelo: falsos positivos | "
            "🔴 Vermelho: fraudes não detectadas | "
            "🟢 Verde: fraudes detectadas corretamente."
        )

        st.markdown("### Em quais faixas de Amount ocorreram os erros?")
        err_df = pd.DataFrame(error["amount_error_buckets_test"])
        st.plotly_chart(px.bar(err_df, x="faixa", y=["FP", "FN"], barmode="group",
                               labels={"faixa": "Amount (moeda desconhecida)", "value": "Erros"}), use_container_width=True)
        st.caption("Números descritivos sobre o teste, sem ajustar o modelo com base neles.")
        st.markdown("### Sensibilidade ao limiar — apenas validação")
        val_options = pd.DataFrame(error["validation_candidates_only"])
        st.dataframe(val_options[["threshold", "precision", "recall", "f1", "tp", "fp", "fn"]],
                     hide_index=True, use_container_width=True)
        st.caption("Valores de referência calculados na validação. O limiar de produção experimental NÃO muda com este gráfico.")
        st.markdown("### Quais variáveis influenciam o modelo?")
        importance = pd.DataFrame(explanation["feature_importances"]).head(12)
        st.plotly_chart(px.bar(importance.iloc[::-1], x="importance", y="feature", orientation="h",
                               labels={"importance": "Peso no estimador (não causal)", "feature": "Variável"}),
                        use_container_width=True)
        st.info(explanation["interpretation"])
        st.warning("V1–V28 são componentes anonimizadas. A importância estatística NÃO identifica causa de negócio. O score ainda NÃO foi calibrado.")
    else:
        st.info("Relatórios ainda não disponíveis nesta versão. Com o CSV e o modelo local, rode `python -m src.diagnostics` e envie os JSON agregados gerados em `reports/` para o GitHub.")

st.divider()
st.caption("Projeto educacional de ML Engineering | Dados históricos anonimizados | Código e resultados experimentais reproduzíveis")
