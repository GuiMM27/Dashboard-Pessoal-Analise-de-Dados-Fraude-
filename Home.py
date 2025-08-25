
import streamlit as st

def add_vertical_space(num_lines: int = 1):
    for _ in range(num_lines):
        st.write("")


st.set_page_config(page_title="Dashboard Profissional | Detecção de Fraudes", page_icon="🧠", layout="wide")

st.title("🧠 Dashboard Pessoal & Análise de Dados")
st.markdown(
    """
    Bem-vindo! Este dashboard foi construído para a **CP1** com duas metas:
    1) Apresentar meu **perfil profissional** 
    
    2) Realizar uma **Análise de Dados**
    sobre o tema **Detecção de Fraudes em Cartões**.
    """
)

with st.container():
        col1, col2 = st.columns([1,2], gap="large")
        with col1:st.subheader("Quem sou eu")
        with col1 :st.write("""
        *Guilherme Machado Moreira* — Estudante de **Engenharia de Software** na FIAP
        """ )
        with col1 :st.subheader("Objetivo Profissional")
        with col1 :st.write("""
        Busco oportunidade de estágio em Desenvolvimento Backend ou Fullstack.
        """ )
        with col2:st.image("fotoarco.jpg", width=300)
        st.write("📍 São Paulo, SP, Brasil")
        st.write("✉️ guimm2013@gmail.com| www.linkedin.com/in/gmoreira27")
        

add_vertical_space(2)
st.info("Use o menu lateral para navegar pelas páginas: Formação, Skills e a Análise de Dados.")
