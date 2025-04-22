import streamlit as st
from data_processing import load_google_ads_data
from visualization import show_google_ads_summary, show_google_ads_analysis
from analysis import show_benchmark_analysis
from chat_interface import chat_interface

# Configuração inicial
st.set_page_config(layout="wide", page_title="📊 Painel de Análise de Google Ads")

# Inicialização do session_state
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("📊 Painel de Análise de Google Ads")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("Carregue seu relatório do Google Ads (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    # Processa o arquivo carregado
    if uploaded_file and st.session_state.df_raw is None:
        with st.spinner("Processando dados do Google Ads..."):
            try:
                st.session_state.df_raw, st.session_state.df_clean = load_google_ads_data(uploaded_file)
                if st.session_state.df_clean is not None:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Relatório do Google Ads carregado com sucesso! {len(st.session_state.df_clean)} campanhas encontradas."
                    })
            except Exception as e:
                st.error(str(e))
    
    # Abas principais
    tab1, tab2, tab3 = st.tabs(["📈 Análise de Campanhas", "📊 Benchmark & Variabilidade", "💬 Chatbot Especializado"])
    
    with tab1:
        if st.session_state.df_clean is not None:
            show_google_ads_summary(st.session_state.df_clean)
            show_google_ads_analysis(st.session_state.df_clean)
        else:
            st.info("Por favor, carregue um relatório do Google Ads para começar a análise.")
    
    with tab2:
        if st.session_state.df_clean is not None:
            show_benchmark_analysis(st.session_state.df_clean)
        else:
            st.info("Por favor, carregue um relatório do Google Ads para análise de benchmark.")
    
    with tab3:
        chat_interface()

if __name__ == "__main__":
    main()
