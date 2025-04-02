import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import StringIO
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configuração inicial
load_dotenv()
st.set_page_config(layout="wide", page_title="📊 Painel de Análise com Chatbot")

# Configuração do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Inicialização do session_state
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Funções de limpeza e análise (mantidas iguais)
def clean_currency(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = re.sub(r'[^\d,-]', '', value)
        value = value.replace('.', '').replace(',', '.').strip()
        if value == '-' or value == '':
            return np.nan
        try:
            return float(value)
        except:
            return np.nan
    return float(value) if pd.notna(value) else np.nan

def clean_dataframe(df):
    money_cols = [col for col in df.columns 
                 if any(word in col.lower() for word in ['invest', 'valor', 'previst', 'realizado', 'total', 'orcamento'])]
    
    for col in money_cols:
        df[col] = df[col].apply(clean_currency)
    
    date_cols = [col for col in df.columns 
                if any(word in col.lower() for word in ['data', 'date', 'periodo'])]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
    
    return df

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado")
            return None, None
        
        return df.copy(), clean_dataframe(df)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None, None

# Funções de análise (mantidas iguais)
def show_data_summary(df):
    st.subheader("Resumo dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primeiras linhas:**")
        st.dataframe(df.head())
    with col2:
        st.write("**Informações básicas:**")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

def show_column_analysis(df, column):
    st.subheader(f"Análise da coluna: {column}")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Estatísticas descritivas:**")
        st.write(df[column].describe())
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write("**Distribuição:**")
            fig = px.histogram(df, x=column, nbins=20)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("**Valores únicos:**")
        st.write(df[column].value_counts().head(10))
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            st.write("**Série temporal:**")
            time_series = df.set_index(column).resample('D').size()
            fig = px.line(time_series, title="Contagem ao longo do tempo")
            st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(df):
    st.subheader("Análise de Correlação")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Não há colunas numéricas suficientes para análise de correlação")

# Funções do chatbot
def generate_ai_response(prompt, df):
    try:
        # Prepara o contexto para o Gemini
        context = f"""
        Você é um assistente de análise de dados. Aqui está uma amostra dos dados:
        {df.head().to_string()}
        
        Colunas disponíveis: {', '.join(df.columns)}
        Tipos de dados: {df.dtypes.to_string()}
        
        Pergunta: {prompt}
        
        Por favor, responda de forma clara e técnica, sugerindo visualizações quando apropriado.
        """
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def chat_interface():
    st.subheader("💬 Chatbot de Análise de Dados")
    
    # Exibir histórico do chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gerar resposta
        with st.spinner("Analisando..."):
            if st.session_state.df_clean is None:
                response = "Por favor, carregue um arquivo de dados primeiro."
            else:
                response = generate_ai_response(prompt, st.session_state.df_clean)
            
            # Adicionar resposta ao histórico
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                st.markdown(response)
        
        st.rerun()

# Interface principal
def main():
    st.title("📊 Painel de Análise com Chatbot")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("Carregue seu arquivo (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    # Processa o arquivo carregado
    if uploaded_file and st.session_state.df_raw is None:
        with st.spinner("Processando dados..."):
            st.session_state.df_raw, st.session_state.df_clean = load_data(uploaded_file)
            if st.session_state.df_clean is not None:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"✅ Dados carregados com sucesso! {len(st.session_state.df_clean)} linhas × {len(st.session_state.df_clean.columns)} colunas"
                })
    
    # Abas principais
    tab1, tab2 = st.tabs(["📈 Análise de Dados", "💬 Chatbot"])
    
    with tab1:
        if st.session_state.df_clean is not None:
            st.sidebar.title("Opções de Análise")
            analysis_type = st.sidebar.selectbox(
                "Selecione o tipo de análise",
                ["Visão Geral", "Análise de Coluna", "Correlações", "Visualização Personalizada"]
            )
            
            if analysis_type == "Visão Geral":
                show_data_summary(st.session_state.df_clean)
                
                st.subheader("Dados Limpos vs Originais")
                tab1, tab2 = st.tabs(["Dados Limpos", "Dados Originais"])
                
                with tab1:
                    st.dataframe(st.session_state.df_clean)
                
                with tab2:
                    st.dataframe(st.session_state.df_raw)
            
            elif analysis_type == "Análise de Coluna":
                selected_column = st.sidebar.selectbox(
                    "Selecione a coluna para análise",
                    st.session_state.df_clean.columns
                )
                show_column_analysis(st.session_state.df_clean, selected_column)
            
            elif analysis_type == "Correlações":
                show_correlation_analysis(st.session_state.df_clean)
            
            elif analysis_type == "Visualização Personalizada":
                st.subheader("Crie sua própria visualização")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Eixo X", st.session_state.df_clean.columns)
                    y_axis = st.selectbox("Eixo Y", st.session_state.df_clean.columns)
                    color = st.selectbox("Cor", ["Nenhum"] + st.session_state.df_clean.columns.tolist())
                
                with col2:
                    chart_type = st.selectbox("Tipo de Gráfico", ["Barras", "Linha", "Dispersão", "Histograma"])
                    if color == "Nenhum":
                        color = None
                
                if st.button("Gerar Visualização"):
                    try:
                        if chart_type == "Barras":
                            fig = px.bar(st.session_state.df_clean, x=x_axis, y=y_axis, color=color)
                        elif chart_type == "Linha":
                            fig = px.line(st.session_state.df_clean, x=x_axis, y=y_axis, color=color)
                        elif chart_type == "Dispersão":
                            fig = px.scatter(st.session_state.df_clean, x=x_axis, y=y_axis, color=color)
                        elif chart_type == "Histograma":
                            fig = px.histogram(st.session_state.df_clean, x=x_axis, color=color)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao gerar visualização: {str(e)}")
        else:
            st.info("Por favor, carregue um arquivo de dados para começar a análise.")
    
    with tab2:
        chat_interface()

if __name__ == "__main__":
    main()
