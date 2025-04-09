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
st.set_page_config(layout="wide", page_title="📊 Painel de Análise de Google Ads")

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

# Funções de limpeza específicas para relatórios do Google Ads
def clean_google_ads_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove caracteres não numéricos exceto ponto e vírgula
        value = re.sub(r'[^\d,.-]', '', value)
        # Remove pontos como separadores de milhar
        value = value.replace('.', '').replace(',', '.').strip()
        if value == '-' or value == '':
            return np.nan
        try:
            return float(value)
        except:
            return np.nan
    return float(value) if pd.notna(value) else np.nan

def clean_google_ads_data(df):
    # Identifica colunas que provavelmente contêm valores monetários ou porcentagens
    money_cols = [col for col in df.columns 
                 if any(word in col.lower() for word in ['cost', 'cpm', 'cpc', 'cpv', 'budget', 'value', 'rate'])]
    
    for col in money_cols:
        df[col] = df[col].apply(clean_google_ads_value)
    
    # Limpeza de colunas específicas
    if 'Interactions' in df.columns:
        df['Interactions'] = df['Interactions'].str.replace(',', '').astype(float)
    
    if 'Impr.' in df.columns:
        df['Impr.'] = df['Impr.'].str.replace(',', '').astype(float)
    
    return df

def load_google_ads_data(uploaded_file):
    try:
        # Lê o arquivo ignorando as duas primeiras linhas
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='utf-8')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=2)
        else:
            st.error("Formato de arquivo não suportado. Por favor, carregue um CSV ou Excel.")
            return None, None
        
        return df.copy(), clean_google_ads_data(df)
    except Exception as e:
        st.error(f"Erro ao carregar os dados do Google Ads: {str(e)}")
        return None, None

# Funções de análise específicas para Google Ads
def show_google_ads_summary(df):
    st.subheader("Resumo do Relatório Google Ads")
    
    # Métricas principais
    st.write("### Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gasto", f"R$ {df['Cost'].sum():,.2f}")
    
    with col2:
        st.metric("Total de Impressões", f"{df['Impr.'].sum():,.0f}")
    
    with col3:
        avg_cpc = df['Cost'].sum() / df['Clicks'].sum() if df['Clicks'].sum() > 0 else 0
        st.metric("CPC Médio", f"R$ {avg_cpc:,.2f}")
    
    with col4:
        ctr = (df['Clicks'].sum() / df['Impr.'].sum()) * 100 if df['Impr.'].sum() > 0 else 0
        st.metric("CTR", f"{ctr:.2f}%")
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Por Campanha", "Performance Temporal"])
    
    with tab1:
        st.write("**Top Campanhas por Gasto**")
        top_campaigns = df.groupby('Campaign')[['Cost', 'Clicks', 'Impr.']].sum().sort_values('Cost', ascending=False).head(10)
        st.dataframe(top_campaigns.style.format({'Cost': 'R$ {:.2f}', 'Impr.': '{:,.0f}'}))
        
        st.write("**Distribuição de Gasto por Tipo de Campanha**")
        fig = px.pie(df, values='Cost', names='Campaign type', title='Gasto por Tipo de Campanha')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        campaign = st.selectbox("Selecione uma Campanha", df['Campaign'].unique())
        campaign_data = df[df['Campaign'] == campaign]
        
        st.write(f"**Performance da Campanha: {campaign}**")
        st.dataframe(campaign_data[['Cost', 'Clicks', 'Impr.', 'Avg. CPC', 'Interaction rate']])
        
        st.write("**Métricas Chave**")
        fig = px.bar(campaign_data, x='Campaign', y=['Cost', 'Clicks', 'Impr.'], barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            time_data = df.groupby('Date')[['Cost', 'Clicks', 'Impr.']].sum().reset_index()
            
            st.write("**Performance ao Longo do Tempo**")
            fig = px.line(time_data, x='Date', y=['Cost', 'Clicks', 'Impr.'], 
                         labels={'value': 'Valor', 'variable': 'Métrica'},
                         title='Métricas ao Longo do Tempo')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados temporais não disponíveis neste relatório")

def show_google_ads_analysis(df):
    st.subheader("Análise Detalhada")
    
    # Filtros
    st.sidebar.header("Filtros")
    campaign_type = st.sidebar.multiselect("Tipo de Campanha", df['Campaign type'].unique())
    status = st.sidebar.multiselect("Status", df['Campaign status'].unique())
    
    # Aplicar filtros
    filtered_df = df.copy()
    if campaign_type:
        filtered_df = filtered_df[filtered_df['Campaign type'].isin(campaign_type)]
    if status:
        filtered_df = filtered_df[filtered_df['Campaign status'].isin(status)]
    
    # Métricas de performance
    st.write("### Métricas de Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Eficiência de Custo**")
        st.dataframe(filtered_df[['Campaign', 'Cost', 'Conversions', 'Conv. value / cost']]
                    .sort_values('Conv. value / cost', ascending=False)
                    .style.format({'Cost': 'R$ {:.2f}', 'Conv. value / cost': '{:.2f}'}))
    
    with col2:
        st.write("**Engajamento**")
        st.dataframe(filtered_df[['Campaign', 'Impr.', 'Clicks', 'Interaction rate']]
                    .sort_values('Interaction rate', ascending=False)
                    .style.format({'Impr.': '{:,.0f}', 'Interaction rate': '{:.2%}'}))
    
    # Visualizações
    st.write("### Visualizações")
    chart_type = st.selectbox("Tipo de Gráfico", ["Barras", "Dispersão", "Linha"])
    
    if chart_type == "Barras":
        x_axis = st.selectbox("Eixo X", df.columns)
        y_axis = st.selectbox("Eixo Y", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
        fig = px.bar(filtered_df, x=x_axis, y=y_axis, color='Campaign type')
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Dispersão":
        x_axis = st.selectbox("Eixo X", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
        y_axis = st.selectbox("Eixo Y", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
        size = st.selectbox("Tamanho", ['None'] + [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
        color = st.selectbox("Cor", ['None', 'Campaign type', 'Campaign status'] + df.columns.tolist())
        
        if size == 'None':
            size = None
        if color == 'None':
            color = None
            
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, size=size, color=color,
                        hover_name='Campaign', hover_data=['Cost', 'Clicks'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Linha":
        if 'Date' in df.columns:
            metrics = st.multiselect("Métricas", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                                   default=['Cost', 'Clicks'])
            time_data = filtered_df.groupby('Date')[metrics].sum().reset_index()
            fig = px.line(time_data, x='Date', y=metrics)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados temporais não disponíveis para gráfico de linha")

# Funções do chatbot
def generate_google_ads_response(prompt, df):
    try:
        # Prepara o contexto para o Gemini
        context = f"""
        Você é um especialista em Google Ads analisando um relatório de campanhas. 
        Aqui está uma amostra dos dados:
        {df.head().to_string()}
        
        Colunas disponíveis: {', '.join(df.columns)}
        Métricas principais:
        - Total gasto: R$ {df['Cost'].sum():,.2f}
        - Total de impressões: {df['Impr.'].sum():,.0f}
        - Total de cliques: {df['Clicks'].sum():,.0f}
        - CTR médio: {(df['Clicks'].sum() / df['Impr.'].sum() * 100 if df['Impr.'].sum() > 0 else 0):.2f}%
        
        Pergunta: {prompt}
        
        Responda de forma técnica, focando em métricas de performance, eficiência de custo e sugestões de otimização.
        Inclua números específicos quando relevante.
        """
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def chat_interface():
    st.subheader("💬 Chatbot de Análise de Google Ads")
    
    # Exibir histórico do chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta sobre os dados do Google Ads..."):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gerar resposta
        with st.spinner("Analisando..."):
            if st.session_state.df_clean is None:
                response = "Por favor, carregue um relatório do Google Ads primeiro."
            else:
                response = generate_google_ads_response(prompt, st.session_state.df_clean)
            
            # Adicionar resposta ao histórico
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                st.markdown(response)
        
        st.rerun()

# Interface principal
def main():
    st.title("📊 Painel de Análise de Google Ads")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("Carregue seu relatório do Google Ads (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    
    # Processa o arquivo carregado
    if uploaded_file and st.session_state.df_raw is None:
        with st.spinner("Processando dados do Google Ads..."):
            st.session_state.df_raw, st.session_state.df_clean = load_google_ads_data(uploaded_file)
            if st.session_state.df_clean is not None:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"✅ Relatório do Google Ads carregado com sucesso! {len(st.session_state.df_clean)} campanhas encontradas."
                })
    
    # Abas principais
    tab1, tab2 = st.tabs(["📈 Análise de Campanhas", "💬 Chatbot Especializado"])
    
    with tab1:
        if st.session_state.df_clean is not None:
            show_google_ads_summary(st.session_state.df_clean)
            show_google_ads_analysis(st.session_state.df_clean)
        else:
            st.info("Por favor, carregue um relatório do Google Ads para começar a análise.")
    
    with tab2:
        chat_interface()

if __name__ == "__main__":
    main()
