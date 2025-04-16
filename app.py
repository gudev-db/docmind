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
def preprocess_google_ads_numbers(value):
    """
    Pré-processa valores numéricos do Google Ads onde:
    - Ponto (.) é o separador decimal
    - Vírgula (,) deve ser ignorada (não é separador de milhar)
    """
    if pd.isna(value) or value == '--':
        return '0'
    
    value = str(value).strip()
    
    # Remove todos os caracteres não numéricos exceto ponto, vírgula e sinal negativo
    value = re.sub(r'[^\d\.,-]', '', value)
    
    # Remove vírgulas (não são separadores de milhar no Google Ads)
    value = value.replace(',', '')
    
    # Mantém o ponto como separador decimal
    # Se houver múltiplos pontos, mantém apenas o último como decimal
    parts = value.split('.')
    if len(parts) > 1:
        integer_part = ''.join(parts[:-1])
        decimal_part = parts[-1]
        value = f"{integer_part}.{decimal_part}" if decimal_part else f"{integer_part}"
    
    # Garante que valores vazios sejam zero
    if value == '' or value == '-':
        return '0'
    
    return value

def clean_google_ads_value(value):
    """
    Limpa e converte um valor do Google Ads para float.
    """
    preprocessed = preprocess_google_ads_numbers(value)
    try:
        return float(preprocessed)
    except ValueError:
        return 0.0

def clean_google_ads_data(df):
    """
    Limpa o dataframe do Google Ads, tratando especialmente os formatos numéricos.
    """
    # Identifica colunas que provavelmente contêm valores monetários ou porcentagens
    money_cols = [col for col in df.columns 
                 if any(word in col.lower() for word in ['cost', 'cpm', 'cpc', 'cpv', 'budget', 'value', 'rate', 'amount'])]
    
    # Aplica o pré-processamento a todas as colunas potencialmente numéricas
    for col in df.columns:
        # Verifica se a coluna parece conter números com formatação especial
        if df[col].dtype == 'object' and df[col].astype(str).str.contains(r'[\d\.\,]').any():
            # Pré-processa todos os valores como texto primeiro
            df[col] = df[col].astype(str).apply(preprocess_google_ads_numbers)
            
            # Tenta converter para numérico
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    # Limpeza específica para colunas numéricas com formato especial
    numeric_cols = ['Clicks', 'Impr.', 'Interactions', 'Viewable impr.', 'Conversions', 'Impressions']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove qualquer caractere não numérico e converte
            df[col] = df[col].astype(str).apply(preprocess_google_ads_numbers)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Garante que todas as colunas numéricas tenham valores válidos
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Converte a coluna 'Interaction rate' para float, tratando valores de porcentagem
    if 'Interaction rate' in df.columns:
        df['Interaction rate'] = (
            df['Interaction rate']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '.', regex=False)  # Se for o caso
            .astype(float) / 100
        )
    
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

def show_google_ads_summary(df):
    st.subheader("Resumo do Relatório Google Ads")
    
    # Verifica se as colunas necessárias existem
    has_cost = 'Cost' in df.columns
    has_impressions = 'Impr.' in df.columns or 'Impressions' in df.columns
    has_clicks = 'Clicks' in df.columns
    
    # Métricas principais
    st.write("### Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    # Usa 'Impressions' se 'Impr.' não estiver disponível
    impressions_col = 'Impr.' if 'Impr.' in df.columns else 'Impressions' if 'Impressions' in df.columns else None
    
    total_cost = df['Cost'].sum() if has_cost else 0
    total_impressions = df[impressions_col].sum() if has_impressions else 0
    total_clicks = df['Clicks'].sum() if has_clicks else 0
    
    with col1:
        st.metric("Total Gasto", f"R$ {total_cost:,.2f}".replace('.', '|').replace(',', '.').replace('|', ','))
    
    with col2:
        st.metric("Total de Impressões", f"{total_impressions:,.0f}".replace(',', '.'))
    
    with col3:
        avg_cpc = total_cost / total_clicks if total_clicks > 0 else 0
        st.metric("CPC Médio", f"R$ {avg_cpc:,.2f}".replace('.', '|').replace(',', '.').replace('|', ','))
    
    with col4:
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        st.metric("CTR", f"{ctr:.2f}%".replace('.', ','))
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Por Campanha", "Performance Temporal"])
    
    with tab1:
        st.write("**Top Campanhas por Gasto**")
        if has_cost:
            top_campaigns = df.groupby('Campaign')[['Cost', 'Clicks', impressions_col if impressions_col else 'Impr.']].sum().sort_values('Cost', ascending=False).head(10)
            st.dataframe(top_campaigns.style.format({'Cost': 'R$ {:.2f}', 'Impr.': '{:,.0f}'}))
        else:
            st.warning("Dados de custo não disponíveis")
        
        st.write("**Distribuição de Gasto por Tipo de Campanha**")
        if 'Campaign type' in df.columns and has_cost:
            fig = px.pie(df, values='Cost', names='Campaign type', title='Gasto por Tipo de Campanha')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados insuficientes para esta visualização")
    
    with tab2:
        if 'Campaign' in df.columns:
            campaign = st.selectbox("Selecione uma Campanha", df['Campaign'].unique())
            campaign_data = df[df['Campaign'] == campaign]
            
            st.write(f"**Performance da Campanha: {campaign}**")
            cols_to_show = []
            if has_cost: cols_to_show.append('Cost')
            if has_clicks: cols_to_show.append('Clicks')
            if has_impressions: cols_to_show.append(impressions_col if impressions_col else 'Impr.')
            if 'Avg. CPC' in df.columns: cols_to_show.append('Avg. CPC')
            if 'Interaction rate' in df.columns: cols_to_show.append('Interaction rate')
            
            if cols_to_show:
                st.dataframe(campaign_data[cols_to_show])
            else:
                st.warning("Dados insuficientes para mostrar detalhes da campanha")
            
            st.write("**Performance por Data**")
            if 'Date' in df.columns:
                df_campaign = campaign_data.groupby('Date')[cols_to_show].sum()
                st.dataframe(df_campaign)
        
        else:
            st.warning("Nenhuma campanha encontrada.")
    
    with tab3:
        if 'Date' in df.columns:
            st.write("**Performance Temporal**")
            fig = px.line(df, x='Date', y=['Cost', 'Impressions', 'Clicks'], title="Performance Temporal")
            st.plotly_chart(fig, use_container_width=True)

def show_google_ads_analysis(df):
    """
    Exibe uma análise dos dados de Google Ads com formatação
    """
    st.subheader("Análise Completa")
    
    # Adicionando formatação percentual na coluna de taxas
    display_df = df.style.format({
        'Cost': 'R$ {:.2f}',
        'CTR': '{:.2%}',
        'Interaction rate': '{:.2%}'  # Aqui garantimos o formato correto
    })
    
    st.dataframe(display_df)

# Carregamento e processamento de arquivo
uploaded_file = st.file_uploader("Carregue seu arquivo do Google Ads", type=["csv", "xlsx"])
if uploaded_file:
    df_raw, df_clean = load_google_ads_data(uploaded_file)
    if df_clean is not None:
        show_google_ads_summary(df_clean)
        show_google_ads_analysis(df_clean)
