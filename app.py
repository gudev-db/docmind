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
            
            st.write("**Métricas Chave**")
            if has_cost and has_clicks and has_impressions:
                fig = px.bar(campaign_data, x='Campaign', y=['Cost', 'Clicks', impressions_col if impressions_col else 'Impr.'], barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados insuficientes para esta visualização")
        else:
            st.warning("Coluna 'Campaign' não encontrada nos dados")
    
    with tab3:
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                time_data = df.groupby('Date')[['Cost', 'Clicks', impressions_col if impressions_col else 'Impr.']].sum().reset_index()
                
                st.write("**Performance ao Longo do Tempo**")
                fig = px.line(time_data, x='Date', y=['Cost', 'Clicks', impressions_col if impressions_col else 'Impr.'], 
                             labels={'value': 'Valor', 'variable': 'Métrica'},
                             title='Métricas ao Longo do Tempo')
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Não foi possível processar dados temporais")
        else:
            st.warning("Dados temporais não disponíveis neste relatório")

def show_google_ads_analysis(df):
    st.subheader("Análise Detalhada")
    
    # Filtros
    st.sidebar.header("Filtros")
    filter_options = []
    
    if 'Campaign type' in df.columns:
        campaign_type = st.sidebar.multiselect("Tipo de Campanha", df['Campaign type'].unique())
        filter_options.append(('Campaign type', campaign_type))
    
    if 'Campaign status' in df.columns:
        status = st.sidebar.multiselect("Status", df['Campaign status'].unique())
        filter_options.append(('Campaign status', status))
    
    # Aplicar filtros
    filtered_df = df.copy()
    for col, values in filter_options:
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    # Métricas de performance
    st.write("### Métricas de Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Eficiência de Custo**")
        cols_to_show = ['Campaign', 'Cost']
        if 'Conversions' in filtered_df.columns: cols_to_show.append('Conversions')
        if 'Conv. value / cost' in filtered_df.columns: cols_to_show.append('Conv. value / cost')
        
        if len(cols_to_show) > 1:
            sort_col = 'Conv. value / cost' if 'Conv. value / cost' in cols_to_show else 'Cost'
            st.dataframe(filtered_df[cols_to_show]
                          .sort_values(sort_col, ascending=False)
                          .style.format({'Cost': 'R$ {:.2f}', 'Conv. value / cost': '{:.2f}'}))
        else:
            st.warning("Dados insuficientes para análise de eficiência de custo")
    
    with col2:
        st.write("**Engajamento**")
        cols_to_show = ['Campaign']
        if 'Impr.' in filtered_df.columns or 'Impressions' in filtered_df.columns: 
            col = 'Impr.' if 'Impr.' in filtered_df.columns else 'Impressions'
            cols_to_show.append(col)
        if 'Clicks' in filtered_df.columns: cols_to_show.append('Clicks')
        if 'Interaction rate' in filtered_df.columns: cols_to_show.append('Interaction rate')
        
        if len(cols_to_show) > 1:
            sort_col = 'Interaction rate' if 'Interaction rate' in cols_to_show else 'Impr.' if 'Impr.' in cols_to_show else 'Impressions'
            st.dataframe(filtered_df[cols_to_show]
                          .sort_values(sort_col, ascending=False)
                          .style.format({'Impr.': '{:,.0f}', 'Impressions': '{:,.0f}', 'Interaction rate': '{:.2%}'}))
        else:
            st.warning("Dados insuficientes para análise de engajamento")
    
    # Visualizações
    st.write("### Visualizações")
    available_numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(available_numeric_cols) > 0:
        chart_type = st.selectbox("Tipo de Gráfico", ["Barras", "Dispersão", "Linha"])
        
        if chart_type == "Barras":
            x_axis = st.selectbox("Eixo X", df.columns)
            y_axis = st.selectbox("Eixo Y", available_numeric_cols)
            color_options = ['Nenhum'] + [col for col in df.columns if df[col].nunique() < 20]
            color = st.selectbox("Cor", color_options)
            
            if color == 'Nenhum':
                color = None
                
            fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Dispersão":
            if len(available_numeric_cols) >= 2:
                x_axis = st.selectbox("Eixo X", available_numeric_cols)
                y_axis = st.selectbox("Eixo Y", available_numeric_cols)
                size = st.selectbox("Tamanho", ['Nenhum'] + available_numeric_cols)
                color_options = ['Nenhum', 'Campaign type', 'Campaign status'] + [col for col in df.columns if df[col].nunique() < 20]
                color = st.selectbox("Cor", color_options)
                
                if size == 'Nenhum':
                    size = None
                if color == 'Nenhum':
                    color = None
                    
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, size=size, color=color,
                                hover_name='Campaign' if 'Campaign' in df.columns else None,
                                hover_data=['Cost', 'Clicks'] if 'Cost' in df.columns and 'Clicks' in df.columns else None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Número insuficiente de colunas numéricas para gráfico de dispersão")
        
        elif chart_type == "Linha":
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                    metrics = st.multiselect("Métricas", available_numeric_cols,
                                           default=['Cost', 'Clicks'] if 'Cost' in available_numeric_cols and 'Clicks' in available_numeric_cols else available_numeric_cols[:2])
                    time_data = filtered_df.groupby('Date')[metrics].sum().reset_index()
                    fig = px.line(time_data, x='Date', y=metrics)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Não foi possível processar dados temporais")
            else:
                st.warning("Dados temporais não disponíveis para gráfico de linha")
    else:
        st.warning("Não há colunas numéricas suficientes para visualizações")

def generate_google_ads_response(prompt, df):
    try:
        # Prepara o contexto para o Gemini
        context = f"""
        Você é um especialista em Google Ads analisando um relatório de campanhas. 
        Aqui está uma amostra dos dados:
        {df.head().to_string()}
        
        Colunas disponíveis: {', '.join(df.columns)}
        """
        
        # Adiciona métricas principais se disponíveis
        metrics_info = []
        if 'Cost' in df.columns:
            metrics_info.append(f"- Total gasto: R$ {df['Cost'].sum():,.2f}")
        if 'Impr.' in df.columns or 'Impressions' in df.columns:
            col = 'Impr.' if 'Impr.' in df.columns else 'Impressions'
            metrics_info.append(f"- Total de impressões: {df[col].sum():,.0f}")
        if 'Clicks' in df.columns:
            metrics_info.append(f"- Total de cliques: {df['Clicks'].sum():,.0f}")
        if ('Impr.' in df.columns or 'Impressions' in df.columns) and 'Clicks' in df.columns:
            impressions_col = 'Impr.' if 'Impr.' in df.columns else 'Impressions'
            total_impressions = df[impressions_col].sum()
            if total_impressions > 0:
                ctr = (df['Clicks'].sum() / total_impressions) * 100
                metrics_info.append(f"- CTR médio: {ctr:.2f}%")
        
        if metrics_info:
            context += "\nMétricas principais:\n" + "\n".join(metrics_info)
        
        context += f"""
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
