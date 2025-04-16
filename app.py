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
    
    # First check if dataframe is valid
    if df.empty:
        st.warning("Nenhum dado disponível para análise")
        return

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

    # Helper function for safe formatting
    def safe_format(value, fmt):
        try:
            if pd.isna(value):
                return ""
            if fmt == "currency":
                return f"R$ {float(value):,.2f}"
            elif fmt == "percent":
                return f"{float(value):.2%}"
            elif fmt == "int":
                return f"{int(value):,}"
            else:
                return str(value)
        except:
            return str(value)
    
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
            display_df = filtered_df[cols_to_show].sort_values(sort_col, ascending=False)
            
            # Apply formatting without using styler
            formatted_df = display_df.copy()
            if 'Cost' in formatted_df.columns:
                formatted_df['Cost'] = formatted_df['Cost'].apply(lambda x: safe_format(x, "currency"))
            if 'Conv. value / cost' in formatted_df.columns:
                formatted_df['Conv. value / cost'] = formatted_df['Conv. value / cost'].apply(lambda x: safe_format(x, "float"))
            
            st.dataframe(formatted_df)
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
            display_df = filtered_df[cols_to_show].sort_values(sort_col, ascending=False)
            
            # Apply formatting without using styler
            formatted_df = display_df.copy()
            if 'Impr.' in formatted_df.columns:
                formatted_df['Impr.'] = formatted_df['Impr.'].apply(lambda x: safe_format(x, "int"))
            if 'Impressions' in formatted_df.columns:
                formatted_df['Impressions'] = formatted_df['Impressions'].apply(lambda x: safe_format(x, "int"))
            if 'Interaction rate' in formatted_df.columns:
                formatted_df['Interaction rate'] = formatted_df['Interaction rate'].apply(
                    lambda x: safe_format(x, "percent") if isinstance(x, (int, float)) else str(x)
                )
            
            st.dataframe(formatted_df)
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
        {df.to_string()}
        
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

def show_benchmark_analysis(df):
    st.subheader("📊 Benchmark de Performance vs Médias")
    
    if df is None or df.empty:
        st.warning("Nenhum dado disponível para análise de benchmark")
        return
    
    # Calcula médias gerais
    metrics = {
        'Cost': {'name': 'Gasto', 'format': 'currency'},
        'Impr.': {'name': 'Impressões', 'format': 'int'},
        'Impressions': {'name': 'Impressões', 'format': 'int'},
        'Clicks': {'name': 'Cliques', 'format': 'int'},
        'Conversions': {'name': 'Conversões', 'format': 'int'},
        'Avg. CPC': {'name': 'CPC Médio', 'format': 'currency'},
        'Avg. CPM': {'name': 'CPM Médio', 'format': 'currency'},
        'CTR': {'name': 'CTR', 'format': 'percent'},
        'Interaction rate': {'name': 'Taxa de Interação', 'format': 'percent'}
    }
    
    # Filtra apenas as métricas disponíveis no dataframe
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not available_metrics:
        st.warning("Não há métricas numéricas disponíveis para análise")
        return
    
    # Calcula médias globais
    global_means = {}
    for metric in available_metrics:
        if pd.api.types.is_numeric_dtype(df[metric]):
            global_means[metric] = df[metric].mean()
    
    # Mostra cards com as médias globais
    st.write("### Médias Globais")
    cols = st.columns(min(4, len(available_metrics)))
    
    for idx, (metric, config) in enumerate(available_metrics.items()):
        if metric in global_means:
            with cols[idx % len(cols)]:
                value = global_means[metric]
                if config['format'] == 'currency':
                    display_value = f"R$ {value:,.2f}"
                elif config['format'] == 'percent':
                    display_value = f"{value:.2%}"
                elif config['format'] == 'int':
                    display_value = f"{int(value):,}"
                else:
                    display_value = str(value)
                
                st.metric(config['name'], display_value)
    
    # Análise de campanhas vs média
    st.write("### Campanhas vs Média")
    
    # Seleciona métrica para análise comparativa
    selected_metric = st.selectbox(
        "Selecione a métrica para análise comparativa",
        options=list(available_metrics.keys()),
        format_func=lambda x: available_metrics[x]['name']
    )
    
    if selected_metric:
        mean_value = global_means[selected_metric]
        df_comparison = df.copy()
        
        # Calcula desvio da média
        df_comparison[f'{selected_metric}_vs_mean'] = (df_comparison[selected_metric] - mean_value) / mean_value
        
        # Classifica campanhas
        df_comparison['Performance'] = np.where(
            df_comparison[selected_metric] > mean_value * 1.2, 'Acima da Média',
            np.where(df_comparison[selected_metric] < mean_value * 0.8, 'Abaixo da Média', 'Na Média')
        )
        
        # Mostra distribuição
        st.write(f"**Distribuição de Performance para {available_metrics[selected_metric]['name']}**")
        performance_dist = df_comparison['Performance'].value_counts(normalize=True) * 100
        st.write(performance_dist)
        
        # Gráfico de violino para mostrar distribuição
        fig = px.violin(df_comparison, y=selected_metric, box=True, points="all",
                        title=f"Distribuição de {available_metrics[selected_metric]['name']}")
        fig.add_hline(y=mean_value, line_dash="dash", line_color="red",
                      annotation_text=f"Média: {mean_value:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com campanhas destacadas
        st.write(f"**Campanhas Destacadas - {available_metrics[selected_metric]['name']}**")
        
        # Filtra apenas campanhas significativas (com pelo menos algum gasto ou impressões)
        significant_campaigns = df_comparison[
            (df_comparison['Cost'] > 0) | 
            (df_comparison.get('Impr.', 0) > 0) |
            (df_comparison.get('Impressions', 0) > 0)
        ]
        
        top_campaigns = significant_campaigns.nlargest(5, selected_metric)
        bottom_campaigns = significant_campaigns.nsmallest(5, selected_metric)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🔝 Top Campanhas")
            st.dataframe(
                top_campaigns[['Campaign', selected_metric]].assign(
                    **{f"% vs Média": ((top_campaigns[selected_metric] - mean_value) / mean_value * 100).round(1)}
                ).style.format({
                    selected_metric: '{:,.2f}',
                    '% vs Média': '{:.1f}%'
                })
            )
        
        with col2:
            st.write("🔻 Campanhas com Baixa Performance")
            st.dataframe(
                bottom_campaigns[['Campaign', selected_metric]].assign(
                    **{f"% vs Média": ((bottom_campaigns[selected_metric] - mean_value) / mean_value * 100).round(1)}
                ).style.format({
                    selected_metric: '{:,.2f}',
                    '% vs Média': '{:.1f}%'
                })
            )
    
    # Análise de variabilidade temporal
    if 'Date' in df.columns:
        st.write("### 📈 Análise de Variabilidade Temporal")
        
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Month'] = df['Date'].dt.month
            
            # Seleciona nível de agregação
            time_agg = st.radio(
                "Nível de Agregação Temporal",
                options=['Diário', 'Semanal', 'Mensal'],
                horizontal=True
            )
            
            if time_agg == 'Semanal':
                time_col = 'Week'
                group_cols = ['Week']
            elif time_agg == 'Mensal':
                time_col = 'Month'
                group_cols = ['Month']
            else:
                time_col = 'Date'
                group_cols = ['Date']
            
            # Seleciona métrica para análise temporal
            temporal_metric = st.selectbox(
                "Selecione a métrica para análise temporal",
                options=list(available_metrics.keys()),
                key='temporal_metric',
                format_func=lambda x: available_metrics[x]['name']
            )
            
            if temporal_metric:
                # Calcula métricas agregadas
                temporal_df = df.groupby(group_cols).agg({
                    temporal_metric: ['mean', 'std', 'count']
                }).reset_index()
                
                # Renomeia colunas
                temporal_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in temporal_df.columns]
                
                # Calcula coeficiente de variação
                temporal_df['CV'] = (temporal_df[f'{temporal_metric}_std'] / temporal_df[f'{temporal_metric}_mean']) * 100
                
                # Mostra métricas de variabilidade
                st.write("**Variabilidade Temporal**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_cv = temporal_df['CV'].mean()
                    st.metric("Coeficiente de Variação Médio", f"{avg_cv:.1f}%")
                
                with col2:
                    max_cv = temporal_df['CV'].max()
                    st.metric("Maior Variabilidade", f"{max_cv:.1f}%")
                
                with col3:
                    min_cv = temporal_df['CV'].min()
                    st.metric("Menor Variabilidade", f"{min_cv:.1f}%")
                
                # Gráfico de linha com variabilidade
                fig = px.line(
                    temporal_df, 
                    x=time_col, 
                    y=f'{temporal_metric}_mean',
                    error_y=f'{temporal_metric}_std',
                    title=f"Variação de {available_metrics[temporal_metric]['name']} ({time_agg})"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostra períodos com maior variabilidade
                st.write("**Períodos com Maior Variabilidade**")
                high_var_periods = temporal_df.nlargest(5, 'CV')
                st.dataframe(
                    high_var_periods[[time_col, f'{temporal_metric}_mean', f'{temporal_metric}_std', 'CV']]
                    .sort_values(time_col)
                    .style.format({
                        f'{temporal_metric}_mean': '{:,.2f}',
                        f'{temporal_metric}_std': '{:,.2f}',
                        'CV': '{:.1f}%'
                    })
                )
        except Exception as e:
            st.warning(f"Não foi possível realizar análise temporal: {str(e)}")

def show_comparative_analysis():
    st.subheader("📊 Análise Comparativa")
    
    if not st.session_state.comparison_data:
        st.info("Carregue pelo menos dois relatórios para realizar a análise comparativa.")
        return
    
    datasets = list(st.session_state.comparison_data.keys())
    
    if len(datasets) < 2:
        st.warning("Você precisa carregar pelo menos dois conjuntos de dados para comparação.")
        return
    
    # Seleção de datasets para comparação
    col1, col2 = st.columns(2)
    with col1:
        dataset1 = st.selectbox("Selecione o primeiro conjunto de dados", datasets, key="dataset1")
    with col2:
        dataset2 = st.selectbox("Selecione o segundo conjunto de dados", [d for d in datasets if d != dataset1], key="dataset2")
    
    df1 = st.session_state.comparison_data[dataset1]
    df2 = st.session_state.comparison_data[dataset2]
    
    # Seleção de métrica para comparação
    common_columns = list(set(df1.columns) & set(df2.columns)
    numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col])]
    
    if not numeric_columns:
        st.warning("Não há colunas numéricas em comum para comparação.")
        return
    
    selected_metric = st.selectbox("Selecione a métrica para comparação", numeric_columns)
    
    # Métricas agregadas
    st.write("### Métricas Agregadas")
    agg_functions = ['sum', 'mean', 'median', 'max', 'min']
    
    # Criar dataframe comparativo
    comparison_df = pd.DataFrame({
        dataset1: df1[selected_metric].agg(agg_functions),
        dataset2: df2[selected_metric].agg(agg_functions)
    }, index=agg_functions)
    
    st.dataframe(comparison_df.style.format("{:,.2f}"))
    
    # Gráfico de comparação
    st.write("### Comparação Visual")
    
    # Opções de visualização
    chart_type = st.radio("Tipo de Gráfico", ["Barras", "Linhas", "Pizza"], horizontal=True)
    
    if chart_type == "Barras":
        fig = px.bar(comparison_df.T, barmode='group', 
                     title=f"Comparação de {selected_metric} entre conjuntos de dados")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Linhas":
        fig = px.line(comparison_df.T, 
                     title=f"Comparação de {selected_metric} entre conjuntos de dados")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Pizza":
        # Usamos apenas a soma para o gráfico de pizza
        pie_data = pd.DataFrame({
            'Dataset': [dataset1, dataset2],
            'Value': [df1[selected_metric].sum(), df2[selected_metric].sum()]
        })
        fig = px.pie(pie_data, values='Value', names='Dataset', 
                     title=f"Distribuição de {selected_metric} entre conjuntos de dados")
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparação por campanha
    st.write("### Comparação por Campanha")
    
    # Encontrar campanhas em comum
    common_campaigns = list(set(df1['Campaign']) & set(df2['Campaign']))
    
    if not common_campaigns:
        st.warning("Não há campanhas em comum entre os conjuntos de dados selecionados.")
        return
    
    selected_campaign = st.selectbox("Selecione uma campanha para detalhes", common_campaigns)
    
    campaign_data1 = df1[df1['Campaign'] == selected_campaign]
    campaign_data2 = df2[df2['Campaign'] == selected_campaign]
    
    # Criar dataframe comparativo para a campanha selecionada
    campaign_comparison = pd.DataFrame({
        dataset1: campaign_data1[numeric_columns].mean(),
        dataset2: campaign_data2[numeric_columns].mean()
    })
    
    st.dataframe(campaign_comparison.style.format("{:,.2f}"))
    
    # Gráfico de radar para comparação de métricas
    st.write("### Comparação de Métricas da Campanha")
    
    # Normalizar os dados para o gráfico de radar
    normalized = campaign_comparison.copy()
    for col in normalized.columns:
        normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
    
    fig = px.line_polar(normalized.reset_index(), r=dataset1, theta='index', 
                        line_close=True, title=f"Comparação normalizada para {selected_campaign}")
    fig.add_trace(px.line_polar(normalized.reset_index(), r=dataset2, theta='index', 
                               line_close=True).data[0])
    st.plotly_chart(fig, use_container_width=True)


# Atualize a função main para incluir a nova aba
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
    
    # Abas principais - ATUALIZADO com nova aba
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise de Campanhas", "📊 Benchmark & Variabilidade", "💬 Chatbot Especializado", "Análise Comparativa"])
    
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

    with tab4:
        show_comparative_analysis()

# Atualize a função main para incluir a nova aba
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
    
    # Abas principais - ATUALIZADO com nova aba
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
