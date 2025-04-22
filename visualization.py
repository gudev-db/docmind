import streamlit as st
import pandas as pd
import plotly.express as px
from config import METRIC_FORMATS

def safe_format(value, fmt):
    """Formata valores de forma segura"""
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

def show_google_ads_summary(df):
    """Mostra um resumo do relatório Google Ads"""
    st.subheader("Resumo do Relatório Google Ads")
    
    # Verifica colunas disponíveis
    has_cost = 'Cost' in df.columns
    has_impressions = 'Impr.' in df.columns or 'Impressions' in df.columns
    has_clicks = 'Clicks' in df.columns
    impressions_col = 'Impr.' if 'Impr.' in df.columns else 'Impressions' if 'Impressions' in df.columns else None
    
    # Métricas principais
    st.write("### Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = df['Cost'].sum() if has_cost else 0
    total_impressions = df[impressions_col].sum() if has_impressions else 0
    total_clicks = df['Clicks'].sum() if has_clicks else 0
    
    with col1:
        st.metric("Total Gasto", f"R$ {total_cost:,.2f}")
    
    with col2:
        st.metric("Total de Impressões", f"{total_impressions:,.0f}")
    
    with col3:
        avg_cpc = total_cost / total_clicks if total_clicks > 0 else 0
        st.metric("CPC Médio", f"R$ {avg_cpc:,.2f}")
    
    with col4:
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        st.metric("CTR", f"{ctr:.2f}%")
    
    # Tabs para visualizações
    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Por Campanha", "Performance Temporal"])
    
    with tab1:
        _show_general_view(df, has_cost, impressions_col)
    
    with tab2:
        _show_campaign_view(df, has_cost, has_clicks, has_impressions, impressions_col)
    
    with tab3:
        _show_temporal_view(df, impressions_col)

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

def _show_general_view(df, has_cost, impressions_col):
    """Mostra visualização geral"""
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

def _show_campaign_view(df, has_cost, has_clicks, has_impressions, impressions_col):
    """Mostra visualização por campanha"""
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

def _show_temporal_view(df, impressions_col):
    """Mostra visualização temporal"""
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
