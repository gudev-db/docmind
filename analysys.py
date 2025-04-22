import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import METRIC_FORMATS

def show_benchmark_analysis(df):
    """Mostra análise de benchmark"""
    st.subheader("📊 Benchmark de Performance vs Médias")
    
    if df is None or df.empty:
        st.warning("Nenhum dado disponível para análise de benchmark")
        return
    
    # Filtra métricas disponíveis
    available_metrics = {k: v for k, v in METRIC_FORMATS.items() if k in df.columns}
    
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
    selected_metric = st.selectbox(
        "Selecione a métrica para análise comparativa",
        options=list(available_metrics.keys()),
        format_func=lambda x: available_metrics[x]['name']
    )
    
    if selected_metric:
        _show_metric_comparison(df, selected_metric, available_metrics, global_means)

def _show_metric_comparison(df, selected_metric, available_metrics, global_means):
    """Mostra comparação de uma métrica específica"""
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
    
    # Gráfico de violino
    fig = px.violin(df_comparison, y=selected_metric, box=True, points="all",
                    title=f"Distribuição de {available_metrics[selected_metric]['name']}")
    fig.add_hline(y=mean_value, line_dash="dash", line_color="red",
                  annotation_text=f"Média: {mean_value:.2f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela com campanhas destacadas
    st.write(f"**Campanhas Destacadas - {available_metrics[selected_metric]['name']}**")
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
