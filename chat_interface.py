import streamlit as st
from config import MODEL

def generate_google_ads_response(prompt, df):
    """Gera resposta do Gemini para análise de Google Ads"""
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
        Inclua números específicos quando relevante. Tire insights sobre as métricas. Retorne os nomes das campanhas que estão abaixo ou acima das médias das colunas numéricas.
        Me traga insights técnicos sobre a performance das campanhas. Gere um relatório sobre as campanhas.
        """
        
        response = MODEL.generate_content(context)
        return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def chat_interface():
    """Interface do chatbot"""
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
