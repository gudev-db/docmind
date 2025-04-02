import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

# Configuração inicial
load_dotenv()
st.set_page_config(layout="wide", page_title="Analisador de Campanhas de Marketing")

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None

def ask_gemini(question, df, api_key, chat_history=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    context = f"""
    Você é um analista de marketing especializado. 
    Dados COMPLETOS das campanhas (formato CSV):
    
    {df.to_csv(index=False)}
    
    Colunas disponíveis: {', '.join(df.columns)}
    """
    
    try:
        response = model.generate_content(
            f"{context}\n\nHistórico:\n{chat_history or 'Nenhum'}\n\nPergunta: {question}"
        )
        return response.text
    except Exception as e:
        return f"Erro na análise: {str(e)}"

def main():
    st.title("📊 Analisador de Campanhas com Gemini")
    st.markdown("Pergunte sobre seus dados em linguagem natural")
    
    # Botão para limpar o chat
    if st.sidebar.button("🧹 Limpar Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat limpo. O que gostaria de saber agora?"
        }]
        st.rerun()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Configure sua API key no arquivo .env")
        return
    
    uploaded_file = st.file_uploader("Carregue seu arquivo", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.info("Carregue um arquivo para começar")
        return
    
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Mostra pré-visualização
    with st.expander("📁 Visualização dos dados (amostra)"):
        st.dataframe(df.head())
        st.write(f"Total de linhas: {len(df)}")
    
    # Inicializa o chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Olá! Analisei {len(df)} registros de campanhas. O que gostaria de saber?"
        }]
    
    # Exibe histórico do chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner(f"Analisando {len(df)} registros..."):
            history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:])
            response = ask_gemini(prompt, df, api_key, history)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
