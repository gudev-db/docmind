import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

def load_campaign_data(file):
    """
    Load campaign data from CSV and return it as a DataFrame.
    """
    df = pd.read_csv(file)

    # Clean columns to ensure numeric operations
    df['Investido'] = df['Investido'].replace({'R\$': '', ',': ''}, regex=True).astype(float)
    df['Previsto'] = df['Previsto'].replace({'R\$': '', ',': ''}, regex=True).astype(float)
    df['Restante'] = df['Restante'].replace({'R\$': '', ',': ''}, regex=True).astype(float)
    
    # Clean '% Pacing Investimento' to numeric values
    df['% Pacing Investimento'] = df['% Pacing Investimento'].replace({'%': '', ',': ''}, regex=True).astype(float)

    return df

def analyze_campaign_data(df: pd.DataFrame, prompt: str):
    """
    Analyze the campaign data based on a custom prompt.
    Args:
        df (DataFrame): The loaded campaign data.
        prompt (str): The analysis instruction or question to ask.
    
    Returns:
        str: The analysis result or answer to the question.
    """
    prompt_lower = prompt.lower()

    # Analyze pacing
    if 'pacing' in prompt_lower:
        pacing = df['% Pacing Investimento'].mean()
        return f"A média do Pacing Investimento das campanhas é {pacing:.2f}%."
    
    # Calculate totals
    if 'total investido' in prompt_lower:
        total_investido = df['Investido'].sum()
        return f"Total Investido: R$ {total_investido:,.2f}"

    if 'total previsto' in prompt_lower:
        total_previsto = df['Previsto'].sum()
        return f"Total Previsto: R$ {total_previsto:,.2f}"

    if 'total restante' in prompt_lower:
        total_restante = df['Restante'].sum()
        return f"Total Restante: R$ {total_restante:,.2f}"

    # Filter by 'Ativa' status
    if 'campanhas ativas' in prompt_lower:
        active_campaigns = df[df['Ativa'] == 'Sim']
        return active_campaigns.to_string(index=False)

    # Filter by 'Mídia' type
    if 'mídia' in prompt_lower:
        media_type = prompt_lower.split('mídia')[-1].strip()
        filtered_media = df[df['Midia'].str.contains(media_type, case=False, na=False)]
        return filtered_media.to_string(index=False)

    # Filter by Investido range
    if 'investido' in prompt_lower:
        investido_range = prompt_lower.split('investido')[-1].strip()
        # Example: filter by Investido > 1000
        if ">" in investido_range:
            value = float(investido_range.split(">")[1].strip().replace("R$", "").replace(",", "."))
            filtered_investido = df[df['Investido'] > value]
            return filtered_investido.to_string(index=False)

    # Summarize campaigns
    if 'resumo de campanhas' in prompt_lower:
        summary = df[['Campanhas', 'Investido', 'Previsto', 'Restante', '% Pacing Investimento']].describe()
        return summary.to_string()

    return "Desculpe, não consegui entender o seu prompt."

def analyze_document(doc, prompt, api_key):
    """
    Use the LLM to analyze the document and return a response.
    Args:
        doc: The content of the document (CSV data).
        prompt: The user's analysis prompt.
        api_key: The Gemini API key.
    
    Returns:
        str: The analysis result.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{prompt}: {doc}")
    return response.text

def main():
    st.title("Analisador de Performance de Campanhas - IA")
    
    st.sidebar.header("Configurações")
    
    # Get the API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    prompt = st.sidebar.selectbox("Selecione o Tipo de Análise", [
        "Análise de Pacing",
        "Soma de valores",
        "Filtrar campanhas",
        "Resumo das campanhas",
        "Prompt Customizado"
    ])
    
    if prompt == "Prompt Customizado":
        prompt = st.sidebar.text_area("Escreva o prompt customizado")
    
    uploaded_file = st.file_uploader("Suba um arquivo CSV de campanha", type=["csv"])
    
    if uploaded_file is not None and api_key:
        st.success("Arquivo CSV subido com sucesso!")
        campaign_data = load_campaign_data(uploaded_file)
        
        if campaign_data is not None:
            # Show the first few rows of the data to the user
            st.subheader("Dados da Campanha:")
            st.dataframe(campaign_data.head())
            
            # Analyze the data based on the prompt
            analysis_result = analyze_campaign_data(campaign_data, prompt)
            st.subheader("Resultado da Análise:")
            st.write(analysis_result)
    elif not api_key:
        st.error("A chave do Gemini API não foi encontrada no arquivo .env.")
    
    st.sidebar.markdown("---")
   
if __name__ == "__main__":
    main()
