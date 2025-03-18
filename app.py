import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os
import pandas as pd
from dotenv import load_dotenv

st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

def load_document(file):
    ext = file.name.split(".")[-1]
    temp_file_path = f"temp_uploaded.{ext}"  # Save file temporarily

    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())  # Write the uploaded file content

    if ext == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif ext == "txt":
        loader = TextLoader(temp_file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(temp_file_path)
    elif ext == "csv":
        # Load CSV using pandas
        df = pd.read_csv(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file
        return df.to_string()  # Return as a string (can be adjusted based on how you want to present the CSV data)
    else:
        st.error("Unsupported file type.")
        return None

    content = loader.load()
    os.remove(temp_file_path)  # Clean up the temporary file
    return content

def analyze_document(doc, prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{prompt}: {doc}")
    return response.text

def main():
    st.title("Análise de Documentos - IA")
    
    st.sidebar.header("Configurações")
    
    # Get the API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    prompt = st.sidebar.selectbox("Select Analysis Type", [
        "Análise Compreensiva de documento",
        "Extrair insights chave",
        "Resumir e identificar perguntas em aberto",
        "Corrija o texto de acordo com as normas gramaticais brasileiras de uma forma que mantenha o sentido do texto e aponte as alterações feitas - Retorne o documento corrigido por completo.",
        "Prompt Customizado"
    ])
    
    if prompt == "Prompt Customizado":
        st.sidebar.markdown("### Chatbot - Interaja abaixo")
        
        # Initialize session state for conversation history if not already set
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Input field for the user's message
        user_input = st.text_input("Você: ", "")

        if user_input:
            # Append user input to the conversation history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get the response from the model
            if api_key:
                response = analyze_document(" ".join([msg["content"] for msg in st.session_state.messages]), user_input, api_key)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display the conversation
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write(f"**Você:** {message['content']}")
                else:
                    st.write(f"**Assistente:** {message['content']}")

    uploaded_file = st.file_uploader("Suba um documento", type=["pdf", "txt", "docx", "csv"])
    if uploaded_file is not None and api_key:
        st.success("Arquivo subido com sucesso!")
        document_content = load_document(uploaded_file)
        
        if document_content:
            analysis_result = analyze_document(document_content, prompt, api_key)
            st.subheader("Resultado:")
            st.write(analysis_result)
    elif not api_key:
        st.error("A chave do Gemini API não foi encontrada no arquivo .env.")
    
    st.sidebar.markdown("---")
   

if __name__ == "__main__":
    main()
