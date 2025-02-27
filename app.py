import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os
import pandas as pd
from dotenv import load_dotenv

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
        prompt = st.sidebar.text_area("Escreva o prompt customizado")
    
    uploaded_file = st.file_uploader("Suba um documento", type=["pdf", "txt", "docx", "csv"])

    if uploaded_file.type == "csv":
        prompt == '''
Siga esse playbook:

Acompanhar resultados
Rotina de acompanhamento dos indicadores dos clientes que você atende

Identificar melhorias de resultados e ações que levaram a estes resultados

Quedas de desempenho e causas raiz

Documentar análises para os próximos relatórios


Em casos de e-commerces, é importante realizar uma análise de produtos:

Buscar oportunidades de vendas na cauda longa

Levantar informações de produtos que vendam bem em lojas físicas e que podem performar melhor no online 

Identificar oportunidades de promoção em produtos com alto potencial de vendas

Realizar estudos de Curva ABC

Relatório
Frequência

Quando uma campanha é encerrada, inicia-se a criação de um relatório

Se a campanha for always on, define-se uma periodicidade destes relatório (minimante mensal,  mas há alguns clientes que possuem relatórios semanais)


Criar Relatório

Seguir modelo de relatório

Não se limitar ao modelo existente! 

Ele é apenas um guia para orientar as informações básicas que devemos apresentar. Mas devemos trazer novas percepções, insights, dicas

Entender como o cliente gosta de visualizar os dados e adaptar às suas preferências, quando for possível

O uso do Diário de Bordo auxiliará na construção desse relatório
Importante! A percepção do cliente é muito positiva quando levamos resultados de testes e dados que corroborem nossa decisão

Entender também quem é a pessoa que irá ler o relatório e adaptar linguagem ao leitor (o padrão não possui essas variações)


Extrair dados das plataformas de mídia e do GA4

Plataformas: investimento total da campanha (custo ou gasto nas plataformas), número de impressões e cliques, alcance de usuários

GA4: 1) lead gen: conversões (form preenchido ou similar), taxas de conversão ou rejeição, sessão, qtde. usuários / 2) ecomm + receita, transação (vol. vendas)

Outros clientes: métricas de vaidade como seguidores, engajamento, visitas à loja (O2O), dados de app (instalação de app, vendas em app, app engagement)


Tratar os dados

Apresentar os dados extraídos em uma apresentação que possibilite o entendimento do cliente: gráficos, layouts, tabelas, big numbers…

Sempre essas comparações trazem dados comparativos entre períodos (XoX), possibilitando uma análise comparativa com a mesma sazonalidade


Fazer as análises dos dados

Fechamento dos números da campanha: planejado x realizado

Identificar pontos de melhoria, propor ideias novas, justificar baixas performances (com fatores externos, p.ex.), defender bons resultados

Mapear pontos positivos e negativos da conta, defendendo sua performance e apresentar planos de ação sobre o que melhorar

Se alguma mudança necessitar de aprovação, defender os argumentos nessa apresentação


Reunir as informações das outras áreas

Criar briefing de relatório com as demais áreas para alinhar a entrega

Fazer um balanço entre o prometido no último mês e o que faremos no próximo

Analisar todas informações e fazer as solicitações necessárias para o cliente melhorar seus próprios resultados:
Importante! Colocar alguma coisa para o cliente fazer também, para ele se tornar corresponsável pela tarefa (se não atingirmos, a culpa é do cliente também)


Buscar dados de mercado

Pesquisa desk sobre o mercado (notícias, alterações de sazonalidade, fatos relevante), benchmarks do mercado, análise de concorrentes de mídia

Buscar noticias que defendam nosso Planejamento e nossas ideias

Entender o momento do cliente, trazendo os destaques, campanhas, momentos e eventos do cliente



Revisar o relatório completo, com os dados das demais áreas

Pontos de Atenção: data de rodapé, comparar e bater os números


Validação

Enviar para Revisão ortográfica

Validar informações e modelo de relatório com o Atendimento


Realizar apresentação dos relatórios

Geralmente são realizadas dentro das weeklies e cada apresentação ocorre em uma semana, para não deixar todos relatórios na mesma semana'''
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
