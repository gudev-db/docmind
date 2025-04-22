import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configuração inicial
load_dotenv()

# Configuração do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-1.5-flash')

# Configurações de formatação
METRIC_FORMATS = {
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
