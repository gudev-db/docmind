import pandas as pd
import numpy as np
import re
from config import METRIC_FORMATS

def preprocess_google_ads_numbers(value):
    """Pré-processa valores numéricos do Google Ads"""
    if pd.isna(value) or value == '--':
        return '0'
    
    value = str(value).strip()
    value = re.sub(r'[^\d\.,-]', '', value)
    value = value.replace(',', '')
    
    parts = value.split('.')
    if len(parts) > 1:
        integer_part = ''.join(parts[:-1])
        decimal_part = parts[-1]
        value = f"{integer_part}.{decimal_part}" if decimal_part else f"{integer_part}"
    
    if value == '' or value == '-':
        return '0'
    
    return value

def clean_google_ads_data(df):
    """Limpa o dataframe do Google Ads mantendo a coluna Campaign intacta"""
    # Lista de colunas que devem SEMPRE ser mantidas como estão
    protected_cols = ['Campaign', 'Ad group', 'Keyword', 'Ad']
    
    # Colunas que sabemos serem numéricas (adicionar outras conforme necessário)
    known_numeric_cols = [
        'Clicks', 'Impr.', 'Interactions', 'Viewable impr.', 
        'Conversions', 'Impressions', 'Cost', 'CPM', 'CPC',
        'CTR', 'Conversion rate', 'Cost per conversion'
    ]
    
    # Colunas monetárias (baseado no nome)
    money_cols = [col for col in df.columns 
                 if any(word in col.lower() for word in ['cost', 'cpm', 'cpc', 'cpv', 'budget', 'value', 'rate', 'amount'])]
    
    # Combina todas as colunas numéricas a serem processadas
    numeric_cols_to_process = list(set(known_numeric_cols + money_cols))
    
    # Remove as colunas protegidas
    numeric_cols_to_process = [col for col in numeric_cols_to_process 
                             if col in df.columns and col not in protected_cols]
    
    # Processa apenas as colunas numéricas identificadas
    for col in numeric_cols_to_process:
        if col in df.columns:
            # Converte para string, processa e tenta converter para numérico
            df[col] = df[col].astype(str).apply(preprocess_google_ads_numbers)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Preenche valores numéricos faltantes (apenas para colunas já numéricas)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

def clean_google_ads_data(df):
    """Limpa o dataframe do Google Ads"""
    # Identifica colunas monetárias
    money_cols = [col for col in df.columns 
                 if any(word in col.lower() for word in ['cost', 'cpm', 'cpc', 'cpv', 'budget', 'value', 'rate', 'amount'])]
    
    # Processa colunas numéricas
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].astype(str).str.contains(r'[\d\.\,]').any():
            df[col] = df[col].astype(str).apply(preprocess_google_ads_numbers)
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    # Limpeza específica para colunas numéricas
    numeric_cols = ['Clicks', 'Impr.', 'Interactions', 'Viewable impr.', 'Conversions', 'Impressions']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(preprocess_google_ads_numbers)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Preenche valores numéricos faltantes
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

def load_google_ads_data(uploaded_file):
    """Carrega dados do Google Ads a partir de um arquivo"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='utf-8', dtype={'Campaign': str})
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=2, dtype={'Campaign': str})
        else:
            raise ValueError("Formato de arquivo não suportado")
        
        if 'Campaign' not in df.columns:
            raise ValueError("O arquivo não contém a coluna 'Campaign' necessária")
        
        return df.copy(), clean_google_ads_data(df)
    except Exception as e:
        raise ValueError(f"Erro ao carregar os dados do Google Ads: {str(e)}")
