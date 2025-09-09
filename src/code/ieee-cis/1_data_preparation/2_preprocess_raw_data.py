#!/usr/bin/env python3
"""
Credit Card Fraud Detection (IEEE-CIS) - Parquet to Enhanced Parquet Processor

Este script carrega o parquet pré-processado do dataset IEEE-CIS e aplica 
feature engineering ADICIONAL, seguindo as diretrizes do projeto GraphSentinel 2.0.

IMPORTANTE: Este script mantém todos os dados em formato BRUTO para preservar
a integridade científica. Normalizações serão aplicadas posteriormente no
pipeline de ML para evitar data leakage.

Features do dataset de entrada (ieee_cis_top100_features.parquet):
- TransactionDT: Timestamp da transação (int32, TOP #1 importance: 25.61%)
- TransactionAmt: Valor da transação (float, TOP #2 importance: 10.42%)
- ProductCD: Tipo de produto (dictionary/categorical, TOP #4 importance: 6.94%)
- card1-card6: Features categóricas de cartão (TOP #3,5-9 importance: 3.41-2.46%)
- addr1, addr2: Features de endereço (TOP #10-11 importance: 1.70-1.40%)
- dist1, dist2: Features de distância (TOP #12-13 importance: 1.29-1.21%)
- P_emaildomain, R_emaildomain: Domínios de email (dictionary/categorical, TOP #14-15 importance: 1.16-0.92%)
- C2-C14: Features categóricas C (11 colunas: C2,C3,C4,C5,C7,C8,C9,C10,C12,C13,C14)
- D2-D15: Features categóricas D (12 colunas: D2,D3,D4,D6,D7,D8,D9,D10,D11,D13,D14,D15)
- V2-V59: Features anonimizadas V (57 colunas não-sequenciais)
- M1-M9: Features categóricas M (dictionary/categorical, 9 colunas)
- isFraud: Target variable (int8, 0=legítima, 1=fraude)

Feature Engineering ADICIONAL implementado (ULTRA REDUZIDO):
- Features temporais ESSENCIAIS (baseadas em TransactionDT) - REDUZIDAS ~26→5
- Features de cartão MÍNIMAS (card1-card6, card4/card6 categóricas) - REDUZIDAS ~43→8
- Features de endereço MÍNIMAS (addr1, addr2) - REDUZIDAS ~19→4
- Features de distância REDUZIDAS (dist1, dist2) - REDUZIDAS ~24→12
- Features de email domain MÍNIMAS (P_emaildomain, R_emaildomain categóricas) - REDUZIDAS ~102→5
- Features categóricas C MÍNIMAS (C2-C14) - REDUZIDAS ~41→1
- Features categóricas D MÍNIMAS (D2-D15) - REDUZIDAS ~49→1
- Features V anonimizadas MÍNIMAS (V2-V59, 57 colunas) - REDUZIDAS ~40→4
- Features de produto MÍNIMAS (ProductCD categórica) - REDUZIDAS ~13→2
- Features M categóricas MÍNIMAS (M1-M9 categóricas) - REDUZIDAS ~13→2
- Features monetárias ESSENCIAIS (log transformation) - REDUZIDAS ~17→4
- Features de interação MÍNIMAS entre grupos de variáveis - REDUZIDAS ~118→3
- Features de outliers MÍNIMAS - REDUZIDAS ~25→2
- Flags de missing values MANTIDAS nas features essenciais

CORREÇÕES APLICADAS:
- Tratamento seguro de features categóricas (dictionary) com conversão para string
- Limpeza de nomes de colunas para evitar caracteres especiais
- Comparações categóricas com fallback para string
- Tratamento de exceções para valores categóricos problemáticos

REDUÇÃO AGRESSIVA: de ~498 para ~150-180 features (~65% de redução)
Focando APENAS nas features mais essenciais com comprovado valor preditivo.

NOTA: Todas as features derivadas são mantidas em escala bruta/natural
para preservar consistência no pipeline de ML e evitar data leakage.

Diretrizes aplicadas:
- Seed fixo (42) para reprodutibilidade
- Estrutura de diretórios definida no projeto
- Paths simplificados (data/parquet/ieee-cis/ieee-cis.parquet)
- Comentários e docstrings claras
- Processamento eficiente sem normalizações prematuras
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuração de reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Caminhos dos arquivos - seguindo estrutura do projeto com NOMES SIMPLIFICADOS
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # 5 níveis para chegar na raiz
INPUT_PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "ieee-cis" / "ieee_cis_top100_features.parquet"
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "ieee-cis" / "ieee-cis.parquet"  # Nome simplificado
METADATA_PATH = PROJECT_ROOT / "data" / "parquet" / "ieee-cis" / "ieee-cis.json"    # Arquivo de metadados


def check_gpu_availability():
    """
    Verifica disponibilidade de GPU para processamento acelerado.
    
    Returns:
        bool: True se GPU estiver disponível e bibliotecas de aceleração instaladas
    """
    try:
        import cudf
        import cupy as cp
        logger.info("RAPIDS cuDF detectado - usando aceleração GPU")
        return True
    except ImportError:
        logger.info("RAPIDS não disponível - usando pandas/CPU")
        return False


def validate_input_file(input_path):
    """
    Valida se o arquivo Parquet de entrada existe e é acessível.
    
    Args:
        input_path (str): Caminho do arquivo Parquet
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo Parquet não encontrado: {input_path}")
    
    file_size = os.path.getsize(input_path) / (1024 * 1024)
    logger.info(f"Arquivo Parquet encontrado - Tamanho: {file_size:.2f} MB")


def create_directories(output_path):
    """
    Cria os diretórios necessários para o arquivo de saída.
    
    Args:
        output_path (str): Caminho do arquivo de saída
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Diretório de saída criado/verificado: {output_dir}")


def load_and_process_data(input_path):
    """
    Carrega dados do parquet pré-processado e aplica feature engineering ADICIONAL.
    
    Args:
        input_path (str): Caminho do arquivo Parquet
    
    Returns:
        pandas.DataFrame: DataFrame processado
    """
    logger.info("=== CARREGANDO E PROCESSANDO DADOS (FEATURE ENGINEERING ADICIONAL) ===")
    
    # Carregar dataset do parquet
    logger.info("Carregando dataset IEEE-CIS do parquet pré-processado...")
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset original: {df.shape}")
    logger.info(f"Colunas disponíveis: {list(df.columns)}")
    
    # VERIFICAR MISSING VALUES
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        logger.info(f"Missing values encontrados: {missing_info[missing_info > 0]}")
    else:
        logger.info("✓ Nenhum missing value encontrado")
    
    # APLICAR FEATURE ENGINEERING ADICIONAL
    
    # 1. FEATURES TEMPORAIS AVANÇADAS (TransactionDT)
    df = create_advanced_temporal_features(df)
    
    # 2. FEATURES MONETÁRIAS AVANÇADAS (TransactionAmt)
    df = create_advanced_amount_features(df)
    
    # 3. FEATURES DE CARTÃO EXPANDIDAS (card1-card6, card4_*)
    df = create_expanded_card_features(df)
    
    # 4. FEATURES DE ENDEREÇO APRIMORADAS (addr1, addr2)
    df = create_enhanced_address_features(df)
    
    # 5. FEATURES DE DISTÂNCIA (dist1, dist2)
    df = create_distance_features(df)
    
    # 6. FEATURES DE EMAIL DOMAIN (P_emaildomain, R_emaildomain)
    df = create_email_features(df)
    
    # 7. FEATURES CATEGÓRICAS C (C2-C14)
    df = create_c_features(df)
    
    # 8. FEATURES CATEGÓRICAS D (D2-D15)
    df = create_d_features(df)
    
    # 9. FEATURES V EXPANDIDAS (V24-V332, não sequenciais)
    df = create_expanded_v_features(df)
    
    # 10. FEATURES DE PRODUTO (ProductCD_*)
    df = create_product_features(df)
    
    # 11. FEATURES M CATEGÓRICAS (M1_F, M2_F/T, M3_F/T)
    df = create_m_features(df)
    
    # 12. FEATURES DE INTERAÇÃO AVANÇADAS
    df = create_advanced_interaction_features(df)
    
    # 13. FEATURES DE OUTLIERS AVANÇADAS
    df = create_advanced_outlier_features(df)
    
    logger.info(f"=== PROCESSAMENTO CONCLUÍDO - Shape final: {df.shape} ===")
    return df


def create_advanced_temporal_features(df):
    """
    Cria features temporais ESSENCIAIS baseadas na coluna TransactionDT.
    TransactionDT é o timestamp em segundos desde uma data de referência.
    
    REDUÇÃO AGRESSIVA: APENAS 5 features temporais mais importantes.
    """
    logger.info("Criando features temporais ESSENCIAIS (TransactionDT)...")
    
    if 'TransactionDT' not in df.columns:
        logger.warning("TransactionDT não encontrado - pulando features temporais")
        return df
    
    # Converter timestamp para datetime (assumindo que é timestamp Unix)
    # Nota: TransactionDT pode estar em formato específico do dataset IEEE-CIS
    df['transaction_timestamp'] = pd.to_datetime(df['TransactionDT'], unit='s', errors='coerce')
    
    # Se conversão falhou, usar método alternativo
    if df['transaction_timestamp'].isnull().all():
        # Usar TransactionDT como segundos desde primeira transação
        min_dt = df['TransactionDT'].min()
        df['transaction_timestamp'] = pd.to_datetime(min_dt + df['TransactionDT'] - min_dt, unit='s')
    
    # APENAS 3 features temporais mais essenciais
    df['trans_hour'] = df['transaction_timestamp'].dt.hour
    df['trans_weekday'] = df['transaction_timestamp'].dt.weekday
    
    # APENAS 1 flag mais importante
    df['is_night'] = ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 5)).astype(int)
    # REMOVER: is_weekend (redundante com weekday + menor valor preditivo)
    
    # APENAS 1 feature cíclica (hora mais importante que weekday)
    df['hour_sin'] = np.sin(2 * np.pi * df['trans_hour'] / 24)
    # REMOVER: hour_cos (sin suficiente), weekday_sin/cos
    
    # REMOVER: time_from_start (alto custo computacional, baixo valor incremental)
    
    logger.info("Features temporais ESSENCIAIS criadas (5 features)")
    return df


def create_advanced_amount_features(df):
    """
    Cria features monetárias ESSENCIAIS baseadas na coluna TransactionAmt.
    
    REDUÇÃO AGRESSIVA: APENAS 5 features mais importantes.
    """
    logger.info("Criando features monetárias ESSENCIAIS (TransactionAmt)...")
    
    if 'TransactionAmt' not in df.columns:
        logger.warning("TransactionAmt não encontrado - pulando features monetárias")
        return df
    
    # APENAS log transformation (mais importante)
    df['amount_log'] = np.log1p(df['TransactionAmt'])
    
    # APENAS 1 flag mais importante (pequeno valor)
    amount_q20 = df['TransactionAmt'].quantile(0.2)
    df['is_small_amount'] = (df['TransactionAmt'] <= amount_q20).astype(int)
    # REMOVER: is_large_amount, amount_category (redundantes)
    
    # APENAS flag de round amount (importante para fraude)
    df['is_round_amount'] = (df['TransactionAmt'] % 1 == 0).astype(int)
    
    # APENAS ranking percentil (mais eficiente que categorização)
    df['amount_rank'] = df['TransactionAmt'].rank(pct=True)
    
    # REMOVER: amount_vs_hour_mean (interaction desnecessária)
    
    logger.info("Features monetárias ESSENCIAIS criadas (4 features)")
    return df


def create_expanded_card_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas de cartão.
    
    ADAPTAÇÃO: card4 e card6 são categóricas (dictionary), não encoded.
    """
    logger.info("Criando features de cartão MÍNIMAS...")
    
    # Colunas de cartão numéricas
    card_numeric_cols = ['card1', 'card2', 'card3', 'card5']
    available_card_cols = [col for col in card_numeric_cols if col in df.columns]
    
    # Colunas de cartão categóricas
    card_categorical_cols = ['card4', 'card6']
    available_card_cat_cols = [col for col in card_categorical_cols if col in df.columns]
    
    logger.info(f"Processando card numéricas: {available_card_cols}")
    logger.info(f"Processando card categóricas: {available_card_cat_cols}")
    
    # Flags de missing para numéricas
    for card_col in available_card_cols:
        df[f'{card_col}_is_missing'] = df[card_col].isnull().astype(int)
    
    # Processar categóricas (card4, card6)
    for card_col in available_card_cat_cols:
        # Flag de missing
        df[f'{card_col}_is_missing'] = df[card_col].isnull().astype(int)
        
        # One-hot para valores mais comuns (top 2)
        if not df[card_col].isnull().all():
            top_values = df[card_col].value_counts().head(2).index.tolist()
            for value in top_values:
                # Limpar nome para usar como coluna
                clean_value = str(value).replace('.', '_').replace(' ', '_').replace('-', '_')
                df[f'{card_col}_{clean_value}'] = (df[card_col].astype(str) == str(value)).astype(int)
    
    # Missing count geral
    all_card_cols = available_card_cols + available_card_cat_cols
    if len(all_card_cols) >= 2:
        df['card_missing_count'] = df[all_card_cols].isnull().sum(axis=1)
    
    logger.info("Features de cartão MÍNIMAS criadas")
    return df


def create_enhanced_address_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas de endereço.
    
    REDUÇÃO AGRESSIVA: APENAS flags de missing essenciais.
    """
    logger.info("Criando features de endereço MÍNIMAS...")
    
    addr_cols = ['addr1', 'addr2']
    available_addr_cols = [col for col in addr_cols if col in df.columns]
    
    if not available_addr_cols:
        logger.info("Nenhuma coluna de endereço encontrada")
        return df
    
    logger.info(f"Processando endereços: {available_addr_cols}")
    
    # APENAS flags de missing (remover frequency e rare)
    for addr_col in available_addr_cols:
        df[f'{addr_col}_is_missing'] = df[addr_col].isnull().astype(int)
    
    # APENAS missing count e comparação básica
    if len(available_addr_cols) == 2:
        df['addr_missing_count'] = df[available_addr_cols].isnull().sum(axis=1)
        
        # Comparação entre addr1 e addr2 (converter para string se necessário)
        try:
            df['addr1_addr2_same'] = ((df['addr1'] == df['addr2']) & 
                                      df['addr1'].notna() & 
                                      df['addr2'].notna()).astype(int)
        except TypeError:
            # Se erro de categoria, converter para string
            df['addr1_addr2_same'] = ((df['addr1'].astype(str) == df['addr2'].astype(str)) & 
                                      df['addr1'].notna() & 
                                      df['addr2'].notna()).astype(int)
    
    logger.info("Features de endereço MÍNIMAS criadas (~4 features)")
    return df


def create_distance_features(df):
    """
    Cria features baseadas nas colunas de distância.
    
    Inclui: dist1, dist2 (importance: 1.29-1.21%).
    """
    logger.info("Criando features de distância...")
    
    dist_cols = ['dist1', 'dist2']
    available_dist_cols = [col for col in dist_cols if col in df.columns]
    
    if not available_dist_cols:
        logger.info("Nenhuma coluna de distância encontrada")
        return df
    
    logger.info(f"Processando distâncias: {available_dist_cols}")
    
    # Para cada coluna de distância - ULTRA OTIMIZADO (50% redução)
    for dist_col in available_dist_cols:
        # Flag de missing (essencial)
        df[f'{dist_col}_is_missing'] = df[dist_col].isnull().astype(int)
        
        if not df[dist_col].isnull().all():
            # APENAS categoria simples (3 níveis em vez de 6)
            dist_q25, dist_q75 = df[dist_col].quantile([0.25, 0.75])
            conditions = [
                (df[dist_col] <= dist_q25),
                (df[dist_col] <= dist_q75)
            ]
            choices = [0, 1]
            df[f'{dist_col}_category'] = np.select(conditions, choices, default=2)
            
            # APENAS flag de zero (mais importante que short/long/very_long)
            df[f'{dist_col}_is_zero'] = (df[dist_col] == 0).astype(int)
            # REMOVER: is_short, is_long, is_very_long, log (redundantes)
    
    # Features de interação ULTRA SIMPLIFICADAS
    if len(available_dist_cols) == 2:
        # APENAS soma (mais importante que diferença/razão)
        df['dist_sum'] = df['dist1'].fillna(0) + df['dist2'].fillna(0)
        
        # APENAS missing count (não precisa de all/any separados)
        df['dist_missing_count'] = df[available_dist_cols].isnull().sum(axis=1)
        # REMOVER: dist_diff, dist_ratio, all_dist_missing, any_dist_missing
    
    logger.info("Features de distância criadas")
    return df


def create_email_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas de email domain.
    
    ADAPTAÇÃO: P_emaildomain e R_emaildomain são categóricas (dictionary), não encoded.
    """
    logger.info("Criando features de email domain MÍNIMAS...")
    
    email_cols = ['P_emaildomain', 'R_emaildomain']
    available_email_cols = [col for col in email_cols if col in df.columns]
    
    if not available_email_cols:
        logger.info("Nenhuma coluna de email encontrada")
        return df
    
    logger.info(f"Processando emails categóricos: {available_email_cols}")
    
    # Para cada coluna de email
    for email_col in available_email_cols:
        # Flag de missing
        df[f'{email_col}_is_missing'] = df[email_col].isnull().astype(int)
        
        # One-hot para domínios mais comuns (top 3)
        if not df[email_col].isnull().all():
            top_domains = df[email_col].value_counts().head(3).index.tolist()
            for domain in top_domains:
                # Limpar nome do domínio para usar como nome de coluna
                clean_domain = str(domain).replace('.', '_').replace('@', '_').replace('-', '_')
                df[f'{email_col}_{clean_domain}'] = (df[email_col].astype(str) == str(domain)).astype(int)
    
    # Features de interação entre emails (apenas se ambos existem)
    if len(available_email_cols) == 2:
        # Comparação entre domínios (converter para string para evitar erro de categorias)
        df['P_R_email_same'] = ((df['P_emaildomain'].astype(str) == df['R_emaildomain'].astype(str)) & 
                                df['P_emaildomain'].notna() & 
                                df['R_emaildomain'].notna()).astype(int)
        
        # Missing count total
        df['email_missing_count'] = df[available_email_cols].isnull().sum(axis=1)
    
    logger.info("Features de email domain MÍNIMAS criadas")
    return df


def create_c_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas categóricas C.
    
    REDUÇÃO AGRESSIVA: APENAS missing count.
    """
    logger.info("Criando features categóricas C MÍNIMAS...")
    
    c_cols = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
    
    if not c_cols:
        logger.info("Nenhuma coluna C encontrada")
        return df
    
    logger.info(f"Processando colunas C: {c_cols}")
    
    # APENAS missing count geral (remover features individuais)
    if len(c_cols) > 1:
        df['c_missing_count'] = df[c_cols].isnull().sum(axis=1)
    
    logger.info("Features categóricas C MÍNIMAS criadas (1 feature)")
    return df


def create_d_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas categóricas D.
    
    REDUÇÃO AGRESSIVA: APENAS missing count.
    """
    logger.info("Criando features categóricas D MÍNIMAS...")
    
    d_cols = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()]
    
    if not d_cols:
        logger.info("Nenhuma coluna D encontrada")
        return df
    
    logger.info(f"Processando colunas D: {d_cols}")
    
    # APENAS missing count geral (remover features individuais)
    if len(d_cols) > 1:
        df['d_missing_count'] = df[d_cols].isnull().sum(axis=1)
    
    logger.info("Features categóricas D MÍNIMAS criadas (1 feature)")
    return df


def create_expanded_v_features(df):
    """
    Cria features MÍNIMAS das colunas V anonimizadas.
    
    ADAPTAÇÃO: V2-V59 (não V24-V332), 57 colunas disponíveis.
    """
    logger.info("Criando features V MÍNIMAS (V2-V59)...")
    
    # Selecionar todas as colunas V presentes no dataset
    v_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
    
    if not v_cols:
        logger.info("Nenhuma coluna V encontrada")
        return df
    
    logger.info(f"Processando {len(v_cols)} colunas V: {v_cols[:10]}...{v_cols[-5:]}")
    
    v_data = df[v_cols]
    
    # Estatística mais importante
    df['v_mean'] = v_data.mean(axis=1)
    
    # Missing count
    df['v_missing_count'] = v_data.isnull().sum(axis=1)
    
    # Interação V mais importante (usar colunas que realmente existem)
    available_v_interactions = []
    if 'V24' in df.columns and 'V38' in df.columns:
        df['V24_V38_product'] = df['V24'] * df['V38']
        available_v_interactions.append('V24_V38_product')
    elif len(v_cols) >= 2:
        # Se V24/V38 não existem, usar as duas primeiras
        df[f'{v_cols[0]}_{v_cols[1]}_product'] = df[v_cols[0]] * df[v_cols[1]]
        available_v_interactions.append(f'{v_cols[0]}_{v_cols[1]}_product')
    
    # Clustering binário simples
    v_mean_median = df['v_mean'].median()
    df['v_profile_cluster'] = (df['v_mean'] > v_mean_median).astype(int)
    
    logger.info(f"Features V MÍNIMAS criadas (4 features, interações: {available_v_interactions})")
    return df


def create_product_features(df):
    """
    Cria features MÍNIMAS baseadas na coluna de produto (ProductCD).
    
    ADAPTAÇÃO: ProductCD é categórica (dictionary), não encoded.
    """
    logger.info("Criando features de produto MÍNIMAS...")
    
    if 'ProductCD' not in df.columns:
        logger.warning("ProductCD não encontrado - pulando features de produto")
        return df
    
    logger.info(f"Processando ProductCD categórica: {df['ProductCD'].unique()}")
    
    # Flag de missing
    df['product_is_missing'] = df['ProductCD'].isnull().astype(int)
    
    # One-hot encoding para valores mais comuns (manter apenas top 3)
    if not df['ProductCD'].isnull().all():
        top_products = df['ProductCD'].value_counts().head(3).index.tolist()
        for product in top_products:
            # Limpar nome para usar como coluna
            clean_product = str(product).replace('.', '_').replace(' ', '_').replace('-', '_')
            df[f'ProductCD_{clean_product}'] = (df['ProductCD'].astype(str) == str(product)).astype(int)
    
    logger.info("Features de produto MÍNIMAS criadas")
    return df


def create_m_features(df):
    """
    Cria features MÍNIMAS baseadas nas colunas M categóricas.
    
    ADAPTAÇÃO: M1-M9 são categóricas (dictionary), não encoded como M1_F, M2_F/T.
    """
    logger.info("Criando features M categóricas MÍNIMAS...")
    
    m_cols = [col for col in df.columns if col.startswith('M') and len(col) <= 3 and col[1:].isdigit()]
    
    if not m_cols:
        logger.info("Nenhuma coluna M encontrada")
        return df
    
    logger.info(f"Processando M features categóricas: {m_cols}")
    
    # Missing count
    df['m_missing_count'] = df[m_cols].isnull().sum(axis=1)
    
    # Para cada M feature, criar flag de valores mais comuns
    for m_col in m_cols[:3]:  # Apenas M1, M2, M3 para redução
        if m_col in df.columns:
            # Flag de missing individual
            df[f'{m_col}_is_missing'] = df[m_col].isnull().astype(int)
            
            # One-hot para valor mais comum (se existir)
            if not df[m_col].isnull().all():
                try:
                    most_common = df[m_col].mode().iloc[0] if len(df[m_col].mode()) > 0 else None
                    if most_common is not None:
                        # Limpar nome para usar como coluna (remover caracteres especiais)
                        clean_value = str(most_common).replace('.', '_').replace(' ', '_').replace('-', '_')
                        df[f'{m_col}_{clean_value}'] = (df[m_col].astype(str) == str(most_common)).astype(int)
                except Exception as e:
                    logger.warning(f"Erro processando {m_col}: {e} - pulando one-hot encoding")
    
    logger.info("Features M categóricas MÍNIMAS criadas")
    return df


def create_advanced_interaction_features(df):
    """
    Cria features MÍNIMAS de interação entre grupos de variáveis.
    REDUÇÃO AGRESSIVA: de 118+ para ~4 features mais essenciais.
    """
    logger.info("Criando features de interação MÍNIMAS...")
    
    # APENAS 2 interações entre features de ALTA importância
    
    # 1. TransactionDT x TransactionAmt (top 2 features) - ESSENCIAL
    if 'TransactionDT' in df.columns and 'TransactionAmt' in df.columns:
        # Razão tempo/valor (mais importante)
        df['dt_per_amt'] = df['TransactionDT'] / (df['TransactionAmt'] + 1)
    
    # 2. APENAS 1 card combination (converter para string para evitar problemas)
    if 'card1' in df.columns and 'card2' in df.columns:
        df['card1_card2_combined'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    
    # 3. Missing patterns ULTRA SIMPLIFICADO
    missing_cols = [col for col in df.columns if '_is_missing' in col]
    if len(missing_cols) >= 2:
        df['total_missing_features'] = df[missing_cols].sum(axis=1)
    
    # REMOVER: TODAS as outras interações (baixo valor incremental)
    
    logger.info("Features de interação MÍNIMAS criadas (3 features)")
    return df


def create_advanced_outlier_features(df):
    """
    Cria features MÍNIMAS para identificação de valores atípicos.
    
    REDUÇÃO AGRESSIVA: APENAS 2 outlier features mais importantes.
    """
    logger.info("Criando features de outliers MÍNIMAS...")
    
    # APENAS 1 outlier de valor (TransactionAmt)
    if 'TransactionAmt' in df.columns:
        amount_q99 = df['TransactionAmt'].quantile(0.99)
        df['is_amount_extreme_high'] = (df['TransactionAmt'] >= amount_q99).astype(int)
        # REMOVER: todas as outras features de outlier (baixo valor incremental)
    
    # APENAS 1 outlier temporal (se existir trans_hour)
    if 'trans_hour' in df.columns:
        # Transações em horários raros (madrugada)
        df['is_rare_hour'] = ((df['trans_hour'] >= 2) & (df['trans_hour'] <= 5)).astype(int)
    
    # REMOVER: todas as outras features de outlier (V, cards, suspicion scores, etc.)
    
    logger.info("Features de outliers MÍNIMAS criadas (2 features)")
    return df


def save_to_parquet(df, output_path):
    """
    Salva DataFrame em formato Parquet otimizado.
    
    Args:
        df (DataFrame): DataFrame para salvar
        output_path (str): Caminho do arquivo de saída
    """
    logger.info(f"Salvando DataFrame em formato Parquet...")
    logger.info(f"Shape final: {df.shape}")
    
    # Otimizações de tipos para reduzir tamanho
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= 0 and df[col].max() <= 4294967295:
                df[col] = df[col].astype('uint32')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    # Salvar com compressão
    df.to_parquet(
        output_path,
        compression='snappy',
        index=False,
        engine='pyarrow'
    )
    
    logger.info(f"Arquivo Parquet salvo: {output_path}")


def generate_metadata(parquet_path, metadata_path, input_path):
    """
    Gera arquivo JSON com metadados do dataset processado.
    """
    logger.info("=== GERANDO METADADOS ===")
    
    # Carregar dataset para análise
    df = pd.read_parquet(parquet_path)
    
    # Metadados base
    metadata = {
        "dataset_name": "ieee-cis",
        "dataset_description": "Credit Card Fraud Detection Dataset (IEEE-CIS) - ENHANCED FEATURES + FEATURE ENGINEERING AVANÇADO",
        "source": "https://www.kaggle.com/c/ieee-fraud-detection",
        "processing_info": {
            "seed": SEED,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_philosophy": "RAW DATA PRESERVATION - Dados mantidos em escala bruta/natural",
            "normalization_note": "Normalizações serão aplicadas no pipeline de ML para evitar data leakage",
            "treatments_applied": [
                "Enhanced temporal features (TransactionDT analysis) - MINIMAL",
                "Essential monetary features (TransactionAmt transformations) - MINIMAL",
                "Minimal card features (card1-card6, card4_* encoded) - MINIMAL",
                "Minimal address features (addr1, addr2) - MINIMAL",
                "Reduced distance features (dist1, dist2) - REDUCED",
                "Minimal email domain features (P_emaildomain, R_emaildomain) - MINIMAL",
                "Minimal categorical C features (C2-C14) - MINIMAL",
                "Minimal categorical D features (D2-D15) - MINIMAL",
                "Minimal V features (V24-V332, 57 non-sequential) - MINIMAL",
                "Minimal product features (ProductCD_*) - MINIMAL",
                "Minimal M categorical features (M1_F, M2_F/T, M3_F/T) - MINIMAL",
                "Minimal interaction features - MINIMAL",
                "Minimal outlier detection - MINIMAL"
            ]
        },
        "file_info": {
            "parquet_path": str(parquet_path),
            "file_size_mb": round(os.path.getsize(parquet_path) / (1024 * 1024), 2),
            "original_parquet": {
                "path": str(input_path),
                "size_mb": round(os.path.getsize(input_path) / (1024 * 1024), 2)
            }
        },
        "data_structure": {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        },
        "target_variable": {
            "name": "isFraud",
            "description": "Fraud indicator (0=legitimate, 1=fraud)",
            "distribution": {
                "legitimate": int((df['isFraud'] == 0).sum()),
                "fraud": int((df['isFraud'] == 1).sum()),
                "fraud_percentage": round((df['isFraud'] == 1).mean() * 100, 4)
            }
        },
        "feature_categories": {
            "original_features": (["TransactionDT", "TransactionAmt", "isFraud"] + 
                                [f"V{i}" for i in range(2, 60) if f"V{i}" in df.columns] +  # V2-V59
                                ["card1", "card2", "card3", "card5", "card6"] +
                                ["card4", "addr1", "addr2", "dist1", "dist2"] +
                                ["P_emaildomain", "R_emaildomain"] +
                                ["C2", "C3", "C4", "C5", "C7", "C8", "C9", "C10", "C12", "C13", "C14"] +
                                ["D2", "D3", "D4", "D6", "D7", "D8", "D9", "D10", "D11", "D13", "D14", "D15"] +
                                ["ProductCD"] +
                                [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]),  # M1-M9
            "temporal_features": [col for col in df.columns if any(x in col.lower() for x in 
                                ['trans_', 'time_', 'hour', 'day', 'weekday', 'weekend', 'night', 'business'])],
            "amount_features": [col for col in df.columns if 'amount' in col.lower()],
            "card_features": [col for col in df.columns if col.startswith('card') and 
                            ('frequency' in col or 'missing' in col or 'rare' in col or 'common' in col)],
            "address_features": [col for col in df.columns if col.startswith('addr') and 
                               ('frequency' in col or 'missing' in col or 'rare' in col or 'rank' in col)],
            "distance_features": [col for col in df.columns if 'dist' in col.lower()],
            "email_features": [col for col in df.columns if 'email' in col.lower()],
            "c_features": [col for col in df.columns if col.startswith('C') or col.startswith('c_')],
            "d_features": [col for col in df.columns if col.startswith('D') or col.startswith('d_')],
            "v_features": [col for col in df.columns if col.startswith('V') or col.startswith('v_')],
            "product_features": [col for col in df.columns if 'product' in col.lower()],
            "m_features": [col for col in df.columns if col.startswith('M') or col.startswith('m_')],
            "interaction_features": [col for col in df.columns if 'interaction' in col or '_combined' in col],
            "outlier_features": [col for col in df.columns if any(x in col for x in 
                               ['outlier', 'extreme', 'suspicious', 'rare', 'zscore'])],
            "flag_features": [col for col in df.columns if col.startswith('is_')]
        },
        "columns": {}
    }
    
    # Análise simplificada por coluna
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique())
        }
        
        # Estatísticas básicas para colunas numéricas
        if df[col].dtype in ['int32', 'int64', 'uint32', 'float32', 'float64', 'int8']:
            col_info.update({
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 6),
                "std": round(float(df[col].std()), 6)
            })
        
        metadata["columns"][col] = col_info
    
    # Salvar metadados
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadados salvos em: {metadata_path}")
    return metadata


def main():
    """
    Função principal para processamento Parquet -> Enhanced Parquet do dataset IEEE-CIS.
    
    Mantém dados em formato bruto com feature engineering avançado para
    preservar integridade científica e evitar data leakage.
    """
    logger.info("=== INICIANDO PROCESSAMENTO PARQUET PARA ENHANCED PARQUET - IEEE-CIS ===")
    logger.info(f"Entrada: {INPUT_PARQUET_PATH}")
    logger.info(f"Saída: {PARQUET_PATH}")
    logger.info(f"Metadados: {METADATA_PATH}")
    
    try:
        # Validações iniciais
        validate_input_file(str(INPUT_PARQUET_PATH))
        create_directories(str(PARQUET_PATH))
        
        # Verificar disponibilidade de GPU
        gpu_available = check_gpu_availability()
        
        # Início da conversão
        start_time = time.time()
        
        # Carregar e processar dados
        processed_df = load_and_process_data(str(INPUT_PARQUET_PATH))
        
        # Salvar em formato Parquet
        save_to_parquet(processed_df, str(PARQUET_PATH))
        
        # Gerar metadados
        metadata = generate_metadata(str(PARQUET_PATH), str(METADATA_PATH), str(INPUT_PARQUET_PATH))
        
        # Relatório final
        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"=== PROCESSAMENTO AVANÇADO CONCLUÍDO EM {elapsed_time:.2f} SEGUNDOS ===")
        logger.info(f"Arquivo Enhanced Parquet: {PARQUET_PATH}")
        logger.info(f"Metadados: {METADATA_PATH}")
        logger.info(f"Dataset final: {metadata['data_structure']['total_rows']:,} registros, {metadata['data_structure']['total_columns']} colunas")
        logger.info(f"Taxa de fraude: {metadata['target_variable']['distribution']['fraud_percentage']}%")
        logger.info(f"Tamanho do arquivo: {metadata['file_info']['file_size_mb']} MB")
        logger.info("IMPORTANTE: Dados mantidos em escala bruta para preservar integridade científica")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
