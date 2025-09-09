"""
Módulo de Engenharia de Features para IEEE-CIS Dataset

Este módulo é responsável por criar e consolidar todas as features (originais, derivadas 
e estatísticas) em um único arquivo parquet que será usado na construção dos grafos.

FUNCIONALIDADES:
1. Carrega dataset original IEEE-CIS (novo formato com 165 colunas)
2. Mantém features originais (TransactionAmt, V2-V59, C2-C14, D2-D15, M1-M9, etc.)
3. Mantém 65+ features derivadas já presentes (temporais, categóricas, agregadas)
4. Cria features estatísticas adicionais baseadas em análises:
   - Features de clustering (cluster_id com one-hot encoding)
   - Features de detecção de anomalias (anomaly_score)
   - Features de PCA (componentes principais)
   - Features temporais derivadas (temporal_position, evitando duplicar hour_sin)
   - Features de correlação (feature_magnitude, high_value_features usando V2-V59)
5. Salva parquet consolidado com todas as features organizadas por categoria

CORREÇÕES CRÍTICAS DE VAZAMENTO DE DADOS:
- TransactionDT usado apenas para criar features temporais derivadas (não como feature direta)
- isFraud mantido apenas como label separado (não como feature)

ADAPTAÇÕES PARA NOVO PARQUET:
- Time → TransactionDT, Amount → TransactionAmt, Class → isFraud
- Features V expandidas: V1-V28 → V2-V59 (55 features V)
- Novas categorias: C2-C14, D2-D15, M1-M9, dist1-dist2, email domains
- 65+ features derivadas pré-existentes mantidas e organizadas

Author: MultiStatGraph Framework
Date: 2025-06-26
Version: 2.0
"""

import logging
import os
import json
import time
import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import gc

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração de seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega o dataset IEEE-CIS a partir do arquivo Parquet.
    
    Args:
        file_path (str): Caminho para o arquivo Parquet
        
    Returns:
        pd.DataFrame: DataFrame contendo os dados do IEEE-CIS
    """
    logger.info(f"Carregando dados de {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    # Otimização de memória: usar tipos de dados mais eficientes
    df = pd.read_parquet(file_path)
    
    # Converter colunas numéricas para tipos mais eficientes quando possível
    for col in df.select_dtypes(include=['float64']).columns:
        if col not in ['TransactionAmt', 'TransactionDT']:  # Manter precisão para colunas críticas
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        if col != 'isFraud':  # Manter isFraud como int64 por segurança
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    logger.info(f"Dados carregados: {df.shape[0]} transações, {df.shape[1]} features")
    logger.info(f"Distribuição de classes: {df['isFraud'].value_counts().to_dict()}")
    
    return df


def load_statistical_insights() -> Dict[str, Any]:
    """
    Carrega os resultados consolidados das análises estatísticas.
    
    Returns:
        Dict[str, Any]: Dicionário com insights estatísticos consolidados
    """
    logger.info("Carregando insights consolidados das análises estatísticas")
    
    # Carregar arquivo consolidado principal
    consolidated_path = "/graphsentinel_2.0/data/statis/ieee-cis/consolidated_multistat_results.json"
    
    try:
        with open(consolidated_path, 'r') as f:
            consolidated_results = json.load(f)
        logger.info(f"✓ Carregado arquivo consolidado: {consolidated_path}")
        
        # Extrair recomendações integradas
        recommendations = consolidated_results.get('integrated_graph_construction_recommendations', {})
        individual_analyses = consolidated_results.get('individual_analyses', {})
        
        # Organizar insights para compatibilidade com o código existente
        stat_insights = {
            'consolidated_recommendations': recommendations,
            'individual_analyses': individual_analyses,
            'dataset_info': consolidated_results.get('dataset_info', {}),
            'summary': consolidated_results.get('summary', {}),
        }
        
        # Carregar dados auxiliares (cluster assignments e anomaly scores) 
        base_path = "/graphsentinel_2.0/data/statis/ieee-cis"
        
        # Carregar cluster assignments
        cluster_path = os.path.join(base_path, "cluster_analysis", "cluster_assignments.csv")
        try:
            if os.path.exists(cluster_path):
                cluster_df = pd.read_csv(cluster_path, index_col=0)
                stat_insights['cluster_assignments'] = cluster_df
                logger.info(f"✓ Carregado cluster assignments: {len(cluster_df)} registros")
            else:
                logger.warning(f"Arquivo cluster assignments não encontrado: {cluster_path}")
                stat_insights['cluster_assignments'] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar cluster assignments: {e}")
            stat_insights['cluster_assignments'] = pd.DataFrame()
        
        # Carregar anomaly scores
        anomaly_path = os.path.join(base_path, "anomaly_scores.csv")
        try:
            if os.path.exists(anomaly_path):
                # Não usar index_col=0 para manter anomaly_score como coluna
                anomaly_df = pd.read_csv(anomaly_path)
                # Definir o índice manualmente se necessário
                anomaly_df.index = range(len(anomaly_df))
                
                stat_insights['anomaly_scores'] = anomaly_df
                logger.info(f"✓ Carregado anomaly scores: {len(anomaly_df)} registros")
                logger.info(f"✓ Colunas disponíveis: {list(anomaly_df.columns)}")
            else:
                logger.warning(f"Arquivo anomaly scores não encontrado: {anomaly_path}")
                stat_insights['anomaly_scores'] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar anomaly scores: {e}")
            stat_insights['anomaly_scores'] = pd.DataFrame()
        
        return stat_insights
        
    except FileNotFoundError:
        logger.error(f"Arquivo consolidado não encontrado: {consolidated_path}")
        logger.warning("Executando com configuração padrão (é recomendado executar 0_run_statistical_analysis.py primeiro)")
        
        # Retornar configuração padrão básica
        return {
            'consolidated_recommendations': {
                'node_attributes': {
                    'final_priority_features': [f'V{i}' for i in range(1, 16)],
                    'recommended_node_features': {
                        'critical': ['V12'],
                        'important': ['V3', 'V7', 'V17'],
                        'moderate': ['V9', 'V16', 'V18']
                    }
                },
                'edge_construction_strategies': {}
            },
            'cluster_assignments': pd.DataFrame(),
            'anomaly_scores': pd.DataFrame()
        }
    
    except Exception as e:
        logger.error(f"Erro ao carregar insights consolidados: {e}")
        raise


def load_real_pca_data() -> pd.DataFrame:
    """
    Carrega os dados de PCA real pré-calculados.
    
    Returns:
        pd.DataFrame: DataFrame com componentes principais ou DataFrame vazio se não disponível
    """
    pca_path = "data/statis/ieee-cis/pca_analysis/pca_transformed_sample.csv"
    
    try:
        if os.path.exists(pca_path):
            # Carregar apenas primeiros 6 componentes com variância significativa
            pca_df = pd.read_csv(pca_path, index_col=0)
            # Filtrar apenas os componentes com variância > 3% (baseado na análise)
            significant_components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
            available_components = [col for col in significant_components if col in pca_df.columns]
            
            if available_components:
                logger.info(f"✓ PCA real carregado: {len(available_components)} componentes significativos")
                return pca_df[available_components]
            else:
                logger.warning("⚠️ Nenhum componente PCA significativo encontrado")
                return pd.DataFrame()
        else:
            logger.warning(f"⚠️ Arquivo PCA não encontrado: {pca_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Erro ao carregar PCA real: {e}")
        return pd.DataFrame()


def create_statistical_features(df_original: pd.DataFrame, stat_insights: Dict[str, Any]) -> pd.DataFrame:
    """
    Cria todas as features estatísticas baseadas nas análises.
    
    Args:
        df_original (pd.DataFrame): DataFrame original com dados das transações
        stat_insights (Dict[str, Any]): Insights das análises estatísticas
        
    Returns:
        pd.DataFrame: DataFrame com features estatísticas adicionadas
    """
    logger.info("🔧 Criando features estatísticas baseadas nas análises")
    
    # Começar com uma cópia das features base (incluindo TODAS as features do parquet exceto TransactionDT e isFraud)
    # Features disponíveis no novo parquet: TransactionAmt, card1-card6, ProductCD, addr1-addr2, dist1-dist2, 
    # P_emaildomain, R_emaildomain, C2-C14, D2-D15, M1-M9, V2-V59, + 65+ features derivadas
    excluded_cols = ['TransactionDT', 'isFraud']
    base_features = [col for col in df_original.columns if col not in excluded_cols]
    df_features = df_original[base_features].copy()
    
    # Carregar dados necessários
    cluster_assignments = stat_insights.get('cluster_assignments')
    anomaly_scores = stat_insights.get('anomaly_scores')
    pca_data = load_real_pca_data()
    
    features_added = []
    
    # 1. FEATURES DE CLUSTERING (one-hot encoding)
    if cluster_assignments is not None and not cluster_assignments.empty:
        logger.info("📊 Adicionando features de clustering...")
        
        # Ajustar índices se necessário
        if len(cluster_assignments) != len(df_original):
            logger.info(f"🔧 Ajustando índices de cluster: {len(cluster_assignments)} -> {len(df_original)}")
            if len(cluster_assignments) < len(df_original):
                cluster_assignments = cluster_assignments.loc[cluster_assignments.index.repeat(len(df_original) // len(cluster_assignments) + 1)][:len(df_original)]
            else:
                cluster_assignments = cluster_assignments.iloc[:len(df_original)]
            cluster_assignments.index = df_original.index
        
        unique_clusters = sorted(cluster_assignments['cluster_id'].unique())
        for cluster in unique_clusters:
            cluster_col = f'cluster_{cluster}'
            df_features[cluster_col] = cluster_assignments['cluster_id'].apply(
                lambda x: 1.0 if x == cluster else 0.0
            )
        features_added.append(f"cluster one-hot ({len(unique_clusters)} clusters)")
    
    # 2. FEATURES DE ANOMALIA
    if anomaly_scores is not None and not anomaly_scores.empty:
        logger.info("🚨 Adicionando features de detecção de anomalias...")
        
        # Ajustar índices se necessário
        if len(anomaly_scores) != len(df_original):
            logger.info(f"🔧 Ajustando índices de anomaly: {len(anomaly_scores)} -> {len(df_original)}")
            if len(anomaly_scores) < len(df_original):
                anomaly_scores = anomaly_scores.loc[anomaly_scores.index.repeat(len(df_original) // len(anomaly_scores) + 1)][:len(df_original)]
            else:
                anomaly_scores = anomaly_scores.iloc[:len(df_original)]
            anomaly_scores.index = df_original.index
        
        if 'anomaly_score' in anomaly_scores.columns:
            df_features['anomaly_score'] = anomaly_scores['anomaly_score']
            features_added.append("anomaly_score")
    
    # 3. FEATURES DE PCA (Componentes Principais)
    if not pca_data.empty:
        logger.info("📈 Adicionando features de PCA...")
        
        # Ajustar índices se necessário
        if len(pca_data) != len(df_original):
            logger.info(f"🔧 Ajustando índices de PCA: {len(pca_data)} -> {len(df_original)}")
            if len(pca_data) < len(df_original):
                pca_data = pca_data.loc[pca_data.index.repeat(len(df_original) // len(pca_data) + 1)][:len(df_original)]
            else:
                pca_data = pca_data.iloc[:len(df_original)]
            pca_data.index = df_original.index
        
        for pc_col in pca_data.columns:
            # Usar nomes consistentes: pc_1, pc_2, etc.
            pc_num = pc_col.replace('PC', '').lower()
            feature_name = f'pc_{pc_num}'
            df_features[feature_name] = pca_data[pc_col]
        features_added.append(f"PCA real ({len(pca_data.columns)} componentes)")
    else:
        logger.warning("⚠️ PCA real não disponível, pulando features PCA")
    
    # 4. FEATURES TEMPORAIS DERIVADAS
    if 'TransactionDT' in df_original.columns:
        logger.info("⏰ Adicionando features temporais derivadas...")
        
        time_vals = df_original['TransactionDT']
        
        # Temporal position: posição relativa normalizada (0-1)
        min_time = time_vals.min()
        max_time = time_vals.max()
        time_range = max_time - min_time
        if time_range > 0:
            df_features['temporal_position'] = (time_vals - min_time) / time_range
        
        # Evitar duplicar hour_sin se já existir (já existe no parquet)
        if 'hour_sin' not in df_original.columns:
            # Hour of day: padrão cíclico (sin/cos encoding)
            hour_of_day = (time_vals % 86400) / 3600  # Converter para hora do dia (0-24)
            df_features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
            features_added.append("features temporais derivadas (temporal_position, hour_sin, hour_cos)")
        else:
            features_added.append("features temporais derivadas (temporal_position)")
    elif 'Time' in df_original.columns:
        # Fallback para compatibilidade com parquets antigos
        logger.info("⏰ Adicionando features temporais derivadas (usando Time como fallback)...")
        
        time_vals = df_original['Time']
        
        # Temporal position: posição relativa normalizada (0-1)
        min_time = time_vals.min()
        max_time = time_vals.max()
        time_range = max_time - min_time
        if time_range > 0:
            df_features['temporal_position'] = (time_vals - min_time) / time_range
        
        # Hour of day: padrão cíclico (sin/cos encoding)
        hour_of_day = (time_vals % 86400) / 3600  # Converter para hora do dia (0-24)
        df_features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        
        features_added.append("features temporais derivadas (temporal_position, hour_sin, hour_cos)")
    
    # 5. FEATURES DE CORRELAÇÃO/MAGNITUDE  
    # Usar todas as features numéricas V disponíveis (V2-V59, excluindo as ausentes)
    v_features = [col for col in df_original.columns if col.startswith('V') and col[1:].isdigit()]
    if v_features:
        logger.info(f"🔗 Adicionando features de correlação usando {len(v_features)} features V...")
        
        # Feature magnitude (proxy para "unusualness")
        feature_values = df_original[v_features].values
        df_features['feature_magnitude'] = np.sqrt(np.sum(feature_values ** 2, axis=1))
        
        # Count of high-value features
        df_features['high_value_features'] = np.sum(np.abs(feature_values) > 1.0, axis=1).astype(float)
        
        features_added.append(f"features de correlação (feature_magnitude, high_value_features) - {len(v_features)} features V")
    
    logger.info(f"✓ Features estatísticas adicionadas:")
    for feature_info in features_added:
        logger.info(f"  - {feature_info}")
    
    logger.info(f"✓ Total de features: {len(df_features.columns)} (originais: {len(base_features)}, novas: {len(df_features.columns) - len(base_features)})")
    
    return df_features


def create_consolidated_dataset(df_original: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Cria dataset consolidado com features originais + derivadas + estatísticas + labels.
    
    Args:
        df_original (pd.DataFrame): DataFrame original (com features derivadas do data preparation)
        df_features (pd.DataFrame): DataFrame com features estatísticas
        
    Returns:
        pd.DataFrame: Dataset consolidado
    """
    logger.info("🔗 Consolidando dataset final...")
    
    # Identificar categorias de features do dataset original
    # Features originais brutas do IEEE-CIS (baseadas nas colunas reais do novo parquet)
    features_originais_brutas = ['TransactionDT', 'TransactionAmt', 'card1', 'ProductCD', 'card2', 'card3', 
                                'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
                                'P_emaildomain', 'R_emaildomain'] + \
                               [f'C{i}' for i in [2,3,4,5,7,8,9,10,12,13,14]] + \
                               [f'D{i}' for i in [2,3,4,6,7,8,9,10,11,13,14,15]] + \
                               [f'M{i}' for i in range(1,10)] + \
                               [col for col in df_original.columns if col.startswith('V') and col[1:].isdigit()] + \
                               ['isFraud']
    
    # Features derivadas do data preparation (novas features engenheiradas do parquet atual)
    features_derivadas_existentes = ['transaction_timestamp', 'trans_hour', 'trans_weekday', 'is_night', 'hour_sin',
                                   'amount_log', 'is_small_amount', 'is_round_amount', 'amount_rank',
                                   'card1_is_missing', 'card2_is_missing', 'card3_is_missing', 'card5_is_missing',
                                   'card4_is_missing', 'card4_visa', 'card4_mastercard', 'card6_is_missing',
                                   'card6_debit', 'card6_credit', 'card_missing_count', 'addr1_is_missing',
                                   'addr2_is_missing', 'addr_missing_count', 'addr1_addr2_same', 'dist1_is_missing',
                                   'dist1_category', 'dist1_is_zero', 'dist2_is_missing', 'dist2_category',
                                   'dist2_is_zero', 'dist_sum', 'dist_missing_count', 'P_emaildomain_is_missing',
                                   'P_emaildomain_gmail_com', 'P_emaildomain_yahoo_com', 'P_emaildomain_hotmail_com',
                                   'R_emaildomain_is_missing', 'R_emaildomain_gmail_com', 'R_emaildomain_hotmail_com',
                                   'R_emaildomain_anonymous_com', 'P_R_email_same', 'email_missing_count',
                                   'c_missing_count', 'd_missing_count', 'v_mean', 'v_missing_count',
                                   'V24_V38_product', 'v_profile_cluster', 'product_is_missing', 'ProductCD_W',
                                   'ProductCD_C', 'ProductCD_R', 'm_missing_count', 'M1_is_missing', 'M1_T',
                                   'M2_is_missing', 'M2_T', 'M3_is_missing', 'M3_T', 'dt_per_amt',
                                   'card1_card2_combined', 'total_missing_features', 'is_amount_extreme_high',
                                   'is_rare_hour']
    
    # Features estatísticas do MultiStatGraph Framework
    features_estatisticas = [col for col in df_features.columns 
                           if col not in features_originais_brutas and col not in features_derivadas_existentes]
    
    logger.info(f"  - Features originais brutas: {len(features_originais_brutas)}")
    logger.info(f"  - Features derivadas (data prep): {len(features_derivadas_existentes)}")
    logger.info(f"  - Features estatísticas (MultiStat): {len(features_estatisticas)}")
    
    # Criar dataset consolidado começando com as features originais brutas (exceto TransactionDT e isFraud)
    features_originais_para_manter = [col for col in features_originais_brutas 
                                     if col not in ['TransactionDT', 'isFraud'] and col in df_original.columns]
    df_consolidated = df_original[features_originais_para_manter].copy()
    
    # Adicionar features derivadas do data preparation (que existem no parquet atual)
    features_derivadas_para_adicionar = [col for col in features_derivadas_existentes 
                                        if col in df_original.columns]
    for col in features_derivadas_para_adicionar:
        df_consolidated[col] = df_original[col]
    
    # Adicionar features estatísticas do MultiStatGraph
    for col in features_estatisticas:
        df_consolidated[col] = df_features[col]
    
    # Adicionar TransactionDT e isFraud como metadados (não como features para o grafo)
    if 'TransactionDT' in df_original.columns:
        df_consolidated['TransactionDT'] = df_original['TransactionDT']
    
    if 'isFraud' in df_original.columns:
        df_consolidated['isFraud'] = df_original['isFraud']
    
    logger.info(f"✓ Dataset consolidado criado: {len(df_consolidated)} transações, {len(df_consolidated.columns)} colunas")
    
    # Organizar colunas por categoria: originais + derivadas + estatísticas + metadados
    cols_originais = features_originais_para_manter
    cols_derivadas = features_derivadas_para_adicionar
    cols_estatisticas = features_estatisticas
    cols_metadados = [col for col in ['TransactionDT', 'isFraud'] if col in df_consolidated.columns]
    
    # Reorganizar dataset com ordem lógica
    df_consolidated = df_consolidated[cols_originais + cols_derivadas + cols_estatisticas + cols_metadados]
    
    logger.info(f"  - Features originais: {len(cols_originais)}")
    logger.info(f"  - Features derivadas: {len(cols_derivadas)}")
    logger.info(f"  - Features estatísticas: {len(cols_estatisticas)}")
    logger.info(f"  - Metadados: {len(cols_metadados)}")
    
    return df_consolidated


def save_consolidated_dataset(df_consolidated: pd.DataFrame, output_path: str) -> None:
    """
    Salva o dataset consolidado em formato Parquet.
    
    Args:
        df_consolidated (pd.DataFrame): Dataset consolidado
        output_path (str): Caminho de saída
    """
    logger.info(f"💾 Salvando dataset consolidado em {output_path}")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar em formato Parquet
    df_consolidated.to_parquet(output_path, index=True)
    
    logger.info(f"✓ Dataset salvo com sucesso:")
    logger.info(f"  - Arquivo: {output_path}")
    logger.info(f"  - Transações: {len(df_consolidated):,}")
    logger.info(f"  - Colunas: {len(df_consolidated.columns)}")
    logger.info(f"  - Tamanho: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def main():
    """
    Função principal que orquestra a criação do dataset consolidado com todas as features.
    """
    start_time = time.time()
    
    logger.info("=== Iniciando Criação de Dataset Consolidado com Features Estatísticas (IEEE-CIS) ===")
    logger.info("🎯 Objetivo: Criar parquet unificado com features originais + derivadas + estatísticas")
    
    try:
        # Configurar caminhos
        input_path = "/graphsentinel_2.0/data/parquet/ieee-cis/ieee-cis.parquet"
        output_path = "/graphsentinel_2.0/data/parquet/ieee-cis/ieee-cis-features.parquet"
        
        # Etapa 1: Carregar dataset original
        logger.info("📁 Etapa 1/5: Carregando dataset original...")
        df_original = load_data(input_path)
        logger.info(f"✓ Dataset carregado: {len(df_original)} transações")
        
        # Etapa 2: Carregar insights estatísticos
        logger.info("📊 Etapa 2/5: Carregando insights estatísticos...")
        stat_insights = load_statistical_insights()
        logger.info("✓ Insights estatísticos carregados")
        
        # Etapa 3: Criar features estatísticas
        logger.info("🔧 Etapa 3/5: Criando features estatísticas...")
        df_features = create_statistical_features(df_original, stat_insights)
        logger.info(f"✓ Features estatísticas criadas: {len(df_features.columns)} colunas")
        
        # Etapa 4: Consolidar dataset
        logger.info("🔗 Etapa 4/5: Consolidando dataset final...")
        df_consolidated = create_consolidated_dataset(df_original, df_features)
        logger.info("✓ Dataset consolidado")
        
        # Etapa 5: Salvar dataset consolidado
        logger.info("💾 Etapa 5/5: Salvando dataset consolidado...")
        save_consolidated_dataset(df_consolidated, output_path)
        logger.info("✓ Dataset salvo")
        
        # Calcular tempo total
        elapsed_time = time.time() - start_time
        logger.info("=== ✅ Criação de Dataset Consolidado IEEE-CIS Concluída com Sucesso ===")
        logger.info(f"⏱️ Tempo total de execução: {elapsed_time:.2f} segundos")
        
        # Estatísticas finais
        logger.info("📈 Estatísticas Finais:")
        logger.info(f"  - Dataset original: {len(df_original):,} transações")
        logger.info(f"  - Total de colunas no dataset consolidado: {len(df_consolidated.columns)}")
        logger.info(f"  - Arquivo de saída: {output_path}")
        
        # Categorizar features para estatísticas detalhadas
        features_originais_brutas = ['TransactionAmt'] + [col for col in df_consolidated.columns if col.startswith('V') and col[1:].isdigit()] + \
                                   ['card1', 'ProductCD', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
                                    'P_emaildomain', 'R_emaildomain'] + \
                                   [f'C{i}' for i in [2,3,4,5,7,8,9,10,12,13,14]] + \
                                   [f'D{i}' for i in [2,3,4,6,7,8,9,10,11,13,14,15]] + \
                                   [f'M{i}' for i in range(1,10)]
        features_derivadas_existentes = ['transaction_timestamp', 'trans_hour', 'trans_weekday', 'is_night', 'hour_sin',
                                       'amount_log', 'is_small_amount', 'is_round_amount', 'amount_rank',
                                       'card1_is_missing', 'card2_is_missing', 'card3_is_missing', 'card5_is_missing',
                                       'card4_is_missing', 'card4_visa', 'card4_mastercard', 'card6_is_missing',
                                       'card6_debit', 'card6_credit', 'card_missing_count', 'addr1_is_missing',
                                       'addr2_is_missing', 'addr_missing_count', 'addr1_addr2_same', 'dist1_is_missing',
                                       'dist1_category', 'dist1_is_zero', 'dist2_is_missing', 'dist2_category',
                                       'dist2_is_zero', 'dist_sum', 'dist_missing_count', 'P_emaildomain_is_missing',
                                       'P_emaildomain_gmail_com', 'P_emaildomain_yahoo_com', 'P_emaildomain_hotmail_com',
                                       'R_emaildomain_is_missing', 'R_emaildomain_gmail_com', 'R_emaildomain_hotmail_com',
                                       'R_emaildomain_anonymous_com', 'P_R_email_same', 'email_missing_count',
                                       'c_missing_count', 'd_missing_count', 'v_mean', 'v_missing_count',
                                       'V24_V38_product', 'v_profile_cluster', 'product_is_missing', 'ProductCD_W',
                                       'ProductCD_C', 'ProductCD_R', 'm_missing_count', 'M1_is_missing', 'M1_T',
                                       'M2_is_missing', 'M2_T', 'M3_is_missing', 'M3_T', 'dt_per_amt',
                                       'card1_card2_combined', 'total_missing_features', 'is_amount_extreme_high',
                                       'is_rare_hour']
        
        # Mostrar tipos de features criadas
        feature_cols = [col for col in df_consolidated.columns if col not in ['TransactionDT', 'isFraud']]
        cluster_features = [col for col in feature_cols if col.startswith('cluster_')]
        pca_features = [col for col in feature_cols if col.startswith('pc_')]
        temporal_features = [col for col in feature_cols if col in ['temporal_position', 'hour_sin', 'hour_cos']]
        correlation_features = [col for col in feature_cols if col in ['feature_magnitude', 'high_value_features']]
        anomaly_features = [col for col in feature_cols if col == 'anomaly_score']
        
        total_features_estatisticas = len(cluster_features) + len(pca_features) + len(temporal_features) + len(correlation_features) + len(anomaly_features)
        
        logger.info("📊 Breakdown Detalhado de Features:")
        logger.info(f"  - Features originais brutas: {len([col for col in features_originais_brutas if col in df_consolidated.columns])}")
        logger.info(f"  - Features derivadas (data prep): {len([col for col in features_derivadas_existentes if col in df_consolidated.columns])}")
        logger.info(f"  - Features estatísticas (MultiStat): {total_features_estatisticas}")
        logger.info(f"    • Clustering: {len(cluster_features)}")
        logger.info(f"    • PCA: {len(pca_features)}")
        logger.info(f"    • Temporais: {len(temporal_features)}")
        logger.info(f"    • Correlação: {len(correlation_features)}")
        logger.info(f"    • Anomalia: {len(anomaly_features)}")
        logger.info(f"  - Metadados (TransactionDT, isFraud): 2")
        logger.info(f"  - TOTAL: {len(df_consolidated.columns)} colunas")
        
        # Liberar memória
        del df_original, df_features, df_consolidated
        gc.collect()
        
    except Exception as e:
        logger.error(f"Erro durante a criação do dataset consolidado: {e}")
        raise


if __name__ == "__main__":
    main()
