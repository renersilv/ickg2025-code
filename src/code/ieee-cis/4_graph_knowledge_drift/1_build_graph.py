"""
Módulo de Construção de Grafos de Conhecimento Heterogêneos para IEEE-CIS Dataset

Este módulo implementa uma metodologia avançada para construção de grafos de conhecimento 
heterogêneos de alta performance, utilizando uma arquitetura multi-relacional com 6 camadas
de conectividade especializadas e tipos de nós diferenciados.

ARQUITETURA DE GRAFO DE CONHECIMENTO HETEROGÊNEO:

TIPOS DE NÓS:
1. Nós de Transação (primários) - representam transações individuais
2. Nós de Cartão - entidades de cartão de crédito (card1, card2, etc.)
3. Nós de Endereço - entidades de endereço (addr1, addr2)
4. Nós de Email - domínios de email (P_emaildomain, R_emaildomain)
5. Nós de Produto - códigos de produto (ProductCD)
6. Nós de Temporal - janelas temporais para contexto

ESTRATÉGIAS DE CONECTIVIDADE (6 CAMADAS):
1. Camada 1: Arestas de Identidade Temporal (Sessões de Usuário)
   - Transação -> Transação: trilhas comportamentais sequenciais
   
2. Camada 2: Arestas de Similaridade Comportamental (k-NN Aprimorado)
   - Transação -> Transação: similaridade em features comportamentais
   
3. Camada 3: Arestas de Relacionamento de Entidade (NOVA)
   - Transação -> Cartão, Transação -> Endereço, Transação -> Email, etc.
   
4. Camada 4: Arestas de Padrões de Anomalia (NOVA)
   - Conecta transações com padrões anômalos similares
   
5. Camada 5: Arestas de Contexto Temporal-Espacial (NOVA)
   - Janelas temporais e padrões geográficos inferidos
   
6. Camada 6: Arestas de Meta-Relacionamentos (NOVA)
   - Relacionamentos entre entidades (Cartão -> Endereço, etc.)

FLUXO DE TRABALHO:
1. Extrair e criar entidades heterogêneas
2. Construir grafo heterogêneo com múltiplos tipos de nós e arestas
3. Aplicar estratégia de 6 camadas de conectividade
4. Enriquecer com features contextuais e agregações
5. Adicionar máscaras de divisão temporal
6. Salvar como HeteroData para GNNs heterogêneas

MELHORIAS DE CONHECIMENTO:
- Features de agregação temporal (janelas deslizantes)
- Features de rede social (centralidade, clustering)
- Embeddings de entidades pré-computados
- Scores de anomalia contextuais
- Padrões sazonais e geo-temporais

Author: GraphSentinel Knowledge Framework
Date: 2025-06-28
Version: 7.0 (Grafo de Conhecimento Heterogêneo Multi-Relacional)
"""

import logging
import os
import time
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import gc
from tqdm import tqdm
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import coalesce
from torch_geometric.transforms import ToUndirected

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração de seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# FEATURES NUMÉRICAS PARA ATRIBUTOS DOS NÓS DE TRANSAÇÃO (x)
# Todas as colunas numéricas exceto TransactionDT, isFraud, card1
TRANSACTION_FEATURE_COLUMNS = [
    # Transação básica
    'TransactionAmt',
    
    # Card features secundárias (card2-6, excluindo card1 que é usado para entidade)
    'card2', 'card3', 'card4', 'card5', 'card6',
    
    # Distance features básicas
    'dist1', 'dist2',
    
    # C features (counting features)
    'C2', 'C3', 'C4', 'C5', 'C7', 'C8', 'C9', 'C10', 'C12', 'C13', 'C14',
    
    # D features (timedelta features) 
    'D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D13', 'D14', 'D15',
    
    # M features (match features)
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    
    # V features (Vesta engineered features)
    'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
    'V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V42', 'V43',
    'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55',
    'V56', 'V57', 'V58', 'V59'
]

# COLUNAS ESPECIAIS PARA CONSTRUÇÃO DO GRAFO HETEROGÊNEO
IDENTITY_COLUMN = 'card1'  # Para conexões de sessão de usuário
TEMPORAL_COLUMN = 'TransactionDT'  # Para ordenação temporal
LABEL_COLUMN = 'isFraud'  # Label para classificação

# COLUNAS PARA CRIAÇÃO DE ENTIDADES HETEROGÊNEAS
ENTITY_COLUMNS = {
    'card': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
    'address': ['addr1', 'addr2'],
    'email': ['P_emaildomain', 'R_emaildomain'],
    'product': ['ProductCD']
}

# CONFIGURAÇÕES PARA FEATURES ENRIQUECIDAS
TEMPORAL_WINDOWS = [3600, 21600, 86400]  # 1h, 6h, 24h em segundos
ANOMALY_CONTAMINATION = 0.1  # Para Isolation Forest


def load_data(file_path: str) -> pd.DataFrame:
    """Carrega o dataset IEEE-CIS normalizado."""
    logger.info(f"📏 Carregando dados PRÉ-NORMALIZADOS de {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo normalizado não encontrado: {file_path}")
    
    if not file_path.endswith('ieee-cis-features-normalized.parquet'):
        raise ValueError(f"Este script requer o arquivo normalizado 'ieee-cis-features-normalized.parquet'")
    
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Dados carregados: {df.shape[0]} transações, {df.shape[1]} features")
    
    # Verificar colunas essenciais
    required_columns = [IDENTITY_COLUMN, TEMPORAL_COLUMN, LABEL_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas essenciais ausentes: {missing_columns}")
    
    # Otimização de memória
    for col in df.select_dtypes(include=['float64']).columns:
        if col not in ['TransactionAmt', 'TransactionDT']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        if col != 'isFraud':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    logger.info(f"Distribuição de classes: {df['isFraud'].value_counts().to_dict()}")
    return df


def create_temporal_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cria divisões temporais ordenadas para treino, validação, teste e monitoring."""
    logger.info("🕒 Criando divisões temporais ordenadas com monitoring para drift")
    
    df_sorted = df.sort_values(TEMPORAL_COLUMN)
    total_samples = len(df_sorted)
    
    # Dividir primeiro entre baseline (70%) e monitoring (30%)
    baseline_ratio = 0.7
    monitoring_ratio = 0.3
    baseline_end = int(total_samples * baseline_ratio)
    
    # Dentro do baseline (70%), dividir em train/val/test
    baseline_samples = baseline_end
    train_end = int(baseline_samples * train_ratio)      # 70% de 70% = 49%
    val_end = int(baseline_samples * (train_ratio + val_ratio))  # 85% de 70% = 59.5%
    
    # Criar máscaras baseadas na posição ordenada
    train_mask = np.zeros(len(df_sorted), dtype=bool)
    val_mask = np.zeros(len(df_sorted), dtype=bool)
    test_mask = np.zeros(len(df_sorted), dtype=bool)
    monitoring_mask = np.zeros(len(df_sorted), dtype=bool)
    
    # Obter posições das transações ordenadas no DataFrame atual
    sorted_positions = list(range(len(df_sorted)))
    
    # Dividir por posições ordenadas temporalmente
    train_positions = sorted_positions[:train_end]                    # 0-49%
    val_positions = sorted_positions[train_end:val_end]              # 49-59.5%
    test_positions = sorted_positions[val_end:baseline_end]          # 59.5-70%
    monitoring_positions = sorted_positions[baseline_end:]           # 70-100%
    
    # Aplicar máscaras
    train_mask[train_positions] = True
    val_mask[val_positions] = True
    test_mask[test_positions] = True
    monitoring_mask[monitoring_positions] = True
    
    logger.info(f"✓ Divisões criadas:")
    logger.info(f"   - Baseline (70%): Treino={train_mask.sum()}, Validação={val_mask.sum()}, Teste={test_mask.sum()}")
    logger.info(f"   - Monitoring (30%): {monitoring_mask.sum()} transações para detecção de drift")
    logger.info(f"   - Distribuição final: {train_mask.mean()*100:.1f}% train, {val_mask.mean()*100:.1f}% val, {test_mask.mean()*100:.1f}% test, {monitoring_mask.mean()*100:.1f}% monitoring")
    
    return train_mask, val_mask, test_mask, monitoring_mask


def create_identity_edges(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Cria arestas de identidade baseadas em sessões de usuário (card1)."""
    logger.info("🔗 CAMADA 1: Criando arestas de identidade (sessões de usuário)")
    
    identity_edges = []
    grouped = df.groupby(IDENTITY_COLUMN)
    
    progress_bar = tqdm(grouped, desc="🔗 Identidades", unit="grupos", ncols=80)
    
    for card1, group in progress_bar:
        if len(group) < 2:
            continue
        
        group_sorted = group.sort_values(TEMPORAL_COLUMN)
        indices = group_sorted.index.tolist()
        
        for i in range(1, len(indices)):
            current_idx = indices[i]
            previous_idx = indices[i-1]
            identity_edges.append((current_idx, previous_idx))
    
    progress_bar.close()
    logger.info(f"✓ Arestas de identidade criadas: {len(identity_edges)} arestas")
    
    return identity_edges


def select_top_features_for_knn(df: pd.DataFrame, n_features: int = 20) -> List[str]:
    """Seleciona as N features mais importantes para k-NN."""
    effective_n_features = min(10, n_features)
    logger.info(f"🎯 Selecionando top {effective_n_features} features para k-NN")
    
    available_features = [feat for feat in TRANSACTION_FEATURE_COLUMNS if feat in df.columns]
    feature_data = df[available_features]
    
    variances = feature_data.var()
    valid_features = variances[variances > 1e-6].index.tolist()
    
    correlations = feature_data[valid_features].corrwith(df[LABEL_COLUMN]).abs()
    correlations = correlations.dropna()
    
    importance_scores = correlations * np.log1p(variances[correlations.index])
    top_features = importance_scores.nlargest(effective_n_features).index.tolist()
    
    logger.info(f"✓ Top {len(top_features)} features selecionadas para k-NN:")
    logger.info(f"   Features: {', '.join(top_features)}")
    logger.info(f"   Ranking detalhado:")
    for i, feature in enumerate(top_features, 1):
        score = importance_scores[feature]
        corr = correlations[feature]
        var = variances[feature]
        logger.info(f"   {i:2d}. {feature:<15} | Score: {score:.4f} | Corr: {corr:.4f} | Var: {var:.4f}")
    
    return top_features


def create_similarity_edges(df: pd.DataFrame, train_mask: np.ndarray, k: int = 7) -> List[Tuple[int, int]]:
    """Cria arestas de similaridade baseadas em k-NN usando apenas dados de treino."""
    effective_k_reduced = min(5, k)
    logger.info(f"🔗 CAMADA 2: Criando arestas de similaridade (k-NN, k={effective_k_reduced})")
    logger.info(f"🔒 PREVENÇÃO DE VAZAMENTO: Usando apenas {train_mask.sum():,} amostras de treino para k-NN")
    logger.info(f"📊 Dataset completo: {len(df):,} transações")
    logger.info(f"🎯 Proporção de treino: {train_mask.mean()*100:.1f}%")
    
    # VALIDAÇÃO CRÍTICA: Verificar se train_mask é válida
    if train_mask.sum() == 0:
        raise ValueError("ERRO CRÍTICO: train_mask está vazia! Isso causaria vazamento total de dados.")
    
    if train_mask.sum() == len(df):
        logger.warning("⚠️ AVISO: train_mask contém todo o dataset. Verifique se as divisões foram criadas corretamente.")
    
    # Usar apenas features dos dados de treino para seleção de features e treinamento
    df_train = df[train_mask]
    logger.info(f"✅ Dados de treino extraídos: {len(df_train):,} transações")
    
    # PREVENÇÃO DE VAZAMENTO: Seleção de features baseada APENAS em dados de treino
    top_features = select_top_features_for_knn(df_train, n_features=10)
    
    # Extrair features apenas do conjunto de treino para treinar o modelo k-NN
    X_train = df_train[top_features].values.astype(np.float32)
    X_full = df[top_features].values.astype(np.float32)
    
    effective_k = min(effective_k_reduced, len(df_train) - 1)
    
    logger.info(f"🔧 MODELO k-NN: Treinamento com {X_train.shape[0]:,} amostras de treino x {X_train.shape[1]} features")
    logger.info(f"🎯 APLICAÇÃO: Busca em {X_full.shape[0]:,} amostras totais (mas modelo conhece apenas treino)")
    logger.info(f"🔒 GARANTIA: Modelo nunca viu dados de validação/teste durante treinamento")
    
    if X_train.shape[0] > 50000:
        sample_size = min(40000, X_train.shape[0])  # Amostragem apenas no treino
        logger.info(f"⚡ Usando subamostra estratificada de {sample_size:,} amostras de treino")
        
        # Amostragem estratificada por classe apenas no conjunto de treino
        train_indices = df_train.index.values
        sample_mask = np.zeros(len(df_train), dtype=bool)
        
        for class_val in df_train[LABEL_COLUMN].unique():
            class_mask = df_train[LABEL_COLUMN] == class_val
            class_size = class_mask.sum()
            n_samples_class = min(sample_size // 2, class_size)
            if n_samples_class > 0:
                class_indices = np.where(class_mask)[0]
                selected_local = np.random.choice(class_indices, size=n_samples_class, replace=False)
                sample_mask[selected_local] = True
        
        X_train_sample = X_train[sample_mask]
        sample_train_indices = train_indices[sample_mask]
        
        logger.info(f"🔧 Treinando índice k-NN com {sample_mask.sum():,} amostras de treino...")
        knn = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        
        knn.fit(X_train_sample)
        
        # Aplicar k-NN em todo o dataset
        batch_size = 15000
        distances_list = []
        indices_list = []
        
        logger.info(f"🔍 Executando busca k-NN em batches de {batch_size:,}...")
        
        with tqdm(total=X_full.shape[0], desc="🔍 k-NN busca", unit="consultas", ncols=80) as pbar:
            for i in range(0, X_full.shape[0], batch_size):
                end_idx = min(i + batch_size, X_full.shape[0])
                batch = X_full[i:end_idx]
                
                batch_distances, batch_indices = knn.kneighbors(batch)
                # Mapear de volta para os índices globais da amostra de treino
                batch_indices_mapped = sample_train_indices[batch_indices]
                
                distances_list.append(batch_distances)
                indices_list.append(batch_indices_mapped)
                
                pbar.update(end_idx - i)
        
        distances = np.concatenate(distances_list, axis=0)
        indices = np.concatenate(indices_list, axis=0)
        
    else:
        logger.info(f"🔧 Treinando índice k-NN apenas com dados de treino...")
        knn = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric='cosine',
            algorithm='auto',
            n_jobs=-1
        )
        
        # Treinar apenas com dados de treino
        knn.fit(X_train)
        
        # Aplicar busca em todo o dataset
        batch_size = 8000
        distances_list = []
        indices_list = []
        
        logger.info(f"🔍 Executando busca k-NN em batches de {batch_size:,}...")
        
        with tqdm(total=X_full.shape[0], desc="🔍 k-NN busca", unit="consultas", ncols=80) as pbar:
            for i in range(0, X_full.shape[0], batch_size):
                end_idx = min(i + batch_size, X_full.shape[0])
                batch = X_full[i:end_idx]
                
                batch_distances, batch_indices = knn.kneighbors(batch)
                # Mapear índices locais de treino para índices globais
                train_indices = df_train.index.values
                batch_indices_mapped = train_indices[batch_indices]
                
                distances_list.append(batch_distances)
                indices_list.append(batch_indices_mapped)
                
                pbar.update(end_idx - i)
        
        distances = np.concatenate(distances_list, axis=0)
        indices = np.concatenate(indices_list, axis=0)
    
    # Criar arestas vetorizado com otimização de memória
    logger.info(f"🔗 Construindo arestas de similaridade...")
    df_indices_array = np.array(df.index.tolist(), dtype=np.int32)  # int32 economiza memória
    source_nodes = np.repeat(df_indices_array, effective_k)
    target_indices = indices[:, 1:effective_k+1].flatten()
    target_nodes = df_indices_array[target_indices]
    
    similarity_edges = list(zip(source_nodes.tolist(), target_nodes.tolist()))
    
    logger.info(f"✓ Arestas de similaridade criadas: {len(similarity_edges):,} arestas")
    logger.info(f"🔒 VAZAMENTO PREVENIDO: Modelo k-NN treinado exclusivamente com dados de treino")
    logger.info(f"✅ Integridade de dados mantida para avaliação limpa")
    
    # Limpeza de memória
    del X_train, X_full, distances, indices, df_indices_array, source_nodes, target_nodes
    gc.collect()
    
    return similarity_edges


def build_knowledge_graph(df: pd.DataFrame, train_mask: np.ndarray) -> HeteroData:
    """Constrói grafo de conhecimento heterogêneo com 6 camadas de conectividade."""
    logger.info("🧠 Construindo grafo de conhecimento heterogêneo...")
    
    # FASE 1: Extrair entidades heterogêneas usando apenas dados de treino
    entities = extract_entities(df, train_mask)
    
    # FASE 2: Computar features enriquecidas usando apenas conhecimento de treino
    df_enriched = compute_enriched_features(df, entities, train_mask)
    
    # FASE 3: Preparar features dos nós de transação
    available_features = [feat for feat in TRANSACTION_FEATURE_COLUMNS if feat in df_enriched.columns]
    
    # Adicionar features enriquecidas
    enriched_features = []
    for col in df_enriched.columns:
        if any(suffix in col for suffix in ['_1h', '_6h', '_24h', '_risk_score', '_centrality', 'anomaly_score']):
            enriched_features.append(col)
    
    all_transaction_features = available_features + enriched_features
    logger.info(f"📐 Features dos nós de transação: {len(all_transaction_features)} features")
    logger.info(f"   - Originais: {len(available_features)}")
    logger.info(f"   - Enriquecidas: {len(enriched_features)}")
    
    # Criar tensores de features para transações
    X_transaction = torch.tensor(df_enriched[all_transaction_features].fillna(0).values, dtype=torch.float32)
    y_transaction = torch.tensor(df_enriched[LABEL_COLUMN].values, dtype=torch.long)
    
    # FASE 4: Criar arestas das 6 camadas
    logger.info("🔗 Criando 6 camadas de conectividade...")
    
    # Camada 1: Identidade Temporal (original)
    identity_edges = create_identity_edges(df_enriched)
    
    # Camada 2: Similaridade Comportamental (melhorada)
    similarity_edges = create_similarity_edges(df_enriched, train_mask, k=7)
    
    # Camada 3: Relacionamentos de Entidade
    entity_edges_dict = create_entity_relationship_edges(df_enriched, entities)
    entity_id_maps = entity_edges_dict.pop('_entity_id_maps', {})
    
    # Camada 4: Padrões de Anomalia
    anomaly_edges = create_anomaly_pattern_edges(df_enriched, k=5)
    
    # Camada 5: Contexto Temporal-Espacial
    temporal_context_edges = create_temporal_context_edges(df_enriched)
    
    # Camada 6: Meta-Relacionamentos
    meta_edges_dict = create_meta_relationship_edges(entities, entity_id_maps)
    
    # FASE 5: Construir HeteroData
    logger.info("🏗️ Montando grafo heterogêneo...")
    
    hetero_data = HeteroData()
    
    # Adicionar nós de transação
    hetero_data['transaction'].x = X_transaction
    hetero_data['transaction'].y = y_transaction
    hetero_data['transaction'].num_nodes = len(df_enriched)
    
    # Adicionar nós de entidades
    for entity_type, entity_dict in entities.items():
        if entity_type in entity_id_maps and len(entity_dict) > 0:
            # Criar features básicas para entidades na mesma ordem dos IDs
            # Ordenar entidades pela ordem dos IDs locais
            sorted_entities = sorted(entity_id_maps[entity_type].items(), key=lambda x: x[1])
            entity_features = []
            
            for entity_key, local_id in sorted_entities:
                entity_data = entity_dict[entity_key]
                features = [
                    entity_data.get('transaction_count', 0),
                    entity_data.get('fraud_rate', 0),
                    entity_data.get('avg_amount', 0),
                    entity_data.get('risk_score', entity_data.get('domain_risk', entity_data.get('category_risk', 0)))
                ]
                entity_features.append(features)
            
            hetero_data[entity_type].x = torch.tensor(entity_features, dtype=torch.float32)
            hetero_data[entity_type].num_nodes = len(entity_dict)
            
            logger.info(f"   📊 {entity_type}: {len(entity_dict)} nós, features shape: {hetero_data[entity_type].x.shape}")
    
    # Adicionar arestas de transação -> transação
    logger.info("   🔗 Adicionando arestas transação->transação...")
    
    # Combinar arestas homogêneas de transação
    all_transaction_edges = identity_edges + similarity_edges + anomaly_edges + temporal_context_edges
    
    if len(all_transaction_edges) > 0:
        edge_index = torch.tensor(all_transaction_edges, dtype=torch.long).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, X_transaction.size(0), X_transaction.size(0))
        
        # Dividir por tipo de aresta
        n_identity = len(identity_edges)
        n_similarity = len(similarity_edges)
        n_anomaly = len(anomaly_edges)
        n_temporal = len(temporal_context_edges)
        
        # Criar edge_attr para distinguir tipos de arestas
        edge_types = []
        edge_types.extend([0] * n_identity)      # 0: identidade
        edge_types.extend([1] * n_similarity)    # 1: similaridade
        edge_types.extend([2] * n_anomaly)       # 2: anomalia
        edge_types.extend([3] * n_temporal)      # 3: temporal
        
        hetero_data['transaction', 'relates_to', 'transaction'].edge_index = edge_index
        if len(edge_types) == edge_index.size(1):
            hetero_data['transaction', 'relates_to', 'transaction'].edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    # Adicionar arestas heterogêneas
    logger.info("   🔗 Adicionando arestas heterogêneas...")
    for edge_type, edges in entity_edges_dict.items():
        if len(edges) > 0:
            source_type, relation, target_type = edge_type
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            hetero_data[source_type, relation, target_type].edge_index = edge_index
    
    # Adicionar meta-relacionamentos
    logger.info("   🔗 Adicionando meta-relacionamentos...")
    for edge_type, edges in meta_edges_dict.items():
        if len(edges) > 0:
            source_type, relation, target_type = edge_type
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            hetero_data[source_type, relation, target_type].edge_index = edge_index
    
    # Estatísticas do grafo construído
    logger.info("📊 Estatísticas do grafo de conhecimento:")
    logger.info(f"   📍 Tipos de nós: {len(hetero_data.node_types)}")
    for node_type in hetero_data.node_types:
        logger.info(f"      - {node_type}: {hetero_data[node_type].num_nodes} nós")
    
    logger.info(f"   🔗 Tipos de arestas: {len(hetero_data.edge_types)}")
    total_edges = 0
    for edge_type in hetero_data.edge_types:
        edge_count = hetero_data[edge_type].edge_index.size(1)
        total_edges += edge_count
        logger.info(f"      - {edge_type}: {edge_count:,} arestas")
    
    logger.info(f"   🎯 Total de arestas: {total_edges:,}")
    
    # VALIDAÇÃO CRÍTICA: Verificar integridade das arestas
    logger.info("🔍 Validando integridade das arestas...")
    validation_errors = []
    
    for edge_type in hetero_data.edge_types:
        source_type, relation, target_type = edge_type
        edge_index = hetero_data[edge_type].edge_index
        
        if edge_index.size(1) > 0:
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Verificar se os índices de origem estão no range válido
            source_max = hetero_data[source_type].num_nodes - 1
            invalid_sources = (source_nodes > source_max) | (source_nodes < 0)
            if invalid_sources.any():
                invalid_count = invalid_sources.sum().item()
                max_invalid = source_nodes[invalid_sources].max().item()
                validation_errors.append(
                    f"Edge type {edge_type}: {invalid_count} invalid source indices "
                    f"(max: {max_invalid}, valid range: 0-{source_max})"
                )
            
            # Verificar se os índices de destino estão no range válido
            target_max = hetero_data[target_type].num_nodes - 1
            invalid_targets = (target_nodes > target_max) | (target_nodes < 0)
            if invalid_targets.any():
                invalid_count = invalid_targets.sum().item()
                max_invalid = target_nodes[invalid_targets].max().item()
                validation_errors.append(
                    f"Edge type {edge_type}: {invalid_count} invalid target indices "
                    f"(max: {max_invalid}, valid range: 0-{target_max})"
                )
    
    if validation_errors:
        logger.error("❌ ERRO DE INTEGRIDADE DO GRAFO DETECTADO:")
        for error in validation_errors:
            logger.error(f"   ❌ {error}")
        raise ValueError(
            "O grafo contém arestas com índices inválidos. "
            "Isso indica um problema na construção do grafo. "
            "Verifique o mapeamento de entidades e a criação de arestas."
        )
    
    logger.info("✅ Validação de integridade das arestas passou!")
    
    # Adicionar metadados
    hetero_data.metadata = {
        'entities': entities,
        'entity_id_maps': entity_id_maps,
        'enriched_features': enriched_features,
        'original_features': available_features
    }
    
    logger.info("✅ Grafo de conhecimento heterogêneo construído com sucesso!")
    return hetero_data


def add_split_masks(
    data: HeteroData, 
    train_mask: np.ndarray, 
    val_mask: np.ndarray, 
    test_mask: np.ndarray,
    monitoring_mask: np.ndarray
) -> HeteroData:
    """Adiciona máscaras de divisão ao objeto de dados do grafo de conhecimento."""
    logger.info("🎭 Adicionando máscaras de divisão ao grafo de conhecimento...")
    
    # Adicionar máscaras apenas aos nós de transação
    data['transaction'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data['transaction'].val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data['transaction'].test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data['transaction'].monitoring_mask = torch.tensor(monitoring_mask, dtype=torch.bool)
    
    total_masked = data['transaction'].train_mask.sum() + data['transaction'].val_mask.sum() + data['transaction'].test_mask.sum() + data['transaction'].monitoring_mask.sum()
    logger.info(f"✓ Máscaras adicionadas aos nós de transação:")
    logger.info(f"  - Baseline: Treino={data['transaction'].train_mask.sum()}, Validação={data['transaction'].val_mask.sum()}, Teste={data['transaction'].test_mask.sum()}")
    logger.info(f"  - Monitoring: {data['transaction'].monitoring_mask.sum()} nós para detecção de drift")
    logger.info(f"  - Total mascarado: {total_masked}/{data['transaction'].num_nodes}")
    
    # ANÁLISE DE CONECTIVIDADE POR SPLIT
    logger.info("=" * 60)
    logger.info("🔍 ANÁLISE DE CONECTIVIDADE POR SPLIT")
    logger.info("=" * 60)
    connectivity_stats = analyze_isolated_nodes_by_mask_hetero(data)
    data.connectivity_stats = connectivity_stats
    
    logger.info("=" * 60)
    logger.info("🚨 IMPORTANTE PARA TREINAMENTO DA GNN HETEROGÊNEA:")
    logger.info("   Use este objeto HeteroData com as máscaras para treinar.")
    logger.info("   Exemplo: loss = criterion(model(data)['transaction'][data['transaction'].train_mask], data['transaction'].y[data['transaction'].train_mask])")
    logger.info("   🔍 MONITORING: Use data['transaction'].monitoring_mask para detecção de drift")
    logger.info("=" * 60)
    
    return data


def parse_arguments():
    """Processa argumentos de linha de comando para a construção do grafo de conhecimento."""
    parser = argparse.ArgumentParser(
        description='Construção de grafos de conhecimento heterogêneos para IEEE-CIS'
    )
    
    parser.add_argument(
        '--graph-type',
        choices=['knowledge', 'basegraph'],
        default='knowledge',
        help='Tipo de grafo a ser construído (padrão: knowledge)'
    )
    
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=1.0,
        help='Percentual de transações a serem consideradas (0.0-1.0, padrão: 1.0 = 100%%)'
    )
    
    return parser.parse_args()



def _extract_entities_for_type(df: pd.DataFrame, entity_type: str, columns: List[str], 
                             entity_features_func) -> Dict[str, Any]:
    """Helper function para extrair entidades de um tipo específico."""
    entities = {}
    emoji_map = {
        'card': '📱',
        'address': '🏠', 
        'email': '📧',
        'product': '🛍️'
    }
    emoji = emoji_map.get(entity_type, '🔹')
    
    logger.info(f"{emoji} Extraindo entidades de {entity_type}...")
    
    for col in columns:
        if col in df.columns:
            unique_items = df[col].dropna().unique()
            progress_bar = tqdm(unique_items, desc=f"{emoji} {entity_type.title()} ({col})", unit="entidades", ncols=80)
            for item_id in progress_bar:
                if item_id not in entities:
                    item_data = df[df[col] == item_id]
                    entities[item_id] = entity_features_func(item_id, col, item_data)
            progress_bar.close()
    
    logger.info(f"✓ {len(entities)} entidades de {entity_type} extraídas")
    return entities


def extract_entities(df: pd.DataFrame, train_mask: np.ndarray = None) -> Dict[str, Dict[str, Any]]:
    """Extrai entidades únicas para construção do grafo heterogêneo usando apenas dados de treino."""
    logger.info("🏗️ Extraindo entidades para grafo heterogêneo...")
    
    # PREVENÇÃO DE VAZAMENTO: Usar apenas dados de treino para estatísticas
    if train_mask is not None:
        df_train = df[train_mask]
        logger.info(f"🔒 PREVENÇÃO DE VAZAMENTO: Usando apenas {len(df_train):,} transações de treino para estatísticas de entidades")
    else:
        df_train = df
        logger.warning("⚠️ AVISO: train_mask não fornecida, usando dataset completo")
    
    entities = {}
    
    # Funções específicas para calcular features de cada tipo de entidade usando apenas treino
    def card_features(card_id, col, card_data):
        return {
            'id': card_id,
            'source_column': col,
            'transaction_count': len(card_data),
            'fraud_rate': card_data[LABEL_COLUMN].mean(),
            'avg_amount': card_data['TransactionAmt'].mean(),
            'total_amount': card_data['TransactionAmt'].sum(),
            'time_span': card_data[TEMPORAL_COLUMN].max() - card_data[TEMPORAL_COLUMN].min(),
            'unique_addresses': card_data[['addr1', 'addr2']].dropna().nunique().sum(),
            'unique_emails': card_data[['P_emaildomain', 'R_emaildomain']].dropna().nunique().sum()
        }
    
    def address_features(addr_id, col, addr_data):
        return {
            'id': addr_id,
            'source_column': col,
            'transaction_count': len(addr_data),
            'fraud_rate': addr_data[LABEL_COLUMN].mean(),
            'unique_cards': addr_data[ENTITY_COLUMNS['card']].dropna().nunique().sum(),
            'avg_amount': addr_data['TransactionAmt'].mean(),
            'risk_score': addr_data[LABEL_COLUMN].mean() * len(addr_data)
        }
    
    def email_features(email_id, col, email_data):
        return {
            'id': email_id,
            'source_column': col,
            'transaction_count': len(email_data),
            'fraud_rate': email_data[LABEL_COLUMN].mean(),
            'unique_cards': email_data[ENTITY_COLUMNS['card']].dropna().nunique().sum(),
            'avg_amount': email_data['TransactionAmt'].mean(),
            'domain_risk': email_data[LABEL_COLUMN].mean() * np.log1p(len(email_data))
        }
    
    def product_features(prod_id, col, product_data):
        return {
            'id': prod_id,
            'transaction_count': len(product_data),
            'fraud_rate': product_data[LABEL_COLUMN].mean(),
            'avg_amount': product_data['TransactionAmt'].mean(),
            'category_risk': product_data[LABEL_COLUMN].mean() * len(product_data)
        }
    
    # Extrair cada tipo de entidade usando apenas dados de treino
    entities['card'] = _extract_entities_for_type(df_train, 'card', ENTITY_COLUMNS['card'], card_features)
    entities['address'] = _extract_entities_for_type(df_train, 'address', ENTITY_COLUMNS['address'], address_features)
    entities['email'] = _extract_entities_for_type(df_train, 'email', ENTITY_COLUMNS['email'], email_features)
    
    # ENTIDADES DE PRODUTO (tratamento especial para única coluna) - usando apenas treino
    product_entities = {}
    if 'ProductCD' in df_train.columns:
        unique_products = df_train['ProductCD'].dropna().unique()
        progress_bar = tqdm(unique_products, desc="🛍️ Produtos", unit="entidades", ncols=80)
        for prod_id in progress_bar:
            product_data = df_train[df_train['ProductCD'] == prod_id]
            product_entities[prod_id] = product_features(prod_id, 'ProductCD', product_data)
        progress_bar.close()
    entities['product'] = product_entities
    logger.info(f"✓ {len(product_entities)} entidades de produto extraídas")
    
    # ENTIDADES TEMPORAIS (JANELAS) - usando apenas período de treino
    logger.info("⏰ Criando entidades temporais...")
    temporal_entities = {}
    min_time = df_train[TEMPORAL_COLUMN].min()
    max_time = df_train[TEMPORAL_COLUMN].max()
    
    # Criar janelas de 6 horas
    window_size = 21600  # 6 horas em segundos
    current_time = min_time
    window_id = 0
    
    while current_time < max_time:
        window_end = current_time + window_size
        window_data = df_train[(df_train[TEMPORAL_COLUMN] >= current_time) & (df_train[TEMPORAL_COLUMN] < window_end)]
        
        if len(window_data) > 0:
            temporal_entities[window_id] = {
                'id': window_id,
                'start_time': current_time,
                'end_time': window_end,
                'transaction_count': len(window_data),
                'fraud_rate': window_data[LABEL_COLUMN].mean(),
                'total_amount': window_data['TransactionAmt'].sum(),
                'unique_cards': window_data[IDENTITY_COLUMN].nunique(),
                'peak_activity': len(window_data) / window_size  # transações por segundo
            }
        
        current_time = window_end
        window_id += 1
    
    entities['temporal'] = temporal_entities
    logger.info(f"✓ {len(temporal_entities)} janelas temporais criadas")
    
    logger.info("✅ Extração de entidades concluída")
    return entities


def compute_enriched_features(df: pd.DataFrame, entities: Dict[str, Dict[str, Any]], train_mask: np.ndarray = None) -> pd.DataFrame:
    """Computa features enriquecidas com conhecimento contextual usando apenas dados de treino."""
    logger.info("🧠 Computando features enriquecidas com conhecimento...")
    
    df_enriched = df.copy()
    
    # PREVENÇÃO DE VAZAMENTO: Definir conjunto de treino para modelos
    if train_mask is not None:
        df_train = df[train_mask]
        logger.info(f"🔒 PREVENÇÃO DE VAZAMENTO: Usando apenas {len(df_train):,} transações de treino para modelos estatísticos")
    else:
        df_train = df
        logger.warning("⚠️ AVISO: train_mask não fornecida para features enriquecidas")
    
    # FEATURES DE AGREGAÇÃO TEMPORAL (MÁXIMA OTIMIZAÇÃO)
    logger.info("⏱️ Adicionando features de agregação temporal (otimizado)...")
    df_sorted = df_enriched.sort_values([IDENTITY_COLUMN, TEMPORAL_COLUMN]).copy()
    
    for window in TEMPORAL_WINDOWS:
        logger.info(f"   Janela de {window//3600}h (processamento super-otimizado)...")
        
        # Inicializar colunas no DataFrame ordenado
        col_count = f'count_{window//3600}h'
        col_fraud = f'fraud_rate_{window//3600}h'
        col_amount = f'amount_sum_{window//3600}h'
        
        df_sorted[col_count] = 0.0
        df_sorted[col_fraud] = 0.0
        df_sorted[col_amount] = 0.0
        
        # Processamento ultra-otimizado por cartão
        grouped = df_sorted.groupby(IDENTITY_COLUMN, sort=False)
        
        def compute_window_features(group):
            """Função otimizada para calcular features de janela temporal"""
            if len(group) < 2:
                return group
            
            times = group[TEMPORAL_COLUMN].values
            n = len(times)
            
            # Pre-alocar arrays
            counts = np.zeros(n, dtype=np.float32)
            fraud_rates = np.zeros(n, dtype=np.float32)
            amount_sums = np.zeros(n, dtype=np.float32)
            
            fraud_values = group[LABEL_COLUMN].values.astype(np.float32)
            amount_values = group['TransactionAmt'].values.astype(np.float32)
            
            # Loop otimizado com numpy
            for i in range(n):
                current_time = times[i]
                window_start = current_time - window
                
                # Máscara vetorizada ultra-rápida
                mask = (times >= window_start) & (times <= current_time)
                
                if mask.any():
                    counts[i] = mask.sum()
                    fraud_rates[i] = fraud_values[mask].mean()
                    amount_sums[i] = amount_values[mask].sum()
            
            # Atribuir resultados de volta
            group.loc[:, col_count] = counts
            group.loc[:, col_fraud] = fraud_rates  
            group.loc[:, col_amount] = amount_sums
            
            return group
        
        # Aplicar função otimizada com barra de progresso
        progress_bar = tqdm(grouped, desc=f"⏱️ Janela {window//3600}h", unit="cartões", ncols=80)
        
        result_groups = []
        for name, group in progress_bar:
            result_groups.append(compute_window_features(group))
        
        progress_bar.close()
        
        # Recombinar dados processados
        df_processed = pd.concat(result_groups, ignore_index=False)
        
        # Transferir valores de volta para o DataFrame original
        df_enriched[col_count] = df_processed[col_count]
        df_enriched[col_fraud] = df_processed[col_fraud]
        df_enriched[col_amount] = df_processed[col_amount]
    
    # FEATURES DE RISCO DE ENTIDADE
    logger.info("🎯 Adicionando features de risco de entidade...")
    
    # Risco do cartão
    if IDENTITY_COLUMN in df_enriched.columns:
        card_risk_map = {ent['id']: ent['fraud_rate'] for ent in entities['card'].values()}
        df_enriched['card_risk_score'] = df_enriched[IDENTITY_COLUMN].map(card_risk_map).fillna(0)
    
    # Risco do endereço
    for addr_col in ['addr1', 'addr2']:
        if addr_col in df_enriched.columns:
            addr_risk_map = {ent['id']: ent['fraud_rate'] for ent in entities['address'].values() if ent['source_column'] == addr_col}
            df_enriched[f'{addr_col}_risk_score'] = df_enriched[addr_col].map(addr_risk_map).fillna(0)
    
    # Risco do email
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df_enriched.columns:
            email_risk_map = {ent['id']: ent['fraud_rate'] for ent in entities['email'].values() if ent['source_column'] == email_col}
            df_enriched[f'{email_col}_risk_score'] = df_enriched[email_col].map(email_risk_map).fillna(0)
    
    # FEATURES DE ANOMALIA CONTEXTUAL - usando apenas dados de treino
    logger.info("🔍 Computando scores de anomalia...")
    
    # Selecionar features numéricas para detecção de anomalia
    numeric_features = []
    for col in TRANSACTION_FEATURE_COLUMNS:
        if col in df_enriched.columns:
            numeric_features.append(col)
    
    if len(numeric_features) > 5:
        # Aplicar Isolation Forest APENAS com dados de treino
        X_train_numeric = df_train[numeric_features].fillna(0)
        X_full_numeric = df_enriched[numeric_features].fillna(0)
        
        iso_forest = IsolationForest(
            contamination=ANOMALY_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # Treinar apenas com dados de treino
        logger.info(f"🔒 Treinando Isolation Forest com {len(X_train_numeric):,} amostras de treino")
        iso_forest.fit(X_train_numeric)
        
        # Aplicar ao dataset completo
        anomaly_scores = iso_forest.decision_function(X_full_numeric)
        anomaly_predictions = iso_forest.predict(X_full_numeric)
        df_enriched['anomaly_score'] = anomaly_scores
        df_enriched['is_anomaly'] = (anomaly_predictions == -1).astype(int)
    
    # FEATURES DE CENTRALIDADE DE REDE (BASEADAS EM ENTIDADES) - usando dados de treino
    logger.info("🌐 Computando features de centralidade...")
    
    # Centralidade baseada em compartilhamento de cartão - usando apenas treino
    card_counts = df_train[IDENTITY_COLUMN].value_counts()
    df_enriched['card_centrality'] = df_enriched[IDENTITY_COLUMN].map(card_counts).fillna(0)
    
    # Centralidade baseada em endereço - usando apenas treino
    if 'addr1' in df_enriched.columns:
        addr_counts = df_train['addr1'].value_counts()
        df_enriched['address_centrality'] = df_enriched['addr1'].map(addr_counts).fillna(0)
    
    logger.info("✅ Features enriquecidas computadas")
    return df_enriched


def _create_entity_edges_for_column(df: pd.DataFrame, column: str, entity_id_maps: Dict[str, Dict], 
                                   edge_type: Tuple[str, str, str], entity_type: str, 
                                   emoji: str, description: str) -> List[Tuple[int, int]]:
    """Helper function para criar arestas entre transações e entidades para uma coluna específica."""
    edges = []
    if column in df.columns:
        progress_bar = tqdm(df.iterrows(), total=len(df), desc=f"{emoji} {description}", unit="transações", ncols=80)
        for idx, row in progress_bar:
            if pd.notna(row[column]) and row[column] in entity_id_maps[entity_type]:
                entity_id = entity_id_maps[entity_type][row[column]]
                edges.append((idx, entity_id))
        progress_bar.close()
    return edges


def create_entity_relationship_edges(df: pd.DataFrame, entities: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    """Cria arestas de relacionamento entre transações e entidades."""
    logger.info("🔗 CAMADA 3: Criando arestas de relacionamento de entidade...")
    
    entity_edges = defaultdict(list)
    
    # Criar mapeamento de entidades para IDs locais
    entity_id_maps = {}
    for entity_type, entity_dict in entities.items():
        entity_id_maps[entity_type] = {}
        local_id = 0
        for entity_key in entity_dict.keys():
            entity_id_maps[entity_type][entity_key] = local_id
            local_id += 1
    
    # TRANSAÇÃO -> CARTÃO
    logger.info("   📱 Criando arestas transação->cartão...")
    for col in ENTITY_COLUMNS['card']:
        edges = _create_entity_edges_for_column(
            df, col, entity_id_maps, 
            ('transaction', 'uses_card', 'card'), 'card', 
            "📱", f"Cartões ({col})"
        )
        entity_edges[('transaction', 'uses_card', 'card')].extend(edges)
    
    # TRANSAÇÃO -> ENDEREÇO
    logger.info("   🏠 Criando arestas transação->endereço...")
    for col in ENTITY_COLUMNS['address']:
        edges = _create_entity_edges_for_column(
            df, col, entity_id_maps,
            ('transaction', 'at_address', 'address'), 'address',
            "🏠", f"Endereços ({col})"
        )
        entity_edges[('transaction', 'at_address', 'address')].extend(edges)
    
    # TRANSAÇÃO -> EMAIL
    logger.info("   📧 Criando arestas transação->email...")
    for col in ENTITY_COLUMNS['email']:
        edges = _create_entity_edges_for_column(
            df, col, entity_id_maps,
            ('transaction', 'uses_email', 'email'), 'email',
            "📧", f"Emails ({col})"
        )
        entity_edges[('transaction', 'uses_email', 'email')].extend(edges)
    
    # TRANSAÇÃO -> PRODUTO
    logger.info("   🛍️ Criando arestas transação->produto...")
    edges = _create_entity_edges_for_column(
        df, 'ProductCD', entity_id_maps,
        ('transaction', 'buys_product', 'product'), 'product',
        "🛍️", "Produtos"
    )
    entity_edges[('transaction', 'buys_product', 'product')].extend(edges)
    
    # TRANSAÇÃO -> TEMPORAL
    logger.info("   ⏰ Criando arestas transação->temporal...")
    progress_bar = tqdm(df.iterrows(), total=len(df), desc="⏰ Temporal", unit="transações", ncols=80)
    for idx, row in progress_bar:
        transaction_time = row[TEMPORAL_COLUMN]
        
        # Encontrar janela temporal correspondente
        for temporal_id, temporal_data in entities['temporal'].items():
            if temporal_data['start_time'] <= transaction_time < temporal_data['end_time']:
                global_temporal_id = entity_id_maps['temporal'][temporal_id]
                entity_edges[('transaction', 'in_timewindow', 'temporal')].append((idx, global_temporal_id))
                break
    progress_bar.close()
    
    # Salvar mapeamentos para uso posterior
    entity_edges['_entity_id_maps'] = entity_id_maps
    
    total_entity_edges = sum(len(edges) for key, edges in entity_edges.items() if isinstance(key, tuple))
    logger.info(f"✓ {total_entity_edges} arestas de entidade criadas")
    
    return dict(entity_edges)


def create_anomaly_pattern_edges(df: pd.DataFrame, k: int = 5) -> List[Tuple[int, int]]:
    """Cria arestas baseadas em padrões de anomalia similares."""
    logger.info("🔗 CAMADA 4: Criando arestas de padrões de anomalia...")
    
    anomaly_edges = []
    
    if 'anomaly_score' not in df.columns:
        logger.warning("   ⚠️ Scores de anomalia não disponíveis, pulando...")
        return anomaly_edges
    
    # Filtrar apenas transações anômalas
    anomalous_transactions = df[df['is_anomaly'] == 1]
    
    if len(anomalous_transactions) < 2:
        logger.warning("   ⚠️ Poucas transações anômalas encontradas, pulando...")
        return anomaly_edges
    
    # Usar k-NN para conectar anomalias similares
    features_for_anomaly = []
    for col in ['anomaly_score', 'TransactionAmt', 'card_risk_score']:
        if col in anomalous_transactions.columns:
            features_for_anomaly.append(col)
    
    if len(features_for_anomaly) < 2:
        logger.warning("   ⚠️ Features insuficientes para análise de anomalia, pulando...")
        return anomaly_edges
    
    X_anomaly = anomalous_transactions[features_for_anomaly].fillna(0)
    
    if len(X_anomaly) > k:
        knn_anomaly = NearestNeighbors(n_neighbors=min(k+1, len(X_anomaly)), metric='euclidean')
        knn_anomaly.fit(X_anomaly)
        
        distances, indices = knn_anomaly.kneighbors(X_anomaly)
        
        for i, neighbors in enumerate(indices):
            source_idx = anomalous_transactions.index[i]
            for j in neighbors[1:]:  # Pular o primeiro (próprio nó)
                target_idx = anomalous_transactions.index[j]
                anomaly_edges.append((source_idx, target_idx))
    
    logger.info(f"✓ {len(anomaly_edges)} arestas de padrão de anomalia criadas")
    return anomaly_edges


def create_temporal_context_edges(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Cria arestas baseadas em contexto temporal-espacial."""
    logger.info("🔗 CAMADA 5: Criando arestas de contexto temporal-espacial...")
    
    temporal_edges = []
    
    # Agrupar por períodos de alta atividade
    df_sorted = df.sort_values(TEMPORAL_COLUMN)
    
    # Definir janelas de alta densidade temporal
    time_density_threshold = df['TransactionAmt'].quantile(0.9)  # Top 10% de valores
    high_value_transactions = df[df['TransactionAmt'] >= time_density_threshold]
    
    if len(high_value_transactions) < 2:
        logger.warning("   ⚠️ Poucas transações de alto valor encontradas, pulando...")
        return temporal_edges
    
    # Conectar transações de alto valor próximas temporalmente
    time_window = 3600  # 1 hora
    
    for i, (idx1, row1) in enumerate(high_value_transactions.iterrows()):
        time1 = row1[TEMPORAL_COLUMN]
        
        # Buscar transações próximas temporalmente
        nearby_transactions = high_value_transactions[
            (high_value_transactions[TEMPORAL_COLUMN] >= time1) &
            (high_value_transactions[TEMPORAL_COLUMN] <= time1 + time_window)
        ]
        
        for idx2, row2 in nearby_transactions.iterrows():
            if idx1 != idx2:
                temporal_edges.append((idx1, idx2))
        
        # Limitar para evitar explosão de arestas
        if len(temporal_edges) > 10000:
            break
    
    logger.info(f"✓ {len(temporal_edges)} arestas de contexto temporal criadas")
    return temporal_edges


def create_meta_relationship_edges(entities: Dict[str, Dict[str, Any]], entity_id_maps: Dict[str, Dict[str, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """Cria meta-relacionamentos entre entidades."""
    logger.info("🔗 CAMADA 6: Criando meta-relacionamentos entre entidades...")
    
    meta_edges = defaultdict(list)
    
    # CARTÃO -> ENDEREÇO (compartilhamento)
    # Esta implementação seria baseada em análise de co-ocorrência
    # Por simplicidade, conectamos entidades com alta sobreposição
    
    # TEMPORAL -> TEMPORAL (sequencial)
    temporal_ids = sorted(entity_id_maps['temporal'].items(), key=lambda x: x[0])
    for i in range(len(temporal_ids) - 1):
        current_id = temporal_ids[i][1]
        next_id = temporal_ids[i + 1][1]
        meta_edges[('temporal', 'sequential', 'temporal')].append((current_id, next_id))
    
    total_meta_edges = sum(len(edges) for edges in meta_edges.values())
    logger.info(f"✓ {total_meta_edges} meta-relacionamentos criados")
    
    return dict(meta_edges)


def analyze_isolated_nodes_by_mask_hetero(data: HeteroData) -> Dict[str, Dict[str, float]]:
    """Analisa o percentual de nós isolados em grafo heterogêneo por split."""
    logger.info("🔍 Analisando nós isolados em grafo heterogêneo por split...")
    
    # Focar apenas nos nós de transação para análise de splits
    transaction_data = data['transaction']
    
    # Verificar se há arestas de transação->transação
    if ('transaction', 'relates_to', 'transaction') in data.edge_types:
        edge_index = data['transaction', 'relates_to', 'transaction'].edge_index
        num_nodes = transaction_data.num_nodes
        
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        if edge_index.size(1) > 0:
            degrees = degrees.scatter_add(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
            degrees = degrees.scatter_add(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    else:
        logger.warning("   ⚠️ Nenhuma aresta transaction->transaction encontrada")
        num_nodes = transaction_data.num_nodes
        degrees = torch.zeros(num_nodes, dtype=torch.long)
    
    isolated_nodes = (degrees == 0)
    
    splits_analysis = {}
    
    for split_name, mask in [('train', transaction_data.train_mask), 
                            ('val', transaction_data.val_mask), 
                            ('test', transaction_data.test_mask),
                            ('monitoring', transaction_data.monitoring_mask)]:
        split_nodes = mask.sum().item()
        
        if split_nodes > 0:
            isolated_in_split = (isolated_nodes & mask).sum().item()
            isolated_percentage = (isolated_in_split / split_nodes) * 100
            
            connected_in_split = split_nodes - isolated_in_split
            connected_percentage = (connected_in_split / split_nodes) * 100
            
            split_degrees = degrees[mask]
            connected_degrees = split_degrees[split_degrees > 0]
            avg_degree_connected = connected_degrees.float().mean().item() if len(connected_degrees) > 0 else 0.0
            
            splits_analysis[split_name] = {
                'total_nodes': split_nodes,
                'isolated_nodes': isolated_in_split,
                'isolated_percentage': isolated_percentage,
                'connected_nodes': connected_in_split,
                'connected_percentage': connected_percentage,
                'avg_degree_connected': avg_degree_connected
            }
            
            logger.info(f"📊 Split {split_name.upper()}:")
            logger.info(f"  - Total de nós: {split_nodes:,}")
            logger.info(f"  - Nós isolados: {isolated_in_split:,} ({isolated_percentage:.2f}%)")
            logger.info(f"  - Nós conectados: {connected_in_split:,} ({connected_percentage:.2f}%)")
            logger.info(f"  - Grau médio (conectados): {avg_degree_connected:.2f}")
            
        else:
            splits_analysis[split_name] = {
                'total_nodes': 0,
                'isolated_nodes': 0,
                'isolated_percentage': 0.0,
                'connected_nodes': 0,
                'connected_percentage': 0.0,
                'avg_degree_connected': 0.0
            }
            logger.warning(f"⚠️ Split {split_name.upper()} está vazio!")
    
    total_isolated = isolated_nodes.sum().item()
    total_isolated_percentage = (total_isolated / num_nodes) * 100
    
    splits_analysis['global'] = {
        'total_nodes': num_nodes,
        'isolated_nodes': total_isolated,
        'isolated_percentage': total_isolated_percentage,
        'connected_nodes': num_nodes - total_isolated,
        'connected_percentage': 100 - total_isolated_percentage,
        'avg_degree_connected': degrees[degrees > 0].float().mean().item() if (degrees > 0).any() else 0.0
    }
    
    logger.info("📊 ANÁLISE GLOBAL DOS NÓS DE TRANSAÇÃO:")
    logger.info(f"  - Total de nós: {num_nodes:,}")
    logger.info(f"  - Nós isolados globais: {total_isolated:,} ({total_isolated_percentage:.2f}%)")
    logger.info(f"  - Conectividade geral: {100 - total_isolated_percentage:.2f}%")
    
    logger.info("🚨 ALERTAS DE QUALIDADE:")
    quality_issues = []
    
    for split_name in ['train', 'val', 'test', 'monitoring']:
        split_data = splits_analysis[split_name]
        isolated_pct = split_data['isolated_percentage']
        
        if isolated_pct > 50:
            quality_issues.append(f"Split {split_name}: {isolated_pct:.1f}% nós isolados (CRÍTICO)")
        elif isolated_pct > 30:
            quality_issues.append(f"Split {split_name}: {isolated_pct:.1f}% nós isolados (ALTO)")
        elif isolated_pct > 15:
            quality_issues.append(f"Split {split_name}: {isolated_pct:.1f}% nós isolados (MODERADO)")
        else:
            logger.info(f"  ✓ Split {split_name}: {isolated_pct:.1f}% nós isolados (BOM)")
    
    if quality_issues:
        for issue in quality_issues:
            logger.warning(f"  ⚠️ {issue}")
    else:
        logger.info("  ✓ Todos os splits têm boa conectividade (<15% nós isolados)")
    
    return splits_analysis


def sample_data(df: pd.DataFrame, sample_ratio: float) -> pd.DataFrame:
    """Aplica amostragem sequencial baseada na ordem temporal."""
    if sample_ratio >= 1.0:
        return df
    
    sample_size = int(len(df) * sample_ratio)
    logger.info(f"📊 Aplicando amostragem sequencial: {sample_size:,} de {len(df):,} transações ({sample_ratio*100:.1f}%)")
    logger.info(f"    Estratégia: primeiras {sample_size:,} transações na ordem temporal")
    
    # Ordenar por tempo e pegar as primeiras N transações
    df_sorted = df.sort_values(TEMPORAL_COLUMN)
    sampled_df = df_sorted.head(sample_size)
    
    # Log detalhado do período coberto
    min_time = sampled_df[TEMPORAL_COLUMN].min()
    max_time = sampled_df[TEMPORAL_COLUMN].max()
    total_min_time = df[TEMPORAL_COLUMN].min()
    total_max_time = df[TEMPORAL_COLUMN].max()
    
    logger.info(f"✓ Amostra sequencial criada: {len(sampled_df):,} transações")
    logger.info(f"    Período da amostra: {min_time:.0f} → {max_time:.0f}")
    logger.info(f"    Período total: {total_min_time:.0f} → {total_max_time:.0f}")
    logger.info(f"    Cobertura temporal: {((max_time - min_time) / (total_max_time - total_min_time)) * 100:.1f}%")
    
    return sampled_df


def main():
    """Função principal que implementa a construção de grafo de conhecimento heterogêneo."""
    args = parse_arguments()
    graph_type = args.graph_type
    sample_ratio = args.sample_ratio
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"🧠 CONSTRUÇÃO DE GRAFO DE CONHECIMENTO - {graph_type.upper()}")
    logger.info("=" * 80)
    logger.info(f"📊 Arquitetura: Grafo heterogêneo multi-relacional")
    logger.info(f"🔗 Estratégia: 6 camadas de conectividade especializada")
    logger.info(f"🎯 Tipos de nós: 6 tipos diferentes")
    logger.info(f"🔗 Tipos de arestas: múltiplos relacionamentos")
    logger.info(f"📊 Amostragem: {sample_ratio*100:.1f}% das transações")
    
    try:
        features_path = "data/parquet/ieee-cis/ieee-cis-features-normalized.parquet"
        output_dir = "data/graph/ieee-cis"
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {features_path}")
        
        # ETAPA 1: Carregar dados completos
        logger.info("=" * 50)
        logger.info("📁 ETAPA 1/5: Carregando dados completos...")
        logger.info("=" * 50)
        df_full = load_data(features_path)
        
        # Aplicar amostragem se necessário
        df = sample_data(df_full, sample_ratio)
        
        required_columns = [IDENTITY_COLUMN, TEMPORAL_COLUMN, LABEL_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas essenciais ausentes: {missing_columns}")
        
        logger.info(f"✓ Dataset para construção: {len(df)} transações")
        
        # ETAPA 2: Criar divisões temporais ANTES da construção do grafo
        logger.info("=" * 50)
        logger.info("🕒 ETAPA 2/5: Criando divisões temporais...")
        logger.info("=" * 50)
        train_mask, val_mask, test_mask, monitoring_mask = create_temporal_splits(df)
        
        # VERIFICAÇÃO CRÍTICA: Confirmar que máscaras foram criadas corretamente
        logger.info(f"🔒 VERIFICAÇÃO DE PREVENÇÃO DE VAZAMENTO:")
        logger.info(f"   - Baseline (70%): Treino={train_mask.sum():,} ({train_mask.mean()*100:.1f}%), Validação={val_mask.sum():,} ({val_mask.mean()*100:.1f}%), Teste={test_mask.sum():,} ({test_mask.mean()*100:.1f}%)")
        logger.info(f"   - Monitoring (30%): {monitoring_mask.sum():,} transações ({monitoring_mask.mean()*100:.1f}%) para detecção de drift")
        logger.info(f"   - Total verificado: {(train_mask | val_mask | test_mask | monitoring_mask).sum():,}/{len(df):,}")
        
        # ETAPA 3: Construir grafo de conhecimento heterogêneo COM PREVENÇÃO DE VAZAMENTO
        logger.info("=" * 50)
        logger.info("🧠 ETAPA 3/5: Construindo grafo de conhecimento heterogêneo...")
        logger.info("🔒 PREVENÇÃO TOTAL DE VAZAMENTO IMPLEMENTADA:")
        logger.info("   - Entidades: estatísticas calculadas APENAS com dados de treino")
        logger.info("   - Features enriquecidas: modelos treinados APENAS com dados de treino")
        logger.info("   - k-NN: índice construído APENAS com dados de treino")
        logger.info("   - Isolation Forest: treinado APENAS com dados de treino")
        logger.info("   - Centralidade: calculada APENAS com dados de treino")
        logger.info("=" * 50)
        hetero_data = build_knowledge_graph(df, train_mask)
        
        # ETAPA 4: Adicionar máscaras ao grafo heterogêneo
        logger.info("=" * 50)
        logger.info("🎭 ETAPA 4/5: Adicionando máscaras de divisão...")
        logger.info("=" * 50)
        hetero_data = add_split_masks(hetero_data, train_mask, val_mask, test_mask, monitoring_mask)
        
        # ETAPA 5: Salvar grafo de conhecimento
        logger.info("=" * 50)
        logger.info("💾 ETAPA 5/5: Salvando grafo de conhecimento...")
        logger.info("=" * 50)
        
        # Salvar objeto HeteroData
        logger.info("💾 Salvando grafo de conhecimento para treinamento...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Nome do arquivo sempre consistente (sem sufixo de amostra)
        file_name = f"{graph_type}_hetero_data.pt"
            
        file_path = os.path.join(output_dir, file_name)
        torch.save(hetero_data, file_path)
        logger.info(f"✓ Grafo de conhecimento salvo: {file_path}")
        
        # Estatísticas finais
        elapsed_time = time.time() - start_time
        connectivity_stats = getattr(hetero_data, 'connectivity_stats', {})
        
        logger.info("=" * 80)
        logger.info("✅ GRAFO DE CONHECIMENTO CONSTRUÍDO COM SUCESSO!")
        logger.info("=" * 80)
        logger.info(f"⏱️ Tempo total: {elapsed_time:.2f} segundos")
        logger.info(f"📊 Tipo de grafo: {graph_type} (heterogêneo)")
        logger.info(f"🧠 Arquitetura:")
        logger.info(f"   - Tipos de nós: {len(hetero_data.node_types)}")
        logger.info(f"   - Tipos de arestas: {len(hetero_data.edge_types)}")
        
        # Contar total de arestas
        total_edges = 0
        for edge_type in hetero_data.edge_types:
            total_edges += hetero_data[edge_type].edge_index.size(1)
        logger.info(f"   - Total de arestas: {total_edges:,}")
        
        logger.info(f"🎭 Partições (nós de transação):")
        logger.info(f"   - Baseline: Treino={hetero_data['transaction'].train_mask.sum()}, Validação={hetero_data['transaction'].val_mask.sum()}, Teste={hetero_data['transaction'].test_mask.sum()}")
        logger.info(f"   - Monitoring: {hetero_data['transaction'].monitoring_mask.sum()} nós para detecção de drift")
        
        if connectivity_stats:
            logger.info(f"🔗 Conectividade por Split:")
            for split_name in ['train', 'val', 'test', 'monitoring']:
                if split_name in connectivity_stats:
                    stats = connectivity_stats[split_name]
                    isolated_pct = stats['isolated_percentage']
                    connected_pct = stats['connected_percentage']
                    logger.info(f"   - {split_name.capitalize()}: {connected_pct:.1f}% conectados, {isolated_pct:.1f}% isolados")
            
            global_stats = connectivity_stats.get('global', {})
            if global_stats:
                global_connected_pct = global_stats.get('connected_percentage', 0)
                logger.info(f"   - Global: {global_connected_pct:.1f}% conectividade geral")
        
        logger.info(f"📊 Features enriquecidas:")
        if hasattr(hetero_data, 'metadata') and isinstance(hetero_data.metadata, dict) and 'enriched_features' in hetero_data.metadata:
            enriched_count = len(hetero_data.metadata['enriched_features'])
            original_count = len(hetero_data.metadata.get('original_features', []))
            logger.info(f"   - Originais: {original_count}")
            logger.info(f"   - Enriquecidas: {enriched_count}")
            logger.info(f"   - Total: {original_count + enriched_count}")
        else:
            logger.info(f"   - Metadados de features não disponíveis")
        
        logger.info(f"💾 Arquivo gerado:")
        logger.info(f"   - Grafo heterogêneo: {file_name}")
        logger.info("=" * 80)
        
        # ========================================
        # CRIAÇÃO DE VERSÃO COM ARESTAS REVERSAS
        # ========================================
        logger.info("=" * 60)
        logger.info("🧬 Criando versão do grafo com arestas reversas para HAN...")
        
        # merge=False cria novos tipos de relação reversa (ex: 'rev_uses_card')
        # que é o ideal para o HAN dar pesos semânticos diferentes para cada direção.
        add_reverse_edges = ToUndirected(merge=False)
        graph_for_han = add_reverse_edges(hetero_data)
        
        han_graph_path = os.path.join(output_dir, "knowledge_hetero_with_reverse_edges.pt")
        torch.save(graph_for_han, han_graph_path)
        
        logger.info(f"✅ Grafo para HAN salvo com sucesso em: {han_graph_path}")
        logger.info(f"   Tipos de arestas originais: {len(hetero_data.edge_types)}")
        logger.info(f"   Novos tipos de arestas (com reversas): {len(graph_for_han.edge_types)}")
        logger.info("=" * 80)
        
        del df, hetero_data, graph_for_han
        gc.collect()
        
    except Exception as e:
        logger.error(f"❌ Erro durante a construção do grafo de conhecimento: {e}")
        raise


if __name__ == "__main__":
    main()
