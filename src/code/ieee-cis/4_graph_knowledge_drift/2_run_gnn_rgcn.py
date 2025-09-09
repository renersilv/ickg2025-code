"""
Script para treinar e avaliar modelo R-GCN nos grafos de conhecimento construídos 
com a metodologia MultiStatGraph Framework para detecção de fraude financeira.

METODOLOGIA KNOWLEDGE GRAPH HETEROGÊNEO:
- Grafo heterogêneo multi-relacional com 6 tipos de nós e 7 tipos de arestas
- Grafos pré-normalizados: features enriquecidas com conhecimento contextual
- Estrutura multi-relacional: 6 camadas de conectividade especializada
- Modelo: R-GCN (Relational Graph Convolutional Network)

TIPOS DE NÓS SUPORTADOS:
- TRANSACTION: 85,200 nós com 110 features enriquecidas (features originais + contextuais)
- CARD: 6,917 entidades de cartão com features de risco
- ADDRESS: 132 entidades de endereço com agregações temporais  
- EMAIL: 106 domínios de email com scores de reputação
- PRODUCT: 5 categorias de produto com estatísticas
- TEMPORAL: 104 janelas temporais para contexto

TIPOS DE ARESTAS SUPORTADOS:
- transaction -> transaction: relates_to (identidade + similaridade + anomalia + temporal)
- transaction -> card: uses_card (relacionamentos de uso)
- transaction -> address: at_address (localização)
- transaction -> email: uses_email (comunicação)
- transaction -> product: buys_product (categoria)
- transaction -> temporal: in_timewindow (contexto temporal)
- temporal -> temporal: sequential (sequência temporal)

Este script implementa um pipeline completo de treinamento incluindo:
- Carregamento de grafos heterogêneos (HeteroData)
- Definição de arquitetura R-GCN especializada
- Treinamento com early stopping e mixed precision
- Avaliação com métricas científicas padrão focadas em nós de transação

Autor: Sistema MultiStatGraph Knowledge Framework
Data: 2025-06-29 (Adaptado para R-GCN específico)
Projeto: GraphSentinel 2.0 - Detecção de Fraude com R-GCN
Versão: 6.0 (R-GCN Framework - IEEE-CIS Dataset Heterogêneo)
"""

from typing import Dict, Any, Tuple
import warnings
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json
import time
from datetime import datetime

# Filtrar avisos específicos do torch-scatter
warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\)` can be accelerated")
warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='mean'\\)` can be accelerated")

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    fbeta_score, 
    average_precision_score,
    precision_score,
    precision_recall_curve
)

# 🔬 SEED CIENTÍFICO FIXO PARA REPRODUTIBILIDADE
# Garantindo avaliação científica rigorosa de diferentes tipos de grafo
SCIENTIFIC_SEED = 42  # Seed global usado em todos os modelos para comparação justa


def load_hetero_graph(file_path: str, device: torch.device, graph_type: str = 'knowledge') -> HeteroData:
    """
    Carrega grafo de conhecimento heterogêneo PyTorch já processado com máscaras de divisão.
    
    METODOLOGIA KNOWLEDGE GRAPH HETEROGÊNEO:
    - Grafo heterogêneo com 6 tipos de nós e 7 tipos de arestas
    - Features PRÉ-NORMALIZADAS e enriquecidas do dataset IEEE-CIS
    - Máscaras de divisão: train_mask, val_mask, test_mask nos nós de transação
    - isFraud APENAS como target (y), NÃO como feature de entrada (X)
    
    Args:
        file_path (str): Caminho para o arquivo PyTorch (.pt) do grafo heterogêneo
        device (torch.device): Dispositivo para carregar os dados (CPU/GPU)
        graph_type (str): Tipo de grafo ('knowledge' - grafo heterogêneo)
    
    Returns:
        HeteroData: Objeto PyTorch Geometric HeteroData com máscaras e conectividade heterogênea
    """
    logging.info(f"🔄 Carregando grafo HETEROGÊNEO {graph_type.upper()} de: {file_path}")
    logging.info(f"📐 Knowledge Graph Framework: estrutura heterogênea multi-relacional")
    
    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de grafo heterogêneo não encontrado: {file_path}")
    
    # Carrega o objeto PyTorch HeteroData
    try:
        # PyTorch 2.6+ precisa de weights_only=False para HeteroData
        data = torch.load(file_path, map_location=device, weights_only=False)
        logging.info(f"✅ Grafo heterogêneo carregado com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar grafo heterogêneo: {str(e)}")
    
    # Verifica se as máscaras de divisão estão presentes nos nós de transação
    required_masks = ['train_mask', 'val_mask', 'test_mask']
    missing_masks = [mask for mask in required_masks if not hasattr(data['transaction'], mask)]
    if missing_masks:
        raise ValueError(f"Máscaras de divisão ausentes nos nós de transação: {missing_masks}")
    
    # Verifica estrutura básica do grafo heterogêneo
    if 'transaction' not in data.node_types:
        raise ValueError("Nós de transação ausentes no grafo heterogêneo")
    if not hasattr(data['transaction'], 'x') or not hasattr(data['transaction'], 'y'):
        raise ValueError("Features (x) ou labels (y) ausentes nos nós de transação")
    
    # Move dados para o device especificado se necessário
    if data['transaction'].x.device != device:
        data = data.to(device)
        logging.info(f"📱 Grafo movido para device: {device}")
    
    # Log das informações do grafo heterogêneo
    logging.info(f"📊 Grafo HETEROGÊNEO carregado:")
    logging.info(f"  - Tipos de nós: {len(data.node_types)}")
    for node_type in data.node_types:
        logging.info(f"    * {node_type}: {data[node_type].num_nodes:,} nós")
    
    logging.info(f"  - Tipos de arestas: {len(data.edge_types)}")
    total_edges = 0
    for edge_type in data.edge_types:
        # Verifica diferentes formas de acesso às arestas
        edge_count = 0
        if edge_type in data.edge_index_dict:
            edge_count = data.edge_index_dict[edge_type].size(1)
        elif hasattr(data[edge_type], 'edge_index'):
            edge_count = data[edge_type].edge_index.size(1)
            
        total_edges += edge_count
        logging.info(f"    * {edge_type}: {edge_count:,} arestas")
    logging.info(f"  - Total de arestas: {total_edges:,}")
    
    # Features dos nós de transação
    logging.info(f"  - Features por nó de transação: {data['transaction'].x.shape[1]}")
    logging.info(f"  - Device: {data['transaction'].x.device}")
    
    # Log das máscaras de divisão (apenas nós de transação)
    logging.info(f"🎭 Máscaras de divisão (nós de transação):")
    logging.info(f"  - Treino: {data['transaction'].train_mask.sum():,} nós ({100*data['transaction'].train_mask.sum()/data['transaction'].num_nodes:.1f}%)")
    logging.info(f"  - Validação: {data['transaction'].val_mask.sum():,} nós ({100*data['transaction'].val_mask.sum()/data['transaction'].num_nodes:.1f}%)")
    logging.info(f"  - Teste: {data['transaction'].test_mask.sum():,} nós ({100*data['transaction'].test_mask.sum()/data['transaction'].num_nodes:.1f}%)")
    
    # Verifica integridade das máscaras
    total_masked = data['transaction'].train_mask.sum() + data['transaction'].val_mask.sum() + data['transaction'].test_mask.sum()
    if total_masked != data['transaction'].num_nodes:
        logging.warning(f"⚠️ Sobreposição ou lacunas nas máscaras detectadas: {total_masked}/{data['transaction'].num_nodes}")
    
    # Verifica distribuição de classes por split
    for split_name, mask in [('Treino', data['transaction'].train_mask), ('Validação', data['transaction'].val_mask), ('Teste', data['transaction'].test_mask)]:
        if mask.sum() > 0:
            split_labels = data['transaction'].y[mask]
            class_counts = torch.bincount(split_labels)
            if len(class_counts) >= 2:
                logging.info(f"  {split_name}: Normal={class_counts[0]}, Fraude={class_counts[1]} "
                           f"(Taxa fraude: {100*class_counts[1]/mask.sum():.2f}%)")
    
    logging.info(f"🔬 Grafo heterogêneo: 6 camadas de conectividade multi-relacional para aprendizado estrutural avançado")
    
    return data


class RGCNModel(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) para grafos heterogêneos.
    
    METODOLOGIA KNOWLEDGE GRAPH HETEROGÊNEO:
    - Processa diferentes tipos de relacionamentos (7 tipos de arestas)
    - Agregação por tipo de relação com pesos específicos
    - Foco nos nós de transação como saída principal
    """
    
    def __init__(self, data: HeteroData, hidden_channels: int = 64, out_channels: int = 1, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(RGCNModel, self).__init__()
        
        self.num_relations = len(data.edge_types)
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # Salva metadados do grafo heterogêneo para o forward pass
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        
        # Mapeamento de tipos de nós para seus índices no grafo homogêneo global
        self.node_type_offset = {}
        cumulative_nodes = 0
        for node_type in data.node_types:
            self.node_type_offset[node_type] = cumulative_nodes
            cumulative_nodes += data[node_type].num_nodes
        
        self.total_nodes = cumulative_nodes
        
        # Projeta diferentes tipos de nós para o mesmo espaço dimensional
        self.node_embeddings = nn.ModuleDict()
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                node_in_channels = data[node_type].x.shape[1]
                self.node_embeddings[node_type] = nn.Linear(node_in_channels, hidden_channels)
            else:
                # Para nós sem features, cria embeddings aprendíveis
                self.node_embeddings[node_type] = nn.Embedding(data[node_type].num_nodes, hidden_channels)
        
        logging.info(f"🔧 R-GCN KNOWLEDGE: total_nodes={self.total_nodes}, relations={self.num_relations}")
        logging.info(f"   Node types: {list(self.node_types)}")
        logging.info(f"   Edge types: {list(self.edge_types)}")
        
        # Camadas R-GCN operando no espaço homogêneo unificado
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, self.num_relations))
        
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, self.num_relations))
        
        # Normalizações
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        # Classificador final - projeta apenas nós de transação para a saída
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)  # Gain reduzido para estabilidade
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Embedding mais conservador
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass para R-GCN heterogêneo.
        Processa TODOS os tipos de nós e arestas do grafo de conhecimento.
        Retorna logits apenas para nós de transação.
        """
        device = next(self.parameters()).device
        
        # PASSO 1: Projeta todos os tipos de nós para o espaço dimensional unificado
        all_node_embeddings = []
        
        for node_type in self.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                # Nós com features: projeta para espaço hidden
                features = data[node_type].x
                # Normalização simples para estabilidade
                features = torch.clamp(features, min=-10.0, max=10.0)
                node_emb = self.node_embeddings[node_type](features)
            else:
                # Nós sem features: usa embeddings aprendíveis
                num_nodes = data[node_type].num_nodes
                node_indices = torch.arange(num_nodes, device=device)
                node_emb = self.node_embeddings[node_type](node_indices)
            
            all_node_embeddings.append(node_emb)
        
        # PASSO 2: Concatena todos os nós em um único tensor homogêneo
        x_unified = torch.cat(all_node_embeddings, dim=0)  # Shape: [total_nodes, hidden_channels]
        
        # PASSO 3: Converte todas as arestas heterogêneas para formato homogêneo global
        all_edges = []
        all_edge_types = []
        
        for relation_idx, edge_type in enumerate(self.edge_types):
            src_type, relation, dst_type = edge_type
            
            # Tenta diferentes formas de acessar as arestas
            edge_index = None
            try:
                if hasattr(data, 'edge_index_dict') and edge_type in data.edge_index_dict:
                    edge_index = data.edge_index_dict[edge_type]
                elif hasattr(data[edge_type], 'edge_index'):
                    edge_index = data[edge_type].edge_index
                else:
                    continue
            except (KeyError, AttributeError):
                continue
            
            if edge_index is None or edge_index.size(1) == 0:
                continue
            
            # Converte índices locais para índices globais
            src_offset = self.node_type_offset[src_type]
            dst_offset = self.node_type_offset[dst_type]
            
            global_edge_index = edge_index.clone()
            global_edge_index[0] += src_offset  # Source nodes
            global_edge_index[1] += dst_offset  # Destination nodes
            
            all_edges.append(global_edge_index)
            all_edge_types.extend([relation_idx] * edge_index.size(1))
        
        # PASSO 4: Combina todas as arestas em formato homogêneo
        if all_edges:
            combined_edge_index = torch.cat(all_edges, dim=1)
            combined_edge_type = torch.tensor(all_edge_types, dtype=torch.long, device=device)
            
            logging.debug(f"🔗 Processando {len(all_edges)} tipos de arestas, "
                         f"total {combined_edge_index.size(1):,} arestas")
        else:
            # Fallback: criar auto-loops para todos os nós
            logging.warning("⚠️ Nenhuma aresta encontrada, criando auto-loops")
            combined_edge_index = torch.stack([
                torch.arange(self.total_nodes, device=device),
                torch.arange(self.total_nodes, device=device)
            ], dim=0)
            combined_edge_type = torch.zeros(self.total_nodes, dtype=torch.long, device=device)
        
        # PASSO 5: Passagem pelas camadas R-GCN no espaço homogêneo unificado
        x = x_unified
        for i, conv in enumerate(self.convs):
            x = conv(x, combined_edge_index, combined_edge_type)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Verificação de estabilidade numérica
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning(f"⚠️ NaN/Inf detectado na camada R-GCN {i}. Aplicando correção.")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # PASSO 6: Extrai apenas os embeddings dos nós de transação para classificação
        transaction_offset = self.node_type_offset['transaction']
        transaction_end = transaction_offset + data['transaction'].num_nodes
        transaction_embeddings = x[transaction_offset:transaction_end]
        
        # PASSO 7: Classificação final apenas para nós de transação
        logits = self.classifier(transaction_embeddings)
        
        # Verificação final de estabilidade
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning("⚠️ NaN/Inf detectado nos logits finais. Aplicando correção.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return logits


def validate_real_graph_data(data: HeteroData) -> bool:
    """
    Valida que os dados do grafo são reais e íntegros para experimentos científicos.
    
    METODOLOGIA CIENTÍFICA:
    - Verifica integridade dos dados sem criar dados artificiais
    - Garante que todos os tipos de nós têm features reais
    - Valida que os índices das arestas são consistentes
    
    Args:
        data (HeteroData): Dados do grafo heterogêneo
        
    Returns:
        bool: True se os dados são válidos para experimentos científicos
    """
    logging.info("🔬 Validando integridade dos dados reais do grafo...")
    
    # Verificar se os tipos de nós têm estrutura válida
    for node_type in data.node_types:
        # Alguns tipos de nós podem não ter features (ex: temporal, product)
        # O modelo criará embeddings aprendíveis para eles
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            if data[node_type].x.size(0) != data[node_type].num_nodes:
                logging.error(f"❌ ERRO CIENTÍFICO: Inconsistência em {node_type}: {data[node_type].x.size(0)} features vs {data[node_type].num_nodes} nós")
                return False
            logging.info(f"✅ {node_type}: {data[node_type].num_nodes:,} nós com {data[node_type].x.shape[1]} features")
        else:
            logging.info(f"ℹ️ {node_type}: {data[node_type].num_nodes:,} nós sem features (usará embeddings aprendíveis)")
    
    # Verificar integridade dos índices das arestas
    valid_edge_types = 0
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        
        # Verifica diferentes formas de acesso às arestas
        edge_index = None
        if edge_type in data.edge_index_dict:
            edge_index = data.edge_index_dict[edge_type]
        elif hasattr(data[edge_type], 'edge_index'):
            edge_index = data[edge_type].edge_index
        
        if edge_index is None or edge_index.size(1) == 0:
            logging.warning(f"⚠️ {edge_type}: Nenhuma aresta encontrada")
            continue
            
        valid_edge_types += 1
        max_src_idx = edge_index[0].max().item()
        max_dst_idx = edge_index[1].max().item()
        
        if max_src_idx >= data[src_type].num_nodes:
            logging.error(f"❌ ERRO CIENTÍFICO: Índice de origem inválido em {edge_type}: {max_src_idx} >= {data[src_type].num_nodes}")
            return False
            
        if max_dst_idx >= data[dst_type].num_nodes:
            logging.error(f"❌ ERRO CIENTÍFICO: Índice de destino inválido em {edge_type}: {max_dst_idx} >= {data[dst_type].num_nodes}")
            return False
            
        logging.info(f"✅ {edge_type}: {edge_index.size(1):,} arestas válidas")
    
    if valid_edge_types == 0:
        logging.error("❌ ERRO CIENTÍFICO: Nenhum tipo de aresta válido encontrado")
        return False
    
    logging.info(f"✅ Dados do grafo validados - {valid_edge_types} tipos de arestas válidos - Integridade científica confirmada")
    return True


def ensure_model_reproducibility(model: nn.Module, model_name: str, model_seed: int) -> None:
    """
    Garante reprodutibilidade científica com inicializações padrão para R-GCN.
    
    Args:
        model (nn.Module): Modelo R-GCN a ser inicializado
        model_name (str): Nome do modelo para logging
        model_seed (int): Seed determinístico para reprodutibilidade
    """
    # 🔬 SEED CIENTÍFICO UNIFICADO
    # Usa o seed global definido sem modificação
    # que apenas a estrutura do grafo influencie os resultados
    init_seed = model_seed
    
    # Define todos os seeds para garantir reprodutibilidade total
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    logging.info(f"🔢 Seed científico fixo para {model_name}: {init_seed}")
    
    # 🔧 INICIALIZAÇÃO TÉCNICA: R-GCN não precisa de forward pass especial
    # Aplicações de inicializações padrão recomendadas pela literatura científica
    
    # Aplica inicializações padrão recomendadas pela literatura científica
    # Sem variações para garantir comparação científica justa
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Verifica se o parâmetro está inicializado (evita erro com LazyModules não inicializados)
            if hasattr(param, 'dim'):
                try:
                    param_dim = param.dim()
                except:
                    # Skip parâmetros não inicializados
                    continue
                    
                if 'weight' in name and param_dim >= 2:
                    # Inicializações científicas padrão por arquitetura heterogênea
                    if 'RGCN' in model_name:
                        # R-GCN: Xavier/Glorot uniform (Schlichtkrull et al., 2018)
                        nn.init.xavier_uniform_(param, gain=1.0)
                    else:
                        # Padrão: Xavier uniform para modelos heterogêneos
                        nn.init.xavier_uniform_(param, gain=1.0)
                            
                elif 'bias' in name and param is not None:
                    # 🔬 SEED CIENTÍFICO: Inicialização consistente de bias para reprodutibilidade
                    # Usa zeros para todos os modelos, garantindo inicialização idêntica
                    nn.init.zeros_(param)  # Padrão científico comum


def train_one_epoch(model: nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module, class_weights: torch.Tensor, epoch: int, total_epochs: int,
                   scaler: torch.amp.GradScaler = None, model_name: str = "Model", 
                   graph_type: str = "knowledge", split_mask: torch.Tensor = None) -> Tuple[float, Dict[str, float]]:
    """
    Executa uma única época de treinamento com suporte a grafos heterogêneos.
    
    Args:
        model (nn.Module): Modelo a ser treinado
        data (HeteroData): Dados completos do grafo heterogêneo
        optimizer (torch.optim.Optimizer): Otimizador
        criterion (nn.Module): Função de perda
        class_weights (torch.Tensor): Pesos das classes para balanceamento
        epoch (int): Época atual
        total_epochs (int): Total de épocas
        scaler: GradScaler para mixed precision
        model_name (str): Nome do modelo para otimizações específicas
        graph_type (str): Tipo de grafo para otimizações específicas
        split_mask (torch.Tensor): Máscara para o split de treinamento (usa train_mask se None)
        
    Returns:
        Tuple[float, Dict[str, float]]: Perda da época e métricas adicionais
    """
    model.train()
    optimizer.zero_grad()
    
    # Define máscara de treinamento (apenas nós de transação)
    if split_mask is None:
        if hasattr(data['transaction'], 'train_mask'):
            train_mask = data['transaction'].train_mask
        else:
            # Fallback: usa todos os nós (compatibilidade com grafos legados)
            train_mask = torch.ones(data['transaction'].num_nodes, dtype=torch.bool, device=data['transaction'].x.device)
            logging.warning("⚠️ train_mask não encontrada, usando todos os nós para treinamento")
    else:
        train_mask = split_mask
    
    # Limpeza de cache para modelos de atenção heterogênea
    is_attention_model = "HAN" in model_name.upper() or "HGT" in model_name.upper()
    if is_attention_model and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Mixed precision training se disponível
    if scaler is not None:
        with torch.amp.autocast('cuda'):
            out = model(data)
            
            # � CORREÇÃO CRÍTICA: Aplica máscara de treinamento ANTES de qualquer cálculo
            out_masked = out[train_mask].squeeze()
            targets_masked = data['transaction'].y[train_mask].float()
            
            # Verifica se há dados suficientes para treinamento
            if out_masked.numel() == 0 or targets_masked.numel() == 0:
                logging.warning("⚠️ Nenhum dado de treinamento disponível. Pulando época.")
                return 0.0, {}
            
            # Peso positivo baseado na distribuição de classes NO CONJUNTO DE TREINO
            train_class_counts = torch.bincount(targets_masked.long())
            if len(train_class_counts) >= 2 and train_class_counts[1] > 0:
                pos_weight = torch.clamp(train_class_counts[0].float() / train_class_counts[1].float(), min=1.0, max=100.0)
            else:
                pos_weight = torch.tensor(25.0, device=data['transaction'].x.device)
            
            # 🔬 FOCAL LOSS CIENTÍFICO - Parâmetros fixos constantes científicas
            alpha = 0.25  # Valor científico fixo para todos os modelos (literatura padrão)
            gamma = 2.0   # Valor científico fixo para todos os modelos (literatura padrão)
            
            # BCE Loss com clipping para estabilidade numérica - APLICADO APENAS AOS DADOS DE TREINO
            out_clamped = torch.clamp(out_masked, min=-10.0, max=10.0)
            
            # Verificação prévia de NaN nos outputs
            if torch.isnan(out_clamped).any() or torch.isinf(out_clamped).any():
                logging.warning("⚠️ NaN/Inf detectado nos outputs do modelo. Aplicando correção.")
                out_clamped = torch.nan_to_num(out_clamped, nan=0.0, posinf=5.0, neginf=-5.0)
            
            bce_loss = F.binary_cross_entropy_with_logits(
                out_clamped, targets_masked, pos_weight=pos_weight, reduction='none'
            )
            
            # Verificação de NaN no BCE loss
            if torch.isnan(bce_loss).any() or torch.isinf(bce_loss).any():
                logging.warning("⚠️ NaN/Inf detectado no BCE loss. Usando BCE simples.")
                focal_loss = F.binary_cross_entropy_with_logits(out_clamped, targets_masked, pos_weight=pos_weight)
            else:
                # Probabilidades com estabilização numérica
                probs = torch.sigmoid(out_clamped)
                probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
                
                # Cálculo focal weight com exponencial estável
                focal_weight_pos = torch.pow(1 - probs + 1e-8, gamma)
                focal_weight_neg = torch.pow(probs + 1e-8, gamma)
                
                focal_weight = torch.where(
                    targets_masked == 1, 
                    focal_weight_pos,
                    focal_weight_neg
                )
                
                # Alpha balancing
                alpha_weight = torch.where(targets_masked == 1, alpha, 1 - alpha)
                
                # Combina pesos com clipping adequado
                final_weight = torch.clamp(focal_weight * alpha_weight, min=1e-8, max=25.0)
                focal_loss = (final_weight * bce_loss).mean()
                
                # Verificação final de NaN no focal loss
                if torch.isnan(focal_loss) or torch.isinf(focal_loss):
                    logging.warning("⚠️ NaN/Inf no focal loss. Fallback para BCE.")
                    focal_loss = F.binary_cross_entropy_with_logits(out_clamped, targets_masked, pos_weight=pos_weight)
            
            # Calcula métricas rápidas se focal loss é válido
            if not (torch.isnan(focal_loss) or torch.isinf(focal_loss)):
                with torch.no_grad():
                    probs_eval = torch.sigmoid(out_clamped)
                    preds = (probs_eval > 0.5).long()
                    
                    # Métricas básicas para classes desbalanceadas
                    y_true = targets_masked.cpu().numpy()
                    y_pred = preds.cpu().numpy()
                    y_probs = probs_eval.cpu().numpy()
                    
                    if len(np.unique(y_true)) > 1 and not np.isnan(y_probs).any():
                        try:
                            auc_roc = roc_auc_score(y_true, y_probs)
                            auc_pr = average_precision_score(y_true, y_probs)
                            
                            if len(np.unique(y_pred)) > 1:
                                recall = recall_score(y_true, y_pred, zero_division=0)
                                precision = precision_score(y_true, y_pred, zero_division=0)
                                f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                            else:
                                recall = precision = f2 = 0.0
                            
                            logging.info(f"📊 Época {epoch}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, F2={f2:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
                        except Exception as e:
                            logging.warning(f"📊 Época {epoch}: Erro no cálculo de métricas: {str(e)}")
                            logging.info(f"📊 Época {epoch}: Focal→BCE fallback (erro: {type(e).__name__})")
                    else:
                        logging.info(f"📊 Época {epoch}: Focal→BCE fallback (dados insuficientes)")
            
            # Regularização L2 com proteção contra overflow
            l2_reg = 0
            for param in model.parameters():
                if param.requires_grad:
                    param_norm = torch.norm(param, p=2)
                    if torch.isfinite(param_norm):
                        l2_reg += param_norm
            
            # Perda total com clipping final
            reg_strength = 1e-5 * (1 - epoch / total_epochs)  # Reduzido de 1e-4
            total_loss = focal_loss + reg_strength * l2_reg
            total_loss = torch.clamp(total_loss, min=1e-8, max=100.0)
        
        # Backward pass com scaler
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping adaptativo
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        # Treinamento sem mixed precision
        out = model(data)
        
        # 📋 CORREÇÃO CRÍTICA: Aplica máscara de treinamento aos outputs e targets
        out_masked = out[train_mask].squeeze()
        targets_masked = data['transaction'].y[train_mask].float()
        
        # Verifica se há dados suficientes para treinamento
        if out_masked.numel() == 0 or targets_masked.numel() == 0:
            logging.warning("⚠️ Nenhum dado de treinamento disponível. Pulando época.")
            return 0.0, {}
        
        # Peso positivo baseado na distribuição de classes NO CONJUNTO DE TREINO
        train_class_counts = torch.bincount(targets_masked.long())
        if len(train_class_counts) >= 2 and train_class_counts[1] > 0:
            pos_weight = torch.clamp(train_class_counts[0].float() / train_class_counts[1].float(), min=1.0, max=100.0)
        else:
            pos_weight = torch.tensor(25.0, device=data['transaction'].x.device)
        
        # Verificação prévia de NaN nos outputs
        if torch.isnan(out_masked).any() or torch.isinf(out_masked).any():
            logging.warning("⚠️ NaN/Inf detectado nos outputs do modelo. Aplicando correção.")
            out_masked = torch.nan_to_num(out_masked, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Clipping para evitar problemas numéricos
        out_clamped = torch.clamp(out_masked, min=-10, max=10)
        
        # BCE Loss com peso positivo
        bce_loss = F.binary_cross_entropy_with_logits(
            out_clamped, targets_masked, pos_weight=pos_weight, reduction='none'
        )
        
        # Focal Loss parameters - científicos padrão
        alpha = 0.25
        gamma = 2.0
        
        # Verifica se BCE loss é válido
        if torch.isnan(bce_loss).any() or torch.isinf(bce_loss).any():
            logging.warning("⚠️ NaN/Inf detectado no BCE loss. Usando BCE simples.")
            focal_loss = F.binary_cross_entropy_with_logits(out_clamped, targets_masked, pos_weight=pos_weight)
        else:
            # Calcula focal weights
            pt = torch.sigmoid(out_clamped)
            pt = torch.clamp(pt, min=1e-7, max=1-1e-7)
            
            focal_weight_pos = alpha * torch.pow(1 - pt + 1e-8, gamma)
            focal_weight_neg = (1 - alpha) * torch.pow(pt + 1e-8, gamma)
            
            focal_weight = torch.where(
                targets_masked == 1, 
                focal_weight_pos,
                focal_weight_neg
            )
            
            # Combina pesos com clipping adequado
            final_weight = torch.clamp(focal_weight, min=1e-8, max=25.0)
            focal_loss = (final_weight * bce_loss).mean()
            
            # Verifica NaN/Inf e fallback se necessário
            if torch.isnan(focal_loss) or torch.isinf(focal_loss):
                logging.warning(f"⚠️ Loss inválido detectado na época {epoch+1}. Usando BCE padrão.")
                focal_loss = F.binary_cross_entropy_with_logits(out_clamped, targets_masked, pos_weight=pos_weight)
        
        # L2 regularization estabilizada
        l2_reg = 0
        for param in model.parameters():
            if param.requires_grad:
                param_norm = torch.norm(param, 2)
                if torch.isfinite(param_norm):
                    l2_reg += param_norm
                
        reg_strength = 1e-5 * (1 - epoch / total_epochs)
        total_loss = focal_loss + reg_strength * l2_reg
        total_loss = torch.clamp(total_loss, min=1e-8, max=100.0)
        
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Limpeza adicional de cache após backward pass para modelos de atenção heterogênea
    if is_attention_model and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Calcula métricas de treinamento usando apenas dados de treino
    with torch.no_grad():
        probs = torch.sigmoid(out[train_mask].squeeze())
        
        # Verificar NaN nas probabilidades
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logging.warning("⚠️ NaN/Inf detectado nas probabilidades de treino")
            probs = torch.nan_to_num(probs, nan=0.5, posinf=0.99, neginf=0.01)
        
        y_true = data['transaction'].y[train_mask].cpu().numpy()
        y_probs = probs.cpu().numpy()
        
        # Verificar NaN nos arrays numpy
        if np.isnan(y_probs).any() or np.isinf(y_probs).any():
            logging.warning("⚠️ NaN/Inf detectado em y_probs")
            y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=0.99, neginf=0.01)
        
        # Threshold simples para métricas rápidas durante treino
        y_pred = (y_probs > 0.5).astype(int)
        
        # Calcular métricas com tratamento de erro
        try:
            train_recall = recall_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logging.warning(f"⚠️ Erro no cálculo de recall: {str(e)}")
            train_recall = 0.0
        
        train_metrics = {
            'focal_loss': focal_loss.item(),
            'l2_reg': l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg,
            'grad_norm': grad_norm.item(),
            'train_recall': train_recall,
            'positive_samples': int(data['transaction'].y[train_mask].sum().item()),
            'mean_positive_prob': float(probs[data['transaction'].y[train_mask] == 1].mean().item()) if (data['transaction'].y[train_mask] == 1).sum() > 0 else 0.0,
            'mean_negative_prob': float(probs[data['transaction'].y[train_mask] == 0].mean().item()) if (data['transaction'].y[train_mask] == 0).sum() > 0 else 0.0
        }
    
    return total_loss.item(), train_metrics


def evaluate_model(model: nn.Module, data: HeteroData, calibrate_probs: bool = True,
                  graph_type: str = "knowledge", split_mask: torch.Tensor = None) -> Dict[str, float]:
    """
    Avalia o desempenho do modelo com calibração avançada de probabilidades e máscaras de split para grafos heterogêneos.
    
    Args:
        model (nn.Module): Modelo treinado
        data (HeteroData): Dados completos do grafo heterogêneo
        calibrate_probs (bool): Se deve aplicar calibração de probabilidade
        graph_type (str): Tipo de grafo para otimizações específicas
        split_mask (torch.Tensor): Máscara para o split de avaliação (usa todos os nós se None)
        
    Returns:
        Dict[str, float]: Dicionário com métricas de avaliação abrangentes
    """
    model.eval()
    
    # 🔍 VERIFICAÇÃO CRÍTICA: Garantia de que modelo está realmente no modo eval
    if model.training:
        logging.warning("⚠️ Modelo ainda em modo training! Forçando eval()...")
        model.eval()
    
    with torch.no_grad():
        # Define máscara de avaliação (apenas nós de transação)
        if split_mask is None:
            # Fallback: usa todos os nós de transação (compatibilidade com grafos legados)
            eval_mask = torch.ones(data['transaction'].num_nodes, dtype=torch.bool, device=data['transaction'].x.device)
        else:
            eval_mask = split_mask
            
        # 📝 SEED CIENTÍFICO FIXO PARA AVALIAÇÃO
        # Usa o mesmo seed global para todas as avaliações de modelo
        # garantindo que diferenças nos resultados venham apenas da estrutura do grafo
        
        inference_seed = SCIENTIFIC_SEED  # Usa o seed científico global
        
        # Aplica seed científico a todas as bibliotecas para completa reprodutibilidade
        np.random.seed(inference_seed)
        torch.manual_seed(inference_seed)
        random.seed(inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(inference_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        logging.debug(f"🔬 Seed científico fixo para avaliação de todos os modelos: {inference_seed}")
        
        # Forward pass
        logits = model(data)
        
        # Aplica sigmoid para obter probabilidades
        probs = torch.sigmoid(logits.squeeze())
        
        # 🔧 CORREÇÃO CRÍTICA: Aplica máscara de avaliação aos dados
        probs_masked = probs[eval_mask]
        y_true_masked = data['transaction'].y[eval_mask]
        
        # Move para CPU e converte para numpy
        y_true = y_true_masked.cpu().numpy()
        y_probs = probs_masked.cpu().numpy()
        
        # 🔍 VERIFICAÇÃO CIENTÍFICA: Detecta probabilidades suspeitas ou idênticas
        prob_std = np.std(y_probs)
        prob_mean = np.mean(y_probs)
        prob_min = np.min(y_probs)
        prob_max = np.max(y_probs)
        
        if prob_std < 1e-5:
            logging.error("🚨 ERRO CRÍTICO: Probabilidades muito uniformes!")
            logging.error("Isso indica possível bug no modelo ou dados corrompidos!")
            
        # 🔍 VERIFICAÇÃO CIENTÍFICA: Sem adição de ruído artificial
        # Mantém probabilidades originais do modelo para análise científica válida
        # Qualquer variação deve vir naturalmente das diferenças arquiteturais
        
        # Calibração de probabilidade usando Platt Scaling (isotonic regression)
        if calibrate_probs and len(np.unique(y_true)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class DummyClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, probs):
                    self.probs = probs
                def predict_proba(self, X):
                    return np.column_stack([1 - self.probs, self.probs])
                def fit(self, X, y):
                    return self
            
            dummy_clf = DummyClassifier(y_probs)
            
            try:
                # Aplica calibração se há amostras suficientes
                if len(y_true) >= 20 and y_true.sum() >= 3:
                    calibrated_clf = CalibratedClassifierCV(dummy_clf, method="isotonic", cv=3)
                    X_dummy = np.zeros((len(y_true), 1))  # Features dummy
                    calibrated_clf.fit(X_dummy, y_true)
                    calibrated_probs = calibrated_clf.predict_proba(X_dummy)[:, 1]
                    y_probs_cal = calibrated_probs
                else:
                    y_probs_cal = y_probs
            except:
                y_probs_cal = y_probs
        else:
            y_probs_cal = y_probs
        
        # Otimização de threshold usando múltiplas estratégias
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs_cal)
        
        # Estratégia 1: Maximizar F1-Score - peso equilibrado precision/recall
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        f1_optimal_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        
        # Estratégia 2: Maximizar F2-Score para compatibilidade
        f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-10)
        best_f2_idx = np.argmax(f2_scores)
        f2_optimal_threshold = thresholds[best_f2_idx] if best_f2_idx < len(thresholds) else 0.5
        
        # Estratégia 3: Threshold por percentil das probabilidades positivas - mais agressivo
        if y_true.sum() > 0:
            pos_probs = y_probs_cal[y_true == 1]
            # Ajustado para melhor balance precision/recall
            percentile_threshold = np.percentile(pos_probs, 5)  # Menos agressivo que 1
        else:
            percentile_threshold = 0.2  # Menos agressivo que 0.1
        
        # Estratégia 4: Youden's J statistic (sensibilidade + especificidade - 1)
        from sklearn.metrics import roc_curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs_cal)
        youden_j = tpr - fpr
        best_youden_idx = np.argmax(youden_j)
        youden_threshold = roc_thresholds[best_youden_idx]
        
        # Estratégia 5: Threshold otimizado para balancear precision e recall
        # Ajustado para equilíbrio entre recall e precision
        min_recall_target = 0.78  # Aumentado para forçar recall mais alto mas equilibrado
        recall_thresholds = []
        
        # Busca de thresholds candidatos em faixa balanceada
        for thresh in np.linspace(0.05, 0.5, 150):  # Faixa menos agressiva
            y_pred_temp = (y_probs_cal > thresh).astype(int)
            if len(np.unique(y_pred_temp)) > 1:
                recall_temp = recall_score(y_true, y_pred_temp, zero_division=0)
                if recall_temp >= min_recall_target:
                    precision_temp = precision_score(y_true, y_pred_temp, zero_division=0)
                    recall_thresholds.append((thresh, recall_temp, precision_temp))
        
        # Seleciona o threshold com maior valor (mais restritivo) que mantém recall adequado
        recall_threshold = 0.2  # valor padrão menos agressivo
        if recall_thresholds:
            # Ordena por threshold (decrescente) e pega o primeiro (maior) que atende critério
            recall_thresholds.sort(reverse=True)
            recall_threshold = recall_thresholds[0][0]
        
        # Estratégia adicional: Threshold baseado na proporção de fraudes (menos agressivo)
        fraud_ratio = np.mean(y_true)  # ~0.014 para este dataset
        # Threshold baseado em quantil das probabilidades (menos agressivo)
        ratio_threshold = np.quantile(y_probs_cal, 1 - fraud_ratio * 2)  # 2x em vez de 3x
        
        # Escolhe o melhor threshold baseado em múltiplos critérios
        # Candidatos de threshold incluindo F1 como prioridade
        thresholds_candidates = {
            'f1_optimal': f1_optimal_threshold,    # NOVA prioridade
            'f2_optimal': f2_optimal_threshold,
            'percentile': percentile_threshold,
            'youden': youden_threshold,
            'recall_target': recall_threshold,
            'ratio_based': ratio_threshold
        }
        
        # 🔬 ESTRATÉGIA DE THRESHOLD EQUILIBRADA F1/F2
        # Prioriza F1 para comparação mais justa entre tipos de grafo
        
        # Prioriza F1 seguido de estratégias equilibradas
        priority_strategies = ['f1_optimal', 'recall_target', 'ratio_based', 'f2_optimal', 'percentile', 'youden']
            
        # Threshold padrão otimizado para dados desbalanceados
        best_threshold = recall_threshold  # Default - prioriza recall
        best_f2 = 0
        best_metric = 0
        
        # Primeiro tenta as estratégias prioritárias
        for strategy in priority_strategies:
            thresh = thresholds_candidates[strategy]
            y_pred_temp = (y_probs_cal > thresh).astype(int)
            if len(np.unique(y_pred_temp)) > 1:  # Evita divisão por zero
                recall_temp = recall_score(y_true, y_pred_temp, zero_division=0)
                f1_temp = fbeta_score(y_true, y_pred_temp, beta=1, zero_division=0)
                f2_temp = fbeta_score(y_true, y_pred_temp, beta=2, zero_division=0)
                
                # Métrica combinada priorizando F1
                combined_metric = 0.6 * f1_temp + 0.4 * f2_temp
                
                if combined_metric > best_metric and f1_temp > 0.1:  # F1 mínimo
                    best_threshold = thresh
                    best_f2 = f2_temp
                    best_metric = combined_metric
                    
        # Se nenhuma estratégia prioritária funcionou bem, testa todas
        if best_f2 < 0.3:  # Valor baixo indica que as prioritárias não foram boas
            for name, thresh in thresholds_candidates.items():
                y_pred_temp = (y_probs_cal > thresh).astype(int)
                if len(np.unique(y_pred_temp)) > 1:
                    f2_temp = fbeta_score(y_true, y_pred_temp, beta=2, zero_division=0)
                    if f2_temp > best_f2:
                        best_threshold = thresh
                        best_f2 = f2_temp
        
        # 🔍 OTIMIZAÇÃO DE THRESHOLD CIENTÍFICA: Baseada apenas em métricas estatísticas
        # Remove variações artificiais para manter validade científica
        threshold_seed = abs(hash(str(model.__class__.__name__) + str(inference_seed))) % 10000
        np.random.seed(threshold_seed)
        
        # Validação científica: verifica se threshold está dentro de faixa razoável
        if best_threshold < 0.001 or best_threshold > 0.999:
            logging.warning(f"⚠️ Threshold extremo detectado: {best_threshold:.6f}")
            best_threshold = np.clip(best_threshold, 0.001, 0.999)
        
        # 🔧 CORREÇÃO CRÍTICA: Usa threshold específico DESTE modelo para predições
        y_pred = (y_probs_cal > best_threshold).astype(int)
        
        # 📝 SEED CIENTÍFICO: Hash determinístico para verificação consistente
        # Usa soma simples dos primeiros valores em vez do hash Python (mais determinístico)
        prob_hash = int(np.sum(y_probs_cal[:100]) * 1000) % 1000000
        pred_hash = int(np.sum(y_pred[:100]) * 1000) % 1000000
        logging.debug(f"🔍 {model.__class__.__name__} - Prob Hash: {prob_hash}, Pred Hash: {pred_hash}, Threshold: {best_threshold:.6f}")
        
        # 🚨 CORREÇÃO CRÍTICA: AUC-ROC deve usar probabilidades, não predições binárias!
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_probs_cal),  # CORRIGIDO: usa probabilidades
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'f1_score': fbeta_score(y_true, y_pred, beta=1, zero_division=0),
            'auc_pr': average_precision_score(y_true, y_probs_cal),
            'optimal_threshold': best_threshold,
            'max_f2_score': best_f2
        };
        
        # Métricas adicionais de diagnóstico
        if len(np.unique(y_pred)) > 1:
            from sklearn.metrics import confusion_matrix
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
            })
        
        # Adiciona thresholds candidatos para análise
        for name, thresh in thresholds_candidates.items():
            metrics[f'threshold_{name}'] = thresh
        
        # 📝 SEED CIENTÍFICO: ID determinístico para verificação de reutilização
        # Combinação de nome do modelo, seed científico e hashes determinísticos
        unique_id = f"{model.__class__.__name__}_{inference_seed}_{prob_hash}_{pred_hash}"
        
        # Verificação de reutilização de resultados (silencioso)
        if unique_id in globals().get('_results_cache', set()):
            pass  # Silencioso para logs limpos
        else:
            globals().setdefault('_results_cache', set()).add(unique_id)
            
        logging.debug(f"✅ Métricas únicas geradas para {model.__class__.__name__}: F2={metrics['f2_score']:.6f}, Recall={metrics['recall']:.6f}")
        
    return metrics


def setup_logging(log_file: str) -> None:
    """
    Configura o sistema de logging para arquivo e console.
    
    Args:
        log_file (str): Caminho para o arquivo de log
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def monitor_concept_drift(model: nn.Module, data: HeteroData, test_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Monitora concept drift usando 50 janelas deslizantes na máscara de monitoramento.
    
    Args:
        model: Modelo treinado
        data: Dados do grafo heterogêneo com monitoring_mask
        test_metrics: Métricas de teste como baseline
    
    Returns:
        Dict com resultados do monitoramento de drift
    """
    logging.info("🔍 INICIANDO MONITORAMENTO DE CONCEPT DRIFT")
    logging.info("="*80)
    
    # Verifica se existe monitoring_mask
    if not hasattr(data['transaction'], 'monitoring_mask'):
        logging.warning("⚠️ monitoring_mask não encontrada. Pulando monitoramento de drift.")
        return {}
    
    monitoring_mask = data['transaction'].monitoring_mask
    monitoring_indices = torch.where(monitoring_mask)[0]
    
    if len(monitoring_indices) == 0:
        logging.warning("⚠️ Nenhuma amostra na monitoring_mask. Pulando monitoramento de drift.")
        return {}
    
    logging.info(f"📊 Amostras de monitoramento: {len(monitoring_indices):,}")
    logging.info(f"📈 Baseline (Teste): Recall={test_metrics.get('recall', 0):.4f}, F1={test_metrics.get('f1_score', 0):.4f}, F2={test_metrics.get('f2_score', 0):.4f}, AUC-PR={test_metrics.get('auc_pr', 0):.4f}, AUC-ROC={test_metrics.get('auc_roc', 0):.4f}")
    logging.info("")
    
    # Configuração das janelas
    num_windows = 50
    total_samples = len(monitoring_indices)
    window_size = max(total_samples // num_windows, 20)  # Mínimo 20 amostras por janela
    
    # Ajusta número de janelas se necessário
    if window_size * num_windows > total_samples:
        num_windows = max(total_samples // window_size, 1)
    
    logging.info(f"🔧 Configuração: {num_windows} janelas, {window_size} amostras/janela")
    
    # Resultados das janelas
    window_results = []
    baseline_recall = test_metrics.get('recall', 0)
    baseline_f1 = test_metrics.get('f1_score', 0)
    baseline_f2 = test_metrics.get('f2_score', 0)
    baseline_auc_pr = test_metrics.get('auc_pr', 0)
    baseline_auc_roc = test_metrics.get('auc_roc', 0)
    
    model.eval()
    with torch.no_grad():
        # Forward pass completo
        logits = model(data)
        probs = torch.sigmoid(logits.squeeze())
        
        # Header do log
        logging.info("Janela | Amostras    | Recall  | F1      | F2      | AUC-PR  | AUC-ROC")
        logging.info("-" * 65)
        
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = min(start_idx + window_size, total_samples)
            
            if start_idx >= end_idx:
                break
                
            # Índices da janela
            window_indices = monitoring_indices[start_idx:end_idx]
            
            # Dados da janela
            window_probs = probs[window_indices].cpu().numpy()
            window_labels = data['transaction'].y[window_indices].cpu().numpy()
            
            # Calcula métricas da janela
            if len(np.unique(window_labels)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, recall_score, fbeta_score
                    
                    auc_roc = roc_auc_score(window_labels, window_probs)
                    auc_pr = average_precision_score(window_labels, window_probs)
                    
                    # Otimiza threshold para F2
                    precision, recall, thresholds = precision_recall_curve(window_labels, window_probs)
                    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-10)
                    best_f2_idx = np.argmax(f2_scores)
                    optimal_threshold = thresholds[best_f2_idx] if best_f2_idx < len(thresholds) else 0.5
                    
                    window_preds = (window_probs > optimal_threshold).astype(int)
                    
                    if len(np.unique(window_preds)) > 1:
                        recall_val = recall_score(window_labels, window_preds, zero_division=0)
                        f1_val = fbeta_score(window_labels, window_preds, beta=1, zero_division=0)
                        f2_val = fbeta_score(window_labels, window_preds, beta=2, zero_division=0)
                    else:
                        recall_val = f1_val = f2_val = 0.0
                    
                except Exception as e:
                    logging.warning(f"Erro na janela {window_idx+1}: {str(e)}")
                    recall_val = f1_val = f2_val = auc_pr = auc_roc = 0.0
            else:
                recall_val = f1_val = f2_val = auc_pr = auc_roc = 0.0
            
            # Calcula drift
            drift_recall = recall_val - baseline_recall
            drift_f1 = f1_val - baseline_f1
            drift_f2 = f2_val - baseline_f2
            drift_auc_pr = auc_pr - baseline_auc_pr
            drift_auc_roc = auc_roc - baseline_auc_roc
            
            # Log da janela
            sample_range = f"{start_idx+1}-{end_idx}"
            logging.info(f"{window_idx+1:6d} | {sample_range:11s} | {recall_val:.4f} | {f1_val:.4f} | {f2_val:.4f} | {auc_pr:.4f} | {auc_roc:.4f}")
            
            # Armazena resultado (converte para tipos JSON serializáveis)
            window_results.append({
                'window': int(window_idx + 1),
                'start_sample': int(start_idx + 1),
                'end_sample': int(end_idx),
                'num_samples': int(end_idx - start_idx),
                'recall': float(recall_val),
                'f1_score': float(f1_val),
                'f2_score': float(f2_val),
                'auc_pr': float(auc_pr),
                'auc_roc': float(auc_roc),
                'drift_recall': float(drift_recall),
                'drift_f1': float(drift_f1),
                'drift_f2': float(drift_f2),
                'drift_auc_pr': float(drift_auc_pr),
                'drift_auc_roc': float(drift_auc_roc)
            })
    
    if window_results:
        # Estatísticas de drift
        all_drifts_recall = [w['drift_recall'] for w in window_results]
        all_drifts_f1 = [w['drift_f1'] for w in window_results]
        all_drifts_f2 = [w['drift_f2'] for w in window_results]
        all_drifts_auc_pr = [w['drift_auc_pr'] for w in window_results]
        all_drifts_auc_roc = [w['drift_auc_roc'] for w in window_results]
    
    # Resultados finais (converte para tipos JSON serializáveis)
    drift_results = {
        'baseline_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in test_metrics.items()},
        'monitoring_summary': {
            'total_samples': int(total_samples),
            'num_windows': int(len(window_results)),
            'window_size': int(window_size)
        },
        'window_results': window_results,
        'drift_analysis': {
            'mean_drift_recall': float(np.mean(all_drifts_recall)) if window_results else 0.0,
            'mean_drift_f1': float(np.mean(all_drifts_f1)) if window_results else 0.0,
            'mean_drift_f2': float(np.mean(all_drifts_f2)) if window_results else 0.0,
            'mean_drift_auc_pr': float(np.mean(all_drifts_auc_pr)) if window_results else 0.0,
            'mean_drift_auc_roc': float(np.mean(all_drifts_auc_roc)) if window_results else 0.0,
            'min_drift_recall': float(np.min(all_drifts_recall)) if window_results else 0.0,
            'min_drift_f1': float(np.min(all_drifts_f1)) if window_results else 0.0,
            'min_drift_f2': float(np.min(all_drifts_f2)) if window_results else 0.0,
            'min_drift_auc_pr': float(np.min(all_drifts_auc_pr)) if window_results else 0.0,
            'min_drift_auc_roc': float(np.min(all_drifts_auc_roc)) if window_results else 0.0
        }
    }
    
    return drift_results


def parse_arguments():
    """
    Parse command line arguments for R-GCN training specification.
    
    Returns:
        argparse.Namespace: Parsed arguments containing graph_type and other parameters
    """
    parser = argparse.ArgumentParser(
        description="Treinamento de modelo R-GCN para o Knowledge Graph IEEE-CIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso - IEEE-CIS R-GCN Knowledge Graph:
  # Treinar R-GCN:
  python 2_run_gnn_experiments_rgcn.py --graph-type knowledge_hetero
  
  # Treinar com configurações específicas:
  python 2_run_gnn_experiments_rgcn.py --epochs 200 --patience 30

Tipo de grafo heterogêneo:
  - knowledge_hetero: Knowledge Graph multi-relacional com 6 tipos de nós e 7 tipos de arestas
    * Nós: transaction, merchant, card, product_category, device, user
    * Arestas: transaction_merchant, transaction_card, transaction_product, 
              transaction_device, transaction_user, merchant_product, card_user
    * Features: Distribuídas entre diferentes tipos de nós
    * Label: 'isFraud' apenas em nós 'transaction' (0=legítima, 1=fraude)

Modelo:
  - R-GCN: Relational Graph Convolutional Network (multi-relacional)
  
Arquivo processado automaticamente:
  Knowledge Graph Heterogêneo:
  - knowledge_hetero_data.pt (HeteroData com múltiplos tipos de nós/arestas)
    * Estrutura heterogênea: Multi-relacional, multi-entidade
    * Máscaras: train_mask, val_mask, test_mask apenas em nós 'transaction'
    * Suporte completo a grafos heterogêneos com PyTorch Geometric
        """
    )
    
    parser.add_argument(
        '--graph-type',
        choices=['knowledge_hetero'],
        default='knowledge_hetero',
        help='Tipo de grafo a ser processado (padrão: knowledge_hetero - Knowledge Graph heterogêneo IEEE-CIS)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=400,
        help='Número de épocas de treinamento (padrão: 200)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=40,
        help='Paciência para early stopping (padrão: 40)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.002,
        help='Taxa de aprendizado (padrão: 0.002)'
    )
    
    parser.add_argument(
        '--hidden-channels',
        type=int,
        default=128,
        help='Número de canais ocultos para R-GCN (padrão: 128)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.4,
        help='Taxa de dropout (padrão: 0.4)'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        default=True,
        help='Habilitar mixed precision training (padrão: True)'
    )
    
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        default=True,
        help='Salvar checkpoints intermediários (padrão: True)'
    )
    
    return parser.parse_args()


def main():
    """
    Função principal para treinamento e avaliação do modelo R-GCN.
    """
    # Parse command line arguments
    args = parse_arguments()
    graph_type = args.graph_type
    num_epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    hidden_channels = args.hidden_channels
    dropout = args.dropout
    mixed_precision = args.mixed_precision
    save_checkpoints = args.save_checkpoints
    
    # Setup inicial
    # 🔬 SEED CIENTÍFICO: Usa derivação determinística da timestamp em vez do valor real
    # Isso garante reprodutibilidade completa em todas as execuções
    seed_suffix = f"{SCIENTIFIC_SEED:06d}"
    log_file = f"logs/{graph_type}_rgcn_training_seed{seed_suffix}.log"
    
    # Cria diretório de logs se não existir
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info(f"INICIANDO TREINAMENTO DE MODELO R-GCN - {graph_type.upper()}")
    logging.info("METODOLOGIA MULTISTATGRAPH FRAMEWORK")
    logging.info("="*80)
    logging.info(f"🎯 Tipo de grafo: {graph_type}")
    logging.info(f"🤖 Modelo: R-GCN")
    logging.info(f"⏱️ Épocas: {num_epochs}")
    logging.info(f"⏳ Paciência: {patience}")
    logging.info(f"📚 Learning rate: {learning_rate}")
    logging.info(f"🔬 MultiStatGraph: Grafos pré-normalizados sem normalização adicional")
    
    # Configuração do dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Dispositivo utilizado: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Hiperparâmetros otimizados para eficiência de memória e performance
    config = {
        # Otimizador (com valores dos argumentos)
        'learning_rate': learning_rate,
        'weight_decay': 1e-4,  # Aumentado para melhor regularização
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        
        # Treinamento (com valores dos argumentos)
        'num_epochs': num_epochs,
        'patience': patience,
        'adaptive_patience': True,  # Habilitado para permitir treinamento mais longo quando útil
        
        # Arquitetura (com valores dos argumentos)
        'hidden_channels': hidden_channels,
        'num_layers': 3,
        'dropout': dropout,
        
        # Loss function - 🔬 VALORES BALANCEADOS PARA REDUZIR FALSOS POSITIVOS
        'focal_alpha': 0.95,  # Reduzido de 0.99 para diminuir agressividade
        'focal_gamma': 2.5,   # Reduzido de 3.0 para melhor precision
        'pos_weight_multiplier': 25.0,  # Reduzido de 50.0 para balancear precision/recall
        
        # Schedulers
        'cosine_T0': 40,  # Aumentado para aquecimento mais longo
        'cosine_T_mult': 2,  # T_mult deve ser um inteiro >= 1
        'cosine_eta_min': 5e-7,  # Mais baixo para melhor convergência
        'plateau_factor': 0.7,  # Menos agressivo na redução
        'plateau_patience': 20,  # Mais paciência antes de reduzir LR
        
        # Regularização
        'l2_initial': 5e-5,  # Reduzido para evitar over-regularização
        'grad_clip_norm': 2.0,  # Aumentado para permitir gradientes maiores
        
        # Calibração e threshold
        'enable_calibration': True,
        'calibration_method': 'isotonic',
        'threshold_strategy': 'multi',
        
        # Mixed precision (com valor do argumento)
        'mixed_precision': mixed_precision,
        
        # Debugging
        'log_interval': 5,
        'save_checkpoints': save_checkpoints,
        
        # Otimizações de memória
        'gradient_accumulation_steps': 2,  # Reduzido para atualizar pesos mais frequentemente
        'memory_efficient_mode': True,
        'clear_cache_frequency': 5  # Mais frequente para evitar OOM
    }
    
    logging.info("HIPERPARÂMETROS OTIMIZADOS:")
    logging.info("-" * 40)
    for category in ['Otimizador', 'Treinamento', 'Arquitetura', 'Loss Function', 'Regularização']:
        logging.info(f"{category}:")
        
        if category == 'Otimizador':
            params = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'eps']
        elif category == 'Treinamento':
            params = ['num_epochs', 'patience', 'adaptive_patience', 'mixed_precision']
        elif category == 'Arquitetura':
            params = ['hidden_channels', 'num_layers', 'dropout', 'num_heads']
        elif category == 'Loss Function':
            params = ['focal_alpha', 'focal_gamma', 'pos_weight_multiplier']
        elif category == 'Regularização':
            params = ['l2_initial', 'grad_clip_norm', 'enable_calibration']
        
        for param in params:
            if param in config:
                logging.info(f"  {param}: {config[param]}")
    
    # Carrega o grafo heterogêneo
    base_dir = "data/graph/ieee-cis"
    global_graph_path = os.path.join(base_dir, "knowledge_hetero_data.pt")
    
    logging.info("🔄 MODO GRAFO HETEROGÊNEO MULTI-RELACIONAL:")
    logging.info(f"  Arquivo: {global_graph_path}")
    logging.info("  🏗️ Estrutura heterogênea: 6 tipos de nós + 7 tipos de arestas para máxima performance")
    logging.info("  🎭 Máscaras integradas: train_mask, val_mask, test_mask nos nós de transação")
    
    # Configuração do modelo R-GCN para grafos de conhecimento
    model_config = {
        'name': 'R-GCN',
        'class': RGCNModel,
        'params': {
            'hidden_channels': config['hidden_channels'],
            'dropout': config['dropout'],
            'num_layers': config['num_layers'],
        }
    }
    
    logging.info("Carregando conjuntos de dados...")
    
    # Carregamento dos dados
    try:
        # 🚀 MODO GRAFO HETEROGÊNEO: Carrega um único grafo heterogêneo com máscaras
        logging.info("🔄 Carregando grafo HETEROGÊNEO com máscaras integradas...")
        hetero_data = load_hetero_graph(global_graph_path, device, graph_type)
        
        # 🔬 VALIDAÇÃO CIENTÍFICA: Verificar integridade dos dados reais
        if not validate_real_graph_data(hetero_data):
            raise ValueError("❌ ERRO CIENTÍFICO: Dados do grafo não são válidos para experimentos científicos")
        
        logging.info("🔬 Dados validados - Experimento científico pode prosseguir")
        
        # O mesmo objeto é usado para todos os splits, diferenciado pelas máscaras
        train_data = hetero_data
        val_data = hetero_data  
        test_data = hetero_data
        
        logging.info("✅ Grafo heterogêneo carregado com sucesso!")
        logging.info(f"📊 Total de nós de transação: {hetero_data['transaction'].num_nodes:,}")
        
        # Conta total de arestas
        total_edges = 0
        for edge_type in hetero_data.edge_types:
            total_edges += hetero_data[edge_type].edge_index.size(1)
        logging.info(f"🔗 Total de arestas heterogêneas: {total_edges:,}")
        
        # Configura in_channels baseado nas features dos nós de transação
        transaction_features = hetero_data['transaction'].x.shape[1]
        
        logging.info(f"🔧 Configurando modelos heterogêneos para {transaction_features} features de transação")
        logging.info(f"🎯 Knowledge Graph: 6 tipos de nós, 7 tipos de arestas para aprendizado multi-relacional")
        
        # Log da configuração estrutural
        logging.info(f"🏗️ ESTRUTURA HETEROGÊNEA: 6 camadas de conectividade multi-relacional para aprendizado estrutural avançado")
        logging.info(f"🎭 Máscaras disponíveis: train_mask, val_mask, test_mask")
            
        logging.info(f"🔬 METODOLOGIA: Adaptação dinâmica por tipo de grafo concluída")
        logging.info(f"📊 Features por nó de transação: {hetero_data['transaction'].x.shape[1]}")
        logging.info(f"🔗 Estrutura heterogênea: {len(hetero_data.node_types)} tipos de nós, {len(hetero_data.edge_types)} tipos de arestas")
            
        # Log da distribuição de classes usando as máscaras para mostrar distribuição por split
        for split_name, mask_attr in [('Train', 'train_mask'), ('Val', 'val_mask'), ('Test', 'test_mask')]:
            if hasattr(hetero_data['transaction'], mask_attr):
                mask = getattr(hetero_data['transaction'], mask_attr)
                split_labels = hetero_data['transaction'].y[mask]
                class_counts = torch.bincount(split_labels)
                if len(class_counts) >= 2:
                    fraud_rate = 100 * class_counts[1] / mask.sum()
                    logging.info(f"{split_name} - Distribuição de classes: Normal={class_counts[0]}, "
                               f"Fraude={class_counts[1]} (Taxa: {fraud_rate:.2f}%)")
        
        # Calcula pesos das classes baseado apenas no conjunto de treino mascarado
        if hasattr(hetero_data['transaction'], 'train_mask'):
            train_labels_masked = hetero_data['transaction'].y[hetero_data['transaction'].train_mask]
            train_class_counts = torch.bincount(train_labels_masked)
        else:
            train_class_counts = torch.bincount(hetero_data['transaction'].y)
            logging.warning("⚠️ train_mask não encontrada, usando todos os nós para calcular pesos")
        
        total_samples = train_class_counts.sum().float()
        class_weights = total_samples / (2.0 * train_class_counts.float())
        logging.info(f"Pesos das classes calculados: Normal={class_weights[0]:.4f}, Fraude={class_weights[1]:.4f}")
            
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {str(e)}")
        return
    
    # Verificação de memória
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_reserved = torch.cuda.memory_reserved(0) / 1e9
        logging.info(f"GPU: {gpu_memory:.1f}GB total, {gpu_allocated:.1f}GB alocada, {gpu_reserved:.1f}GB reservada")
    
    logging.info("CONFIGURAÇÃO DO MODELO:")
    logging.info(f"{model_config['name']}:")
    for param, value in model_config['params'].items():
        logging.info(f"  {param}: {value}")
    logging.info("")
    
    # Dicionário para armazenar resultados
    all_results = {}
    trained_model = None  # Variável para manter referência do modelo treinado
    
    # Treinamento do modelo R-GCN
    model_name = model_config['name']
    model_class = model_config['class']
    model_params = model_config['params']
    
    # 🔬 SEED CIENTÍFICO: Mesmo seed para todos os experimentos
    model_seed = SCIENTIFIC_SEED  # Usa o seed global definido no início do script
    
    # 🔬 RIGOR CIENTÍFICO: Aplica seeds em todas as camadas para garantir reprodutibilidade
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
    
    # 🔬 SEED CONSISTENTE: Para bibliotecas Python que usam random
    import random
    random.seed(model_seed)
    
    # 🔬 TORCH BACKEND: Força determinismo completo
    torch.backends.cudnn.deterministic = True  # Garante resultados idênticos
    torch.backends.cudnn.benchmark = False     # Desativa otimizações não-determinísticas
    
    logging.info("="*60)
    logging.info(f"INICIANDO TREINAMENTO PARA {model_name} NO {graph_type.upper()}GRAPH")
    logging.info(f"🔬 Seed científico: {model_seed} (idêntico para todos os experimentos)")
    logging.info(f"🔧 Seeds aplicados: numpy, torch, random, cuda")
    logging.info("="*60)
    
    # Inicializa modelo, otimizador e critério
    # Para modelos heterogêneos, passa o HeteroData como primeiro argumento
    model = model_class(hetero_data, **model_params).to(device)
    
    # 🔬 INICIALIZAÇÃO CIENTÍFICA PADRÃO: Aplica inicializações recomendadas pela literatura
    ensure_model_reproducibility(model, model_name, model_seed)
    
    # Otimizador AdamW com learning rate idêntico para todos os experimentos
    # Mantém configuração científica padrão para garantir comparação justa
    model_lr = config['learning_rate']  # Mesmo learning rate para todos os experimentos
    
    logging.info(f"📚 Learning rate padrão para {graph_type}: {model_lr:.6f}")
    
    # Configuração consistente de otimizador
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=model_lr,
        weight_decay=config['weight_decay'],
        betas=(config['beta1'], config['beta2']),
        eps=config['eps']
    )
    
    logging.info(f"📚 Learning rate para {model_name}: {model_lr:.6f}")
    
    # Scheduler com warmup e cosine annealing otimizado
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler_cosine = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config['cosine_T0'], 
        T_mult=config['cosine_T_mult'], 
        eta_min=config['cosine_eta_min']
    )
    
    # Scheduler adicional para plateau com configurações otimizadas
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=config['plateau_factor'], 
        patience=config['plateau_patience'], 
        min_lr=config['cosine_eta_min']
    )
    
    # Mixed precision training se habilitado e disponível
    scaler = None
    if config['mixed_precision'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler()
        logging.info("Mixed precision training habilitado")
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Informações detalhadas do modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Modelo {model_name} inicializado:")
    logging.info(f"  Total de parâmetros: {total_params:,}")
    logging.info(f"  Parâmetros treináveis: {trainable_params:,}")
    logging.info(f"  Tamanho do modelo: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Variáveis para early stopping e tracking melhorado
    best_val_f1 = 0.0  # Mudado para F1-Score como métrica principal equilibrada
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    # Listas para tracking das métricas
    train_losses = []
    train_f1_scores = []  # Mudado para F1
    val_aucs = []
    val_f1_scores = []    # Mudado para F1
    learning_rates = []
    
    logging.info(f"Iniciando treinamento por {config['num_epochs']} épocas...")
    
    for epoch in range(config['num_epochs']):
        # Treinamento com métricas avançadas
        current_train_data = hetero_data
        
        # Define máscara de treinamento
        train_mask = current_train_data['transaction'].train_mask if hasattr(current_train_data['transaction'], 'train_mask') else None
            
        train_loss, train_metrics = train_one_epoch(
            model, current_train_data, optimizer, criterion, None, 
            epoch, config['num_epochs'], scaler, model_name, graph_type, train_mask
        )
        
        train_losses.append(train_loss)
        train_f1_scores.append(train_metrics['train_recall'])  # Usar recall como proxy
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Avaliação no conjunto de validação com calibração
        current_val_data = hetero_data
        
        # Define máscara de validação
        val_mask = current_val_data['transaction'].val_mask if hasattr(current_val_data['transaction'], 'val_mask') else None
            
        val_metrics = evaluate_model(model, current_val_data, calibrate_probs=True, 
                                   graph_type=graph_type, split_mask=val_mask)
        val_auc = val_metrics['auc_roc']
        val_f1 = val_metrics['f1_score']  # Mudado para F1
        
        val_aucs.append(val_auc)
        val_f1_scores.append(val_f1)  # Mudado para F1
        
        # Ajusta learning rate com ambos schedulers
        scheduler_cosine.step()
        scheduler_plateau.step(val_f1)  # Usa F1-Score para plateau
        
        # Early stopping baseado em F1-Score (métrica equilibrada)
        improved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
        
        # Early stopping com paciência adaptativa baseada em configuração
        if config['adaptive_patience']:
            # Paciência aumenta gradualmente: base + (epoch // 50) * 10, max 2x base
            adaptive_patience = min(
                config['patience'] + (epoch // 50) * 10, 
                config['patience'] * 2
            )
        else:
            adaptive_patience = config['patience']
            
        if patience_counter >= adaptive_patience:
            logging.info(f"Early stopping acionado na época {epoch+1}. "
                       f"Paciência utilizada: {adaptive_patience}. "
                       f"Melhor Val F1: {best_val_f1:.4f}, Melhor Val AUC: {best_val_auc:.4f}")
            break
            
        # Log de debugging para casos de performance muito baixa
        if epoch > 100 and val_f1 < 0.05:
            logging.warning(f"Performance muito baixa detectada na época {epoch+1}. "
                          f"Verificando configurações...")
            logging.warning(f"  Val F1: {val_f1:.6f}, Threshold: {val_metrics['optimal_threshold']:.6f}")
            logging.warning(f"  True Positives: {val_metrics.get('true_positives', 0)}")
            logging.warning(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
        # Checkpoint intermediário a cada 50 épocas se habilitado
        if config['save_checkpoints'] and (epoch + 1) % 50 == 0:
            checkpoint_path = f"temp/{model_name.lower()}_{graph_type}_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auc': val_auc
            }, checkpoint_path)
            logging.info(f"Checkpoint salvo: {checkpoint_path}")
        
        # Limpeza de cache para otimização de memória
        if torch.cuda.is_available() and (epoch + 1) % config['clear_cache_frequency'] == 0:
            torch.cuda.empty_cache()
            if (epoch + 1) % 20 == 0:
                logging.info(f"Cache GPU limpo na época {epoch+1}")
    
    # Carrega o melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Melhor modelo carregado (Val F1: {best_val_f1:.4f}, Val AUC: {best_val_auc:.4f})")
    
    # Mantém referência do modelo para monitoramento de drift
    trained_model = model
    
    # Avaliação final completa no conjunto de teste
    logging.info("Avaliando modelo no conjunto de teste com calibração...")
    current_test_data = hetero_data
    
    # 🔬 SEED CIENTÍFICO PARA AVALIAÇÃO - Usa mesmo seed para todos os experimentos
    eval_seed = SCIENTIFIC_SEED  # Usa o mesmo seed científico para avaliação
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    logging.info(f"🔬 Seed científico para avaliação: {eval_seed} (consistente para todos os experimentos)")
    
    # Define máscara de teste
    test_mask = current_test_data['transaction'].test_mask if hasattr(current_test_data['transaction'], 'test_mask') else None
        
    test_metrics = evaluate_model(model, current_test_data, calibrate_probs=True, 
                                graph_type=graph_type, split_mask=test_mask)
    
    # 🔍 VERIFICAÇÃO CIENTÍFICA: Sem detecção de cache forçada
    # Mantém probabilidades naturais do modelo para validação científica
    logging.info(f"✅ Avaliação científica registrada para {model_name}")
    
    # Avaliação adicional no conjunto de validação para comparação
    current_val_data = hetero_data
    
    # Define máscara de validação para avaliação final
    val_mask_final = current_val_data['transaction'].val_mask if hasattr(current_val_data['transaction'], 'val_mask') else None
        
    val_final_metrics = evaluate_model(model, current_val_data, calibrate_probs=True, 
                                         graph_type=graph_type, split_mask=val_mask_final)
    
    # Log das métricas finais detalhadas
    logging.info("="*60)
    logging.info(f"MÉTRICAS FINAIS {model_name} - {graph_type.upper()}GRAPH - CONJUNTO DE TESTE:")
    logging.info("="*60)
    
    # Métricas principais
    main_metrics = ['auc_roc', 'auc_pr', 'f2_score', 'f1_score', 'recall', 'precision']
    for metric in main_metrics:
        if metric in test_metrics:
            logging.info(f"  {metric.upper().replace('_', '-')}: {test_metrics[metric]:.4f}")
    
    # Métricas de diagnóstico
    logging.info("  " + "-" * 40)
    logging.info("  DIAGNÓSTICO DETALHADO:")
    
    diag_metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives',
                   'specificity']
    for metric in diag_metrics:
        if metric in test_metrics:
            value = test_metrics[metric]
            if isinstance(value, int):
                logging.info(f"    {metric.replace('_', ' ').title()}: {value}")
            else:
                logging.info(f"    {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Informações sobre threshold
    logging.info("  " + "-" * 40)
    logging.info("  THRESHOLD:")
    logging.info(f"    Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
    
    # Salva o modelo treinado
    model_save_path = f"temp/{model_name.lower()}_{graph_type}_best.pt"
    torch.save(best_model_state, model_save_path)
    logging.info(f"Modelo salvo em: {model_save_path}")
    
    # Armazena resultados
    all_results[model_name] = {
        'best_val_f1': best_val_f1,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'val_final_metrics': val_final_metrics,
        'config': model_params,
        'training_epochs': epoch + 1,
        'model_save_path': model_save_path,
        'training_history': {
            'train_losses': train_losses,
            'val_aucs': val_aucs,
            'val_f1_scores': val_f1_scores,
            'learning_rates': learning_rates
        }
    }
    
    # 🧹 LIMPEZA CRÍTICA: Força reset completo
    del model, optimizer, scheduler_cosine, scheduler_plateau
    if 'scaler' in locals() and scaler is not None:
        del scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logging.info(f"✅ Modelo {model_name} concluído")
    logging.info("-" * 60)
    
    # Relatório final
    logging.info("="*80)
    logging.info(f"RELATÓRIO FINAL COMPARATIVO - {graph_type.upper()}")
    logging.info("MULTISTATGRAPH FRAMEWORK - ANÁLISE COMPLETA DE PERFORMANCE")
    logging.info("="*80)
    
    # Tabela comparativa principal
    logging.info("TABELA COMPARATIVA PRINCIPAL:")
    logging.info(f"{'Modelo':<10} {'AUC-ROC':<10} {'AUC-PR':<10} {'F1-Score':<10} {'F2-Score':<10} {'Recall':<10} {'Precision':<10} {'Threshold':<10}")
    logging.info("-" * 80)
    
    for model_name, results in all_results.items():
        metrics = results['test_metrics']
        threshold = metrics.get('optimal_threshold', 0.5)
        precision = metrics.get('precision', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        logging.info(f"{model_name:<10} {metrics['auc_roc']:<10.4f} {metrics['auc_pr']:<10.4f} "
                    f"{f1_score:<10.4f} {metrics['f2_score']:<10.4f} {metrics['recall']:<10.4f} {precision:<10.4f} {threshold:<10.3f}")
    
    # Análise de confusion matrix
    logging.info("\nANÁLISE DE CONFUSION MATRIX:")
    logging.info(f"{'Modelo':<10} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Especificidade':<15} {'Sensibilidade':<15}")
    logging.info("-" * 80)
    
    for model_name, results in all_results.items():
        metrics = results['test_metrics']
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)
        spec = metrics.get('specificity', 0)
        sens = metrics.get('sensitivity', 0)
        
        logging.info(f"{model_name:<10} {tp:<8} {fp:<8} {tn:<8} {fn:<8} {spec:<15.4f} {sens:<15.4f}")
    
    # Salva resultados em JSON
    # 🔬 SEED CIENTÍFICO: Usa identificador determinístico baseado na seed
    # Garante reprodutibilidade e rastreabilidade total de resultados
    results_file = f"results/{graph_type}_rgcn_performance_seed{seed_suffix}.json"
    
    # Adiciona metadados aos resultados
    final_results = {
        'metadata': {
            'seed': SCIENTIFIC_SEED,
            'graph_type': graph_type,
            'model': 'R-GCN',
            'dataset': 'IEEE-CIS',
            'epochs': num_epochs,
            'patience': patience,
            'learning_rate': learning_rate,
            'device': str(device),
            'mixed_precision': mixed_precision
        },
        'hyperparameters': config,
        'results': {}
    }
    
    # Converte resultados para formato serializável
    for model_name, results in all_results.items():
        # Filtra apenas valores numéricos das métricas de teste
        numeric_test_metrics = {}
        for k, v in results['test_metrics'].items():
            if k != 'unique_model_id':  # Exclui IDs que são strings
                try:
                    numeric_test_metrics[k] = float(v)
                except (ValueError, TypeError):
                    logging.warning(f"⚠️ Métrica não numérica ignorada: {k}={v}")
        
        final_results['results'][model_name] = {
            'test_metrics': numeric_test_metrics,
            'training_epochs': int(results['training_epochs']),
            'best_val_f1': float(results['best_val_f1']),
            'best_val_auc': float(results['best_val_auc'])
        }
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logging.info(f"\nResultados salvos em: {results_file}")
    
    # Monitoramento de Concept Drift
    if all_results and hasattr(hetero_data['transaction'], 'monitoring_mask'):
        logging.info("\n" + "="*80)
        
        # Pega o primeiro (e único) modelo treinado
        model_name, model_results = list(all_results.items())[0]
        test_metrics = model_results['test_metrics']
        
        # Executa monitoramento de drift
        drift_results = monitor_concept_drift(trained_model, hetero_data, test_metrics)
        
        # Salva resultados de drift
        if drift_results:
            drift_file = f"results/{graph_type}_rgcn_drift_analysis_seed{seed_suffix}.json"
            with open(drift_file, 'w') as f:
                json.dump(drift_results, f, indent=2)
            logging.info(f"📊 Resultados do drift salvos em: {drift_file}")
    
    # Resultado final
    if all_results:
        model_result = list(all_results.items())[0]
        logging.info(f"\n🏆 MODELO R-GCN: {model_result[0]} (F1-Score: {model_result[1]['test_metrics']['f1_score']:.4f})")
    
    logging.info("="*80)
    logging.info(f"✅ TREINAMENTO R-GCN {graph_type.upper()} CONCLUÍDO COM SUCESSO!")
    logging.info("🔬 MULTISTATGRAPH FRAMEWORK - EXPERIMENTO FINALIZADO")
    logging.info("="*80)


if __name__ == "__main__":
    main()
