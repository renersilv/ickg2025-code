"""
Temporal analysis module for IEEE-CIS fraud d    if 'TransactionDT' not in X.column    for i in range(0, int(df_sorted['TransactionDT'].max()), window_size):
        window_end = i + window_size
        window_data = df_sorted[(df_sorted['TransactionDT'] >= i) & (df_sorted['TransactionDT'] < window_end)]
        
        if len(window_data) > 0:
            time_windows.append({
                'start_time': i,
                'end_time': window_end,
                'transaction_count': len(window_data),
                'fraud_count': int(window_data['isFraud'].sum()),
                'fraud_rate': float(window_data['isFraud'].mean()),
                'avg_amount': float(window_data['TransactionAmt'].mean()) if 'TransactionAmt' in window_data.columns else 0.0
            })gger.warning("Coluna 'TransactionDT' não encontrada, criando índice temporal sintético")
        X = X.copy()
        X['TransactionDT'] = range(len(X))
        available_temporal.append('TransactionDT')
    
    # Criar DataFrame combinado para análise
    df = X[available_temporal].copy()
    df['isFraud'] = y
    
    # Ordenar por tempo
    df_sorted = df.sort_values('TransactionDT').copy()
    
    # Analisar distribuição temporal de fraudes
    df_sorted['Time_hour'] = (df_sorted['TransactionDT'] // 3600) % 24  # Hora do dia
    hourly_fraud_rate = df_sorted.groupby('Time_hour')['isFraud'].mean().
"""

import os
import numpy as np
import pandas as pd
import gc
import json

# Import utils functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import optimize_dataframe_memory
from feature_config import get_features_for_analysis, validate_features_in_dataframe, FEATURE_CATEGORIES

# Constants
RANDOM_STATE = 42


def temporal_analysis(X, y, log_dir, logger, **kwargs):
    """
    Executa análise temporal dos padrões de fraude.
    
    Args:
        X: DataFrame com features (deve incluir coluna 'TransactionDT')
        y: Series com variável target (isFraud)
        log_dir: Diretório para salvar resultados
        logger: Logger object
        **kwargs: Argumentos adicionais
    
    Returns:
        Dictionary com resultados da análise temporal
    """
    logger.info("Executando análise temporal")
    
    # Otimizar memória
    X = optimize_dataframe_memory(X)
    
    # Get temporal features for enhanced analysis
    temporal_features = FEATURE_CATEGORIES['temporal_features']
    available_temporal, missing_temporal = validate_features_in_dataframe(X, temporal_features)
    
    if missing_temporal:
        logger.warning(f"Missing temporal features: {missing_temporal}")
    
    logger.info(f"Using {len(available_temporal)} temporal features: {available_temporal}")
    
    # Verificar se temos coluna TransactionDT
    if 'TransactionDT' not in X.columns:
        logger.warning("Coluna 'TransactionDT' não encontrada, criando índice temporal sintético")
        X = X.copy()
        X['TransactionDT'] = range(len(X))
        available_temporal.append('TransactionDT')
    
    # Criar DataFrame combinado para análise
    df = X[available_temporal].copy()
    df['isFraud'] = y
    
    # Ordenar por tempo
    df_sorted = df.sort_values('TransactionDT').copy()
    
    # Convert TransactionDT to appropriate type for calculations to avoid overflow
    df_sorted['TransactionDT'] = df_sorted['TransactionDT'].astype('int64')
    
    # Analisar distribuição temporal de fraudes
    df_sorted['Time_hour'] = (df_sorted['TransactionDT'] // 3600) % 24  # Hora do dia
    hourly_fraud_rate = df_sorted.groupby('Time_hour')['isFraud'].mean()
    
    # Análise de janelas temporais
    window_size = 3600  # 1 hora em segundos
    time_windows = []
    
    logger.info(f"Analisando janelas temporais de {window_size}s...")
    
    for i in range(0, int(df_sorted['TransactionDT'].max()), window_size):
        window_end = i + window_size
        window_data = df_sorted[(df_sorted['TransactionDT'] >= i) & (df_sorted['TransactionDT'] < window_end)]
        
        if len(window_data) > 0:
            time_windows.append({
                'start_time': i,
                'end_time': window_end,
                'transaction_count': len(window_data),
                'fraud_count': int(window_data['isFraud'].sum()),
                'fraud_rate': float(window_data['isFraud'].mean()),
                'avg_amount': float(window_data['TransactionAmt'].mean()) if 'TransactionAmt' in window_data.columns else 0.0
            })
    
    # Identificar janelas de alto risco
    high_risk_windows = [w for w in time_windows if w['fraud_rate'] > 0.1]
    
    # Identificar padrões temporais
    peak_hours = hourly_fraud_rate.nlargest(5).index.tolist()
    low_hours = hourly_fraud_rate.nsmallest(5).index.tolist()
    
    # Features temporais relevantes para construção de grafos
    temporal_features = ['TransactionDT', 'Time_hour']
    if 'TransactionAmt' in df.columns:
        # Calcular variações temporais do TransactionAmt
        df_sorted['Amount_rolling_mean'] = df_sorted['TransactionAmt'].rolling(window=100, min_periods=1).mean()
        temporal_features.extend(['TransactionAmt', 'Amount_rolling_mean'])
    
    # Graph Construction Insights
    graph_insights = {
        "node_attributes": {
            "priority_features": temporal_features,
            "temporal_encoding": {
                "hour_of_day": True,
                "time_since_start": True,
                "rolling_statistics": ["mean", "std"]
            }
        },
        "edge_construction": {
            "temporal_edges": {
                "time_window_seconds": window_size,
                "temporal_threshold": 1800,  # 30 minutos
                "fraud_time_boost": 2.0
            },
            "temporal_similarity": {
                "same_hour_bonus": 0.3,
                "time_decay_factor": 0.1
            }
        },
        "subgraph_analysis": {
            "high_fraud_time_windows": [
                {
                    "start_time": w['start_time'],
                    "end_time": w['end_time'],
                    "fraud_rate": w['fraud_rate']
                } for w in high_risk_windows
            ],
            "temporal_clustering": {
                "peak_fraud_hours": peak_hours,
                "low_fraud_hours": low_hours
            }
        }
    }
    
    # Salvar análise temporal
    temporal_analysis_file = os.path.join(log_dir, "temporal_analysis.json")
    temporal_data = {
        'hourly_fraud_rates': hourly_fraud_rate.to_dict(),
        'time_windows': time_windows,
        'high_risk_windows': high_risk_windows,
        'peak_fraud_hours': peak_hours,
        'low_fraud_hours': low_hours,
        'total_time_span': float(df_sorted['TransactionDT'].max() - df_sorted['TransactionDT'].min()),
        'window_size_seconds': window_size
    }
    
    with open(temporal_analysis_file, 'w') as f:
        json.dump(temporal_data, f, indent=2)
    
    results = {
        'hourly_fraud_rates': hourly_fraud_rate.to_dict(),
        'time_windows': time_windows,
        'high_risk_windows': high_risk_windows,
        'window_size_seconds': window_size,
        'total_time_span': float(df_sorted['TransactionDT'].max() - df_sorted['TransactionDT'].min()),
        'peak_fraud_hours': peak_hours,
        'low_fraud_hours': low_hours,
        'graph_construction_insights': graph_insights
    }
    
    logger.info(f"Análise temporal concluída: {len(high_risk_windows)} janelas de alto risco identificadas")
    logger.info(f"Horários de pico de fraude: {peak_hours}")
    logger.info(f"Período total analisado: {results['total_time_span']:.0f} segundos")
    
    # Limpeza de memória
    del df, df_sorted
    gc.collect()
    
    return results
