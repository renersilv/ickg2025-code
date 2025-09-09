"""
Anomaly detection module for IEEE-CIS fraud detection dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import gc

# Import utils functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import optimize_dataframe_memory
from feature_config import get_features_for_analysis, validate_features_in_dataframe

# Constants
RANDOM_STATE = 42


def anomaly_detection(X, y, log_dir, logger, contamination=0.05, **kwargs):
    """
    Executa detecção de anomalias usando Isolation Forest.
    
    Args:
        X: DataFrame com features
        y: Series com variável target (isFraud)
        log_dir: Diretório para salvar resultados
        logger: Logger object
        contamination: Taxa de contaminação esperada
        **kwargs: Argumentos adicionais
    
    Returns:
        Tuple (results_dict, anomaly_scores_df)
    """
    logger.info("Executando detecção de anomalias")
    
    # Otimizar memória
    X = optimize_dataframe_memory(X)
    
    # Get optimized feature list for anomaly detection
    target_features = get_features_for_analysis('anomaly', exclude_time=True)
    valid_features, missing_features = validate_features_in_dataframe(X, target_features)
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    logger.info(f"Using {len(valid_features)} features for anomaly detection (from {len(target_features)} expected)")
    
    # Preparar dados
    # Filter to keep only numeric columns for anomaly detection
    X_analysis = X[valid_features]
    numeric_columns = X_analysis.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Filtering to {len(numeric_columns)} numeric columns for anomaly detection")
    
    X_features = X_analysis[numeric_columns].fillna(0)
    
    # Normalizar dados
    logger.info("Normalizando dados para detecção de anomalias...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Aplicar Isolation Forest
    logger.info(f"Aplicando Isolation Forest com contaminação={contamination}")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
        n_jobs=-1
    )
    
    # Treinar o modelo
    iso_forest.fit(X_scaled)
    
    # Calcular scores e labels
    anomaly_scores = iso_forest.decision_function(X_scaled)
    anomaly_labels = iso_forest.predict(X_scaled)
    
    # Criar DataFrame com scores
    anomaly_results = pd.DataFrame({
        'anomaly_score': -anomaly_scores,  # Inverter para que maior = mais anômalo
        'TransactionDT': X['TransactionDT'].values if 'TransactionDT' in X.columns else range(len(X)),
        'isFraud': y.values,
        'TransactionAmt': X['TransactionAmt'].values if 'TransactionAmt' in X.columns else np.zeros(len(X))
    }, index=X.index)
    
    # Analisar anomalias
    n_anomalies = np.sum(anomaly_labels == -1)
    anomaly_fraud_rate = np.mean(y.loc[anomaly_labels == -1]) if n_anomalies > 0 else 0
    
    # Top anomalias
    top_anomalies = anomaly_results.nlargest(int(len(X) * contamination), 'anomaly_score')
    
    # Identificar features que mais contribuem para anomalias
    anomaly_mask = anomaly_labels == -1
    key_anomaly_features = numeric_columns[:8]  # Default fallback using numeric columns
    
    if np.sum(anomaly_mask) > 0:
        anomaly_feature_importance = {}
        normal_mean = X_scaled[~anomaly_mask].mean(axis=0)
        anomaly_mean = X_scaled[anomaly_mask].mean(axis=0)
        
        for i, feat in enumerate(numeric_columns):
            importance = abs(anomaly_mean[i] - normal_mean[i])
            anomaly_feature_importance[feat] = importance
        
        # Top features que distinguem anomalias
        key_anomaly_features_scores = sorted(anomaly_feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:8]
        key_anomaly_features = [feat for feat, _ in key_anomaly_features_scores]
    
    # Graph Construction Insights
    graph_insights = {
        "node_attributes": {
            "key_anomaly_features": key_anomaly_features,
            "anomaly_score": "anomaly_score"
        },
        "edge_construction": {
            "anomaly_based_edges": {
                "anomaly_similarity_threshold": 0.7,
                "normal_similarity_threshold": 0.5,
                "anomaly_boost_factor": 2.0
            }
        },
        "subgraph_analysis": {
            "anomaly_clusters": {
                "high_anomaly_threshold": float(top_anomalies['anomaly_score'].quantile(0.8)),
                "anomaly_fraud_correlation": float(anomaly_fraud_rate)
            }
        }
    }
    
    # Salvar anomaly scores
    anomaly_scores_file = os.path.join(log_dir, "anomaly_scores.csv")
    anomaly_results.to_csv(anomaly_scores_file, index=False)
    
    results = {
        'anomaly_scores_file': 'anomaly_scores.csv',
        'n_anomalies': int(n_anomalies),
        'anomaly_fraud_rate': float(anomaly_fraud_rate),
        'contamination': contamination,
        'top_anomaly_threshold': float(top_anomalies['anomaly_score'].min()),
        'graph_construction_insights': graph_insights
    }
    
    logger.info(f"Detecção de anomalias concluída: {n_anomalies} anomalias encontradas")
    logger.info(f"Taxa de fraude em anomalias: {anomaly_fraud_rate:.4f}")
    logger.info(f"Features chave para anomalias: {key_anomaly_features[:5]}")
    
    # Limpeza de memória
    del X_scaled, anomaly_scores, anomaly_labels
    gc.collect()
    
    return results, anomaly_results
