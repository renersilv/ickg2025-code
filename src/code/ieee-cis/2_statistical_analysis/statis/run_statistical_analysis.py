"""
Modular statistical analysis orchestrator for IEEE-CIS fraud detection dataset.
This script coordinates all statistical analyses using the modular framework.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
np.seterr(divide='ignore', invalid='ignore')

# Add the framework directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from utils import setup_logging, optimize_dataframe_memory
from correlation_analysis import correlation_analysis
from pca_analysis import pca_analysis
from cluster_analysis import cluster_analysis
from anomaly_detection import anomaly_detection
from temporal_analysis import temporal_analysis
from feature_config import get_feature_info
from validation_utils import (
    validate_reproducibility_seeds, 
    validate_data_integrity,
    validate_statistical_consistency,
    validate_no_file_duplications,
    save_validation_report,
    check_memory_usage,
    log_environment_info
)

# Constants
RANDOM_STATE = 42


def load_ieee_cis_data(data_path: str) -> pd.DataFrame:
    """Carrega dados do IEEE-CIS com otimizações de memória."""
    logger = logging.getLogger(__name__)
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {data_path}")
    
    # Otimização de memória: converter tipos de dados para economizar RAM
    logger.info("Otimizando tipos de dados para economizar memória...")
    
    # Check for constant columns that may cause issues
    logger.info("Checking for problematic columns...")
    constant_cols = []
    for col in df.columns:
        if col != 'isFraud':
            try:
                # Check numeric columns for constant/near-constant values
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique() <= 1 or (df[col].std() < 1e-10 if not df[col].isna().all() else False):
                        constant_cols.append(col)
                # Check categorical columns for constant values
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                    if df[col].nunique() <= 1:
                        constant_cols.append(col)
            except Exception as e:
                logger.warning(f"Could not check column {col} for constants: {e}")
                continue
    
    if constant_cols:
        logger.warning(f"Found {len(constant_cols)} constant/near-constant columns: {constant_cols[:5]}...")
        # Note: We keep them for now but analyses will handle them appropriately
    
    # Convert float64 to float32 (exceto colunas críticas)
    float_cols = df.select_dtypes(include=['float64']).columns
    critical_cols = {'TransactionAmt', 'TransactionDT'}  # Manter precisão para essas colunas
    
    for col in float_cols:
        if col not in critical_cols:
            df[col] = df[col].astype('float32')
    
    # Convert int64 to int32 quando possível (exceto isFraud)
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col != 'isFraud':
            # Verificar se os valores cabem em int32
            if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
    
    logger.info(f"Otimização de memória concluída. Economia estimada: ~25% RAM")
    
    return df


def consolidate_results(corr_results, pca_results, cluster_results, anomaly_results, 
                       temporal_results, df_info):
    """Consolida todos os resultados das análises e gera recomendações integradas."""
    logger = logging.getLogger(__name__)
    logger.info("Consolidando resultados de todas as análises...")
    
    # Extrair features prioritárias de cada análise
    top_correlation_features = set()
    top_pca_features = set()
    top_cluster_features = set()
    top_anomaly_features = set()
    top_temporal_features = set()
    
    # Extrair features de cada análise com verificação de segurança
    if 'graph_construction_insights' in corr_results:
        insights = corr_results['graph_construction_insights']
        if 'node_attributes' in insights and 'priority_features' in insights['node_attributes']:
            top_correlation_features = set(insights['node_attributes']['priority_features'])
    
    if 'graph_construction_insights' in pca_results:
        insights = pca_results['graph_construction_insights']
        if 'node_attributes' in insights and 'priority_features' in insights['node_attributes']:
            top_pca_features = set(insights['node_attributes']['priority_features'])
    
    if 'graph_construction_insights' in cluster_results:
        insights = cluster_results['graph_construction_insights']
        if 'node_attributes' in insights and 'distinctive_features' in insights['node_attributes']:
            top_cluster_features = set(insights['node_attributes']['distinctive_features'])
    
    if 'graph_construction_insights' in anomaly_results:
        insights = anomaly_results['graph_construction_insights']
        if 'node_attributes' in insights and 'key_anomaly_features' in insights['node_attributes']:
            top_anomaly_features = set(insights['node_attributes']['key_anomaly_features'])
    
    if 'graph_construction_insights' in temporal_results:
        insights = temporal_results['graph_construction_insights']
        if 'node_attributes' in insights and 'priority_features' in insights['node_attributes']:
            top_temporal_features = set(insights['node_attributes']['priority_features'])
    
    # Combinar todas as features prioritárias com pesos por tipo de análise
    all_features = (top_correlation_features | top_pca_features | top_cluster_features | 
                   top_anomaly_features | top_temporal_features)
    
    # Calcular consenso entre análises com pesos ajustados
    consensus_features = []
    feature_weights = {
        'correlation_analysis': 1.8,  # Correlação é importante para fraude
        'anomaly_detection': 1.8,  # Anomalias são críticas para fraude
        'cluster_analysis': 1.2,  # Clustering oferece perspectiva diferente
        'pca_analysis': 1.2,  # PCA mostra variância
        'temporal_analysis': 1.2   # Temporal mostra padrões temporais
    }
    
    for feature in all_features:
        weighted_score = 0
        count = 0
        
        if feature in top_correlation_features:
            weighted_score += feature_weights.get('correlation_analysis', 1.0)
            count += 1
        if feature in top_pca_features:
            weighted_score += feature_weights.get('pca_analysis', 1.0)
            count += 1
        if feature in top_cluster_features:
            weighted_score += feature_weights.get('cluster_analysis', 1.0)
            count += 1
        if feature in top_anomaly_features:
            weighted_score += feature_weights.get('anomaly_detection', 1.0)
            count += 1
        if feature in top_temporal_features:
            weighted_score += feature_weights.get('temporal_analysis', 1.0)
            count += 1
        
        consensus_features.append((feature, count, weighted_score))
    
    # Ordenar por score ponderado, depois por contagem
    consensus_features = sorted(consensus_features, key=lambda x: (x[2], x[1]), reverse=True)
    
    # Features que aparecem em pelo menos 1 análise (mais flexível para fraud detection)
    final_priority_features = [feat for feat, count, score in consensus_features if count >= 1]
    
    # Features de alto consenso (aparecem em múltiplas análises)
    high_consensus_features = [feat for feat, count, score in consensus_features if count >= 2]
    
    # Recomendações integradas para construção de grafos
    integrated_recommendations = {
        "node_attributes": {
            "final_priority_features": final_priority_features,
            "feature_consensus_scores": dict([(feat, {'count': count, 'weighted_score': score}) for feat, count, score in consensus_features]),
            "recommended_node_features": {
                "critical": final_priority_features[:5],
                "important": final_priority_features[5:15],
                "moderate": final_priority_features[15:25]
            }
        },
        "edge_construction_strategies": {
            "similarity_based": {
                "primary_features": final_priority_features[:10],
                "similarity_threshold": 0.8,
                "use_weighted_features": True
            },
            "cluster_based": {
                "high_risk_clusters": cluster_results.get('high_risk_clusters', []),
                "cluster_similarity_boost": 1.5
            },
            "anomaly_based": {
                "anomaly_threshold": anomaly_results.get('top_anomaly_threshold', 0.1),
                "anomaly_edge_boost": 2.0
            },
            "temporal_based": {
                "time_window_seconds": temporal_results.get('window_size_seconds', 3600),
                "fraud_time_boost": 1.8
            }
        },
        "subgraph_analysis": {
            "recommended_subgraph_types": [
                "high_risk_clusters",
                "anomaly_networks", 
                "temporal_fraud_windows",
                "feature_similarity_groups"
            ],
            "subgraph_priority_order": [
                "anomaly_networks",
                "high_risk_clusters", 
                "temporal_fraud_windows",
                "feature_similarity_groups"
            ]
        }
    }
    
    # Consolidar resultados finais
    consolidated_results = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset_info": df_info,
        "individual_analyses": {
            "correlation_analysis": corr_results,
            "pca_analysis": pca_results,
            "cluster_analysis": cluster_results,
            "anomaly_detection": anomaly_results,
            "temporal_analysis": temporal_results
        },
        "integrated_graph_construction_recommendations": integrated_recommendations,
        "summary": {
            "total_features_analyzed": len(all_features),
            "consensus_features": len(final_priority_features),
            "high_consensus_features": len(high_consensus_features),
            "recommended_strategies": 4,  # similarity, cluster, anomaly, temporal
            "feature_analysis_breakdown": {
                "correlation_features": len(top_correlation_features),
                "pca_features": len(top_pca_features),
                "cluster_features": len(top_cluster_features),
                "anomaly_features": len(top_anomaly_features),
                "temporal_features": len(top_temporal_features)
            }
        }
    }
    
    return consolidated_results


def save_results(results: dict, output_dir: str, filename: str):
    """Salva resultados em arquivo JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Resultados salvos em {filepath}")


def main():
    """Função principal que executa todas as análises estatísticas usando framework modular."""
    # Setup logging
    log_dir = "data/statis/ieee-cis"
    logger, log_path = setup_logging(log_dir)
    
    logger.info("=== Iniciando Análises Estatísticas Modulares para IEEE-CIS ===")
    
    # Log environment information for reproducibility
    log_environment_info(logger)
    
    # Validate reproducibility seeds
    if not validate_reproducibility_seeds():
        logger.error("✗ Seed validation failed - results may not be reproducible!")
        raise ValueError("Reproducibility validation failed")
    
    try:
        # Configurar caminhos
        data_path = "data/parquet/ieee-cis/ieee-cis.parquet"
        output_dir = "data/statis/ieee-cis"
        
        # Check initial memory usage
        initial_memory = check_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory['rss_mb']:.1f} MB RSS, {initial_memory['percent']:.1f}%")
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}")
        df = load_ieee_cis_data(data_path)
        
        # Validate initial data integrity
        data_validation = validate_data_integrity(df, "initial_data")
        save_validation_report(data_validation, output_dir, "data_integrity_validation.json")
        
        if not data_validation['validation_passed']:
            logger.error("✗ Initial data validation failed!")
            for issue in data_validation['issues']:
                logger.error(f"  - {issue}")
            raise ValueError("Data integrity validation failed")
        
        # Otimizar memória
        df = optimize_dataframe_memory(df)
        
        # Preparar dados
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Validate processed data
        processed_validation = validate_data_integrity(X, "processed_features")
        if not processed_validation['validation_passed']:
            logger.warning("⚠️ Processed data validation issues detected")
        
        # Informações do dataset para consolidação
        df_info = {
            "path": data_path,
            "shape": df.shape,
            "features": df.shape[1] - 1,  # Excluindo isFraud
            "samples": df.shape[0],
            "target_column": "isFraud",
            "fraud_rate": float(y.mean())
        }
        
        logger.info(f"Dataset carregado: {df_info['shape']} | Taxa de fraude: {df_info['fraud_rate']:.4f}")
        
        # Log feature configuration info
        feature_info = get_feature_info()
        logger.info("=== Configuração de Features ===")
        logger.info(f"Total de features disponíveis para análise: {feature_info['total_features']}")
        logger.info(f"Features por categoria:")
        for category, count in feature_info['by_category'].items():
            logger.info(f"  - {category}: {count} features")
        logger.info(f"Features excluídas: {feature_info['excluded_features']}")
        logger.info("=== Iniciando Análises ===")
        
        # Dictionary to store all analysis results for consistency validation
        all_analysis_results = {}
        
        # Executar análises estatísticas modulares
        logger.info("Executando análises estatísticas modulares...")
        
        # 1. Análise de Correlação
        logger.info("--- 1. Análise de Correlação ---")
        corr_results = correlation_analysis(X, y, output_dir, logger)
        save_results(corr_results, output_dir, "correlation_results.json")
        all_analysis_results['correlation_analysis'] = corr_results  # Store results for validation
        
        # 2. Análise PCA
        logger.info("--- 2. Análise PCA ---")
        pca_results = pca_analysis(X, y, output_dir, logger)
        save_results(pca_results, output_dir, "pca_results.json")
        all_analysis_results['pca_analysis'] = pca_results  # Store results for validation
        
        # 3. Análise de Clustering
        logger.info("--- 3. Análise de Clustering ---")
        cluster_results, cluster_assignments = cluster_analysis(X, y, output_dir, logger)
        save_results(cluster_results, output_dir, "cluster_results.json")
        all_analysis_results['cluster_analysis'] = cluster_results  # Store results for validation
        # Note: cluster_assignments.csv is already saved in cluster_analysis function in cluster_analysis/ subdirectory
        
        # 4. Detecção de Anomalias
        logger.info("--- 4. Detecção de Anomalias ---")
        anomaly_results, anomaly_scores = anomaly_detection(X, y, output_dir, logger)
        save_results(anomaly_results, output_dir, "anomaly_results.json")
        all_analysis_results['anomaly_detection'] = anomaly_results  # Store results for validation
        # Note: anomaly_scores.csv is already saved in anomaly_detection function
        
        # 5. Análise Temporal
        logger.info("--- 5. Análise Temporal ---")
        temporal_results = temporal_analysis(X, y, output_dir, logger)
        save_results(temporal_results, output_dir, "temporal_analysis_results.json")
        all_analysis_results['temporal_analysis'] = temporal_results  # Store results for validation
        
        # 6. Consolidação de Resultados e Recomendações Integradas
        logger.info("--- 6. Consolidação de Resultados ---")
        consolidated_results = consolidate_results(
            corr_results, pca_results, cluster_results, 
            anomaly_results, temporal_results, df_info
        )
        save_results(consolidated_results, output_dir, "consolidated_multistat_results.json")
        
        logger.info("=== Todas as Análises Estatísticas Modulares Concluídas ===")
        
        # Resumo detalhado
        recommendations = consolidated_results['integrated_graph_construction_recommendations']
        logger.info("Resumo das Análises Modulares:")
        logger.info(f"  - Clusters de alto risco: {len(cluster_results.get('high_risk_clusters', []))}")
        logger.info(f"  - Anomalias detectadas: {anomaly_results['n_anomalies']}")
        logger.info(f"  - Janelas temporais de risco: {len(temporal_results['high_risk_windows'])}")
        logger.info(f"  - Pares de alta correlação: {len(corr_results.get('high_correlation_pairs', []))}")
        logger.info("")
        logger.info("Recomendações Integradas para Construção de Grafos:")
        logger.info(f"  - Features prioritárias (consenso): {len(recommendations['node_attributes']['final_priority_features'])}")
        logger.info(f"  - Features críticas: {len(recommendations['node_attributes']['recommended_node_features']['critical'])}")
        logger.info(f"  - Estratégias de arestas: {consolidated_results['summary']['recommended_strategies']}")
        logger.info(f"  - Arquivo consolidado: consolidated_multistat_results.json")
        
        # Validate statistical consistency across analyses
        logger.info("--- 6. Validação da Consistência Estatística ---")
        consistency_report = validate_statistical_consistency(all_analysis_results)
        save_validation_report(consistency_report, output_dir, "statistical_consistency_validation.json")
        
        if not consistency_report['validation_passed']:
            logger.warning("⚠️ Questões de consistência estatística detectadas:")
            for rec in consistency_report['recommendations']:
                logger.warning(f"  - {rec}")
        else:
            logger.info("✓ Validação de consistência estatística passou")
        
        # Validate no file duplications
        logger.info("--- 7. Validação de Duplicação de Arquivos ---")
        duplication_report = validate_no_file_duplications(output_dir)
        save_validation_report(duplication_report, output_dir, "file_duplication_validation.json")
        
        if not duplication_report['validation_passed']:
            logger.error("✗ Arquivos duplicados detectados:")
            for issue in duplication_report['issues']:
                logger.error(f"  - {issue}")
            raise ValueError("File duplication validation failed - check output structure")
        else:
            logger.info("✓ Validação de duplicação de arquivos passou")
        
        # Final memory usage report
        final_memory = check_memory_usage()
        logger.info(f"Final memory usage: {final_memory['rss_mb']:.1f} MB RSS, {final_memory['percent']:.1f}%")
        memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
        logger.info(f"Memory increase during analysis: {memory_increase:.1f} MB")
        
        logger.info("=== TODAS AS ANÁLISES E VALIDAÇÕES CONCLUÍDAS COM SUCESSO ===")
        
    except Exception as e:
        logger.error(f"Erro durante as análises estatísticas modulares: {e}")
        raise


if __name__ == "__main__":
    main()
