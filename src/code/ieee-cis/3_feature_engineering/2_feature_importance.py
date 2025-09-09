"""
Feature importance analysis module for IEEE-CIS fraud detection dataset.
Standalone script for feature engineering phase.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
np.seterr(divide='ignore', invalid='ignore')

# Try to import optional packages
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from sklearn.inspection import permutation_importance
    HAS_PERMUTATION = True
except ImportError:
    HAS_PERMUTATION = False

# Constants
RANDOM_STATE = 42


def setup_logging(log_dir):
    """Set up logging to file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"ieee_cis_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('IEEECISFeatureImportance')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path


def optimize_dataframe_memory(df):
    """Otimiza o uso de memória do DataFrame."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
        # Manter colunas categóricas como estão para evitar problemas
    return df


def load_ieee_cis_data(data_path: str) -> pd.DataFrame:
    """Carrega dados do IEEE-CIS com otimizações de memória."""
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {data_path}")
    
    # Otimização de memória
    df = optimize_dataframe_memory(df)
    return df


def advanced_feature_importance_analysis(X, y, log_dir, logger, k=15, **kwargs):
    """
    Executa análise avançada de importância de features com múltiplos métodos.
    
    Args:
        X: DataFrame com features
        y: Series com variável target (Class)
        log_dir: Diretório para salvar resultados
        logger: Logger object
        k: Número de features principais para selecionar
        **kwargs: Argumentos adicionais
    
    Returns:
        Dictionary com resultados da análise
    """
    logger.info("Executando análise avançada de importância de features")
    
    # Otimizar memória
    X = optimize_dataframe_memory(X)
    
    # Preparar dados e remover features constantes
    feature_cols = [col for col in X.columns if col not in ['TransactionDT', 'isFraud']]
    X_features = X[feature_cols].copy()
    
    # Tratar valores nulos de forma apropriada para cada tipo de coluna
    for col in X_features.columns:
        try:
            if X_features[col].dtype.name == 'category':
                # Para colunas categóricas, usar o valor mais frequente ou uma categoria específica
                mode_value = X_features[col].mode()
                if len(mode_value) > 0:
                    X_features[col] = X_features[col].fillna(mode_value[0])
                else:
                    # Se não há modo, converter para string e preencher com 'unknown'
                    X_features[col] = X_features[col].astype(str).fillna('unknown')
            elif pd.api.types.is_numeric_dtype(X_features[col]):
                # Para colunas numéricas, usar 0
                X_features[col] = X_features[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(X_features[col]):
                # Para colunas datetime, converter para timestamp numérico
                X_features[col] = X_features[col].astype('int64', errors='ignore')
                X_features[col] = X_features[col].fillna(0)
            elif pd.api.types.is_timedelta64_dtype(X_features[col]):
                # Para colunas timedelta, converter para segundos
                X_features[col] = X_features[col].dt.total_seconds().fillna(0)
            else:
                # Para outros tipos (string, object), usar 'unknown'
                X_features[col] = X_features[col].fillna('unknown')
        except Exception as e:
            logger.warning(f"Erro ao processar coluna {col}: {e}")
            # Em caso de erro, tentar converter para string
            X_features[col] = X_features[col].astype(str).fillna('unknown')
    
    # Converter colunas categóricas para numéricas para análise de importância
    categorical_cols = X_features.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        logger.info(f"Convertendo {len(categorical_cols)} colunas categóricas para numéricas...")
        
        for col in categorical_cols:
            try:
                if X_features[col].nunique() > 100:
                    # Para colunas com muitas categorias, usar hash ou truncar
                    logger.warning(f"Coluna {col} tem {X_features[col].nunique()} categorias únicas - convertendo para hash")
                    X_features[col] = X_features[col].astype(str).apply(lambda x: hash(x) % 10000)
                else:
                    # Para colunas com poucas categorias, usar LabelEncoder
                    le = LabelEncoder()
                    X_features[col] = le.fit_transform(X_features[col].astype(str))
            except Exception as e:
                logger.warning(f"Erro ao converter coluna categórica {col}: {e}")
                # Fallback: usar hash simples
                X_features[col] = X_features[col].astype(str).apply(lambda x: hash(str(x)) % 1000)
    
    # Categorizar features por tipo para melhor análise
    feature_categories = {
        'original_pca': [col for col in feature_cols if col.startswith('V') and col[1:].isdigit()],  # V2-V59
        'transaction_basic': ['TransactionAmt', 'TransactionDT'],
        'card_features': [col for col in feature_cols if col.startswith('card')],
        'product_features': [col for col in feature_cols if col.startswith('ProductCD')],
        'address_features': [col for col in feature_cols if col.startswith('addr')],
        'distance_features': [col for col in feature_cols if col.startswith('dist')],
        'email_features': [col for col in feature_cols if 'email' in col.lower()],
        'c_features': [col for col in feature_cols if col.startswith('C') and col[1:].isdigit()],  # C2-C14
        'd_features': [col for col in feature_cols if col.startswith('D') and col[1:].isdigit()],  # D2-D15
        'm_features': [col for col in feature_cols if col.startswith('M') and col[1:].isdigit()],  # M1-M9
        'temporal_engineered': ['trans_hour', 'trans_weekday', 'is_night', 'hour_sin', 
                               'temporal_position', 'transaction_timestamp', 'is_rare_hour'],
        'amount_engineered': ['amount_log', 'is_small_amount', 'is_round_amount', 'amount_rank',
                             'is_amount_extreme_high', 'dt_per_amt'],
        'statistical_features': ['v_mean', 'v_missing_count', 'c_missing_count', 'd_missing_count', 
                                'm_missing_count', 'total_missing_features', 'V24_V38_product'],
        'interaction_features': ['card1_card2_combined', 'addr1_addr2_same', 'P_R_email_same'],
        'clustering_features': [f'cluster_{i}' for i in range(10)] + ['v_profile_cluster'],  # Expandido
        'anomaly_features': ['anomaly_score'],
        'pca_features': [f'pc_{i}' for i in range(1, 7)],  # pc_1 to pc_6
        'magnitude_features': ['feature_magnitude', 'high_value_features']
    }
    
    logger.info(f"Dataset com features expandidas - Total: {len(feature_cols)} features")
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in feature_cols]
        if available_features:
            logger.info(f"  {category}: {len(available_features)} features")
    
    # Remove constant features que podem causar problemas na análise estatística
    logger.info("Verificando features constantes...")
    non_constant_cols = []
    for col in feature_cols:
        try:
            # Verificar se a coluna tem mais de um valor único
            if X_features[col].nunique() > 1:
                # Para colunas numéricas, verificar desvio padrão
                if pd.api.types.is_numeric_dtype(X_features[col]):
                    if X_features[col].std() > 1e-10:
                        non_constant_cols.append(col)
                    else:
                        logger.warning(f"Removendo feature constante (baixo std): {col}")
                # Para colunas não-numéricas, apenas verificar se tem variação
                else:
                    non_constant_cols.append(col)
            else:
                logger.warning(f"Removendo feature constante (valores únicos): {col}")
        except Exception as e:
            # Se não conseguir processar a coluna, pular
            logger.warning(f"Removendo feature com erro de processamento: {col} - {str(e)}")
            continue
    
    X_features = X_features[non_constant_cols]
    logger.info(f"Using {len(non_constant_cols)} non-constant features for importance analysis")
    
    # Verificação final: garantir que todas as colunas sejam numéricas
    logger.info("Verificação final de tipos de dados...")
    for col in X_features.columns:
        if not pd.api.types.is_numeric_dtype(X_features[col]):
            logger.warning(f"Forçando conversão numérica para coluna: {col} (tipo: {X_features[col].dtype})")
            try:
                X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0)
            except:
                # Última tentativa: hash dos valores
                X_features[col] = X_features[col].astype(str).apply(lambda x: hash(str(x)) % 1000)
    
    logger.info(f"Análise preparada com {X_features.shape[1]} features numéricas")
    
    # Dividir dados para treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    results = {}
    
    # Método 1: Random Forest Feature Importance
    logger.info("Calculando importância com Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Obter importâncias
    rf_importance = pd.Series(rf.feature_importances_, index=non_constant_cols).sort_values(ascending=False)
    
    # Calcular performance
    y_pred = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_pred)
    
    logger.info(f"Random Forest AUC: {rf_auc:.4f}")
    
    # Método 2: Statistical Feature Selection (SelectKBest)
    logger.info("Executando seleção estatística de features...")
    selector = SelectKBest(score_func=f_classif, k=min(k, len(non_constant_cols)))
    X_selected = selector.fit_transform(X_train, y_train)
    selected_features = [non_constant_cols[i] for i in selector.get_support(indices=True)]
    
    logger.info(f"Features selecionadas estatisticamente: {len(selected_features)}")
    
    # Método 3: Permutation Importance (opcional)
    if HAS_PERMUTATION:
        logger.info("Calculando permutation importance...")
        try:
            # Reduzir n_repeats e usar menos jobs para evitar timeouts
            perm_importance = permutation_importance(
                rf, X_test, y_test, 
                n_repeats=5,  # Reduzido de 10 para 5
                random_state=RANDOM_STATE,
                n_jobs=2  # Reduzido de -1 para 2 para evitar sobrecarga
            )
            
            perm_scores = pd.Series(
                perm_importance.importances_mean, 
                index=non_constant_cols
            ).sort_values(ascending=False)
            
            top_perm_features = perm_scores.head(k).index.tolist()
            results['permutation_importance'] = {feat: float(perm_scores[feat]) for feat in top_perm_features}
            
            logger.info(f"Permutation importance calculada para {len(top_perm_features)} features")
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de permutation importance: {e}")
            results['permutation_importance'] = {}
    else:
        logger.info("Permutation importance não disponível")
        results['permutation_importance'] = {}
    
    # Método 4: SHAP values (opcional)
    if HAS_SHAP and len(X_features) < 50000:
        logger.info("Calculando SHAP values...")
        try:
            # Usar amostra menor para SHAP
            sample_size = min(1000, len(X_test))
            X_shap_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_shap_sample)
            
            # Se binary classification, pegar valores da classe positiva
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calcular importância média absoluta
            shap_importance = pd.Series(
                np.mean(np.abs(shap_values), axis=0),
                index=X_shap_sample.columns
            ).sort_values(ascending=False)
            
            top_shap_features = shap_importance.head(k).index.tolist()
            results['shap_importance'] = {feat: float(shap_importance[feat]) for feat in top_shap_features}
            
            logger.info(f"SHAP values calculados para {len(top_shap_features)} features")
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de SHAP values: {e}")
            results['shap_importance'] = {}
    else:
        logger.info("SHAP values pulados (dataset muito grande ou SHAP não disponível)")
        results['shap_importance'] = {}
    
    # Consolidar features de todos os métodos
    all_methods_features = set()
    all_methods_features.update(rf_importance.head(k).index)
    all_methods_features.update(selected_features)
    
    if 'permutation_importance' in results and results['permutation_importance']:
        all_methods_features.update(results['permutation_importance'].keys())
    
    if 'shap_importance' in results and results['shap_importance']:
        all_methods_features.update(results['shap_importance'].keys())
    
    # Calcular consenso entre métodos
    consensus_scores = {}
    for feature in all_methods_features:
        score = 0
        
        # RF importance
        rf_rank = rf_importance.index.get_loc(feature) if feature in rf_importance.index else len(non_constant_cols)
        score += max(0, k - rf_rank)
        
        # Statistical importance
        if feature in selected_features:
            score += k // 2
        
        # Permutation importance
        if feature in results.get('permutation_importance', {}):
            perm_rank = list(results['permutation_importance'].keys()).index(feature)
            score += max(0, k - perm_rank)
        
        # SHAP importance
        if feature in results.get('shap_importance', {}):
            shap_rank = list(results['shap_importance'].keys()).index(feature)
            score += max(0, k - shap_rank)
        
        consensus_scores[feature] = score
    
    # Selecionar top features por consenso
    final_top_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    final_feature_list = [feat for feat, _ in final_top_features]
    
    # Graph Construction Insights com categorias expandidas
    # Categorizar features finais por tipo
    final_feature_categories = {
        'critical_original_pca': [f for f in final_feature_list if f.startswith('V') and f[1:].isdigit()],
        'critical_card': [f for f in final_feature_list if f.startswith('card')],
        'critical_c_features': [f for f in final_feature_list if f.startswith('C') and f[1:].isdigit()],
        'critical_d_features': [f for f in final_feature_list if f.startswith('D') and f[1:].isdigit()],
        'critical_m_features': [f for f in final_feature_list if f.startswith('M') and f[1:].isdigit()],
        'critical_temporal': [f for f in final_feature_list if f in ['trans_hour', 'trans_weekday', 'is_night', 
                             'hour_sin', 'temporal_position', 'transaction_timestamp', 'is_rare_hour']],
        'critical_amount': [f for f in final_feature_list if 'amount' in f.lower()],
        'critical_statistical': [f for f in final_feature_list if f.startswith('v_') or f.endswith('_missing_count')],
        'critical_clustering': [f for f in final_feature_list if f.startswith('cluster_') or f == 'v_profile_cluster'],
        'critical_pca': [f for f in final_feature_list if f.startswith('pc_')],
        'critical_anomaly': [f for f in final_feature_list if 'anomaly' in f.lower()],
        'critical_magnitude': [f for f in final_feature_list if 'magnitude' in f.lower() or 'high_value' in f.lower()]
    }
    
    graph_insights = {
        "node_attributes": {
            "priority_features": final_feature_list,
            "feature_importance_weights": {
                feat: float(rf_importance.get(feat, 0)) for feat in final_feature_list
            },
            "consensus_scores": dict(final_top_features),
            "feature_categories": {
                "critical": final_feature_list[:10],  # Top 10 mais críticas
                "important": final_feature_list[10:20],  # 10-20
                "moderate": final_feature_list[20:]  # 20+
            },
            "feature_type_distribution": {
                category: len(features) for category, features in final_feature_categories.items() if features
            }
        },
        "edge_construction": {
            "importance_based_weighting": {
                "use_feature_weights": True,
                "weight_scaling_factor": 2.0,
                "min_importance_threshold": 0.005,  # Reduzido para dataset maior
                "multi_method_consensus": True,
                "feature_category_weights": {
                    "original_pca": 1.0,
                    "temporal_engineered": 1.2,
                    "amount_engineered": 1.1,
                    "anomaly_features": 1.3,
                    "clustering_features": 1.15
                }
            }
        },
        "subgraph_analysis": {
            "feature_driven_clusters": {
                "critical_feature_subgraphs": final_feature_list[:10],
                "feature_interaction_analysis": True,
                "importance_method_validation": len([k for k in results.keys() if results[k]]),
                "feature_type_subgraphs": final_feature_categories
            }
        },
        "dataset_characteristics": {
            "total_features_analyzed": len(non_constant_cols),
            "selected_features": len(final_feature_list),
            "feature_reduction_ratio": len(final_feature_list) / len(non_constant_cols),
            "engineered_features_ratio": len([f for f in final_feature_list if not f.startswith('V')]) / len(final_feature_list)
        }
    }
    
    # Salvar resultados
    feature_importance_file = os.path.join(log_dir, "feature_importance_analysis.json")
    with open(feature_importance_file, 'w') as f:
        json.dump({
            'top_features': final_feature_list,
            'rf_importances': rf_importance.head(k).to_dict(),
            'statistical_scores': {non_constant_cols[i]: float(selector.scores_[i]) 
                                  for i in selector.get_support(indices=True)},
            'consensus_scores': dict(final_top_features),
            'rf_auc_score': float(rf_auc)
        }, f, indent=2)
    
    results.update({
        'top_features': final_feature_list,
        'rf_importances': rf_importance.head(k).to_dict(),
        'statistical_scores': {non_constant_cols[i]: float(selector.scores_[i]) 
                              for i in selector.get_support(indices=True)},
        'consensus_scores': dict(final_top_features),
        'methods_used': list(results.keys()),
        'k_selected': k,
        'rf_auc_score': float(rf_auc),
        'graph_construction_insights': graph_insights
    })
    
    logger.info(f"Análise avançada de importância concluída: {len(final_feature_list)} features selecionadas")
    logger.info(f"Métodos utilizados: {len([k for k in results.keys() if results[k] and k.endswith('_importance')])}")
    logger.info(f"Top 5 features: {final_feature_list[:5]}")
    
    # Limpeza de memória
    del X_train, X_test, y_train, y_test, rf
    gc.collect()
    
    return results


def save_results(results: dict, output_dir: str, filename: str):
    """Salva resultados em arquivo JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Resultados salvos em {filepath}")


def main():
    """Função principal para executar análise de importância de features."""
    # Configurar caminhos absolutos baseados no diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    
    # Setup logging
    log_dir = os.path.join(project_root, "data", "statis", "ieee-cis")
    logger, log_path = setup_logging(log_dir)
    
    logger.info("=== Iniciando Análise de Importância de Features para IEEE-CIS ===")
    logger.info(f"Diretório do script: {script_dir}")
    logger.info(f"Raiz do projeto: {project_root}")
    logger.info(f"Diretório de logs: {log_dir}")
    
    try:
        # Configurar caminhos
        data_path = os.path.join(project_root, "data", "parquet", "ieee-cis", "ieee-cis-features.parquet")
        output_dir = os.path.join(project_root, "data", "statis", "ieee-cis")
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
        df = load_ieee_cis_data(data_path)
        
        # Preparar dados
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Informações do dataset expandido
        df_info = {
            "path": data_path,
            "shape": df.shape,
            "features": df.shape[1] - 1,  # Excluindo coluna isFraud
            "samples": df.shape[0],
            "target_column": "isFraud",
            "fraud_rate": float(y.mean()),
            "dataset_type": "IEEE-CIS with engineered features",
            "feature_types": {
                "original_pca_features": len([col for col in df.columns if col.startswith('V') and col[1:].isdigit()]),
                "amount_features": len([col for col in df.columns if 'amount' in col.lower()]),
                "card_features": len([col for col in df.columns if col.startswith('card')]),
                "addr_features": len([col for col in df.columns if col.startswith('addr')]),
                "c_features": len([col for col in df.columns if col.startswith('C') and col[1:].isdigit()]),
                "d_features": len([col for col in df.columns if col.startswith('D') and col[1:].isdigit()]),
                "m_features": len([col for col in df.columns if col.startswith('M') and col[1:].isdigit()]),
                "email_features": len([col for col in df.columns if 'email' in col.lower()]),
                "temporal_features": len([col for col in df.columns if any(temporal in col.lower() for temporal in ['trans_', 'hour', 'temporal', 'time', 'night'])]),
                "statistical_features": len([col for col in df.columns if col.startswith('v_') or col.endswith('_missing_count')]),
                "clustering_features": len([col for col in df.columns if col.startswith('cluster_') or col == 'v_profile_cluster']),
                "pca_features": len([col for col in df.columns if col.startswith('pc_')]),
                "anomaly_features": len([col for col in df.columns if 'anomaly' in col.lower()]),
                "magnitude_features": len([col for col in df.columns if 'magnitude' in col.lower() or 'high_value' in col.lower()])
            }
        }
        
        logger.info(f"Dataset expandido carregado: {df_info['shape']} | Taxa de fraude: {df_info['fraud_rate']:.4f}")
        logger.info(f"Tipos de features:")
        logger.info(f"  - Original PCA (V2-V59): {df_info['feature_types']['original_pca_features']}")
        logger.info(f"  - Card features: {df_info['feature_types']['card_features']}")
        logger.info(f"  - Address features: {df_info['feature_types']['addr_features']}")
        logger.info(f"  - C features: {df_info['feature_types']['c_features']}")
        logger.info(f"  - D features: {df_info['feature_types']['d_features']}")
        logger.info(f"  - M features: {df_info['feature_types']['m_features']}")
        logger.info(f"  - Email features: {df_info['feature_types']['email_features']}")
        logger.info(f"  - Temporal features: {df_info['feature_types']['temporal_features']}")
        logger.info(f"  - Amount features: {df_info['feature_types']['amount_features']}")
        logger.info(f"  - Clustering features: {df_info['feature_types']['clustering_features']}")
        logger.info(f"  - PCA features: {df_info['feature_types']['pca_features']}")
        logger.info(f"  - Anomaly features: {df_info['feature_types']['anomaly_features']}")
        logger.info(f"  - Magnitude features: {df_info['feature_types']['magnitude_features']}")
        
        # Executar análise de importância de features
        logger.info("--- Executando Análise de Importância de Features ---")
        feature_results = advanced_feature_importance_analysis(X, y, output_dir, logger, k=30)
        save_results(feature_results, output_dir, "feature_importance_results.json")
        
        logger.info("=== Análise de Importância de Features Concluída ===")
        
        # Resumo da análise expandida
        logger.info("Resumo da Análise de Features Expandidas:")
        logger.info(f"  - Total de features analisadas: {df_info['features']}")
        logger.info(f"  - Features importantes identificadas: {len(feature_results['top_features'])}")
        logger.info(f"  - Métodos utilizados: {len([k for k in feature_results.keys() if feature_results[k] and k.endswith('_importance')])}")
        logger.info(f"  - AUC Random Forest: {feature_results['rf_auc_score']:.4f}")
        logger.info(f"  - Top 10 features: {feature_results['top_features'][:10]}")
        logger.info(f"  - Taxa de redução de features: {feature_results['graph_construction_insights']['dataset_characteristics']['feature_reduction_ratio']:.3f}")
        logger.info(f"  - Proporção de features engineered: {feature_results['graph_construction_insights']['dataset_characteristics']['engineered_features_ratio']:.3f}")
        logger.info(f"  - Arquivo salvo: feature_importance_results.json")
        
    except Exception as e:
        logger.error(f"Erro durante análise de importância de features: {e}")
        raise


if __name__ == "__main__":
    main()
