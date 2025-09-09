"""
Feature normalization module for IEEE-CIS fraud detection dataset.
Implements complete standardization for Graph Neural Network compatibility.
Standalone script for feature engineering phase.
"""

import os
import sys
import json
import logging
import warnings
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import gc

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
np.seterr(divide='ignore', invalid='ignore')

# Constants
RANDOM_STATE = 42


def setup_logging(log_dir):
    """Set up logging to file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"ieee_cis_normalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('IEEECISNormalization')
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


def analyze_feature_distributions(df, logger):
    """Analisa distribuições das features para escolha da normalização adequada."""
    logger.info("Analisando distribuições das features...")
    
    # Excluir TransactionDT e isFraud da análise
    feature_cols = [col for col in df.columns if col not in ['TransactionDT', 'isFraud']]
    
    # Categorizar features por tipo (incluindo novas features do data preparation)
    feature_categories = {
        'original_pca': [f'V{i}' for i in range(1, 29) if f'V{i}' in feature_cols],
        'amount_features': [col for col in feature_cols if 'amount' in col.lower()],
        'card_features': [col for col in feature_cols if col.startswith('card')],
        'addr_features': [col for col in feature_cols if col.startswith('addr')],
        'device_features': [col for col in feature_cols if 'device' in col.lower()],
        'user_features': [col for col in feature_cols if col.startswith('user_')],
        'temporal_engineered': [col for col in feature_cols if any(t in col.lower() for t in ['time', 'hour', 'temporal', 'time_diff', 'tx_velocity'])],
        'statistical_features': [col for col in feature_cols if col.startswith('v_')],
        'binary_features': [col for col in feature_cols if df[col].nunique() == 2],
        'clustering_features': [col for col in feature_cols if col.startswith('cluster_')],
        'pca_features': [col for col in feature_cols if col.startswith('pc_')],
        'anomaly_features': [col for col in feature_cols if 'anomaly' in col.lower()],
        'magnitude_features': [col for col in feature_cols if any(m in col.lower() for m in ['magnitude', 'high_value'])]
    }
    
    # Análise estatística por categoria
    distribution_analysis = {}
    
    for category, features in feature_categories.items():
        if not features:
            continue
            
        category_stats = {
            'features': features,
            'count': len(features),
            'mean_range': [],
            'std_range': [],
            'skewness_issues': [],
            'outlier_presence': []
        }
        
        for feature in features:
            if feature in df.columns:
                data = df[feature].dropna()
                
                # Verificar se a feature é numérica antes de calcular estatísticas
                if pd.api.types.is_numeric_dtype(data):
                    # Estatísticas básicas para colunas numéricas
                    mean_val = float(data.mean())
                    std_val = float(data.std())
                    skew_val = float(data.skew())
                    
                    # Detecção de outliers (IQR method)
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_threshold = 1.5 * IQR
                    outliers_count = ((data < (Q1 - outlier_threshold)) | (data > (Q3 + outlier_threshold))).sum()
                    outlier_percentage = (outliers_count / len(data)) * 100
                    
                    category_stats['mean_range'].append(abs(mean_val))
                    category_stats['std_range'].append(std_val)
                    
                    if abs(skew_val) > 2:  # Highly skewed
                        category_stats['skewness_issues'].append(feature)
                    
                    if outlier_percentage > 5:  # More than 5% outliers
                        category_stats['outlier_presence'].append(feature)
                else:
                    # Para colunas categóricas/não-numéricas, pular análise estatística
                    logger.info(f"  Pulando análise estatística para coluna categórica: {feature}")
                    continue
        
        distribution_analysis[category] = category_stats
        logger.info(f"{category}: {len(features)} features")
        if category_stats['skewness_issues']:
            logger.info(f"  - Features com alta assimetria: {len(category_stats['skewness_issues'])}")
        if category_stats['outlier_presence']:
            logger.info(f"  - Features com muitos outliers: {len(category_stats['outlier_presence'])}")
    
    return distribution_analysis, feature_categories


def normalize_features_for_gnn(df, distribution_analysis, feature_categories, logger, scaler_type='standard_clipped', clip_percentile=98.5):
    """
    Normaliza todas as features (exceto TransactionDT, isFraud e card1) para compatibilidade com GNN.
    VERSÃO OTIMIZADA: StandardScaler com clipping robusto de outliers extremos.
    
    Args:
        df: DataFrame com features
        distribution_analysis: Análise de distribuições
        feature_categories: Categorização das features
        logger: Logger object
        scaler_type: Tipo de scaler ('standard_clipped', 'standard', 'minmax', 'robust')
        clip_percentile: Percentil para clipping de outliers (padrão: 98.5%)
    
    Returns:
        Tuple (df_normalized, scalers_dict, normalization_metadata)
    """
    logger.info(f"🔧 Iniciando normalização OTIMIZADA para GNN com {scaler_type} scaler...")
    
    # Features a serem excluídas da normalização (TransactionDT, isFraud e card1)
    # card1 é usado como IDENTITY_COLUMN para agrupamento no grafo
    exclude_features = ['TransactionDT', 'isFraud', 'card1']
    
    # Features a serem normalizadas (todas as outras)
    feature_cols = [col for col in df.columns if col not in exclude_features]
    
    logger.info(f"Total de features a normalizar: {len(feature_cols)}")
    logger.info(f"Features excluídas: {exclude_features}")
    
    # Preparar DataFrame para normalização
    df_normalized = df.copy()
    X_features = df[feature_cols].copy()
    
    # Tratar colunas categóricas e valores nulos antes da normalização
    categorical_cols = []
    logger.info("Preparando features para normalização...")
    
    for col in feature_cols:
        if isinstance(X_features[col].dtype, pd.CategoricalDtype) or X_features[col].dtype == 'object':
            categorical_cols.append(col)
            logger.info(f"Convertendo coluna categórica para numérica: {col}")
            
            if isinstance(X_features[col].dtype, pd.CategoricalDtype):
                # Para colunas categóricas, usar os códigos diretos
                X_features[col] = X_features[col].cat.codes.astype('int32')
                # Códigos categóricas já usam -1 para NaN, então não precisa fillna
            else:
                # Para colunas object, converter usando LabelEncoder simples
                unique_values = X_features[col].dropna().unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                X_features[col] = X_features[col].map(value_map).fillna(-1).astype('int32')  # -1 para valores nulos
        elif pd.api.types.is_datetime64_any_dtype(X_features[col]):
            logger.info(f"Convertendo coluna datetime para timestamp: {col}")
            X_features[col] = X_features[col].astype('int64', errors='ignore').astype('float64')
        elif pd.api.types.is_timedelta64_dtype(X_features[col]):
            logger.info(f"Convertendo coluna timedelta para segundos: {col}")
            X_features[col] = X_features[col].dt.total_seconds().astype('float64')
    
    # Preencher valores nulos com 0 para estabilidade
    X_features = X_features.fillna(0)
    
    # Verificar se todas as colunas são numéricas agora
    non_numeric_cols = []
    for col in X_features.columns:
        if not pd.api.types.is_numeric_dtype(X_features[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        logger.warning(f"Forçando conversão numérica para colunas: {non_numeric_cols}")
        for col in non_numeric_cols:
            X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0)
    
    logger.info(f"Convertidas {len(categorical_cols)} colunas categóricas para numéricas")
    
    # TRATAMENTO ROBUSTO DE OUTLIERS EXTREMOS
    clipping_stats = {}
    
    if scaler_type == 'standard_clipped':
        logger.info(f"🚨 APLICANDO CLIPPING DE OUTLIERS ({clip_percentile}% dos dados preservados)")
        
        outliers_found = 0
        features_with_outliers = []
        
        for col in X_features.columns:
            data = X_features[col].dropna()
            if len(data) > 0:
                # Calcular thresholds de clipping baseados em percentis
                lower_threshold = np.percentile(data, (100 - clip_percentile) / 2)
                upper_threshold = np.percentile(data, clip_percentile + (100 - clip_percentile) / 2)
                
                # Contar outliers antes do clipping
                outliers_mask = (data < lower_threshold) | (data > upper_threshold)
                outliers_count = outliers_mask.sum()
                outliers_pct = (outliers_count / len(data)) * 100
                
                if outliers_count > 0:
                    outliers_found += outliers_count
                    features_with_outliers.append(col)
                    
                    # Valores extremos antes do clipping
                    extreme_values = data[outliers_mask].values
                    extreme_sample = extreme_values[:5] if len(extreme_values) > 5 else extreme_values
                    
                    clipping_stats[col] = {
                        'lower_threshold': float(lower_threshold),
                        'upper_threshold': float(upper_threshold),
                        'outliers_count': int(outliers_count),
                        'outliers_percentage': float(outliers_pct),
                        'original_range': [float(data.min()), float(data.max())],
                        'extreme_values_sample': [float(x) for x in extreme_sample]
                    }
                    
                    # Aplicar clipping
                    X_features[col] = np.clip(X_features[col], lower_threshold, upper_threshold)
                    
                    # Log para features com muitos outliers
                    if outliers_pct > 1.0:
                        logger.info(f"   🔧 {col}: {outliers_count} outliers ({outliers_pct:.2f}%) "
                                  f"[{data.min():.3f}, {data.max():.3f}] → "
                                  f"[{lower_threshold:.3f}, {upper_threshold:.3f}]")
        
        logger.info(f"✓ Clipping concluído: {outliers_found:,} outliers em {len(features_with_outliers)} features")
        
        # Usar StandardScaler após clipping (melhor para GNN)
        scaler = StandardScaler()
        logger.info("   Usando StandardScaler após clipping (otimizado para GNN)")
        
    elif scaler_type == 'robust_clipped':
        logger.info(f"🚨 APLICANDO CLIPPING DE OUTLIERS ({clip_percentile}% dos dados preservados)")
        
        outliers_found = 0
        features_with_outliers = []
        
        for col in X_features.columns:
            data = X_features[col].dropna()
            if len(data) > 0:
                # Calcular thresholds de clipping baseados em percentis
                lower_threshold = np.percentile(data, (100 - clip_percentile) / 2)
                upper_threshold = np.percentile(data, clip_percentile + (100 - clip_percentile) / 2)
                
                # Contar outliers antes do clipping
                outliers_mask = (data < lower_threshold) | (data > upper_threshold)
                outliers_count = outliers_mask.sum()
                outliers_pct = (outliers_count / len(data)) * 100
                
                if outliers_count > 0:
                    outliers_found += outliers_count
                    features_with_outliers.append(col)
                    
                    # Valores extremos antes do clipping
                    extreme_values = data[outliers_mask].values
                    extreme_sample = extreme_values[:5] if len(extreme_values) > 5 else extreme_values
                    
                    clipping_stats[col] = {
                        'lower_threshold': float(lower_threshold),
                        'upper_threshold': float(upper_threshold),
                        'outliers_count': int(outliers_count),
                        'outliers_percentage': float(outliers_pct),
                        'original_range': [float(data.min()), float(data.max())],
                        'extreme_values_sample': [float(x) for x in extreme_sample]
                    }
                    
                    # Aplicar clipping
                    X_features[col] = np.clip(X_features[col], lower_threshold, upper_threshold)
                    
                    # Log para features com muitos outliers
                    if outliers_pct > 1.0:
                        logger.info(f"   🔧 {col}: {outliers_count} outliers ({outliers_pct:.2f}%) "
                                  f"[{data.min():.3f}, {data.max():.3f}] → "
                                  f"[{lower_threshold:.3f}, {upper_threshold:.3f}]")
        
        logger.info(f"✓ Clipping concluído: {outliers_found:,} outliers em {len(features_with_outliers)} features")
        
        # Usar RobustScaler após clipping
        scaler = RobustScaler()
        logger.info("   Usando RobustScaler após clipping (otimizado para GNN)")
        
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        logger.info("Usando StandardScaler (μ=0, σ=1) - Recomendado para GNN")
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        logger.info("Usando MinMaxScaler [0,1]")
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        logger.info("Usando RobustScaler (robusto a outliers)")
    else:
        raise ValueError(f"Scaler type não suportado: {scaler_type}")
    
    # Aplicar normalização
    logger.info("Aplicando normalização...")
    X_normalized = scaler.fit_transform(X_features)
    
    # Atualizar DataFrame com features normalizadas
    for i, col in enumerate(feature_cols):
        df_normalized[col] = X_normalized[:, i].astype('float32')
    
    # Manter features excluídas inalteradas
    for col in exclude_features:
        if col in df.columns:
            df_normalized[col] = df[col]
    
    # Metadados da normalização EXPANDIDOS
    normalization_metadata = {
        'scaler_type': scaler_type,
        'clip_percentile': clip_percentile if scaler_type == 'robust_clipped' else None,
        'features_normalized': feature_cols,
        'features_excluded': exclude_features,
        'original_shape': df.shape,
        'normalized_shape': df_normalized.shape,
        'total_features_normalized': len(feature_cols),
        'normalization_timestamp': datetime.now().isoformat(),
        'clipping_statistics': clipping_stats,
        'outlier_summary': {
            'total_outliers_clipped': sum(stats.get('outliers_count', 0) for stats in clipping_stats.values()),
            'features_with_outliers': len(clipping_stats),
            'most_problematic_features': sorted(
                clipping_stats.items(), 
                key=lambda x: x[1]['outliers_percentage'], 
                reverse=True
            )[:5]  # Top 5 features com mais outliers
        } if scaler_type == 'robust_clipped' else {},
        'feature_statistics': {
            'pre_normalization': {
                col: {
                    'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0,
                    'std': float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0,
                    'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0,
                    'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                } for col in feature_cols[:10]  # Primeiro 10 para economizar espaço
            },
            'post_normalization': {
                col: {
                    'mean': float(df_normalized[col].mean()),
                    'std': float(df_normalized[col].std()),
                    'min': float(df_normalized[col].min()),
                    'max': float(df_normalized[col].max())
                } for col in feature_cols[:10]
            }
        },
        'distribution_analysis_summary': {
            category: {
                'feature_count': stats['count'],
                'skewed_features': len(stats['skewness_issues']),
                'outlier_features': len(stats['outlier_presence'])
            } for category, stats in distribution_analysis.items()
        }
    }
    
    # Estatísticas pós-normalização
    logger.info("Verificando resultados da normalização...")
    for i, col in enumerate(feature_cols[:5]):  # Mostrar primeiras 5
        if pd.api.types.is_numeric_dtype(df[col]):
            original_stats = f"Original: μ={df[col].mean():.3f}, σ={df[col].std():.3f}"
        else:
            original_stats = f"Original: categórica convertida"
        normalized_stats = f"Normalizado: μ={df_normalized[col].mean():.3f}, σ={df_normalized[col].std():.3f}"
        logger.info(f"  {col}: {original_stats} → {normalized_stats}")
    
    # Verificação de integridade
    assert df_normalized.shape == df.shape, "Shape alterado durante normalização"
    assert not df_normalized[feature_cols].isnull().any().any(), "NaN introduzidos durante normalização"
    
    logger.info(f"Normalização concluída com sucesso!")
    logger.info(f"Features normalizadas: {len(feature_cols)}")
    logger.info(f"Shape final: {df_normalized.shape}")
    
    return df_normalized, scaler, normalization_metadata


def save_normalized_data(df_normalized, scaler, metadata, output_dir, logger):
    """Salva dados normalizados e metadados."""
    logger.info("Salvando dados normalizados...")
    
    # Otimizar memória antes de salvar
    df_normalized = optimize_dataframe_memory(df_normalized)
    
    # Salvar dataset normalizado
    output_file = os.path.join(output_dir, "ieee-cis-features-normalized.parquet")
    df_normalized.to_parquet(output_file, index=False, compression='snappy')
    logger.info(f"Dataset normalizado salvo: {output_file}")
    
    # Salvar scaler para uso futuro
    scaler_file = os.path.join(output_dir, "feature_scaler.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler salvo: {scaler_file}")
    
    # Salvar metadados
    metadata_file = os.path.join(output_dir, "normalization_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadados salvos: {metadata_file}")
    
    return output_file, scaler_file, metadata_file


def main():
    """Função principal para executar normalização de features."""
    # Configurar caminhos absolutos baseados no diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    
    # Setup logging
    log_dir = os.path.join(project_root, "data", "statis", "ieee-cis")
    logger, log_path = setup_logging(log_dir)
    
    logger.info("=== Iniciando Normalização de Features para IEEE-CIS (GNN) ===")
    logger.info(f"Diretório do script: {script_dir}")
    logger.info(f"Raiz do projeto: {project_root}")
    logger.info(f"Diretório de logs: {log_dir}")
    
    try:
        # Configurar caminhos
        data_path = os.path.join(project_root, "data", "parquet", "ieee-cis", "ieee-cis-features.parquet")
        output_dir = os.path.join(project_root, "data", "parquet", "ieee-cis")
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
        df = load_ieee_cis_data(data_path)
        
        # Informações do dataset
        logger.info(f"Dataset carregado: {df.shape}")
        logger.info(f"Taxa de fraude: {df['isFraud'].mean():.4f}")
        
        # Log detalhado das categorias de features
        feature_cols = [col for col in df.columns if col not in ['TransactionDT', 'isFraud']]
        logger.info(f"Total de features a processar: {len(feature_cols)}")
        logger.info("Tipos de features encontradas:")
        logger.info(f"  - Original PCA (V2-V59): {len([col for col in feature_cols if col.startswith('V') and col[1:].isdigit()])}")
        logger.info(f"  - Card features: {len([col for col in feature_cols if col.startswith('card')])}")
        logger.info(f"  - Address features: {len([col for col in feature_cols if col.startswith('addr')])}")
        logger.info(f"  - C features: {len([col for col in feature_cols if col.startswith('C') and col[1:].isdigit()])}")
        logger.info(f"  - D features: {len([col for col in feature_cols if col.startswith('D') and col[1:].isdigit()])}")
        logger.info(f"  - M features: {len([col for col in feature_cols if col.startswith('M') and col[1:].isdigit()])}")
        logger.info(f"  - Email features: {len([col for col in feature_cols if 'email' in col.lower()])}")
        logger.info(f"  - Temporal features: {len([col for col in feature_cols if any(t in col.lower() for t in ['trans_', 'hour', 'temporal', 'time', 'night'])])}")
        logger.info(f"  - Amount features: {len([col for col in feature_cols if 'amount' in col.lower()])}")
        
        # Análise de distribuições
        logger.info("--- Analisando Distribuições das Features ---")
        distribution_analysis, feature_categories = analyze_feature_distributions(df, logger)
        
        # Normalização para GNN
        logger.info("--- Executando Normalização Completa para GNN ---")
        df_normalized, scaler, metadata = normalize_features_for_gnn(
            df, distribution_analysis, feature_categories, logger, scaler_type='standard_clipped', clip_percentile=98.5
        )
        
        # Salvar resultados
        logger.info("--- Salvando Resultados ---")
        output_file, scaler_file, metadata_file = save_normalized_data(
            df_normalized, scaler, metadata, output_dir, logger
        )
        
        # VALIDAÇÃO FINAL DA NORMALIZAÇÃO
        logger.info("--- Validação Final da Normalização ---")
        feature_cols = [col for col in df_normalized.columns if col not in ['TransactionDT', 'isFraud', 'card1']]
        all_values = []
        
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df_normalized[col]):
                all_values.extend(df_normalized[col].dropna().values)
        
        if all_values:
            all_values = np.array(all_values)
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            global_min = np.min(all_values)
            global_max = np.max(all_values)
            
            # Análise de normalização
            values_in_normal_range = np.sum((all_values >= -3) & (all_values <= 3))
            pct_in_normal_range = (values_in_normal_range / len(all_values)) * 100
            
            logger.info(f"✓ Validação da normalização:")
            logger.info(f"   - Range global: [{global_min:.3f}, {global_max:.3f}]")
            logger.info(f"   - Média global: {global_mean:.6f}")
            logger.info(f"   - Desvio padrão: {global_std:.6f}")
            logger.info(f"   - Valores em [-3, +3]: {values_in_normal_range:,}/{len(all_values):,} ({pct_in_normal_range:.1f}%)")
            
            # Verificar se normalização foi bem-sucedida
            mean_ok = abs(global_mean) < 0.01
            std_ok = abs(global_std - 1.0) < 0.1
            range_ok = global_max < 6 and global_min > -6  # Mais rigoroso ainda
            outliers_ok = pct_in_normal_range > 99.7  # Pelo menos 99.7% em [-3,+3]
            
            if mean_ok and std_ok and range_ok and outliers_ok:
                logger.info("🎉 NORMALIZAÇÃO BEM-SUCEDIDA!")
            else:
                logger.warning("⚠️ Normalização pode ter problemas:")
                if not mean_ok:
                    logger.warning(f"   - Média não centrada: {global_mean:.6f}")
                if not std_ok:
                    logger.warning(f"   - Desvio padrão não unitário: {global_std:.6f}")
                if not range_ok:
                    logger.warning(f"   - Range muito amplo: [{global_min:.3f}, {global_max:.3f}]")
                if not outliers_ok:
                    logger.warning(f"   - Muitos outliers: apenas {pct_in_normal_range:.1f}% em [-3,+3]")
        
        logger.info("=== Normalização de Features Concluída ===")
        
        # Resumo final
        logger.info("Resumo da Normalização:")
        logger.info(f"  - Dataset original: {df.shape}")
        logger.info(f"  - Dataset normalizado: {df_normalized.shape}")
        logger.info(f"  - Features normalizadas: {metadata['total_features_normalized']}")
        logger.info(f"  - Features excluídas: {metadata['features_excluded']}")
        logger.info(f"  - Scaler utilizado: {metadata['scaler_type']}")
        logger.info(f"  - Arquivo de saída: {output_file}")
        logger.info(f"  - Scaler salvo em: {scaler_file}")
        logger.info(f"  - Metadados em: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Erro durante normalização de features: {e}")
        raise


if __name__ == "__main__":
    main()
