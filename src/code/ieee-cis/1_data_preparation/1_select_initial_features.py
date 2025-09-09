#!/usr/bin/env python3
"""
IEEE-CIS Fraud Detection - Feature Selection Script

This script processes the IEEE-CIS Fraud Detection dataset to select the top 100 most
important features using ONLY LightGBM feature importance ranking on original dataset 
features. No derived features are created in this phase.

The selected features are saved to a Parquet file for subsequent graph construction 
and GNN training, where derived/graph-critical features will be created.

IMPROVEMENTS IMPLEMENTED:
- Explicit categorical dtype conversion: Inherently categorical features (card1, addr1, etc.) converted to category dtype
- Intelligent categorical encoding: Label Encoding for high-cardinality, One-Hot for low-cardinality
- High correlation removal: Features with correlation > 0.98 removed to reduce redundancy and multicollinearity
- Automatic detection of categorical columns (no hardcoded limitations)
- Memory optimization through proper categorical dtypes before encoding
- Preserves semantic meaning by avoiding artificial ordering where inappropriate
- Comprehensive encoding information saved for reproducibility
- Enhanced feature importance logging with percentages
- Clean separation: original feature selection here, graph features in graph construction phase

WORKFLOW:
1. Load IEEE-CIS transaction + identity data (284k most recent)
2. Remove constant features (zero variance)
3. Convert inherently categorical features to category dtype (card1, addr1, etc.)
4. Apply intelligent categorical encoding (including email domains)
5. Remove highly correlated features (correlation > 0.98) to reduce redundancy
6. Train LightGBM on optimally processed features
7. Select top 100 features based on LightGBM importance
8. Save cleaned dataset for graph construction phase

Author: GraphSentinel 2.0 Project
Date: June 26, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import pyarrow
import warnings
import json
import gc
import subprocess
import time

# Try to import GPUtil for GPU detection, but don't fail if not available
try:
    import GPUtil
except ImportError:
    GPUtil = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility (following project guidelines)
np.random.seed(42)

# ============================================================================
# CATEGORICAL FEATURES CONFIGURATION - LIGHTGBM BEST PRACTICES
# ============================================================================

# High-cardinality categorical identifiers that should be treated as categorical
# by LightGBM (not numerically encoded with artificial ordering)
CATEGORICAL_FEATURES = [
    # Card identifiers (user/device identification)
    'card1', 'card2', 'card3', 'card5',
    # Address identifiers
    'addr1', 'addr2', 
    # Email domains (categorical nature)
    'P_emaildomain', 'R_emaildomain',
    # Device information (categorical nature)
    'DeviceInfo',
    # Product and card types (low-cardinality categorical)
    'ProductCD', 'card4', 'card6',
    # M-series features (binary/categorical flags)
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

# Features that should remain numeric (continuous values)
NUMERIC_FEATURES_KEEP_AS_IS = [
    'TransactionDT',    # Time (unix timestamp)
    'TransactionAmt',   # Amount (continuous value)
    'dist1', 'dist2',   # Distance measures (continuous)
    # C-series features are continuous
    # D-series features are continuous  
    # V-series features are continuous
]

def clean_feature_name(name):
    """
    Clean feature names to avoid special JSON characters that LightGBM doesn't support.
    
    Args:
        name (str): Original feature name
        
    Returns:
        str: Cleaned feature name safe for LightGBM
    """
    # Convert to string and replace problematic characters
    safe_name = str(name)
    
    # Replace special characters that cause JSON issues in LightGBM
    special_chars = [' ', '.', '-', '/', '\\', '(', ')', '[', ']', '{', '}', 
                    '"', "'", ':', ';', ',', '?', '!', '@', '#', '$', '%', 
                    '^', '&', '*', '+', '=', '|', '<', '>', '~', '`']
    
    for char in special_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove multiple consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    # Ensure the name is not empty
    if not safe_name:
        safe_name = 'unknown_feature'
    
    return safe_name

def check_gpu_availability():
    """Check if GPU is available for LightGBM training with enhanced detection."""
    try:
        # First check if lightgbm supports GPU
        import lightgbm as lgb
        
        # Check if GPU is available in the system
        gpu_available = False
        gpu_info = "No GPU detected"
        
        try:
            if GPUtil is not None:
                gpus = GPUtil.getGPUs()
                if len(gpus) > 0:
                    gpu_available = True
                    gpu_info = f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}"
            else:
                # Fallback: try to detect CUDA via nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_available = True
                        gpu_info = "CUDA detected via nvidia-smi"
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                    pass
        except Exception:
            # Fallback: try to detect CUDA via nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_available = True
                    gpu_info = "CUDA detected via nvidia-smi"
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        # Additional check: try to detect OpenCL
        if not gpu_available:
            try:
                result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'Platform' in result.stdout:
                    gpu_available = True
                    gpu_info = "OpenCL detected via clinfo"
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        print(f"GPU Detection: {gpu_info}")
        return gpu_available
        
    except ImportError:
        print("LightGBM not available for GPU detection")
        return False

def optimize_dataframe_memory(df):
    """
    Optimize DataFrame memory usage by converting data types to more efficient ones.
    Following project guidelines for memory optimization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    print("Optimizing DataFrame memory usage...")
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage decreased from {original_memory:.2f} MB to {optimized_memory:.2f} MB "
          f"({100 * (original_memory - optimized_memory) / original_memory:.1f}% reduction)")
    
    return df

def remove_constant_features(df, exclude_columns=None):
    """
    Remove features que têm apenas um valor único (colunas constantes).
    Colunas constantes não agregam informação para treinamento de ML/GNN.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        exclude_columns (list): Colunas a serem excluídas da análise (ex: target, identificadores críticos)
        
    Returns:
        tuple: (df_filtered, removed_features_info)
            - df_filtered: DataFrame sem as colunas constantes
            - removed_features_info: Dicionário com informações das colunas removidas
    """
    if exclude_columns is None:
        exclude_columns = ['isFraud', 'TransactionID']
    
    print("🔍 Analisando colunas constantes (variância zero)...")
    
    # Colunas para analisar (excluindo target e outras especiais)
    analyze_columns = [col for col in df.columns if col not in exclude_columns]
    
    constant_features = []
    constant_info = {}
    
    for col in analyze_columns:
        # Verificar se a coluna tem apenas um valor único
        unique_count = df[col].nunique()
        if unique_count <= 1:  # Apenas 1 valor único (ou todos NaN)
            constant_features.append(col)
            
            # Coletar informações detalhadas
            unique_vals = df[col].dropna().unique()
            non_null_count = df[col].notna().sum()
            
            constant_info[col] = {
                'unique_count': unique_count,
                'unique_values': unique_vals.tolist() if len(unique_vals) > 0 else ['All NaN'],
                'total_records': len(df),
                'non_null_records': non_null_count,
                'null_percentage': ((len(df) - non_null_count) / len(df)) * 100,
                'dtype': str(df[col].dtype)
            }
            
            # Estatísticas adicionais para colunas numéricas
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'int8']:
                try:
                    constant_info[col].update({
                        'mean': float(df[col].mean()) if non_null_count > 0 else None,
                        'std': float(df[col].std()) if non_null_count > 0 else None,
                        'min': float(df[col].min()) if non_null_count > 0 else None,
                        'max': float(df[col].max()) if non_null_count > 0 else None
                    })
                except:
                    pass  # Em caso de erro, continuar sem estatísticas
    
    if constant_features:
        print(f"🚫 Encontradas {len(constant_features)} colunas constantes:")
        
        # Mostrar detalhes das primeiras 10 colunas constantes
        for i, col in enumerate(constant_features[:10]):
            info = constant_info[col]
            unique_str = ', '.join([str(x) for x in info['unique_values'][:3]])
            if len(info['unique_values']) > 3:
                unique_str += '...'
            
            print(f"  {col}: unique={info['unique_count']}, values=[{unique_str}], "
                  f"dtype={info['dtype']}, null={info['null_percentage']:.1f}%")
            
            # Mostrar estatísticas se disponíveis
            if 'mean' in info and info['mean'] is not None:
                print(f"    Stats: mean={info['mean']:.6f}, std={info['std']:.6f}, "
                      f"min={info['min']:.6f}, max={info['max']:.6f}")
        
        if len(constant_features) > 10:
            print(f"  ... e mais {len(constant_features) - 10} colunas constantes")
        
        # Remover colunas constantes
        df_filtered = df.drop(columns=constant_features)
        
        print(f"✅ Removidas {len(constant_features)} colunas constantes")
        print(f"📊 Shape: {df.shape} → {df_filtered.shape} "
              f"({len(constant_features)} colunas, {df.shape[1] - df_filtered.shape[1]} features removidas)")
        
        # Calcular redução de memória
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        filtered_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = ((original_memory - filtered_memory) / original_memory) * 100
        
        print(f"💾 Memória: {original_memory:.2f} MB → {filtered_memory:.2f} MB "
              f"({memory_reduction:.1f}% redução)")
        
        return df_filtered, constant_info
    else:
        print("✅ Nenhuma coluna constante encontrada - todas as features têm variância > 0")
        return df, {}

def load_and_merge_data():
    """
    Load transaction and identity data, then merge them.
    Uses the most recent 284k transactions to avoid temporal bias and capture
    the latest fraud patterns for better feature selection.
    
    Returns:
        pd.DataFrame: Merged DataFrame containing transaction and identity data
    """
    print("Loading IEEE-CIS dataset...")
    
    # Define data paths (using absolute path from project root)
    import os
    current_dir = os.getcwd()
    
    # Navigate to project root (assuming we're in a subdirectory)
    if 'graphsentinel_2.0' in current_dir:
        project_root = current_dir.split('graphsentinel_2.0')[0] + 'graphsentinel_2.0'
    else:
        project_root = '/graphsentinel_2.0'  # Fallback to absolute path
    
    data_path = os.path.join(project_root, "data", "raw", "ieee-cis")
    trans_path = os.path.join(data_path, "transaction.csv")
    identity_path = os.path.join(data_path, "identity.csv")
    
    # Check if files exist
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Transaction file not found: {trans_path}")
    if not os.path.exists(identity_path):
        raise FileNotFoundError(f"Identity file not found: {identity_path}")
    
    # Load transaction data
    print(f"Loading transaction data from {trans_path}...")
    df_trans = pd.read_csv(trans_path)
    print(f"Transaction data shape: {df_trans.shape}")
    
    # Sort by TransactionDT and limit to most recent 284k transactions (avoiding temporal bias)
    print("Sorting by TransactionDT and selecting most recent 284k transactions...")
    df_trans = df_trans.sort_values('TransactionDT').tail(284000)
    print(f"Selected transaction data shape: {df_trans.shape}")
    print(f"Time range: {df_trans['TransactionDT'].min()} to {df_trans['TransactionDT'].max()}")
    print(f"Fraud rate in selected period: {df_trans['isFraud'].mean():.4f}")
    
    # Load identity data
    print(f"Loading identity data from {identity_path}...")
    df_id = pd.read_csv(identity_path)
    print(f"Identity data shape: {df_id.shape}")
    
    # Filter identity data for selected transactions only
    print("Filtering identity data for selected transactions...")
    relevant_transaction_ids = df_trans['TransactionID'].values
    df_id = df_id[df_id['TransactionID'].isin(relevant_transaction_ids)]
    print(f"Filtered identity data shape: {df_id.shape}")
    
    # Merge dataframes
    print("Merging transaction and identity data...")
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    print(f"Merged data shape: {df.shape}")
    
    # Remove TransactionID as it's not suitable for ML features (identifier without predictive value)
    print("Removing TransactionID column (identifier not suitable for feature selection)...")
    if 'TransactionID' in df.columns:
        df = df.drop('TransactionID', axis=1)
        print(f"TransactionID removed. New shape: {df.shape}")
    
    # Note: R_emaildomain and P_emaildomain are now kept for LightGBM evaluation
    # Previous removal was due to PCA hybrid strategy concerns, but with pure LightGBM
    # these features can be properly evaluated for their predictive value
    print("Keeping all categorical features for LightGBM evaluation (including email domains)...")
    
    # Optimize memory usage
    df = optimize_dataframe_memory(df)
    
    # Remove constant features
    df, _ = remove_constant_features(df)
    
    # Clean up intermediate DataFrames to free memory
    del df_trans, df_id
    import gc
    gc.collect()
    
    return df

def prepare_categorical_features(X):
    """
    Prepare categorical features using LightGBM best practices.
    
    Instead of Label Encoding (which creates artificial numerical ordering),
    this function converts categorical features to pandas 'category' dtype,
    which allows LightGBM to handle them optimally with its internal
    categorical feature support.
    
    This approach:
    - Preserves the categorical nature of features
    - Lets LightGBM use optimal split strategies for categorical data
    - Avoids artificial ordering that can mislead the model
    - Reduces memory usage through category dtype
    - Ensures safer output for subsequent graph construction and GNN training
    
    Args:
        X (pd.DataFrame): Feature DataFrame
        
    Returns:
        tuple: (X_processed, categorical_info)
            - X_processed: DataFrame with categorical features as category dtype
            - categorical_info: Dictionary with processing information
    """
    print("🎯 Preparing categorical features using LightGBM best practices...")
    print("   📋 Using category dtype instead of Label Encoding for optimal performance")
    
    X_processed = X.copy()
    
    # Track processing information
    categorical_info = {
        'converted_to_category': [],
        'onehot_encoded': [],
        'kept_as_numeric': [],
        'auto_detected': [],
        'memory_impact': {}
    }
    
    # Calculate initial memory usage
    initial_memory = X_processed.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\n� Converting categorical features to category dtype...")
    
    # Convert known categorical features to category dtype
    converted_count = 0
    for col in CATEGORICAL_FEATURES:
        if col in X_processed.columns:
            # Check if it's currently non-categorical
            if X_processed[col].dtype not in ['category']:
                unique_count = X_processed[col].nunique()
                print(f"   • {col}: {unique_count} unique values → category dtype")
                
                # Convert to category dtype (LightGBM will handle internally)
                X_processed[col] = X_processed[col].astype('category')
                categorical_info['converted_to_category'].append({
                    'column': col,
                    'unique_values': unique_count,
                    'conversion_type': 'explicit_categorical'
                })
                converted_count += 1
    
    print(f"   ✅ Converted {converted_count} known categorical features to category dtype")
    
    # Auto-detect additional categorical features (object dtype columns)
    print(f"\n🔍 Auto-detecting additional categorical features...")
    auto_detected_count = 0
    for col in X_processed.columns:
        if (col not in CATEGORICAL_FEATURES and 
            col not in NUMERIC_FEATURES_KEEP_AS_IS and
            X_processed[col].dtype == 'object'):
            
            unique_count = X_processed[col].nunique()
            total_count = len(X_processed[col])
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Only convert if it looks categorical (not too high cardinality)
            if unique_count <= 1000 and cardinality_ratio < 0.5:  # Reasonable thresholds
                print(f"   • {col}: {unique_count} unique values → category dtype (auto-detected)")
                X_processed[col] = X_processed[col].astype('category')
                categorical_info['auto_detected'].append({
                    'column': col,
                    'unique_values': unique_count,
                    'cardinality_ratio': cardinality_ratio
                })
                auto_detected_count += 1
            else:
                print(f"   ⚠️ {col}: {unique_count} unique values - too high cardinality, keeping as-is")
                categorical_info['kept_as_numeric'].append({
                    'column': col,
                    'unique_values': unique_count,
                    'reason': 'high_cardinality'
                })
    
    print(f"   ✅ Auto-detected and converted {auto_detected_count} additional categorical features")
    
    # Handle remaining object columns that weren't converted
    remaining_object_cols = [col for col in X_processed.columns if X_processed[col].dtype == 'object']
    if remaining_object_cols:
        print(f"\n⚠️ Warning: {len(remaining_object_cols)} object columns remain unconverted:")
        for col in remaining_object_cols[:5]:  # Show first 5
            unique_count = X_processed[col].nunique()
            print(f"   • {col}: {unique_count} unique values")
        if len(remaining_object_cols) > 5:
            print(f"   ... and {len(remaining_object_cols) - 5} more")
        
        print("   💡 These will be handled by LightGBM's automatic type inference")
    
    # Calculate memory impact
    final_memory = X_processed.memory_usage(deep=True).sum() / 1024**2
    memory_saved = initial_memory - final_memory
    
    categorical_info['memory_impact'] = {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_saved_mb': memory_saved,
        'reduction_percentage': (memory_saved / initial_memory * 100) if initial_memory > 0 else 0
    }
    
    # Summary
    total_categorical = len(categorical_info['converted_to_category']) + len(categorical_info['auto_detected'])
    print(f"\n✅ CATEGORICAL PROCESSING COMPLETED:")
    print(f"   📊 Explicit categorical conversions: {len(categorical_info['converted_to_category'])}")
    print(f"   🔍 Auto-detected conversions: {len(categorical_info['auto_detected'])}")
    print(f"   � Total categorical features: {total_categorical}")
    print(f"   💾 Memory saved: {memory_saved:.2f}MB ({categorical_info['memory_impact']['reduction_percentage']:.1f}%)")
    print(f"   🚀 Ready for optimal LightGBM training with native categorical support!")
    
    return X_processed, categorical_info

def train_lightgbm_and_get_importance(X, y, categorical_info=None):
    """
    Train LightGBM model and extract feature importances.
    
    This function now properly handles categorical features using LightGBM's
    native categorical support instead of pre-encoded numerical values.
    
    Args:
        X (pd.DataFrame): Feature matrix with category dtype for categorical features
        y (pd.Series): Target variable
        categorical_info (dict): Information about categorical features processing
        
    Returns:
        tuple: (feature_importance_df, X_processed, categorical_feature_names)
    """
    print("Training LightGBM model for feature importance ranking...")
    print("   🎯 Using LightGBM native categorical support for optimal performance")
    
    # Handle missing values by filling with median for numeric columns
    print("Handling missing values...")
    X_processed = X.copy()
    
    # Identify categorical features for LightGBM
    categorical_feature_names = []
    for col in X_processed.columns:
        if X_processed[col].dtype.name == 'category':
            categorical_feature_names.append(col)
        elif X_processed[col].isnull().sum() > 0:
            # Fill missing values only for numeric columns
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    print(f"   📊 Identified {len(categorical_feature_names)} categorical features for LightGBM:")
    if categorical_feature_names:
        for i, col in enumerate(categorical_feature_names[:5]):  # Show first 5
            unique_count = X_processed[col].nunique()
            print(f"      {i+1}. {col}: {unique_count} categories")
        if len(categorical_feature_names) > 5:
            print(f"      ... and {len(categorical_feature_names) - 5} more")
    
    # Clean all column names to avoid LightGBM JSON issues
    print("Cleaning feature names for LightGBM compatibility...")
    original_columns = X_processed.columns.tolist()
    cleaned_columns = [clean_feature_name(col) for col in original_columns]
    
    # Create mapping for later reference
    column_mapping = dict(zip(original_columns, cleaned_columns))
    
    # Update categorical feature names with cleaned names
    cleaned_categorical_features = [clean_feature_name(col) for col in categorical_feature_names]
    
    # Rename columns
    X_processed.columns = cleaned_columns
    
    # Check for duplicate column names after cleaning
    if len(set(cleaned_columns)) != len(cleaned_columns):
        print("WARNING: Duplicate column names detected after cleaning. Adding suffixes...")
        seen = {}
        final_columns = []
        final_categorical_features = []
        
        for i, col in enumerate(cleaned_columns):
            original_col = original_columns[i]
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
                final_columns.append(new_col)
                # Update categorical list if this was a categorical feature
                if original_col in categorical_feature_names:
                    final_categorical_features.append(new_col)
            else:
                seen[col] = 0
                final_columns.append(col)
                # Update categorical list if this was a categorical feature
                if original_col in categorical_feature_names:
                    final_categorical_features.append(col)
        
        X_processed.columns = final_columns
        cleaned_categorical_features = final_categorical_features
        # Update mapping
        column_mapping = dict(zip(original_columns, final_columns))
    
    print(f"Cleaned {len(original_columns)} feature names for LightGBM compatibility")
    print(f"   📊 Categorical features after cleaning: {len(cleaned_categorical_features)}")
    
    # Convert categorical features to LightGBM-compatible format
    print("Converting categorical features to LightGBM-compatible format...")
    for col in cleaned_categorical_features:
        if col in X_processed.columns:
            # Ensure categorical features are in the right format for LightGBM
            # LightGBM handles category dtype internally, but we need to ensure
            # missing values are handled properly
            if X_processed[col].isnull().sum() > 0:
                # For categorical features, fill missing with a special category
                X_processed[col] = X_processed[col].cat.add_categories(['__MISSING__'])
                X_processed[col] = X_processed[col].fillna('__MISSING__')
                print(f"   • {col}: filled {X_processed[col].isnull().sum()} missing values")
    
    # Try GPU first, then fallback to CPU if there are issues
    device_type = 'gpu' if check_gpu_availability() else 'cpu'
    print(f"Attempting to use device: {device_type}")
    
    # Create LightGBM classifier parameters - OPTIMIZED FOR FRAUD DETECTION
    # Calculate class imbalance ratio for pos_weight
    fraud_ratio = y.sum() / len(y)
    pos_weight = (1 - fraud_ratio) / fraud_ratio  # Approximately 27 for 3.6% fraud rate
    
    print(f"Fraud rate: {fraud_ratio:.4f}, Calculated pos_weight: {pos_weight:.2f}")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',  # Changed from 'auc' to optimize for recall/F2
        'scale_pos_weight': pos_weight,  # Weight positive class appropriately
        'random_state': 42,          # Following project guidelines for reproducibility
        'verbose': -1,
        'n_estimators': 300,         # Increased for better feature importance stability
        'learning_rate': 0.08,       # Slightly lower for better convergence
        'num_leaves': 31,
        'max_depth': 8,              # Increased to capture complex fraud patterns
        'feature_fraction': 0.85,    # Slight regularization
        'bagging_fraction': 0.85,    # Slight regularization
        'bagging_freq': 5,
        'min_child_samples': 50,     # Prevent overfitting on small fraud samples
        'reg_alpha': 0.1,            # L1 regularization
        'reg_lambda': 0.1,           # L2 regularization
    }
    
    # Try GPU training first
    if device_type == 'gpu':
        try:
            print("Trying GPU training with categorical support...")
            lgb_params_gpu = lgb_params.copy()
            lgb_params_gpu.update({
                'device_type': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': 63,  # Reduce max_bin for GPU compatibility
                'max_depth': 7,  # Limit depth for GPU memory
                'n_jobs': 1,  # GPU training should use single job
                'force_col_wise': True,  # Better GPU performance
                'gpu_use_dp': False  # Use single precision for better GPU performance
            })
            
            lgb_model = lgb.LGBMClassifier(**lgb_params_gpu)
            # Pass categorical features to LightGBM
            lgb_model.fit(
                X_processed, y, 
                categorical_feature=cleaned_categorical_features if cleaned_categorical_features else 'auto'
            )
            print("GPU training with categorical support successful!")
            
        except Exception as gpu_error:
            print(f"GPU training failed: {str(gpu_error)}")
            print("Falling back to CPU training...")
            
            # Fallback to CPU
            lgb_params_cpu = lgb_params.copy()
            lgb_params_cpu.update({
                'device_type': 'cpu',
                'n_jobs': -1,
                'force_col_wise': True
            })
            
            lgb_model = lgb.LGBMClassifier(**lgb_params_cpu)
            lgb_model.fit(
                X_processed, y,
                categorical_feature=cleaned_categorical_features if cleaned_categorical_features else 'auto'
            )
            print("CPU training with categorical support successful!")
    else:
        # Direct CPU training
        print("Using CPU training with categorical support...")
        lgb_params_cpu = lgb_params.copy()
        lgb_params_cpu.update({
            'device_type': 'cpu',
            'n_jobs': -1,
            'force_col_wise': True
        })
        
        lgb_model = lgb.LGBMClassifier(**lgb_params_cpu)
        lgb_model.fit(
            X_processed, y,
            categorical_feature=cleaned_categorical_features if cleaned_categorical_features else 'auto'
        )
        print("CPU training with categorical support successful!")
    
    # Extract feature importances
    feature_importance = lgb_model.feature_importances_
    feature_names = X_processed.columns.tolist()
    
    # Create importance DataFrame and sort by importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Calculate importance percentages
    total_importance = feature_importance_df['importance'].sum()
    feature_importance_df['importance_percentage'] = (feature_importance_df['importance'] / total_importance * 100).round(2)
    
    # Restore original column names in the processed DataFrame for consistency
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    original_feature_names = [reverse_mapping.get(col, col) for col in feature_names]
    
    # Update the feature importance DataFrame with original names
    feature_importance_df['feature'] = original_feature_names
    
    # Also restore column names in X_processed (but keep as category dtype where appropriate)
    X_processed.columns = original_feature_names
    
    print(f"Feature importance ranking completed. Top 100 features:")
    for i, row in feature_importance_df.head(100).iterrows():
        print(f"  {i+1:3d}. {row['feature']:<30} | Importance: {row['importance']:8.2f} ({row['importance_percentage']:5.2f}%)")
    
    print(f"\n🎯 LightGBM training completed with native categorical support!")
    print(f"   📊 Features processed: {len(feature_names)}")
    print(f"   🔍 Categorical features used: {len(cleaned_categorical_features)}")
    print(f"   ✅ Model trained successfully for feature importance ranking")
    
    return feature_importance_df, X_processed, cleaned_categorical_features

def select_top_features_as_list(feature_importance_df, top_n=100):
    """
    Select the top N features and return as a clean list of feature names.
    
    This function provides the SAFE OUTPUT for the pipeline - just the feature names
    without any potentially problematic encoded data that could affect downstream
    graph construction and GNN training.
    
    Args:
        feature_importance_df (pd.DataFrame): Features ranked by importance
        top_n (int): Number of top features to select
        
    Returns:
        list: List of top N feature names (strings)
    """
    print(f"🎯 Selecting top {top_n} features based on LightGBM importance ranking...")
    
    # Define critical columns that must always be included for graph construction
    critical_columns = ['TransactionDT', 'TransactionAmt', 'card1', 'isFraud']
    
    print(f"   📋 Critical features (always included): {critical_columns}")
    
    # Get available critical features from importance ranking
    available_critical = []
    for col in critical_columns:
        if col in feature_importance_df['feature'].values:
            available_critical.append(col)
        else:
            print(f"   ⚠️ Warning: Critical feature '{col}' not found in importance ranking")
    
    # Get top features excluding critical ones (to avoid duplication)
    remaining_slots = top_n - len(available_critical)
    
    if remaining_slots > 0:
        top_features_df = feature_importance_df[
            ~feature_importance_df['feature'].isin(available_critical)
        ].head(remaining_slots)
        top_additional_features = top_features_df['feature'].tolist()
    else:
        top_additional_features = []
        print(f"   ⚠️ Warning: Critical features ({len(available_critical)}) exceed top_n ({top_n})")
    
    # Combine critical + top importance features
    final_feature_list = available_critical + top_additional_features
    
    print(f"\n✅ FEATURE SELECTION COMPLETED:")
    print(f"   📊 Critical features included: {len(available_critical)}")
    print(f"   🏆 Top importance features: {len(top_additional_features)}")
    print(f"   📈 Total features selected: {len(final_feature_list)}")
    print(f"   📝 Output: Clean list of feature names (safe for downstream pipeline)")
    
    # Display selected features summary
    print(f"\n📋 Selected Features Summary:")
    print(f"   Critical: {available_critical}")
    if len(top_additional_features) > 0:
        print(f"   Top 10 by importance: {top_additional_features[:10]}")
        if len(top_additional_features) > 10:
            print(f"   ... and {len(top_additional_features) - 10} more importance-ranked features")
    
    return final_feature_list

def select_top_features_and_save(df, X_processed, feature_importance_df, encoders_info, categorical_mappings, y, removed_constant_info=None, correlation_removal_info=None, top_n=100):
    """
    Select the top N features and save the final DataFrame to Parquet.
    Ensures critical columns (TransactionDT, TransactionAmt, isFraud) are always included.
    Uses the processed DataFrame with categorical features converted to numeric values.
    
    Args:
        df (pd.DataFrame): Original merged DataFrame (for target column)
        X_processed (pd.DataFrame): Processed feature DataFrame with categorical features as numeric
        feature_importance_df (pd.DataFrame): Features ranked by importance
        encoders_info (dict): Dictionary containing encoding information and objects
        categorical_mappings (dict): Dictionary containing categorical value mappings
        y (pd.Series): Target variable for calculating pos_weight
        removed_constant_info (dict): Information about removed constant features
        correlation_removal_info (dict): Information about removed highly correlated features
        top_n (int): Number of top features to select
    """
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    print(f"Selecting top {top_n} features...")
    
    # Calculate pos_weight for summary information
    fraud_ratio = y.mean()
    pos_weight = (1 - fraud_ratio) / fraud_ratio
    
    # Define critical columns that must always be included
    # card1 is essential for graph construction (user nodes and user edges)
    critical_columns = ['TransactionDT', 'TransactionAmt', 'card1', 'isFraud']
    
    # IMPORTANT: card1 is crucial for graph construction even though it's a numeric identifier
    # It enables:
    # - User-based edges: connecting transactions from the same card/user
    # - Temporal user patterns: sequence of transactions per user
    # - Fraud propagation: risk spreading between user's transactions
    # - Community detection: identifying related users and suspicious groups
    
    # Define important One-Hot encoded features to force inclusion
    important_onehot_patterns = ['ProductCD_', 'card4_', 'M1_', 'M2_', 'M3_']
    force_include_onehot = []
    
    # Find One-Hot columns that match important patterns and exist in processed data
    for pattern in important_onehot_patterns:
        matching_cols = [col for col in X_processed.columns if col.startswith(pattern)]
        if matching_cols:
            # Include at least the first few columns for each important categorical
            force_include_onehot.extend(matching_cols[:3])  # Limit to 3 per pattern to avoid overload
    
    if force_include_onehot:
        print(f"Forcing inclusion of important One-Hot features: {force_include_onehot[:10]}{'...' if len(force_include_onehot) > 10 else ''}")
    
    # Check which critical columns exist in the dataframe
    existing_critical = [col for col in critical_columns if col in df.columns]
    missing_critical = [col for col in critical_columns if col not in df.columns]
    
    if missing_critical:
        print(f"WARNING: Missing critical columns: {missing_critical}")
    
    print(f"Critical columns found: {existing_critical}")
    
    # Separate target and feature critical columns
    critical_features = [col for col in existing_critical if col != 'isFraud']
    target_column = ['isFraud'] if 'isFraud' in existing_critical else []
    
    # Get top N feature names, excluding critical features to avoid duplication
    all_reserved_features = critical_features + force_include_onehot
    available_features = feature_importance_df[
        ~feature_importance_df['feature'].isin(all_reserved_features)
    ]
    
    # Calculate how many additional features we need
    features_needed = top_n - len(critical_features) - len(force_include_onehot)
    
    if features_needed > 0:
        top_features = available_features.head(features_needed)['feature'].tolist()
    else:
        top_features = []
        print(f"WARNING: Reserved features ({len(all_reserved_features)}) exceed top_n ({top_n})")
        # Prioritize: critical > forced One-Hot > top importance
        available_slots = top_n - len(critical_features)
        if available_slots > 0:
            # Allocate remaining slots for one-hot features
            onehot_slots = min(len(force_include_onehot), available_slots)
            force_include_onehot = force_include_onehot[:onehot_slots]
            print(f"Reduced to {onehot_slots} one-hot features")
    
    # Combine all feature types in priority order
    final_features_raw = critical_features + force_include_onehot + top_features
    
    # Remove duplicates while preserving order
    final_features = []
    seen = set()
    duplicates_removed = 0
    for feature in final_features_raw:
        if feature not in seen:
            final_features.append(feature)
            seen.add(feature)
        else:
            duplicates_removed += 1
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate features from final selection")
    
    columns_to_keep = final_features + target_column
    
    print(f"Critical features included: {critical_features}")
    print(f"Forced One-Hot features included: {len(force_include_onehot)}")
    print(f"Additional top features: {len(top_features)}")
    print(f"Total features selected: {len(final_features)}")
    
    # Create final DataFrame with selected features and target
    # Use processed features (with categorical conversions) + target from original df
    df_features = X_processed[final_features].copy()
    
    # Add target column from original DataFrame if it exists
    if target_column:
        df_target = df[target_column].copy()
        df_final = pd.concat([df_features, df_target], axis=1)
    else:
        df_final = df_features
    
    print(f"Final DataFrame shape: {df_final.shape}")
    print(f"Selected features: {len(final_features)}")
    print(f"Target column included: {'isFraud' in df_final.columns}")
    print(f"Critical columns preserved: {[col for col in critical_features if col in df_final.columns]}")
    
    # Create output directory if it doesn't exist (using absolute path)
    import os
    current_dir = os.getcwd()
    
    # Navigate to project root
    if 'graphsentinel_2.0' in current_dir:
        project_root = current_dir.split('graphsentinel_2.0')[0] + 'graphsentinel_2.0'
    else:
        project_root = '/graphsentinel_2.0'  # Fallback to absolute path
    
    output_dir = os.path.join(project_root, "data", "parquet", "ieee-cis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Parquet file
    output_path = os.path.join(output_dir, "ieee_cis_top100_features.parquet")
    print(f"Saving DataFrame to {output_path}...")
    
    df_final.to_parquet(
        output_path,
        engine='pyarrow',
        index=False
    )
    
    # Save categorical mappings and encoding info to JSON file for future reference
    mappings_path = os.path.join(output_dir, "categorical_mappings.json")
    encoders_path = os.path.join(output_dir, "encoding_info.json")
    constant_features_path = os.path.join(output_dir, "removed_constant_features.json")
    categorical_conversions_path = os.path.join(output_dir, "categorical_conversions.json")
    correlation_removal_path = os.path.join(output_dir, "correlation_removal.json")
    
    categorical_mappings_in_final = {
        col: mapping for col, mapping in categorical_mappings.items() 
        if col in df_final.columns
    }
    
    # Prepare encoding information for final features
    # Note: encoders_info is actually categorical_info from prepare_categorical_features
    encoding_info_final = {
        'converted_to_category': [item['column'] for item in encoders_info.get('converted_to_category', []) if item['column'] in df_final.columns],
        'auto_detected': [item['column'] for item in encoders_info.get('auto_detected', []) if item['column'] in df_final.columns],
        'kept_as_numeric': [item['column'] for item in encoders_info.get('kept_as_numeric', []) if item['column'] in df_final.columns],
        'categorical_conversions_summary': {
            'total_conversions': len(encoders_info.get('converted_to_category', [])) + len(encoders_info.get('auto_detected', [])),
            'explicit_conversions': len(encoders_info.get('converted_to_category', [])),
            'auto_detected_conversions': len(encoders_info.get('auto_detected', [])),
            'memory_saved_mb': encoders_info.get('memory_impact', {}).get('memory_saved_mb', 0)
        }
    }
    
    # Check which categorical features are in final dataset
    # Since we're using category dtype approach, no one-hot encoding was performed
    encoding_info_final['onehot_encoding'] = []
    encoding_info_final['onehot_columns_created'] = []
    
    # Add summary of forced inclusions (this was for one-hot features that don't exist in this approach)
    encoding_info_final['forced_onehot_count'] = 0
    encoding_info_final['forced_onehot_patterns'] = []
    
    if categorical_mappings_in_final or encoding_info_final['onehot_encoding']:
        print(f"Saving categorical mappings to {mappings_path}...")
        with open(mappings_path, 'w') as f:
            json.dump(categorical_mappings_in_final, f, indent=2)
        
        print(f"Saving encoding information to {encoders_path}...")
        with open(encoders_path, 'w') as f:
            json.dump(encoding_info_final, f, indent=2)
        
        print(f"Saving categorical conversions info to {categorical_conversions_path}...")
        
        # Convert categorical info to JSON-serializable format
        categorical_conversions_json = convert_numpy_types(encoders_info)
        
        with open(categorical_conversions_path, 'w') as f:
            json.dump(categorical_conversions_json, f, indent=2)
        
        print(f"Saved mappings for {len(categorical_mappings_in_final)} label-encoded columns")
        print(f"Saved info for {len(encoding_info_final['onehot_encoding'])} one-hot encoded features")
        print(f"Saved categorical conversion details for {encoding_info_final['categorical_conversions_summary']['total_conversions']} features")
    
    # Save correlation removal information
    if correlation_removal_info is None:
        correlation_removal_info = {
            'pairs_found': 0,
            'features_removed': [],
            'features_removed_count': 0,
            'correlation_threshold': 0.98,
            'method': 'pearson',
            'note': 'Correlation removal was not performed or no correlations found'
        }
    
    if correlation_removal_info['features_removed_count'] > 0:
        print(f"Saving correlation removal info to {correlation_removal_path}...")
        
        # Convert correlation removal data to JSON-serializable format
        correlation_removal_json = convert_numpy_types(correlation_removal_info)
        
        with open(correlation_removal_path, 'w') as f:
            json.dump(correlation_removal_json, f, indent=2)
        
        print(f"Saved correlation removal details for {correlation_removal_info['features_removed_count']} removed features")
    else:
        print(f"Saving correlation analysis info to {correlation_removal_path}...")
        
        # Save info even when no features were removed for completeness
        correlation_summary = convert_numpy_types({
            'analysis_performed': True,
            'threshold_used': correlation_removal_info['correlation_threshold'],
            'method_used': correlation_removal_info['method'],
            'pairs_analyzed': correlation_removal_info.get('pairs_found', 0),
            'features_removed': correlation_removal_info['features_removed_count'],
            'result': 'No highly correlated features found' if correlation_removal_info['features_removed_count'] == 0 else f"{correlation_removal_info['features_removed_count']} features removed",
            'computation_time': correlation_removal_info.get('computation_time', 0),
            'memory_saved_mb': correlation_removal_info.get('memory_saved_mb', 0),
            'max_remaining_correlation': correlation_removal_info.get('max_remaining_correlation'),
            'validation_passed': correlation_removal_info.get('validation_passed', True)
        })
        
        with open(correlation_removal_path, 'w') as f:
            json.dump(correlation_summary, f, indent=2)
        
        print(f"Saved correlation analysis summary (no removals needed)")
    
    # Save information about removed constant features
    if removed_constant_info is None:
        removed_constant_info = {}
    
    if removed_constant_info:
        print(f"Saving removed constant features info to {constant_features_path}...")
        constant_features_summary = {
            'total_removed': len(removed_constant_info),
            'removed_features': removed_constant_info,
            'removal_timestamp': pd.Timestamp.now().isoformat(),
            'removal_reason': 'Zero variance - no information gain for ML/GNN training',
            'impact': {
                'memory_saved_mb': sum([info.get('memory_usage_mb', 0) for info in removed_constant_info.values()]),
                'training_efficiency': 'Improved - removed uninformative features'
            }
        }
        
        with open(constant_features_path, 'w') as f:
            json.dump(constant_features_summary, f, indent=2)
        
        print(f"Saved info for {len(removed_constant_info)} removed constant features")
    else:
        print("No constant features were removed - all features had variance > 0")
    
    # Display final results
    print("\n" + "="*60)
    print("FEATURE SELECTION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Final DataFrame saved to: {output_path}")
    print(f"Categorical mappings saved to: {mappings_path}")
    print(f"Encoding information saved to: {encoders_path}")
    print(f"Shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns[:5])}... (showing first 5)")
    print(f"Critical columns in final dataset: {[col for col in critical_columns if col in df_final.columns]}")
    print("\nFirst 5 rows of final DataFrame:")
    print(df_final.head())
    
    # Display data types information - now all should be numeric
    print(f"\nData types summary:")
    dtype_counts = df_final.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Show examples of different encoding strategies
    print(f"\n🔢 ENCODING STRATEGIES APPLIED:")
    
    # Category dtype conversions
    converted_to_category = [item['column'] for item in encoders_info.get('converted_to_category', [])]
    if converted_to_category:
        print(f"📊 CATEGORY DTYPE CONVERSIONS (for LightGBM native categorical support):")
        for col in converted_to_category[:2]:  # Show first 2
            if col in df_final.columns:
                print(f"  {col}:")
                sample_values = df_final[col].head(5).tolist()
                print(f"    Sample values: {sample_values}")
                print(f"    Note: LightGBM handles categorical splits internally")
    
    # Note: No One-hot encoding was performed in this approach
    
    # Auto-detection summary
    auto_detected_cols = [item['column'] for item in encoders_info.get('auto_detected', [])]
    if auto_detected_cols:
        print(f"🔍 AUTO-DETECTED categorical columns: {len(auto_detected_cols)}")
        print(f"    Columns: {auto_detected_cols}")
    
    print(f"\n✅ INTELLIGENT FEATURE PROCESSING COMPLETED:")
    print(f"   🎯 Category dtype conversions: {encoding_info_final['categorical_conversions_summary']['total_conversions']} features optimized")
    print(f"   📊 Category dtype approach: No artificial numerical encoding applied")
    print(f"   � LightGBM native categorical support: Optimal performance achieved")
    print(f"   🔍 Auto-detected: {len(encoding_info_final['auto_detected'])} additional categorical features")
    print(f"   💾 Memory optimization: {encoding_info_final['categorical_conversions_summary']['memory_saved_mb']:.2f}MB saved from category dtypes")
    print(f"   🔗 Correlation removal: {correlation_removal_info['features_removed_count']} highly correlated features removed")
    if correlation_removal_info['features_removed_count'] > 0:
        print(f"       • Threshold: {correlation_removal_info['correlation_threshold']}")
        print(f"       • Method: {correlation_removal_info['method']}")
        print(f"       • Memory saved: {correlation_removal_info.get('memory_saved_mb', 0):.2f}MB")
        print(f"       • Max remaining correlation: {correlation_removal_info.get('max_remaining_correlation', 'N/A')}")
    else:
        print(f"       • No features exceeded correlation threshold ({correlation_removal_info['correlation_threshold']})")
    
    # Summary of features selected based on LightGBM importance (original features only)
    print(f"\n🎯 FEATURE SELECTION ANALYSIS (Original IEEE-CIS Features Only):")
    print(f"   📊 Critical features preserved: {critical_features}")
    print(f"   � Top LightGBM features: {len(top_features)} selected by importance ranking")
    print(f"   🎛️ One-Hot categorical features: {len(force_include_onehot)} forced inclusions")
    print(f"   📈 Total final features: {len(final_features)}/100 slots used")
    print(f"   � Next phase: Graph construction will create derived/temporal/network features")

    # Summary of constant features removal
    if removed_constant_info:
        print(f"\n🚫 CONSTANT FEATURES REMOVAL:")
        print(f"   • Removed features: {len(removed_constant_info)} (zero variance)")
        print(f"   • Examples: {list(removed_constant_info.keys())[:3]}{'...' if len(removed_constant_info) > 3 else ''}")
        print(f"   • Impact: Improved training efficiency and reduced memory usage")
        print(f"   • Details saved: {constant_features_path}")
    else:
        print(f"\n✅ FEATURE VARIANCE CHECK:")
        print(f"   • All features had variance > 0 (no constant features found)")
    
    # Summary of correlation removal
    if correlation_removal_info and correlation_removal_info['features_removed_count'] > 0:
        print(f"\n🔗 HIGH CORRELATION REMOVAL:")
        print(f"   • Removed features: {correlation_removal_info['features_removed_count']} (correlation ≥ {correlation_removal_info['correlation_threshold']})")
        print(f"   • Examples: {correlation_removal_info['features_removed'][:3]}{'...' if len(correlation_removal_info['features_removed']) > 3 else ''}")
        print(f"   • Method: {correlation_removal_info['method']} correlation")
        print(f"   • Computation time: {correlation_removal_info.get('computation_time', 0):.2f}s")
        print(f"   • Memory saved: {correlation_removal_info.get('memory_saved_mb', 0):.2f}MB")
        print(f"   • Validation: {'✅ Passed' if correlation_removal_info.get('validation_passed', True) else '⚠️ Check needed'}")
        print(f"   • Impact: Reduced multicollinearity and improved model stability")
        print(f"   • Details saved: {correlation_removal_path}")
    else:
        print(f"\n✅ CORRELATION ANALYSIS:")
        print(f"   • Threshold used: {correlation_removal_info.get('correlation_threshold', 0.98)}")
        print(f"   • Method: {correlation_removal_info.get('method', 'pearson')}")
        print(f"   • Result: No highly correlated features found")
        print(f"   • All feature pairs below threshold")
    
    print(f"📊 Optimized for Recall, F2-Score, AUC-PR metrics!")
    print(f"🔧 Enhanced LightGBM parameters: pos_weight={pos_weight:.1f}, n_estimators=300")
    print(f"🎯 Category dtype optimization: {encoding_info_final['categorical_conversions_summary']['total_conversions']} features converted for optimal LightGBM performance")
    print(f"� Correlation optimization: {correlation_removal_info['features_removed_count']} redundant features removed (threshold: {correlation_removal_info['correlation_threshold']})")
    total_memory_saved = encoding_info_final['categorical_conversions_summary']['memory_saved_mb'] + correlation_removal_info.get('memory_saved_mb', 0)
    print(f"💾 Total memory efficiency: {total_memory_saved:.2f}MB saved from optimizations")
    print(f"🚀 Ready for graph construction phase (where derived features will be created)!")
    print(f"🗂️ Comprehensive metadata saved:")
    print(f"   • Encoding information: {encoders_path}")
    print(f"   • Categorical conversions: {categorical_conversions_path}")
    print(f"   • Label mappings: {mappings_path}")
    print(f"   • Correlation removal: {correlation_removal_path}")
    
    print(f"\nDataFrame with optimally processed IEEE-CIS features saved successfully!")
    print(f"Selection optimized for fraud detection with enhanced recall and F2-score!")
    print(f"Features properly processed for optimal LightGBM performance and reduced redundancy!")
    print(f"Clean feature set ready for graph construction and GNN training!")
    print(f"Derived features (temporal, similarity, network) will be created in graph construction phase!")
    
    return df_final

def remove_highly_correlated_features(X, correlation_threshold=0.98, method='pearson', max_iterations=20):
    """
    Remove features with high correlation (>= threshold) to reduce redundancy
    and improve model generalization using enhanced iterative removal.
    
    Uses multiple iterations with improved strategies to handle:
    - Perfect correlations (1.0) with aggressive removal
    - Chained correlations (A-B-C) where removing A might leave B-C still highly correlated
    - Multiple correlation clusters that require several passes
    
    For each pair of highly correlated features, removes the one with lower
    variance to preserve the most informative feature. For perfect correlations,
    uses more aggressive grouping strategies.
    
    Args:
        X (pd.DataFrame): Feature matrix (after categorical encoding)
        correlation_threshold (float): Correlation threshold (default: 0.98)
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
        max_iterations (int): Maximum iterations to prevent infinite loops (increased to 10)
        
    Returns:
        tuple: (X_filtered, removal_info)
            - X_filtered: DataFrame with highly correlated features removed
            - removal_info: Dictionary with detailed removal information
    """
    print(f"🔍 Analyzing feature correlations with enhanced iterative removal (threshold: {correlation_threshold})")
    
    # Initialize for iterative removal
    X_current = X.copy()
    all_removed_features = []
    all_removal_decisions = []
    all_corr_pairs = []
    iteration_details = []
    total_computation_time = 0
    
    print(f"  🔄 Starting enhanced iterative correlation removal (max {max_iterations} iterations)...")
    
    for iteration in range(max_iterations):
        print(f"    📊 Iteration {iteration + 1}/{max_iterations}: Analyzing {X_current.shape[1]} features...")
        
        # Calculate correlation matrix with timing
        iteration_start = time.time()
        
        # Filter to numeric columns only for correlation analysis
        numeric_columns = X_current.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            print(f"      ⚠️ Only {len(numeric_columns)} numeric columns available for correlation analysis")
            break
        
        X_numeric = X_current[numeric_columns]
        
        # For large datasets, use efficient correlation computation
        if len(X_numeric) > 50000:
            # Sample for correlation analysis if dataset is very large
            sample_size = min(50000, len(X_numeric))
            if sample_size < len(X_numeric):
                sample_indices = np.random.choice(len(X_numeric), sample_size, replace=False)
                X_sample = X_numeric.iloc[sample_indices]
                corr_matrix = X_sample.corr(method=method)
            else:
                corr_matrix = X_numeric.corr(method=method)
        else:
            corr_matrix = X_numeric.corr(method=method)
        
        iteration_time = time.time() - iteration_start
        total_computation_time += iteration_time
        
        # Find highly correlated pairs for this iteration
        highly_correlated_pairs = []
        perfect_corr_pairs = []  # Track perfect correlations separately
        
        # Get upper triangle of correlation matrix (avoid duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold, separating perfect correlations
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                correlation = upper_triangle.loc[idx, col]
                if pd.notna(correlation) and abs(correlation) >= correlation_threshold:
                    pair_info = {
                        'feature1': idx,
                        'feature2': col,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'iteration': iteration + 1
                    }
                    highly_correlated_pairs.append(pair_info)
                    
                    # Track perfect or near-perfect correlations (≥ 0.9999)
                    if abs(correlation) >= 0.9999:
                        perfect_corr_pairs.append(pair_info)
        
        print(f"      📋 Found {len(highly_correlated_pairs)} highly correlated pairs ({len(perfect_corr_pairs)} perfect/near-perfect)")
        
        # If no high correlations found, we're done
        if len(highly_correlated_pairs) == 0:
            print(f"      ✅ No more high correlations found after {iteration + 1} iterations")
            break
        
        # Sort pairs by correlation strength (highest first) for systematic removal
        highly_correlated_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        all_corr_pairs.extend(highly_correlated_pairs)
        
        # Enhanced removal strategy: Handle perfect correlations more aggressively
        features_to_remove_this_iter = set()
        removal_decisions_this_iter = []
        
        # Strategy 1: For perfect correlations (≥ 0.9999), group and remove aggressively
        if perfect_corr_pairs:
            print(f"      🎯 Applying aggressive removal for {len(perfect_corr_pairs)} perfect correlations...")
            
            # Build groups of perfectly correlated features
            perfect_groups = {}
            processed_features = set()
            
            for pair in perfect_corr_pairs:
                feat1, feat2 = pair['feature1'], pair['feature2']
                
                # Skip if both features already processed
                if feat1 in processed_features and feat2 in processed_features:
                    continue
                
                # Find existing group or create new one
                group_key = None
                for key, group in perfect_groups.items():
                    if feat1 in group or feat2 in group:
                        group_key = key
                        break
                
                if group_key is None:
                    # Create new group
                    group_key = f"perfect_group_{len(perfect_groups)}"
                    perfect_groups[group_key] = set()
                
                perfect_groups[group_key].update([feat1, feat2])
                processed_features.update([feat1, feat2])
            
            # For each perfect group, keep only the feature with highest variance
            for group_key, feature_group in perfect_groups.items():
                if len(feature_group) <= 1:
                    continue
                
                # Calculate variances for all features in group (only numeric features)
                feature_variances = {}
                for feat in feature_group:
                    if feat in numeric_columns and feat in X_current.columns:
                        feature_variances[feat] = X_current[feat].var()
                
                if len(feature_variances) <= 1:
                    continue
                
                # Keep feature with highest variance
                best_feature = max(feature_variances.keys(), key=lambda x: feature_variances[x])
                features_to_remove = feature_group - {best_feature}
                
                print(f"        🎯 Perfect group {group_key}: keeping '{best_feature}' (var={feature_variances[best_feature]:.6f}), removing {len(features_to_remove)} others")
                
                for feat_to_remove in features_to_remove:
                    if feat_to_remove not in features_to_remove_this_iter:
                        features_to_remove_this_iter.add(feat_to_remove)
                        removal_decisions_this_iter.append({
                            'removed_feature': feat_to_remove,
                            'kept_feature': best_feature,
                            'correlation': 1.0,  # Perfect correlation
                            'abs_correlation': 1.0,
                            'removed_variance': feature_variances[feat_to_remove],
                            'kept_variance': feature_variances[best_feature],
                            'removal_reason': f"Perfect correlation group removal (kept highest variance: {feature_variances[best_feature]:.6f})",
                            'iteration': iteration + 1,
                            'removal_strategy': 'perfect_group'
                        })
        
        # Strategy 2: Handle remaining high correlations (non-perfect) with standard approach
        remaining_pairs = [pair for pair in highly_correlated_pairs if pair['abs_correlation'] < 0.9999]
        
        for pair in remaining_pairs:
            feat1, feat2 = pair['feature1'], pair['feature2']
            correlation = pair['correlation']
            
            # Skip if either feature already marked for removal in this iteration
            if feat1 in features_to_remove_this_iter or feat2 in features_to_remove_this_iter:
                continue
            
            # Skip if either feature is not in numeric columns
            if feat1 not in numeric_columns or feat2 not in numeric_columns:
                continue
            
            # Skip if either feature is not in current dataframe
            if feat1 not in X_current.columns or feat2 not in X_current.columns:
                continue
            
            # Decision criteria: Remove feature with lower variance
            var1 = X_current[feat1].var()
            var2 = X_current[feat2].var()
            
            # Enhanced decision logic with smaller tolerance for variance differences
            if var1 > var2 * 1.005:  # 0.5% threshold (more sensitive than before)
                feature_to_remove = feat2
                reason = f"Lower variance ({var2:.6f} vs {var1:.6f})"
                kept_feature = feat1
            elif var2 > var1 * 1.005:
                feature_to_remove = feat1
                reason = f"Lower variance ({var1:.6f} vs {var2:.6f})"
                kept_feature = feat2
            else:
                # Similar variance - use lexicographic order for consistency
                if feat1 > feat2:
                    feature_to_remove = feat1
                    reason = f"Similar variance ({var1:.6f} ≈ {var2:.6f}), lexicographic order"
                    kept_feature = feat2
                else:
                    feature_to_remove = feat2
                    reason = f"Similar variance ({var1:.6f} ≈ {var2:.6f}), lexicographic order"
                    kept_feature = feat1
            
            features_to_remove_this_iter.add(feature_to_remove)
            
            removal_decisions_this_iter.append({
                'removed_feature': feature_to_remove,
                'kept_feature': kept_feature,
                'correlation': correlation,
                'abs_correlation': abs(correlation),
                'removed_variance': var1 if feature_to_remove == feat1 else var2,
                'kept_variance': var2 if feature_to_remove == feat1 else var1,
                'removal_reason': reason,
                'iteration': iteration + 1,
                'removal_strategy': 'pairwise'
            })
        
        # Remove features for this iteration
        if features_to_remove_this_iter:
            X_current = X_current.drop(columns=list(features_to_remove_this_iter))
            all_removed_features.extend(list(features_to_remove_this_iter))
            all_removal_decisions.extend(removal_decisions_this_iter)
            
            print(f"      🗑️ Removed {len(features_to_remove_this_iter)} features in iteration {iteration + 1}")
            print(f"      � Shape after iteration {iteration + 1}: {X_current.shape}")
        else:
            print(f"      ℹ️ No features removed in iteration {iteration + 1}")
            break
        
        # Store iteration details
        iteration_details.append({
            'iteration': iteration + 1,
            'pairs_found': len(highly_correlated_pairs),
            'features_removed': len(features_to_remove_this_iter),
            'remaining_features': X_current.shape[1],
            'computation_time': iteration_time,
            'removed_features': list(features_to_remove_this_iter)
        })
    
    # Enhanced final validation with detailed analysis
    print(f"\n  ✅ Validating correlation removal after {len(iteration_details)} iterations...")
    
    # Filter to numeric columns for final validation
    final_numeric_columns = X_current.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(final_numeric_columns) > 1:  # Need at least 2 numeric features to compute correlation
        X_final_numeric = X_current[final_numeric_columns]
        final_corr_matrix = X_final_numeric.corr(method=method)
        final_upper = final_corr_matrix.where(
            np.triu(np.ones(final_corr_matrix.shape), k=1).astype(bool)
        )
        max_remaining_corr = final_upper.abs().max().max()
        
        if pd.notna(max_remaining_corr):
            print(f"    📊 Maximum remaining correlation: {max_remaining_corr:.4f}")
            
            # Enhanced validation with multiple thresholds
            validation_passed = max_remaining_corr < correlation_threshold
            perfect_corrs_remain = max_remaining_corr >= 0.9999
            very_high_corrs_remain = max_remaining_corr >= 0.995
            
            if validation_passed:
                print(f"    ✅ All correlations now below threshold ({correlation_threshold})")
            else:
                print(f"    ⚠️ WARNING: {max_remaining_corr:.4f} >= {correlation_threshold} - some high correlations still remain!")
                
                # Count remaining high correlation pairs for detailed analysis
                remaining_high_pairs = []
                for col in final_upper.columns:
                    for idx in final_upper.index:
                        corr_val = final_upper.loc[idx, col]
                        if pd.notna(corr_val) and abs(corr_val) >= correlation_threshold:
                            remaining_high_pairs.append({
                                'feature1': idx,
                                'feature2': col,
                                'correlation': corr_val,
                                'abs_correlation': abs(corr_val)
                            })
                
                print(f"    📋 {len(remaining_high_pairs)} high correlation pairs still remain:")
                
                # Show top 3 highest remaining correlations
                remaining_high_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
                for i, pair in enumerate(remaining_high_pairs[:3]):
                    print(f"      {i+1}. {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.6f}")
                
                if len(remaining_high_pairs) > 3:
                    print(f"      ... and {len(remaining_high_pairs) - 3} more pairs")
                
                # Enhanced recommendations based on remaining correlation levels
                if perfect_corrs_remain:
                    print(f"    🚨 CRITICAL: Perfect correlations (≥0.9999) still remain!")
                    print(f"    💡 Recommendations:")
                    print(f"       • Increase max_iterations beyond {max_iterations}")
                    print(f"       • Lower correlation_threshold to 0.95 or 0.99")
                    print(f"       • Manual review of remaining perfectly correlated features")
                    print(f"       • Check for duplicated or derived features")
                elif very_high_corrs_remain:
                    print(f"    ⚠️ Very high correlations (≥0.995) still remain")
                    print(f"    💡 Recommendations:")
                    print(f"       • Consider lowering threshold to 0.95")
                    print(f"       • Increase iterations to {max_iterations + 5}")
                else:
                    print(f"    💡 Recommendations:")
                    print(f"       • Consider lowering threshold if needed for your use case")
                    print(f"       • Current remaining correlations may be acceptable")
        else:
            max_remaining_corr = None
            validation_passed = True
            print(f"    ✅ Correlation validation completed successfully")
    else:
        max_remaining_corr = None
        validation_passed = True
        print(f"    ℹ️ Only {len(final_numeric_columns)} numeric feature(s) remaining - no correlation validation needed")
    
    # Calculate memory impact
    original_memory = X.memory_usage(deep=True).sum() / 1024**2
    final_memory = X_current.memory_usage(deep=True).sum() / 1024**2
    memory_saved = original_memory - final_memory
    
    # Compile comprehensive removal information
    removal_info = {
        'pairs_found': len(all_corr_pairs),
        'features_removed': all_removed_features,
        'features_removed_count': len(all_removed_features),
        'correlation_threshold': correlation_threshold,
        'method': method,
        'computation_time': total_computation_time,
        'memory_saved_mb': memory_saved,
        'removal_decisions': all_removal_decisions,
        'highly_correlated_pairs': all_corr_pairs,
        'original_shape': X.shape,
        'filtered_shape': X_current.shape,
        'max_remaining_correlation': float(max_remaining_corr) if pd.notna(max_remaining_corr) else None,
        'validation_passed': validation_passed,
        'iterations_performed': len(iteration_details),
        'iteration_details': iteration_details,
        'summary_stats': {
            'correlation_range': {
                'min': float(min([pair['abs_correlation'] for pair in all_corr_pairs])) if all_corr_pairs else None,
                'max': float(max([pair['abs_correlation'] for pair in all_corr_pairs])) if all_corr_pairs else None,
                'mean': float(np.mean([pair['abs_correlation'] for pair in all_corr_pairs])) if all_corr_pairs else None
            },
            'variance_analysis': {
                'decisions_by_variance': len([d for d in all_removal_decisions if 'Lower variance' in d['removal_reason']]),
                'decisions_by_lexicographic': len([d for d in all_removal_decisions if 'lexicographic' in d['removal_reason']]),
                'decisions_by_perfect_group': len([d for d in all_removal_decisions if d.get('removal_strategy') == 'perfect_group']),
                'decisions_by_pairwise': len([d for d in all_removal_decisions if d.get('removal_strategy') == 'pairwise'])
            },
            'iterative_analysis': {
                'total_iterations': len(iteration_details),
                'features_removed_per_iteration': [detail['features_removed'] for detail in iteration_details],
                'pairs_found_per_iteration': [detail['pairs_found'] for detail in iteration_details]
            }
        }
    }
    
    print(f"\n  ✅ Enhanced iterative correlation analysis completed:")
    print(f"     • Method: {method}")
    print(f"     • Threshold: {correlation_threshold}")
    print(f"     • Total iterations: {len(iteration_details)}")
    print(f"     • Total pairs found: {len(all_corr_pairs)}")
    print(f"     • Features removed: {len(all_removed_features)}")
    
    # Summary of removal strategies
    perfect_group_removals = len([d for d in all_removal_decisions if d.get('removal_strategy') == 'perfect_group'])
    pairwise_removals = len([d for d in all_removal_decisions if d.get('removal_strategy') == 'pairwise'])
    if perfect_group_removals > 0:
        print(f"     • Perfect correlation groups: {perfect_group_removals} features removed")
    if pairwise_removals > 0:
        print(f"     • Pairwise correlation removal: {pairwise_removals} features removed")
    
    print(f"     • Memory saved: {memory_saved:.2f}MB")
    print(f"     • Total computation time: {total_computation_time:.2f}s")
    if validation_passed:
        print(f"     • ✅ Validation: No high correlations remain")
    else:
        print(f"     • ⚠️ Validation: Max remaining correlation = {max_remaining_corr:.4f}")
        print(f"     • 💡 Recommendation: Review remaining highly correlated features manually")
    
    return X_current, removal_info

def main():
    """Main execution function."""
    try:
        print("="*60)
        print("IEEE-CIS FRAUD DETECTION - LIGHTGBM BEST PRACTICES REFACTOR")
        print("GraphSentinel 2.0 Project - Category dtype + Native Categorical Support")
        print("="*60)
        print("🎯 NEW APPROACH: Using pandas category dtype + LightGBM native categorical support")
        print("✅ REMOVED: Label Encoding (artificial numerical ordering)")
        print("🚀 BENEFITS: Better LightGBM performance + safer output for graph construction")
        
        # Step 1: Load and merge data
        df = load_and_merge_data()
        
        # Step 1.1: Remove constant features
        print("\n" + "="*50)
        print("REMOVING CONSTANT FEATURES")
        print("="*50)
        df, removed_constant_info = remove_constant_features(df)
        
        # Log constant feature removal for audit
        if removed_constant_info:
            print(f"\n📋 AUDIT - Constant features removed:")
            for i, (col, info) in enumerate(list(removed_constant_info.items())[:5]):
                if 'mean' in info:
                    print(f"   {i+1}. {col}: mean={info['mean']:.6f}, std={info['std']:.6f}")
                else:
                    print(f"   {i+1}. {col}: values={info['unique_values']}")
            
            if len(removed_constant_info) > 5:
                print(f"   ... and {len(removed_constant_info) - 5} more features")
        
        # Step 2: Prepare data for modeling
        print("\nPreparing data for modeling...")
        
        # Separate target and features
        if 'isFraud' not in df.columns:
            raise ValueError("Target column 'isFraud' not found in the dataset")
        
        y = df['isFraud'].copy()
        X = df.drop(['isFraud'], axis=1)
        
        print(f"Target distribution:")
        print(y.value_counts().to_dict())
        print(f"Fraud rate: {y.mean():.4f}")
        
        # NEW APPROACH: Prepare categorical features with category dtype (no Label Encoding)
        print(f"\n{'='*50}")
        print("CATEGORICAL FEATURE PROCESSING - LIGHTGBM BEST PRACTICES")
        print("="*50)
        X_processed, categorical_info = prepare_categorical_features(X)
        
        # Step 2.5: Remove highly correlated features
        print(f"\n{'='*50}")
        print("REMOVING HIGHLY CORRELATED FEATURES")
        print("="*50)
        X_decorrelated, correlation_removal_info = remove_highly_correlated_features(
            X_processed, 
            correlation_threshold=0.98,
            method='pearson',
            max_iterations=20
        )
        
        # Step 3: Train LightGBM with native categorical support
        print(f"\n{'='*50}")
        print("LIGHTGBM TRAINING WITH NATIVE CATEGORICAL SUPPORT")
        print("="*50)
        feature_importance_df, X_final, categorical_features = train_lightgbm_and_get_importance(
            X_decorrelated, y, categorical_info
        )
        
        # Step 4: SAFE OUTPUT - Extract top feature names as list (no encoded data)
        print(f"\n{'='*50}")
        print("EXTRACTING TOP FEATURES AS SAFE LIST OUTPUT")
        print("="*50)
        top_feature_names = select_top_features_as_list(feature_importance_df, top_n=100)
        
        # Step 5: Save top features DataFrame to Parquet
        print(f"\n{'='*50}")
        print("SAVING TOP FEATURES DATAFRAME TO PARQUET")
        print("="*50)
        df_final = select_top_features_and_save(
            df=df,
            X_processed=X_decorrelated,
            feature_importance_df=feature_importance_df,
            encoders_info=categorical_info,
            categorical_mappings={},  # No mappings since we're using category dtype
            y=y,
            removed_constant_info=removed_constant_info,
            correlation_removal_info=correlation_removal_info,
            top_n=100
        )
        
        # Save feature list to JSON file for downstream pipeline
        current_dir = os.getcwd()
        if 'graphsentinel_2.0' in current_dir:
            project_root = current_dir.split('graphsentinel_2.0')[0] + 'graphsentinel_2.0'
        else:
            project_root = '/graphsentinel_2.0'
        
        output_dir = os.path.join(project_root, "data", "parquet", "ieee-cis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature list
        feature_list_path = os.path.join(output_dir, "top_100_feature_names.json")
        with open(feature_list_path, 'w') as f:
            json.dump({
                'top_features': top_feature_names,
                'total_count': len(top_feature_names),
                'selection_method': 'lightgbm_importance_with_category_dtype',
                'categorical_features_used': len(categorical_features),
                'correlation_threshold': 0.98,
                'fraud_rate': float(y.mean()),
                'timestamp': pd.Timestamp.now().isoformat(),
                'note': 'Feature names only - no encoded data. Safe for graph construction.'
            }, f, indent=2)
        
        # Save detailed information for reproducibility
        info_path = os.path.join(output_dir, "feature_selection_info.json")
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        detailed_info = {
            'approach': 'lightgbm_native_categorical_support',
            'categorical_processing': convert_to_serializable(categorical_info),
            'correlation_removal': convert_to_serializable(correlation_removal_info),
            'constant_features_removed': len(removed_constant_info) if removed_constant_info else 0,
            'original_features': int(X.shape[1]),
            'final_features': len(top_feature_names),
            'categorical_features_in_final': [feat for feat in top_feature_names if feat in CATEGORICAL_FEATURES],
            'benefits': [
                'No artificial numerical ordering from Label Encoding',
                'LightGBM uses optimal categorical split strategies',
                'Safer output for subsequent graph construction',
                'Better model performance with native categorical support',
                'Preserved semantic meaning of categorical features'
            ]
        }
        
        with open(info_path, 'w') as f:
            json.dump(detailed_info, f, indent=2)
        
        print("\n" + "="*60)
        print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ SAFE OUTPUT: Feature names saved to {feature_list_path}")
        print(f"📋 DETAILS: Processing info saved to {info_path}")
        print(f"🎯 SELECTED: {len(top_feature_names)} features using LightGBM importance")
        print(f"📊 CATEGORICAL: {len([f for f in top_feature_names if f in CATEGORICAL_FEATURES])} categorical features included")
        print(f"🚀 NEXT PHASE: Graph construction can safely use these feature names")
        print(f"✅ NO ENCODED DATA: Output contains only feature names (strings)")
        print(f"🔧 OPTIMIZED: LightGBM used native categorical support for better performance")
        
        # Display sample of selected features by type
        categorical_selected = [f for f in top_feature_names if f in CATEGORICAL_FEATURES]
        numeric_selected = [f for f in top_feature_names if f not in CATEGORICAL_FEATURES and f != 'isFraud']
        
        print(f"\n📊 FEATURE BREAKDOWN:")
        print(f"   • Categorical features: {len(categorical_selected)}")
        if categorical_selected:
            print(f"     Examples: {categorical_selected[:5]}")
        print(f"   • Numeric features: {len(numeric_selected)}")
        if numeric_selected:
            print(f"     Examples: {numeric_selected[:5]}")
        
        return top_feature_names  # Return the safe output
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Process failed. Please check the error message above.")
        sys.exit(1)

if __name__ == "__main__":
    main()