"""
Correlation analysis for IEEE-CIS fraud detection dataset.
"""

# Force CPU mode - disable all GPU functionality
import os
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUPY_DISABLE_HIP'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
from utils import save_json, save_csv
from feature_config import get_features_for_analysis, validate_features_in_dataframe


def correlation_analysis(X, y, log_dir, logger, correlation_threshold=0.8):
    """
    Perform correlation analysis between features and target variable.
    
    Args:
        X: DataFrame with features
        y: Series with target variable (fraud indicator)
        log_dir: Directory to save outputs
        logger: Logger object
        correlation_threshold: Threshold for high correlation pairs
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting correlation analysis...")
    
    # Create directory for correlation analysis outputs
    corr_dir = os.path.join(log_dir, 'correlation_analysis')
    os.makedirs(corr_dir, exist_ok=True)
    
    results = {}
    
    # Get optimized feature list for correlation analysis
    target_features = get_features_for_analysis('correlation', exclude_time=True)
    valid_features, missing_features = validate_features_in_dataframe(X, target_features)
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    logger.info(f"Analyzing correlations for {len(valid_features)} features (from {len(target_features)} expected)")
    
    # Use only valid features that exist in the dataset and are numeric
    X_analysis = X[valid_features]
    
    # Filter to keep only numeric columns for correlation analysis
    numeric_columns = X_analysis.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Filtering to {len(numeric_columns)} numeric columns for correlation analysis")
    
    X_analysis = X_analysis[numeric_columns]
    
    # Remove columns with zero variance (constant values) to avoid division by zero
    logger.info("Checking for constant features...")
    non_constant_cols = []
    for col in numeric_columns:  # Use numeric_columns instead of valid_features
        try:
            # Only check std for numeric columns that exist in X_analysis
            if col in X_analysis.columns and pd.api.types.is_numeric_dtype(X_analysis[col]):
                if X_analysis[col].nunique() > 1 and X_analysis[col].std() > 1e-10:  # Not constant and not near-zero variance
                    non_constant_cols.append(col)
                else:
                    logger.warning(f"Removing constant/near-constant numeric feature: {col}")
        except Exception as e:
            logger.warning(f"Could not check feature {col}: {e}")
            continue
    
    logger.info(f"Kept {len(non_constant_cols)} non-constant features for correlation analysis")
    X_analysis = X_analysis[non_constant_cols]
    
    # Calculate correlation matrix with only non-constant features
    logger.info("Calculating correlation matrix...")
    with np.errstate(invalid='ignore', divide='ignore'):  # Suppress warnings
        correlation_matrix = X_analysis.corr()
    
    # Replace any NaN values that might still occur with 0
    correlation_matrix = correlation_matrix.fillna(0)
    
    # Find correlations with TransactionAmt (if exists)
    amount_correlations = {}
    if 'TransactionAmt' in correlation_matrix.columns:
        amount_corr_series = correlation_matrix['TransactionAmt'].abs().sort_values(ascending=False)
        # Remove NaN values and keep only significant correlations
        amount_corr_clean = amount_corr_series.dropna()
        amount_correlations = {k: v for k, v in amount_corr_clean.to_dict().items() if v > 0.1}
    
    # Find high correlation pairs between features
    logger.info(f"Finding high correlation pairs (threshold: {correlation_threshold})...")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            
            # Skip NaN or infinite values
            if pd.isna(corr_val) or not np.isfinite(corr_val):
                continue
                
            abs_corr = abs(corr_val)
            if abs_corr > correlation_threshold:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': float(abs_corr)
                })
    
    # Calculate correlations with target (isFraud) - using only non-constant features
    logger.info("Calculating correlations with target variable...")
    target_correlations = {}
    for col in non_constant_cols:  # Use non_constant_cols instead of numeric_cols
        try:
            # Use safe correlation calculation
            with np.errstate(invalid='ignore', divide='ignore'):
                corr_with_target = X[col].corr(y)
            
            if not pd.isna(corr_with_target) and np.isfinite(corr_with_target):
                target_correlations[col] = abs(corr_with_target)
            else:
                target_correlations[col] = 0
        except Exception as e:
            logger.warning(f"Error calculating correlation for {col}: {e}")
            target_correlations[col] = 0
    
    # Get top features correlated with fraud
    top_fraud_corr_features = sorted(target_correlations.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
    
    # Extract priority features for graph construction
    priority_features = [feat for feat, _ in top_fraud_corr_features[:5]]
    similarity_features = list(set([pair['feature1'] for pair in high_corr_pairs[:5]] + 
                                 [pair['feature2'] for pair in high_corr_pairs[:5]]))
    
    # Graph Construction Insights
    graph_insights = {
        "node_attributes": {
            "priority_features": priority_features,
            "feature_weights": dict(top_fraud_corr_features[:10])
        },
        "edge_construction": {
            "similarity_features": similarity_features,
            "correlation_threshold": correlation_threshold,
            "recommended_similarity_metrics": ["cosine", "correlation"]
        },
        "subgraph_analysis": {
            "high_correlation_clusters": [pair for pair in high_corr_pairs if pair['correlation'] > 0.9]
        }
    }
    
    # Compile results
    results = {
        'high_amount_correlations': amount_correlations,
        'high_correlation_pairs': high_corr_pairs,
        'target_correlations': target_correlations,
        'correlation_summary': {
            'total_features': len(valid_features),
            'high_corr_pairs_count': len(high_corr_pairs),
            'correlation_threshold': correlation_threshold
        },
        'graph_construction_insights': graph_insights
    }
    
    # Save correlation matrix to CSV
    corr_matrix_file = save_csv(correlation_matrix, 'correlation_matrix.csv', corr_dir)
    logger.info(f"Saved correlation matrix to {corr_matrix_file}")
    
    # Save target correlations
    target_corr_df = pd.DataFrame(list(target_correlations.items()), 
                                  columns=['Feature', 'Correlation_with_Target'])
    target_corr_df = target_corr_df.sort_values('Correlation_with_Target', ascending=False)
    target_corr_file = save_csv(target_corr_df, 'target_correlations.csv', corr_dir)
    logger.info(f"Saved target correlations to {target_corr_file}")
    
    # Save high correlation pairs
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_file = save_csv(high_corr_df, 'high_correlation_pairs.csv', corr_dir)
        logger.info(f"Saved high correlation pairs to {high_corr_file}")
    
    logger.info(f"Correlation analysis completed: {len(high_corr_pairs)} high correlation pairs found")
    logger.info(f"Top features correlated with fraud: {priority_features}")
    
    return results
