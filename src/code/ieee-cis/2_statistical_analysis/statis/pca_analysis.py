"""
PCA analysis for IEEE-CIS fraud detection dataset.
"""

# Force CPU mode - disable all GPU functionality
import os
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUPY_DISABLE_HIP'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import save_json, save_csv
from feature_config import get_features_for_analysis, validate_features_in_dataframe
import gc


def pca_analysis(X, y, log_dir, logger, n_components=10):
    """
    Perform PCA analysis on the dataset.
    
    Args:
        X: DataFrame with features
        y: Series with target variable (fraud indicator)
        log_dir: Directory to save outputs
        logger: Logger object
        n_components: Number of principal components
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting PCA analysis...")
    
    # Create directory for PCA analysis outputs
    pca_dir = os.path.join(log_dir, 'pca_analysis')
    os.makedirs(pca_dir, exist_ok=True)
    
    # Get optimized feature list for PCA analysis (exclude categorical/flag features)
    target_features = get_features_for_analysis('pca', exclude_time=True)
    valid_features, missing_features = validate_features_in_dataframe(X, target_features)
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    logger.info(f"Using {len(valid_features)} continuous features for PCA (from {len(target_features)} expected)")
    
    # Prepare data (fill NaN values and remove constant features)
    logger.info("Preparing data for PCA...")
    
    # Filter to keep only numeric columns for PCA
    X_analysis = X[valid_features]
    numeric_columns = X_analysis.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Filtering to {len(numeric_columns)} numeric columns for PCA analysis")
    
    X_clean = X_analysis[numeric_columns].fillna(0)
    
    # Remove constant features that can cause issues in PCA
    non_constant_cols = []
    for col in numeric_columns:  # Use numeric_columns instead of valid_features
        try:
            # Only check std for numeric columns
            if pd.api.types.is_numeric_dtype(X_clean[col]):
                if X_clean[col].nunique() > 1 and X_clean[col].std() > 1e-10:
                    non_constant_cols.append(col)
                else:
                    logger.warning(f"Removing constant/near-constant numeric feature for PCA: {col}")
        except Exception as e:
            logger.warning(f"Could not check feature {col} for PCA: {e}")
            continue
    
    X_for_pca = X_clean[non_constant_cols]
    logger.info(f"Using {len(non_constant_cols)} non-constant features for PCA")
    
    # Standardize data
    logger.info("Standardizing features for PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_for_pca)
    
    # Adjust n_components if necessary
    max_components = min(n_components, len(non_constant_cols), X_scaled.shape[0])
    if max_components < n_components:
        logger.warning(f"Reducing PCA components from {n_components} to {max_components} due to data constraints")
        n_components = max_components
    
    # Apply PCA
    logger.info(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate variance explained
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Identify most important features for each component
    components_importance = []
    top_pca_features = set()
    
    for i, component in enumerate(pca.components_):
        # Get top 5 features for this component using non_constant_cols
        top_features_idx = np.argsort(np.abs(component))[-5:]
        top_features = [(non_constant_cols[idx], abs(component[idx])) for idx in top_features_idx]
        
        components_importance.append({
            f'PC{i+1}': top_features
        })
        
        # Collect top 3 features from each component
        for feat, _ in top_features[-3:]:
            top_pca_features.add(feat)
    
    # Limit to top 8 features
    priority_features = list(top_pca_features)[:8]
    
    # Graph Construction Insights
    graph_insights = {
        "node_attributes": {
            "priority_features": priority_features,
            "dimensionality_reduction": {
                "recommended_components": min(8, n_components),
                "variance_threshold": 0.95
            }
        },
        "edge_construction": {
            "pca_based_similarity": {
                "use_principal_components": True,
                "n_components_for_similarity": min(5, n_components)
            }
        },
        "subgraph_analysis": {
            "component_based_grouping": {
                "high_variance_components": [i for i, var in enumerate(explained_variance_ratio) if var > 0.1]
            }
        }
    }
    
    # Compile results
    results = {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'components_importance': components_importance,
        'n_components': n_components,
        'total_variance_explained': float(cumulative_variance[-1]),
        'graph_construction_insights': graph_insights
    }
    
    # Save PCA components
    components_df = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=non_constant_cols  # Use non_constant_cols (features actually used in PCA)
    )
    components_file = save_csv(components_df, 'pca_components.csv', pca_dir)
    logger.info(f"Saved PCA components to {components_file}")
    
    # Save explained variance
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance_Ratio': explained_variance_ratio,
        'Cumulative_Variance': cumulative_variance
    })
    variance_file = save_csv(variance_df, 'explained_variance.csv', pca_dir)
    logger.info(f"Saved explained variance to {variance_file}")
    
    # Save transformed data (sample)
    if X_pca.shape[0] > 10000:
        # Save only a sample if dataset is large
        sample_indices = np.random.choice(X_pca.shape[0], size=10000, replace=False)
        X_pca_sample = X_pca[sample_indices]
        y_sample = y.iloc[sample_indices]
    else:
        X_pca_sample = X_pca
        y_sample = y
    
    pca_transformed_df = pd.DataFrame(
        X_pca_sample,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_transformed_df['isFraud'] = y_sample.values
    transformed_file = save_csv(pca_transformed_df, 'pca_transformed_sample.csv', pca_dir)
    logger.info(f"Saved PCA transformed data sample to {transformed_file}")
    
    # Clean up memory
    del X_scaled, X_pca, pca_transformed_df
    gc.collect()
    
    logger.info(f"PCA analysis completed: {n_components} components explain {cumulative_variance[-1]:.2%} of variance")
    logger.info(f"Priority features identified: {priority_features}")
    
    return results
