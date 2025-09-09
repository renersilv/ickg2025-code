"""
Cluster analysis for IEEE-CIS fraud detection dataset.
"""

# Force CPU mode - disable all GPU functionality
import os
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUPY_DISABLE_HIP'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils import save_json, save_csv
from feature_config import get_features_for_analysis, validate_features_in_dataframe
import gc


def cluster_analysis(X, y, log_dir, logger, n_clusters=5, random_state=42):
    """
    Perform cluster analysis on the dataset.
    
    Args:
        X: DataFrame with features
        y: Series with target variable (fraud indicator)
        log_dir: Directory to save outputs
        logger: Logger object
        n_clusters: Number of clusters for K-means
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with analysis results and cluster assignments DataFrame
    """
    logger.info("Starting cluster analysis...")
    
    # Create directory for cluster analysis outputs
    cluster_dir = os.path.join(log_dir, 'cluster_analysis')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Get optimized feature list for clustering analysis
    target_features = get_features_for_analysis('clustering', exclude_time=True)
    valid_features, missing_features = validate_features_in_dataframe(X, target_features)
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    logger.info(f"Using {len(valid_features)} features for clustering (from {len(target_features)} expected)")
    
    # Prepare data with optimized memory usage
    logger.info("Preparing data for clustering...")
    
    # Filter to keep only numeric columns for clustering
    X_analysis = X[valid_features]
    numeric_columns = X_analysis.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Filtering to {len(numeric_columns)} numeric columns for clustering analysis")
    
    X_clean = X_analysis[numeric_columns].fillna(0)
    
    # Convert to float32 for faster computation (if not already)
    if X_clean.dtypes.iloc[0] == 'float64':
        X_clean = X_clean.astype('float32')
    
    # Standardize data
    logger.info("Standardizing features for clustering...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Apply K-Means clustering with optimized parameters
    logger.info(f"Applying K-Means with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=3,  # Reduced from 10 to 3 for speed
        max_iter=100,  # Reduced from 300 to 100
        algorithm='elkan'  # Faster algorithm for dense data
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create DataFrame with cluster assignments
    cluster_assignments = pd.DataFrame({
        'cluster_id': cluster_labels,
        'isFraud': y.values
    }, index=X.index)
    
    # Analyze clusters by fraud rate
    cluster_stats = cluster_assignments.groupby('cluster_id').agg({
        'isFraud': ['count', 'sum', 'mean']
    }).round(4)
    cluster_stats.columns = ['total_count', 'fraud_count', 'fraud_rate']
    
    # Identify high-risk clusters
    high_risk_threshold = 0.05  # Adjusted for IEEE-CIS's higher fraud rate (~3.5%)
    high_risk_clusters = cluster_stats[cluster_stats['fraud_rate'] >= high_risk_threshold].index.tolist()
    
    # Analyze distinctive features of clusters
    cluster_centroids = kmeans.cluster_centers_
    distinctive_features = []
    
    # Make sure we only use the columns that were actually used in clustering
    used_features = X_clean.columns.tolist()  # Use the columns from the DataFrame before scaling
    
    for cluster_id in range(n_clusters):
        centroid = cluster_centroids[cluster_id]
        # Features with extreme values (above 1 std from center)
        extreme_features = []
        for i, (feat, val) in enumerate(zip(used_features, centroid)):
            if abs(val) > 1.0:  # Normalized value > 1 std
                extreme_features.append(feat)
        distinctive_features.extend(extreme_features[:3])  # Top 3 per cluster
    
    # Remove duplicates and limit
    distinctive_features = list(set(distinctive_features))[:10]
    
    # Graph Construction Insights
    graph_insights = {
        "node_attributes": {
            "distinctive_features": distinctive_features,
            "cluster_membership": "cluster_id"
        },
        "edge_construction": {
            "cluster_based_edges": {
                "same_cluster_threshold": 0.8,
                "cross_cluster_threshold": 0.3,
                "high_risk_cluster_boost": 1.5
            }
        },
        "subgraph_analysis": {
            "high_risk_clusters": high_risk_clusters,
            "cluster_isolation": {
                int(cluster_id): float(stats['fraud_rate']) 
                for cluster_id, stats in cluster_stats.iterrows()
            }
        }
    }
    
    # Calculate cluster quality metrics (optimized sampling for large datasets)
    logger.info("Calculating cluster quality metrics...")
    try:
        cluster_quality = {}
        
        # Use sampling for large datasets to speed up quality metrics
        if len(X_scaled) > 50000:
            logger.info("Large dataset detected - using sampling for quality metrics")
            sample_size = min(10000, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_indices]
            labels_sample = cluster_labels[sample_indices]
        else:
            X_sample = X_scaled
            labels_sample = cluster_labels
        
        cluster_quality['silhouette_score'] = silhouette_score(X_sample, labels_sample)
        cluster_quality['calinski_harabasz_score'] = calinski_harabasz_score(X_sample, labels_sample)
        cluster_quality['davies_bouldin_score'] = davies_bouldin_score(X_sample, labels_sample)
        
        logger.info(f"Cluster quality metrics:")
        logger.info(f"  - Silhouette Score: {cluster_quality['silhouette_score']:.4f}")
        logger.info(f"  - Calinski-Harabasz Score: {cluster_quality['calinski_harabasz_score']:.2f}")
        logger.info(f"  - Davies-Bouldin Score: {cluster_quality['davies_bouldin_score']:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate cluster quality metrics: {e}")
        cluster_quality = {}
        cluster_quality = {}
    
    # Compile results
    results = {
        'cluster_assignments_file': 'cluster_assignments.csv',
        'cluster_statistics': cluster_stats.to_dict(),
        'high_risk_clusters': high_risk_clusters,
        'n_clusters': n_clusters,
        'high_risk_threshold': high_risk_threshold,
        'cluster_quality_metrics': cluster_quality,
        'graph_construction_insights': graph_insights
    }
    
    # Save cluster assignments
    assignments_file = save_csv(cluster_assignments, 'cluster_assignments.csv', cluster_dir)
    logger.info(f"Saved cluster assignments to {assignments_file}")
    
    # Save cluster statistics
    stats_file = save_csv(cluster_stats, 'cluster_statistics.csv', cluster_dir)
    logger.info(f"Saved cluster statistics to {stats_file}")
    
    # Save cluster centroids
    centroids_df = pd.DataFrame(
        cluster_centroids.T,
        columns=[f'Cluster_{i}' for i in range(n_clusters)],
        index=used_features  # Use used_features (the actual numeric features used in clustering)
    )
    centroids_file = save_csv(centroids_df, 'cluster_centroids.csv', cluster_dir)
    logger.info(f"Saved cluster centroids to {centroids_file}")
    
    # Analyze cluster-specific feature characteristics
    logger.info("Analyzing cluster-specific features...")
    cluster_feature_analysis = {}
    
    # Pre-calculate global means once
    global_means = X_clean.mean()
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()
        
        if cluster_size > 10:  # Only analyze clusters with sufficient data
            # Use vectorized operations for speed
            cluster_data = X_clean[cluster_mask]
            cluster_means = cluster_data.mean()
            
            # Calculate deviation from global mean (vectorized)
            deviations = (cluster_means - global_means) / (global_means.abs() + 1e-10)
            
            # Get top 3 deviating features (reduced from 5 for speed)
            top_deviations = deviations.abs().nlargest(3)
            cluster_feature_analysis[cluster_id] = {
                'top_deviating_features': top_deviations.index.tolist(),
                'deviations': top_deviations.to_dict()
            }
    
    # Save cluster feature analysis
    feature_analysis_file = save_json(cluster_feature_analysis, 'cluster_feature_analysis.json', cluster_dir)
    logger.info(f"Saved cluster feature analysis to {feature_analysis_file}")
    
    # Clean up memory
    del X_scaled, cluster_centroids
    if 'X_sample' in locals():
        del X_sample, labels_sample
    gc.collect()
    
    logger.info(f"Cluster analysis completed: {len(high_risk_clusters)} high-risk clusters identified")
    logger.info(f"High-risk clusters: {high_risk_clusters}")
    logger.info(f"Distinctive features: {distinctive_features}")
    
    return results, cluster_assignments
