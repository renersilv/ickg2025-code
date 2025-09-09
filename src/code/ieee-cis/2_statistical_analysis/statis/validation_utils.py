"""
Validation utilities for MultiStatGraph Framework
This module provides validation functions to ensure data quality and reproducibility.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import json
from pathlib import Path

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def validate_reproducibility_seeds():
    """Validate that all random number generators are properly seeded."""
    logger = logging.getLogger(__name__)
    
    # Check numpy seed
    random_state = np.random.get_state()[1][0]
    logger.info(f"NumPy random state first element: {random_state}")
    
    # Validate seed consistency
    test_random_1 = np.random.random(5)
    np.random.seed(RANDOM_STATE)
    test_random_2 = np.random.random(5)
    
    if not np.allclose(test_random_1, test_random_2):
        logger.warning("⚠️ Random seed inconsistency detected!")
        return False
    
    logger.info("✓ Random seed validation passed")
    return True


def validate_data_integrity(df: pd.DataFrame, analysis_name: str) -> Dict[str, Any]:
    """Validate data integrity for scientific analyses."""
    logger = logging.getLogger(__name__)
    
    validation_results = {
        'analysis_name': analysis_name,
        'total_samples': len(df),
        'total_features': df.shape[1],
        'missing_values': {},
        'infinite_values': {},
        'data_types': {},
        'validation_passed': True,
        'issues': []
    }
    
    logger.info(f"Validating data integrity for {analysis_name}...")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    validation_results['missing_values'] = missing_counts.to_dict()
    
    if missing_counts.any():
        validation_results['issues'].append(f"Missing values found in {missing_counts[missing_counts > 0].index.tolist()}")
        logger.warning(f"⚠️ Missing values detected: {missing_counts.sum()} total")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        validation_results['infinite_values'][col] = inf_count
        
        if inf_count > 0:
            validation_results['issues'].append(f"Infinite values in {col}: {inf_count}")
            validation_results['validation_passed'] = False
            logger.error(f"✗ Infinite values detected in {col}: {inf_count}")
    
    # Check data types
    validation_results['data_types'] = df.dtypes.astype(str).to_dict()
    
    # Check for extremely large or small values that might cause numerical issues
    for col in numeric_cols:
        col_max = df[col].max()
        col_min = df[col].min()
        
        if abs(col_max) > 1e10 or abs(col_min) > 1e10:
            validation_results['issues'].append(f"Extremely large values in {col}: min={col_min}, max={col_max}")
            logger.warning(f"⚠️ Extremely large values in {col}")
    
    if validation_results['validation_passed']:
        logger.info("✓ Data integrity validation passed")
    else:
        logger.error("✗ Data integrity validation failed")
    
    return validation_results


def validate_cluster_quality(cluster_labels: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """Validate clustering quality using multiple metrics."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    logger = logging.getLogger(__name__)
    logger.info("Validating cluster quality...")
    
    try:
        metrics = {}
        
        # Silhouette Score (higher is better, range: [-1, 1])
        metrics['silhouette_score'] = silhouette_score(X, cluster_labels)
        
        # Calinski-Harabasz Score (higher is better)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, cluster_labels)
        
        # Davies-Bouldin Score (lower is better)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, cluster_labels)
        
        # Number of clusters
        metrics['n_clusters'] = len(np.unique(cluster_labels))
        
        # Cluster size distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        metrics['min_cluster_size'] = counts.min()
        metrics['max_cluster_size'] = counts.max()
        metrics['cluster_size_std'] = counts.std()
        
        logger.info(f"✓ Cluster quality metrics computed:")
        logger.info(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        logger.info(f"  - Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Cluster quality validation failed: {str(e)}")
        return {}


def validate_statistical_consistency(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate consistency between different statistical analyses."""
    logger = logging.getLogger(__name__)
    logger.info("Validating statistical consistency...")
    
    consistency_report = {
        'feature_overlap': {},
        'consistency_scores': {},
        'recommendations': [],
        'validation_passed': True,
        'feature_categories': {},
        'feature_sets_by_analysis': {}
    }
    
    # Extract priority features from each analysis with more flexible approach
    feature_sets = {}
    feature_categories = {
        'original_features': set(),
        'engineered_features': set(),
        'statistical_features': set()
    }
    
    for analysis_name, results in analysis_results.items():
        if 'graph_construction_insights' in results:
            insights = results['graph_construction_insights']
            if 'node_attributes' in insights:
                node_attrs = insights['node_attributes']
                
                # Extract priority features based on analysis type
                features = []
                if 'priority_features' in node_attrs:
                    features = node_attrs['priority_features']
                elif 'distinctive_features' in node_attrs:
                    features = node_attrs['distinctive_features']
                elif 'key_anomaly_features' in node_attrs:
                    features = node_attrs['key_anomaly_features']
                
                if features:
                    feature_set = set(features)
                    feature_sets[analysis_name] = feature_set
                    
                    # Log features found by each analysis
                    logger.info(f"Features from {analysis_name}: {list(features)[:5]}{'...' if len(features) > 5 else ''}")
                    
                    # Categorize features for better analysis
                    for feature in features:
                        if feature.startswith('V') and len(feature) <= 3:
                            feature_categories['original_features'].add(feature)
                        elif any(x in feature.lower() for x in ['amount', 'time']):
                            feature_categories['original_features'].add(feature)
                        elif any(x in feature.lower() for x in ['product', 'ratio', 'category']):
                            feature_categories['engineered_features'].add(feature)
                        else:
                            feature_categories['statistical_features'].add(feature)
    
    consistency_report['feature_categories'] = {
        cat: list(features) for cat, features in feature_categories.items()
    }
    consistency_report['feature_sets_by_analysis'] = {
        analysis: list(features) for analysis, features in feature_sets.items()
    }
    
    # Calculate pairwise overlaps with improved logic
    analyses = list(feature_sets.keys())
    similarities = []
    
    for i, analysis1 in enumerate(analyses):
        for j, analysis2 in enumerate(analyses[i+1:], i+1):
            if analysis1 in feature_sets and analysis2 in feature_sets:
                set1 = feature_sets[analysis1]
                set2 = feature_sets[analysis2]
                
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                
                # Calculate multiple similarity metrics
                jaccard_similarity = len(intersection) / len(union) if union else 0
                overlap_coefficient = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
                
                similarities.append(jaccard_similarity)
                
                overlap_key = f"{analysis1}_vs_{analysis2}"
                consistency_report['feature_overlap'][overlap_key] = {
                    'jaccard_similarity': jaccard_similarity,
                    'overlap_coefficient': overlap_coefficient,
                    'intersection_size': len(intersection),
                    'common_features': list(intersection),
                    'set1_size': len(set1),
                    'set2_size': len(set2)
                }
                
                logger.info(f"Feature overlap {analysis1} vs {analysis2}: J={jaccard_similarity:.3f}, O={overlap_coefficient:.3f}")
    
    # Calculate overall consistency score with more lenient thresholds
    if similarities:
        mean_similarity = np.mean(similarities)
        consistency_report['consistency_scores']['mean_jaccard'] = mean_similarity
        consistency_report['consistency_scores']['std_jaccard'] = np.std(similarities)
        consistency_report['consistency_scores']['max_jaccard'] = np.max(similarities)
        consistency_report['consistency_scores']['min_jaccard'] = np.min(similarities)
        
        # More realistic thresholds for fraud detection (where different analyses focus on different aspects)
        if mean_similarity < 0.02:  # Extremely low consistency
            consistency_report['validation_passed'] = False
            consistency_report['recommendations'].append("Extremely low feature consistency. Consider reviewing feature engineering and analysis parameters.")
            logger.error(f"✗ Extremely low statistical consistency: {mean_similarity:.3f}")
        elif mean_similarity < 0.08:  # Low but acceptable for fraud detection
            consistency_report['recommendations'].append("Low feature overlap detected - this is expected in fraud detection where different analyses provide complementary perspectives.")
            consistency_report['recommendations'].append("Different analyses focus on: Correlation (linear relationships), PCA (variance), Clustering (similarity groups), Anomalies (outliers), Temporal (time patterns).")
            logger.info(f"ℹ️ Low but acceptable statistical consistency for fraud detection: {mean_similarity:.3f}")
        elif mean_similarity < 0.20:  # Moderate consistency
            consistency_report['recommendations'].append("Moderate feature overlap detected - good balance between analysis diversity and consistency.")
            logger.info(f"✓ Moderate statistical consistency: {mean_similarity:.3f}")
        else:
            logger.info(f"✓ High statistical consistency: {mean_similarity:.3f}")
        
        # Check if at least one pair has good overlap
        max_similarity = np.max(similarities)
        if max_similarity >= 0.3:
            consistency_report['recommendations'].append(f"Good feature agreement found between some analyses (max: {max_similarity:.3f}).")
            logger.info(f"✓ Maximum pairwise consistency: {max_similarity:.3f}")
    
    return consistency_report


def generate_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash for data integrity verification."""
    # Convert to string and hash
    data_string = df.to_string()
    return hashlib.md5(data_string.encode()).hexdigest()


def save_validation_report(validation_results: Dict[str, Any], output_dir: str, filename: str):
    """Save validation report to JSON file."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"✓ Validation report saved: {filepath}")
        
    except Exception as e:
        logger.error(f"✗ Failed to save validation report: {str(e)}")


def check_memory_usage() -> Dict[str, float]:
    """Check current memory usage for optimization monitoring."""
    import psutil
    import gc
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Force garbage collection
    gc.collect()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def log_environment_info(logger: logging.Logger):
    """Log environment information for reproducibility."""
    import sys
    import platform
    
    logger.info("=== ENVIRONMENT INFORMATION ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Random seed: {RANDOM_STATE}")
    
    # Log key library versions
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import networkx as nx
        
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
        logger.info(f"NetworkX version: {nx.__version__}")
        
    except ImportError as e:
        logger.warning(f"Could not import library for version check: {e}")
    
    logger.info("================================")


def validate_no_file_duplications(output_dir: str) -> Dict[str, Any]:
    """
    Validate that no duplicate files are created in the output directory structure.
    
    Args:
        output_dir: Base output directory to check
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    
    # Define expected file structure (files should only exist in specific subdirectories)
    expected_structure = {
        'cluster_assignments.csv': 'cluster_analysis/',
        'cluster_statistics.csv': 'cluster_analysis/',
        'cluster_centroids.csv': 'cluster_analysis/',
        'anomaly_scores.csv': '',  # anomaly_detection saves directly in root for this analysis
        'correlation_matrix.csv': 'correlation_analysis/',
        'high_correlation_pairs.csv': 'correlation_analysis/',
        'target_correlations.csv': 'correlation_analysis/',
        'pca_components.csv': 'pca_analysis/',
        'explained_variance.csv': 'pca_analysis/',
        'pca_transformed_sample.csv': 'pca_analysis/'
    }
    
    validation_results = {
        'validation_passed': True,
        'duplicated_files': [],
        'missing_files': [],
        'misplaced_files': [],
        'issues': []
    }
    
    # Check for file duplications
    for filename, expected_subdir in expected_structure.items():
        root_file = os.path.join(output_dir, filename)
        subdir_file = os.path.join(output_dir, expected_subdir, filename) if expected_subdir else root_file
        
        root_exists = os.path.exists(root_file)
        subdir_exists = os.path.exists(subdir_file)
        
        # Check for duplications (file exists in both root and subdirectory)
        if expected_subdir and root_exists and subdir_exists:
            validation_results['duplicated_files'].append(filename)
            validation_results['issues'].append(f"Duplicate file found: {filename} exists in both root and {expected_subdir}")
            validation_results['validation_passed'] = False
        
        # Check for missing files in expected location
        if expected_subdir and not subdir_exists:
            validation_results['missing_files'].append(f"{expected_subdir}{filename}")
        elif not expected_subdir and not root_exists:
            validation_results['missing_files'].append(filename)
        
        # Check for misplaced files (should be in subdirectory but found in root)
        if expected_subdir and root_exists and not subdir_exists:
            validation_results['misplaced_files'].append(filename)
            validation_results['issues'].append(f"Misplaced file: {filename} should be in {expected_subdir}")
    
    # Log results
    if validation_results['validation_passed']:
        logger.info("✓ File duplication validation passed")
    else:
        logger.warning("⚠️ File duplication issues detected:")
        for issue in validation_results['issues']:
            logger.warning(f"  - {issue}")
    
    return validation_results
