"""
Utilities for statistical analysis of IEEE-CIS dataset.
"""

# Force CPU mode - disable all GPU functionality
import os
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUPY_DISABLE_HIP'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import gc  # For garbage collection


def setup_logging(log_dir):
    """Set up logging to file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"ieee_cis_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('IEEECISStatisticalAnalysis')
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


def load_dataset(file_path):
    """Load the IEEE-CIS dataset with memory optimizations."""
    try:
        # Import pyarrow if available for better memory efficiency
        try:
            import pyarrow.parquet as pq
            print("Using pyarrow for memory-efficient Parquet reading")
            table = pq.read_table(file_path)
            df = table.to_pandas(split_blocks=True, self_destruct=True)
            del table  # Free memory
            gc.collect()
        except ImportError:
            # Fall back to pandas if pyarrow not available
            print("Pyarrow not available, using pandas for Parquet reading")
            df = pd.read_parquet(file_path)
        
        print(f"Initial dataframe memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize memory usage of DataFrame."""
    print("Optimizing DataFrame memory usage...")
    
    initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Optimize integer columns
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:  # Only positive values
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:  # Positive and negative values
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    # Optimize float columns to float32
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        if df[col].isnull().sum() == 0:  # Skip if has NaN values
            df[col] = df[col].astype('float32')
    
    # Convert object columns with low cardinality to categorical
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Final memory usage: {final_memory:.2f} MB (reduction of {initial_memory - final_memory:.2f} MB)")
    
    # Force garbage collection
    gc.collect()
    
    return df


def prepare_data_for_analysis(df, target_col='isFraud'):
    """Prepare data for statistical analysis."""
    print("Preparing data for analysis...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Time', target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Fill NaN values with 0 (IEEE-CIS shouldn't have NaN, but safety check)
    X = X.fillna(0)
    
    print(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def save_json(data, filename, output_dir):
    """Save data to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath


def save_csv(df, filename, output_dir):
    """Save DataFrame to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=True)
    
    return filepath


def generate_graph_construction_insights(results_dict, analysis_type):
    """Generate graph construction insights for a specific analysis type."""
    insights = {
        "node_attributes": {},
        "edge_construction": {},
        "subgraph_analysis": {}
    }
    
    if analysis_type == "correlation":
        insights = {
            "node_attributes": {
                "priority_features": results_dict.get('priority_features', []),
                "feature_weights": results_dict.get('feature_weights', {})
            },
            "edge_construction": {
                "similarity_features": results_dict.get('similarity_features', []),
                "correlation_threshold": 0.8,
                "recommended_similarity_metrics": ["cosine", "correlation"]
            },
            "subgraph_analysis": {
                "high_correlation_clusters": results_dict.get('high_correlation_pairs', [])
            }
        }
    
    elif analysis_type == "pca":
        insights = {
            "node_attributes": {
                "priority_features": results_dict.get('priority_features', []),
                "dimensionality_reduction": {
                    "recommended_components": results_dict.get('n_components', 10),
                    "variance_threshold": 0.95
                }
            },
            "edge_construction": {
                "pca_based_similarity": {
                    "use_principal_components": True,
                    "n_components_for_similarity": min(5, results_dict.get('n_components', 10))
                }
            },
            "subgraph_analysis": {
                "component_based_grouping": {
                    "high_variance_components": results_dict.get('high_variance_components', [])
                }
            }
        }
    
    return insights


def consolidate_all_results(individual_results, dataset_info):
    """Consolidate results from all statistical analyses."""
    print("Consolidating results from all analyses...")
    
    # Extract priority features from each analysis
    all_features = set()
    feature_consensus = {}
    
    for analysis_name, results in individual_results.items():
        if 'graph_construction_insights' in results:
            insights = results['graph_construction_insights']
            if 'node_attributes' in insights:
                priority_features = insights['node_attributes'].get('priority_features', [])
                for feature in priority_features:
                    all_features.add(feature)
                    if feature not in feature_consensus:
                        feature_consensus[feature] = 0
                    
                    # Standard weight for all analyses
                    weight = 1
                    feature_consensus[feature] += weight
    
    # Sort features by consensus score
    consensus_features = sorted(feature_consensus.items(), key=lambda x: x[1], reverse=True)[:12]
    final_priority_features = [feat for feat, _ in consensus_features]
    
    # Generate integrated recommendations
    integrated_recommendations = {
        "node_attributes": {
            "final_priority_features": final_priority_features,
            "feature_consensus_scores": dict(consensus_features),
            "recommended_node_features": {
                "critical": [feat for feat, score in consensus_features if score >= 4][:5],
                "important": [feat for feat, score in consensus_features if score == 3][:5],
                "moderate": [feat for feat, score in consensus_features if score == 2][:5]
            }
        },
        "edge_construction": {
            "multi_strategy_approach": {
                "similarity_based": {
                    "features": final_priority_features[:8],
                    "similarity_metrics": ["cosine", "euclidean"],
                    "threshold_range": [0.7, 0.9]
                },
                "cluster_based": {
                    "use_cluster_membership": True,
                    "same_cluster_threshold": 0.8,
                    "cross_cluster_threshold": 0.3
                },
                "anomaly_based": {
                    "anomaly_boost_factor": 2.0,
                    "anomaly_threshold": individual_results.get('anomaly_detection', {}).get('top_anomaly_threshold', 0.5)
                },
                "temporal_based": {
                    "time_window_seconds": 3600,
                    "temporal_decay": True
                }
            }
        },
        "graph_parameters": {
            "node_selection": {
                "include_all_transactions": True,
                "feature_encoding": "multi_hot"
            },
            "edge_weighting": {
                "combine_strategies": True,
                "weight_normalization": "min_max",
                "edge_pruning_threshold": 0.1
            }
        }
    }
    
    # Create consolidated results
    consolidated_results = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset_info": dataset_info,
        "individual_analyses": individual_results,
        "integrated_graph_construction_recommendations": integrated_recommendations,
        "summary": {
            "total_features_analyzed": len(all_features),
            "consensus_features": len(final_priority_features),
            "high_consensus_features": len([f for f, s in consensus_features if s >= 3]),
            "recommended_strategies": 4  # similarity, cluster, anomaly, temporal
        }
    }
    
    return consolidated_results
