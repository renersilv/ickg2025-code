"""
Feature configuration for IEEE-CIS dataset analysis.
Defines consistent feature categories and filtering rules across all modules.
Updated to match the actual parquet dataset structure with 165 columns.
"""

# Features to always exclude from analysis (target variable)
EXCLUDED_FEATURES = ['isFraud']  # isFraud is the target variable

# Features that require special handling for temporal analysis
TEMPORAL_CONTEXT_FEATURES = ['TransactionDT', 'transaction_timestamp', 'trans_hour', 'trans_weekday', 'is_night', 'hour_sin']

# Categorized features for specialized analysis based on actual dataset structure
FEATURE_CATEGORIES = {
    # Original V features from dataset (54 total)
    'v_features': [
        'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
        'V12', 'V13', 'V14', 'V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
        'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V33', 'V34', 'V35',
        'V36', 'V37', 'V38', 'V39', 'V40', 'V42', 'V43', 'V44', 'V45', 'V46',
        'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
        'V57', 'V58', 'V59'
    ],
    
    # Enhanced V features (derived)
    'v_derived_features': [
        'v_mean', 'v_missing_count', 'V24_V38_product', 'v_profile_cluster'
    ],
    
    # Transaction amount features
    'amount_features': [
        'TransactionAmt', 'amount_log', 'is_small_amount', 'is_round_amount', 
        'amount_rank', 'is_amount_extreme_high', 'dt_per_amt'
    ],
    
    # Temporal features
    'temporal_features': [
        'TransactionDT', 'transaction_timestamp', 'trans_hour', 'trans_weekday', 
        'is_night', 'hour_sin', 'is_rare_hour'
    ],
    
    # Card features (original + derived)
    'card_features': [
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'card1_is_missing', 'card2_is_missing', 'card3_is_missing', 
        'card4_is_missing', 'card5_is_missing', 'card6_is_missing',
        'card4_visa', 'card4_mastercard', 'card6_debit', 'card6_credit',
        'card_missing_count', 'card1_card2_combined'
    ],
    
    # Address features
    'address_features': [
        'addr1', 'addr2', 'addr1_is_missing', 'addr2_is_missing',
        'addr_missing_count', 'addr1_addr2_same'
    ],
    
    # Distance features
    'distance_features': [
        'dist1', 'dist2', 'dist1_is_missing', 'dist1_category', 'dist1_is_zero',
        'dist2_is_missing', 'dist2_category', 'dist2_is_zero', 'dist_sum', 'dist_missing_count'
    ],
    
    # Email domain features
    'email_features': [
        'P_emaildomain', 'R_emaildomain', 'P_emaildomain_is_missing',
        'P_emaildomain_gmail_com', 'P_emaildomain_yahoo_com', 'P_emaildomain_hotmail_com',
        'R_emaildomain_is_missing', 'R_emaildomain_gmail_com', 'R_emaildomain_hotmail_com',
        'R_emaildomain_anonymous_com', 'P_R_email_same', 'email_missing_count'
    ],
    
    # C features (count features)
    'c_features': [
        'C2', 'C3', 'C4', 'C5', 'C7', 'C8', 'C9', 'C10', 'C12', 'C13', 'C14'
    ],
    
    # C derived features
    'c_derived_features': [
        'c_missing_count'
    ],
    
    # D features (time delta features)
    'd_features': [
        'D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D13', 'D14', 'D15'
    ],
    
    # D derived features
    'd_derived_features': [
        'd_missing_count'
    ],
    
    # M features (match features - original)
    'm_features': [
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ],
    
    # M derived features
    'm_derived_features': [
        'm_missing_count', 'M1_is_missing', 'M1_T', 'M2_is_missing', 'M2_T', 'M3_is_missing', 'M3_T'
    ],
    
    # Product features
    'product_features': [
        'ProductCD', 'product_is_missing', 'ProductCD_W', 'ProductCD_C', 'ProductCD_R'
    ],
    
    # Global aggregation features
    'aggregation_features': [
        'total_missing_features'
    ]
}

# All available features (excluding target)
ALL_ANALYSIS_FEATURES = (
    FEATURE_CATEGORIES['v_features'] + 
    FEATURE_CATEGORIES['v_derived_features'] + 
    FEATURE_CATEGORIES['amount_features'] + 
    FEATURE_CATEGORIES['temporal_features'] + 
    FEATURE_CATEGORIES['card_features'] +
    FEATURE_CATEGORIES['address_features'] +
    FEATURE_CATEGORIES['distance_features'] +
    FEATURE_CATEGORIES['email_features'] +
    FEATURE_CATEGORIES['c_features'] +
    FEATURE_CATEGORIES['c_derived_features'] +
    FEATURE_CATEGORIES['d_features'] +
    FEATURE_CATEGORIES['d_derived_features'] +
    FEATURE_CATEGORIES['m_features'] +
    FEATURE_CATEGORIES['m_derived_features'] +
    FEATURE_CATEGORIES['product_features'] +
    FEATURE_CATEGORIES['aggregation_features']
)

# Remove duplicates and excluded features
ALL_ANALYSIS_FEATURES = list(set(ALL_ANALYSIS_FEATURES) - set(EXCLUDED_FEATURES))

# Legacy feature mapping for backward compatibility
LEGACY_FEATURE_MAPPING = {
    'Amount': 'TransactionAmt',
    'Time': 'TransactionDT',
    'Class': 'isFraud'
}

def get_features_for_analysis(analysis_type='general', exclude_time=True):
    """
    Get appropriate features for specific analysis types.
    
    Args:
        analysis_type: Type of analysis ('general', 'temporal', 'correlation', etc.)
        exclude_time: Whether to exclude TransactionDT feature (default True for most analyses)
    
    Returns:
        List of features appropriate for the analysis
    """
    if analysis_type == 'temporal':
        # For temporal analysis, include all temporal features
        return [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES]
    
    elif analysis_type == 'correlation':
        # For correlation, include all numeric features
        features = [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES]
        if exclude_time:
            features = [f for f in features if f != 'TransactionDT']
        return features
    
    elif analysis_type == 'clustering':
        # For clustering, exclude TransactionDT but include all other features
        return [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES + (['TransactionDT'] if exclude_time else [])]
    
    elif analysis_type == 'anomaly':
        # For anomaly detection, include all features except TransactionDT
        return [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES + (['TransactionDT'] if exclude_time else [])]
    
    elif analysis_type == 'pca':
        # For PCA, use all numeric features except TransactionDT for better results
        features = [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES]
        if exclude_time:
            features = [f for f in features if f != 'TransactionDT']
        return features
    
    else:  # 'general'
        # Default: all features except TransactionDT and excluded ones
        features = [f for f in ALL_ANALYSIS_FEATURES if f not in EXCLUDED_FEATURES]
        if exclude_time:
            features = [f for f in features if f != 'TransactionDT']
        return features


def validate_features_in_dataframe(df, features_list):
    """
    Validate that all specified features exist in the DataFrame.
    
    Args:
        df: DataFrame to check
        features_list: List of feature names to validate
    
    Returns:
        Tuple of (valid_features, missing_features)
    """
    available_features = set(df.columns)
    valid_features = [f for f in features_list if f in available_features]
    missing_features = [f for f in features_list if f not in available_features]
    
    return valid_features, missing_features


def get_feature_info():
    """Get summary information about feature categories."""
    info = {
        'total_features': len(ALL_ANALYSIS_FEATURES),
        'by_category': {cat: len(features) for cat, features in FEATURE_CATEGORIES.items()},
        'excluded_features': EXCLUDED_FEATURES,
        'temporal_context_features': TEMPORAL_CONTEXT_FEATURES,
        'legacy_mapping': LEGACY_FEATURE_MAPPING,
        'dataset_columns': len(ALL_ANALYSIS_FEATURES) + len(EXCLUDED_FEATURES)  # Total columns including target
    }
    return info


def apply_legacy_mapping(feature_list):
    """
    Apply legacy feature name mapping for backward compatibility.
    
    Args:
        feature_list: List of feature names that may contain legacy names
        
    Returns:
        List of features with legacy names mapped to new names
    """
    return [LEGACY_FEATURE_MAPPING.get(feature, feature) for feature in feature_list]


def get_features_by_category(category_name):
    """
    Get features from a specific category.
    
    Args:
        category_name: Name of the feature category
        
    Returns:
        List of features in the specified category, or empty list if category doesn't exist
    """
    return FEATURE_CATEGORIES.get(category_name, [])
