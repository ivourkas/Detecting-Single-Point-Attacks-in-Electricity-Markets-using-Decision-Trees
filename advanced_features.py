def engineer_advanced_features(df):
    """
    Apply advanced feature engineering techniques to improve attack detection
    
    """
    print("\n=== Performing advanced feature engineering ===")
    import numpy as np
    import pandas as pd
    
    # Store original shape for reporting
    original_shape = df.shape
    
    # ====================== DOMAIN-SPECIFIC FEATURE ENGINEERING ======================
    print("\n=== Performing domain-specific feature engineering ===")

    # 1. Basic session and time features
    df['day_in_ssn'] = df.groupby('ssn').cumcount() + 1  # Start at Day 1

    # 2. Calculate OPF sensitivity features
    # These help identify attacks on specific OPF parameters

    # 2.1 Cost function sensitivity indicators
    # Create features that might detect manipulation of generation costs (Type 4 attack)
    df['fval_per_unit_load'] = df['fval'] / df.groupby('ssn')['fval'].transform('mean')
    df['fval_normalized_by_ssn'] = df.groupby('ssn')['fval'].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))

    # 2.2 Ramp rate indicators 
    # Create features that might detect manipulation of ramp rates (Type 1 attack)
    df['fval_change'] = df.groupby('ssn')['fval'].diff().fillna(0)
    df['fval_change_rate'] = df['fval_change'] / df['fval'].shift(1).fillna(1)
    df['fval_acceleration'] = df.groupby('ssn')['fval_change'].diff().fillna(0)

    # Create rolling metrics to detect unusual ramp behavior
    for window in [2, 3, 5]:
        # Rolling standard deviation of changes (volatility)
        df[f'fval_change_std_{window}d'] = df.groupby('ssn')['fval_change'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
        
        # Maximum change in the window
        df[f'fval_max_change_{window}d'] = df.groupby('ssn')['fval_change'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max().fillna(0))
        
        # Minimum change in the window
        df[f'fval_min_change_{window}d'] = df.groupby('ssn')['fval_change'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min().fillna(0))

    # 2.3 Generation limit indicators
    # Create features that might detect manipulation of upper/lower limits (Type 2 & 3 attacks)
    df['fval_peak_ratio'] = df['fval'] / df.groupby('ssn')['fval'].transform('max')
    df['fval_trough_ratio'] = df['fval'] / df.groupby('ssn')['fval'].transform('min')

    # Use quantiles to detect limits being approached
    df['fval_quantile_in_ssn'] = df.groupby('ssn')['fval'].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop').astype(float))

    # 3. Create specialized detection features for each attack type
    # These target the specific mechanisms of each attack type

    # 3.1 Features for detecting sudden value changes
    # Look for discontinuities in how fast values can change
    df['change_threshold_exceeded'] = (df['fval_change'].abs() > 
                                      df.groupby('ssn')['fval_change'].transform('std')).astype(int)

    # 3.2 Features for detecting boundary approaches
    # Look for values that approach the extremes of observed ranges
    df['max_boundary_proximity'] = 1 - (df['fval'] / df.groupby('ssn')['fval'].transform('max'))
    df['min_boundary_proximity'] = (df['fval'] / df.groupby('ssn')['fval'].transform('min')) - 1

    # 3.3 Features for detecting cost manipulation (Type 4)
    # Look for cost-inefficient dispatches that shouldn't happen under normal cost functions
    df['cost_efficiency'] = df['fval'] / df.groupby('ssn')['fval'].transform('mean')
    df['cost_anomaly_score'] = df.groupby('ssn')['cost_efficiency'].transform(
        lambda x: (x - x.mean()).abs() / (x.std() if x.std() != 0 else 1))

    # 4. Inter-session comparison features
    # These help identify if a session is behaving differently from others

    # 4.1 Calculate average fval across all sessions for each day
    day_avg = df.groupby('day_in_ssn')['fval'].transform('mean')
    day_std = df.groupby('day_in_ssn')['fval'].transform('std')

    # 4.2 Compare each session's values to the average across all sessions
    df['fval_day_deviation'] = (df['fval'] - day_avg) / (day_std if day_std.any() != 0 else 1)
    df['fval_day_pct_diff'] = (df['fval'] - day_avg) / day_avg

    # 5. Statistical anomaly detection features
    # These help identify outliers regardless of attack mechanism

    # 5.1 Z-scores and modified Z-scores for more robust outlier detection
    df['fval_median_dev'] = df.groupby('ssn')['fval'].transform(
        lambda x: (x - x.median()) / (x.max() - x.min() if (x.max() - x.min()) != 0 else 1))

    # 6. Cumulative features to detect subtle long-term manipulations
    df['fval_cumsum'] = df.groupby('ssn')['fval'].cumsum()
    df['fval_cummean'] = df.groupby('ssn')['fval'].expanding().mean().reset_index(level=0, drop=True)
    df['fval_cum_deviation'] = df['fval'] - df['fval_cummean']

    # 7. Trajectory features to detect changes in patterns
    for lag in range(1, 4):
        df[f'fval_lag_{lag}'] = df.groupby('ssn')['fval'].shift(lag).fillna(0)
        df[f'fval_diff_lag_{lag}'] = df['fval'] - df[f'fval_lag_{lag}']
    
    # ====================== 1. ENHANCED MINIMUM BOUNDARY DETECTION FEATURES ======================
    print("\n1. Creating specialized minimum boundary detection features...")
    
    # 1.1 Improved minimum boundary proximity features
    # More granular detection of approaches to minimum boundaries
    df['min_boundary_proximity_squared'] = df['min_boundary_proximity'] ** 2
    df['min_boundary_proximity_cubed'] = df['min_boundary_proximity'] ** 3
    
    # 1.2 Minimum boundary violation likelihood
    # Higher values indicate higher likelihood of minimum boundary violation
    df['min_boundary_violation_likelihood'] = (
        (df['fval'] < df.groupby('ssn')['fval'].transform('min') * 1.05).astype(int)
    )
    
    # 1.3 Exponentially weighted minimum boundary proximity
    # Gives more weight to values closer to the minimum boundary
    df['exp_min_boundary_proximity'] = np.exp(df['min_boundary_proximity'] * 10) - 1
    
    # 1.4 Minimum boundary proximity change rate
    # Detects rapid approaches to boundaries
    df['min_boundary_proximity_change'] = df.groupby('ssn')['min_boundary_proximity'].diff().fillna(0)
    df['min_boundary_approach_rate'] = (
        df['min_boundary_proximity_change'] / df['min_boundary_proximity'].shift(1).fillna(0.1)
    ).replace([np.inf, -np.inf], 0)
    
    # 1.5 Minimum boundary threshold crossings
    # Counts how many times values get close to the minimum boundary in a session
    threshold = 0.05  # 5% above minimum
    df['near_min_boundary'] = (df['min_boundary_proximity'] < threshold).astype(int)
    df['min_boundary_crossings'] = df.groupby('ssn')['near_min_boundary'].cumsum()
    
    # 1.6 Minimum boundary temporal pattern detection
    # Detects patterns of gradually approaching boundaries
    for window in [2, 3, 5]:
        # Rolling mean of proximity to minimum boundary
        df[f'min_boundary_proximity_mean_{window}d'] = df.groupby('ssn')['min_boundary_proximity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0)
        )
        
        # Trend in proximity to minimum boundary
        df[f'min_boundary_proximity_trend_{window}d'] = (
            df['min_boundary_proximity'] - df[f'min_boundary_proximity_mean_{window}d']
        )
    
    # ====================== 3. LARGE CHANGES WITHOUT THRESHOLD FEATURES ======================
    print("3. Creating features for large changes without threshold activation...")
    
    # 3.1 Large change but no threshold indicator
    # This targets patterns where large changes occur without triggering thresholds
    df['large_change_no_threshold'] = (
        (df['fval_change'].abs() > df.groupby('ssn')['fval_change'].transform('std') * 1.5) & 
        (df['change_threshold_exceeded'] == 0)
    ).astype(int)
    
    # 3.2 Large change magnitude without threshold
    df['large_change_no_threshold_magnitude'] = df['fval_change'].abs() * (1 - df['change_threshold_exceeded'])
    
    # 3.3 Cumulative large changes without thresholds
    df['cum_large_changes_no_threshold'] = df.groupby('ssn')['large_change_no_threshold'].cumsum()
    
    # 3.4 Ratio of change to standard deviation without triggering threshold
    df['change_std_ratio_no_threshold'] = (
        df['fval_change'].abs() / 
        (df['fval_change_std_3d'] + 0.1)  # Adding small constant to avoid division by zero
    ) * (1 - df['change_threshold_exceeded'])
    
    # ====================== 4. TEMPORAL PATTERN DETECTION FEATURES ======================
    print("4. Creating temporal pattern detection features...")
    
    # 4.1 Enhanced lag features with more lags and differences
    for lag in range(4, 8):  # Adding more lag periods
        df[f'fval_lag_{lag}'] = df.groupby('ssn')['fval'].shift(lag).fillna(0)
        df[f'fval_diff_lag_{lag}'] = df['fval'] - df[f'fval_lag_{lag}']
    
    # 4.2 Change acceleration features
    # Detect changes in the rate of change
    df['fval_change_acceleration'] = df.groupby('ssn')['fval_change'].diff().fillna(0)
    df['fval_change_jerk'] = df.groupby('ssn')['fval_change_acceleration'].diff().fillna(0)  # Rate of change of acceleration
    
    # 4.3 Autoregressive features
    # Predict value from previous values and calculate residuals
    for p in [3, 5]:  # AR(3) and AR(5) models
        # Create lag features for AR model
        lag_cols = [f'fval_lag_{i}' for i in range(1, p+1)]
        
        # Avoid the first p rows in each session where we don't have enough lags
        for ssn in df['ssn'].unique():
            ssn_idx = df[df['ssn'] == ssn].index
            
            if len(ssn_idx) <= p:
                continue
                
            # Skip the first p rows of each session
            valid_idx = ssn_idx[p:]
            
            # Simple AR approximation: mean of previous p values
            df.loc[valid_idx, f'fval_ar{p}_pred'] = df.loc[valid_idx][lag_cols].mean(axis=1)
            
            # Residual (error of prediction)
            df.loc[valid_idx, f'fval_ar{p}_residual'] = df.loc[valid_idx, 'fval'] - df.loc[valid_idx, f'fval_ar{p}_pred']
    
    # Fill NaN values with 0
    for col in [f'fval_ar{p}_{suffix}' for p in [3, 5] for suffix in ['pred', 'residual']]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 4.4 Fourier features for seasonal patterns
    # Extract seasonal components that might reveal manipulation patterns
    for period in [5, 10, 20]:
        # Calculate sine and cosine components
        df[f'fval_sin_{period}d'] = np.sin(2 * np.pi * df['day_in_ssn'] / period)
        df[f'fval_cos_{period}d'] = np.cos(2 * np.pi * df['day_in_ssn'] / period)
        
        # Interact them with the fval
        df[f'fval_sin_{period}d_interaction'] = df['fval'] * df[f'fval_sin_{period}d']
        df[f'fval_cos_{period}d_interaction'] = df['fval'] * df[f'fval_cos_{period}d']
    
    # ====================== 5. COMPOSITE ANOMALY DETECTION FEATURES ======================
    print("5. Creating composite anomaly detection features...")
    
    # 5.1 Minimum Boundary Approach composite score
    df['min_boundary_approach_score'] = (
        df['min_boundary_proximity'] * 0.3 +
        df['exp_min_boundary_proximity'] * 0.3 +
        df['min_boundary_approach_rate'].clip(0, 10) * 0.2 +
        df['near_min_boundary'] * 0.2
    )
    
    # 5.2 Large Change without Threshold composite score
    df['large_change_no_threshold_score'] = (
        df['large_change_no_threshold'] * 0.4 +
        df['large_change_no_threshold_magnitude'].clip(0, 1000) / 1000 * 0.3 +
        df['change_std_ratio_no_threshold'].clip(0, 10) / 10 * 0.3
    )
    
    # 5.4 Weighted anomaly score combining all anomaly types
    df['weighted_anomaly_score'] = (
        df['min_boundary_approach_score'] * 0.6 +  # More weight to boundary violations (our biggest weakness)
        df['large_change_no_threshold_score'] * 0.4
    )
    
    # ====================== 6. INTERACTION FEATURES WITH NEW FEATURES ======================
    print("6. Creating interaction features with new composite features...")
    
    # 6.1 List of important existing features
    key_existing_features = [
        'fval_change', 
        'fval_diff_lag_1',
        'ramp_constraint_active',
        'fval_change_std_3d',
        'day_in_ssn'
    ]
    
    # 6.2 List of new important composite features
    key_new_features = [
        'min_boundary_approach_score',
        'large_change_no_threshold_score',
        'weighted_anomaly_score'
    ]
    
    # 6.3 Create interactions between key existing and new features
    interaction_count = 0
    for existing_feature in key_existing_features:
        for new_feature in key_new_features:
            if existing_feature in df.columns and new_feature in df.columns:
                interaction_name = f"{existing_feature}_x_{new_feature}"
                df[interaction_name] = df[existing_feature] * df[new_feature]
                interaction_count += 1
    
    # 6.4 Create interactions among the new composite features
    for i in range(len(key_new_features)):
        for j in range(i+1, len(key_new_features)):
            feature1 = key_new_features[i]
            feature2 = key_new_features[j]
            if feature1 in df.columns and feature2 in df.columns:
                interaction_name = f"{feature1}_x_{feature2}"
                df[interaction_name] = df[feature1] * df[feature2]
                interaction_count += 1
    
    print(f"Created {interaction_count} new interaction features")
    
    # ====================== 7. FINAL CLEANING AND REPORTING ======================
    # Clean any NaN values
    df = df.fillna(0)
    
    # Report on new features added
    new_shape = df.shape
    features_added = new_shape[1] - original_shape[1]
    print(f"\nAdvanced feature engineering complete. Added {features_added} new features.")
    print(f"New DataFrame shape: {new_shape}")
    
    return df
