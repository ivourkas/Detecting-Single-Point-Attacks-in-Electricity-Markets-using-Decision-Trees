
def engineer_advanced_features(df):
    """
    Apply advanced feature engineering techniques to improve attack detection,
    with special focus on Lower Limit Attacks and vulnerable generators
    """
    print("\n=== Performing advanced feature engineering ===")
    import numpy as np
    
    # Store original shape for reporting
    original_shape = df.shape
    
    # ====================== 1. ENHANCED LOWER LIMIT ATTACK DETECTION FEATURES ======================
    print("\n1. Creating specialized Lower Limit Attack detection features...")
    
    # 1.1 Improved lower limit proximity features
    # More granular detection of approaches to lower limits
    df['lower_limit_proximity_squared'] = df['lower_limit_proximity'] ** 2
    df['lower_limit_proximity_cubed'] = df['lower_limit_proximity'] ** 3
    
    # 1.2 Lower limit violation likelihood
    # Higher values indicate higher likelihood of lower limit violation
    df['lower_limit_violation_likelihood'] = (
        (df['fval'] < df.groupby('ssn')['fval'].transform('min') * 1.05).astype(int)
    )
    
    # 1.3 Exponentially weighted lower limit proximity
    # Gives more weight to values closer to the lower limit
    df['exp_lower_limit_proximity'] = np.exp(df['lower_limit_proximity'] * 10) - 1
    
    # 1.4 Lower limit proximity change rate
    # Detects rapid approaches to lower limits
    df['lower_limit_proximity_change'] = df.groupby('ssn')['lower_limit_proximity'].diff().fillna(0)
    df['lower_limit_approach_rate'] = (
        df['lower_limit_proximity_change'] / df['lower_limit_proximity'].shift(1).fillna(0.1)
    ).replace([np.inf, -np.inf], 0)
    
    # 1.5 Lower limit threshold crossings
    # Counts how many times values get close to the lower limit in a session
    threshold = 0.05  # 5% above minimum
    df['near_lower_limit'] = (df['lower_limit_proximity'] < threshold).astype(int)
    df['lower_limit_crossings'] = df.groupby('ssn')['near_lower_limit'].cumsum()
    
    # 1.6 Lower limit temporal pattern detection
    # Detects patterns of gradually approaching limits
    for window in [2, 3, 5]:
        # Rolling mean of proximity to lower limit
        df[f'lower_limit_proximity_mean_{window}d'] = df.groupby('ssn')['lower_limit_proximity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0)
        )
        
        # Trend in proximity to lower limit
        df[f'lower_limit_proximity_trend_{window}d'] = (
            df['lower_limit_proximity'] - df[f'lower_limit_proximity_mean_{window}d']
        )
    
    # # ====================== 2. VULNERABLE GENERATOR SPECIFIC FEATURES ======================
    # print("2. Creating vulnerable generator specific features...")
    
    # # List of vulnerable generators based on our analysis
    # vulnerable_generators = [7, 25, 31, 44, 52, 60, 69]
    
    # # 2.1 Vulnerable generator indicator
    # df['is_vulnerable_generator'] = df['gen_attacked'].isin(vulnerable_generators).astype(int)
    
    # # 2.2 Generator-specific anomaly scores
    # # For each vulnerable generator, calculate targeted anomaly scores
    # for gen in vulnerable_generators:
    #     # Create a mask for the specific generator
    #     gen_mask = (df['gen_attacked'] == gen)
        
    #     if gen_mask.sum() > 0:  # Only process if this generator exists in the data
    #         # Generator-specific fval statistics
    #         gen_fval_mean = df[gen_mask]['fval'].mean()
    #         gen_fval_std = df[gen_mask]['fval'].std() if df[gen_mask]['fval'].std() > 0 else 1
            
    #         # Calculate generator-specific z-scores
    #         df[f'gen{gen}_fval_zscore'] = 0.0  # Initialize with zeros
    #         df.loc[gen_mask, f'gen{gen}_fval_zscore'] = (df.loc[gen_mask, 'fval'] - gen_fval_mean) / gen_fval_std
            
    #         # Calculate generator-specific change anomaly
    #         gen_change_mean = df[gen_mask]['fval_change'].mean()
    #         gen_change_std = df[gen_mask]['fval_change'].std() if df[gen_mask]['fval_change'].std() > 0 else 1
            
    #         df[f'gen{gen}_change_zscore'] = 0.0  # Initialize with zeros
    #         df.loc[gen_mask, f'gen{gen}_change_zscore'] = (df.loc[gen_mask, 'fval_change'] - gen_change_mean) / gen_change_std
    
    # ====================== 3. LARGE CHANGES WITHOUT RAMP CONSTRAINT FEATURES ======================
    print("3. Creating features for large changes without ramp constraint activation...")
    
    # 3.1 Large change but no ramp constraint indicator
    # This targets the pattern we observed in missed attacks
    df['large_change_no_constraint'] = (
        (df['fval_change'].abs() > df.groupby('ssn')['fval_change'].transform('std') * 1.5) & 
        (df['ramp_constraint_active'] == 0)
    ).astype(int)
    
    # 3.2 Large change magnitude without ramp constraint
    df['large_change_no_constraint_magnitude'] = df['fval_change'].abs() * (1 - df['ramp_constraint_active'])
    
    # 3.3 Cumulative large changes without constraints
    df['cum_large_changes_no_constraint'] = df.groupby('ssn')['large_change_no_constraint'].cumsum()
    
    # 3.4 Ratio of change to standard deviation without triggering constraint
    df['change_std_ratio_no_constraint'] = (
        df['fval_change'].abs() / 
        (df['fval_change_std_3d'] + 0.1)  # Adding small constant to avoid division by zero
    ) * (1 - df['ramp_constraint_active'])
    
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
    
    # ====================== 5. COMPOSITE ATTACK-SPECIFIC FEATURES ======================
    print("5. Creating composite attack-specific features...")
    
    # 5.1 Lower Limit Attack composite score
    df['lower_limit_attack_score'] = (
        df['lower_limit_proximity'] * 0.3 +
        df['exp_lower_limit_proximity'] * 0.3 +
        df['lower_limit_approach_rate'].clip(0, 10) * 0.2 +
        df['near_lower_limit'] * 0.2
    )
    
    # 5.2 Large Change without Constraint composite score
    df['large_change_no_constraint_score'] = (
        df['large_change_no_constraint'] * 0.4 +
        df['large_change_no_constraint_magnitude'].clip(0, 1000) / 1000 * 0.3 +
        df['change_std_ratio_no_constraint'].clip(0, 10) / 10 * 0.3
    )
    
    # # 5.3 Vulnerable Generator Attack composite score
    # # This will be 0 for non-vulnerable generators
    # df['vulnerable_gen_attack_score'] = 0.0
    
    # # Only calculate for the vulnerable generators
    # for gen in vulnerable_generators:
    #     gen_mask = (df['gen_attacked'] == gen)
        
    #     if gen_mask.sum() > 0 and f'gen{gen}_fval_zscore' in df.columns and f'gen{gen}_change_zscore' in df.columns:
    #         # Calculate the score only for rows with this generator
    #         df.loc[gen_mask, 'vulnerable_gen_attack_score'] = (
    #             df.loc[gen_mask, f'gen{gen}_fval_zscore'].abs().clip(0, 5) / 5 * 0.5 +
    #             df.loc[gen_mask, f'gen{gen}_change_zscore'].abs().clip(0, 5) / 5 * 0.5
    #         )
    
    # 5.4 Weighted anomaly score combining all attack types
    df['weighted_attack_score'] = (
        df['lower_limit_attack_score'] * 0.6 +  # More weight to lower limit attacks (our biggest weakness)
        df['large_change_no_constraint_score'] * 0.4
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
        'lower_limit_attack_score',
        'large_change_no_constraint_score',
        'weighted_attack_score'
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