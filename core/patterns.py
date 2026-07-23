# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine V5
# FIX: Same defensive merge pattern as fetcher and backtest

import pandas as pd
import numpy as np
from data.fetcher import (
    build_features_through, 
    build_defensive_features_through, 
    get_weekly_player_stats,
    _load_weekly_raw,
    normalize_columns
)
from core.scoring import generate_nfl_projections, BOOM_WEIGHTS

# ============================================================================
# TRACKED FEATURES (whitelist)
# ============================================================================
TRACKED_FEATURES = [
    'target_share',
    'yds_per_tgt', 
    'td_per_tgt',
    'adot',
    'routes',
    'boom_score',
    'def_yds_per_tgt',
    'def_td_per_tgt',
]

def run_pattern_analysis(season=2025, max_weeks=18):
    """
    Analyze which features correlate with actual TD hits.
    Features from weeks 1 to N-1, outcomes from week N.
    """
    correlations = []
    
    for week in range(2, max_weeks + 1):
        try:
            # Build features from weeks 1 to N-1 (NO LEAKAGE)
            features = build_features_through(week, season)
            if features.empty:
                continue
            
            # FIX STEP 1: Attach this week's opponent FIRST
            all_data = _load_weekly_raw(season)
            all_data = normalize_columns(all_data)
            week_n = all_data[all_data['week'] == week][['player_id', 'opponent_team']].drop_duplicates('player_id')
            
            if week_n.empty:
                continue
            
            features = features.merge(week_n, on='player_id', how='inner')
            if features.empty:
                continue
            
            # FIX STEP 2: THEN attach the defense they FACE
            def_features = build_defensive_features_through(week, season)
            if not def_features.empty:
                features = features.merge(
                    def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']]
                        .rename(columns={'team': 'opponent_team'}),
                    on='opponent_team',
                    how='left'
                )
            
            # Generate projections
            projections = generate_nfl_projections(features, current_week=week)
            if projections.empty:
                continue
            
            # Get ACTUAL outcomes from week N
            actuals = get_weekly_player_stats(week, season)
            if actuals.empty:
                continue
            
            actuals = actuals[['player_id', 'player_name', 'team', 'receiving_tds']].copy()
            actuals = actuals.rename(columns={'receiving_tds': 'actual_tds'}).fillna(0)
            
            # Merge on player_id
            test_df = projections.merge(actuals, on=['player_id', 'player_name', 'team'], how='inner')
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            
            # Calculate correlation of each feature with hit_td
            for feature in TRACKED_FEATURES:
                if feature in test_df.columns:
                    feature_vals = pd.to_numeric(test_df[feature], errors='coerce').fillna(0)
                    if feature_vals.std() > 0:
                        corr = feature_vals.corr(test_df['hit_td'])
                        correlations.append({
                            'Week': week,
                            'Feature': feature,
                            'Correlation': round(corr, 3),
                            'N': len(test_df),
                        })
        except Exception:
            continue
    
    if not correlations:
        return pd.DataFrame()
    
    corr_df = pd.DataFrame(correlations)
    
    # Aggregate
    summary = corr_df.groupby('Feature').agg({
        'Correlation': ['mean', 'std', 'count'],
    }).reset_index()
    summary.columns = ['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']
    summary['Avg Correlation'] = summary['Avg Correlation'].round(3)
    summary['Std Dev'] = summary['Std Dev'].round(3)
    
    summary['Abs Correlation'] = summary['Avg Correlation'].abs()
    summary = summary.sort_values('Abs Correlation', ascending=False)
    
    return summary[['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']]

def get_proposed_weights(pattern_results):
    """
    Based on pattern analysis, propose conservative weight adjustments.
    Uses BOOM_WEIGHTS single source of truth.
    """
    if pattern_results.empty:
        return None
    
    proposed = {}
    
    for _, row in pattern_results.iterrows():
        feature = row['Feature']
        corr = row['Avg Correlation']
        weeks = row['Weeks Sampled']
        
        # Only propose adjustments for features with enough evidence
        if weeks >= 5 and abs(corr) >= 0.05:
            if feature in BOOM_WEIGHTS:
                current = BOOM_WEIGHTS[feature]
                # Scale adjustment to the weight itself
                adjustment = current * corr * 0.5
                new_weight = current + adjustment
                new_weight = max(0.10, min(0.60, new_weight))
                
                if abs(new_weight - current) >= 0.005:
                    proposed[feature] = {
                        'current': round(current, 2),
                        'proposed': round(new_weight, 2),
                        'evidence': f"{corr:+.3f} over {weeks} weeks",
                    }
    
    return proposed if proposed else None
