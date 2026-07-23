# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine V7.1
# FIX: Track both raw and shrunk versions to test shrinkage effectiveness

import pandas as pd
import numpy as np
from data.fetcher import (
    build_features_through, 
    build_defensive_features_through, 
    get_weekly_player_stats,
    load_prior_rates_from_season,
    _load_weekly_raw,
    normalize_columns
)
from core.scoring import generate_nfl_projections, BOOM_WEIGHTS

# FIX: Track both raw and shrunk versions
# This directly answers "is my shrinkage helping or hurting?"
TRACKED_FEATURES = [
    'target_share', 'shrunk_target_share',
    'yds_per_tgt', 'shrunk_yds_per_tgt',
    'td_per_tgt', 'shrunk_td_per_tgt',
    'adot', 'routes', 'boom_score',
    'def_yds_per_tgt', 'def_td_per_tgt',
    'proj_targets', 'team_avg_pass_attempts',
]

def run_pattern_analysis(season=2025, max_weeks=18):
    """Analyze which features correlate with actual TD hits."""
    correlations = []
    
    # Load prior rates once for early weeks
    prior_rates = load_prior_rates_from_season(season - 1)
    
    # Start at week 1 (now that it works)
    for week in range(1, max_weeks + 1):
        try:
            # Build features (with prior rates for weeks 1-3)
            features = build_features_through(week, season, prior_rates=prior_rates if week <= 3 else None)
            if features.empty:
                continue
            
            # Attach opponent
            all_data = _load_weekly_raw(season)
            all_data = normalize_columns(all_data)
            week_n = all_data[all_data['week'] == week][['player_id', 'opponent_team']].drop_duplicates('player_id')
            
            if week_n.empty:
                continue
            
            features = features.merge(week_n, on='player_id', how='inner')
            if features.empty:
                continue
            
            # Attach defense
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
            
            # Get actuals
            actuals = get_weekly_player_stats(week, season)
            if actuals.empty:
                continue
            
            actuals = actuals[['player_id', 'player_name', 'team', 'receiving_tds']].copy()
            actuals = actuals.rename(columns={'receiving_tds': 'actual_tds'}).fillna(0)
            
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
    """Based on pattern analysis, propose conservative weight adjustments."""
    if pattern_results.empty:
        return None
    
    proposed = {}
    
    for _, row in pattern_results.iterrows():
        feature = row['Feature']
        corr = row['Avg Correlation']
        weeks = row['Weeks Sampled']
        
        if weeks >= 5 and abs(corr) >= 0.05:
            if feature in BOOM_WEIGHTS:
                current = BOOM_WEIGHTS[feature]
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
