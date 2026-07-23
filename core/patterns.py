# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine
# Ported from MLB: tracks which features actually predict winning props,
# proposes weight adjustments based on accumulated evidence.

import pandas as pd
import numpy as np
from data.fetcher import get_weekly_player_stats
from core.scoring import generate_nfl_projections

# ============================================================================
# TRACKED FEATURES (whitelist — derived, not hand-maintained)
# These are the features we analyze for correlation with outcomes.
# CRITICAL: Do NOT include model outputs (proj_*, prob_*) — a feature
# can't cite itself as its own evidence.
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

# ============================================================================
# PATTERN ANALYSIS
# ============================================================================
def run_pattern_analysis(season=2025, max_weeks=18):
    """
    Analyze which features correlate with actual TD hits.
    Returns a DataFrame of feature correlations.
    """
    correlations = []
    
    for week in range(1, max_weeks + 1):
        try:
            # Get raw data with actuals
            raw_data = get_weekly_player_stats(week, year=season)
            if raw_data.empty:
                continue
            
            # Generate projections
            projections = generate_nfl_projections(raw_data, current_week=week)
            if projections.empty:
                continue
            
            # Merge with actuals
            actuals = raw_data[['player_name', 'team', 'receiving_tds']].copy()
            actuals = actuals.rename(columns={'receiving_tds': 'actual_tds'}).fillna(0)
            
            test_df = projections.merge(actuals, on=['player_name', 'team'], how='left')
            test_df['actual_tds'] = test_df['actual_tds'].fillna(0)
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            
            # Calculate correlation of each feature with hit_td
            for feature in TRACKED_FEATURES:
                if feature in test_df.columns:
                    feature_vals = pd.to_numeric(test_df[feature], errors='coerce').fillna(0)
                    if feature_vals.std() > 0:  # Only if there's variance
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
    
    # Aggregate: average correlation across weeks
    summary = corr_df.groupby('Feature').agg({
        'Correlation': ['mean', 'std', 'count'],
    }).reset_index()
    summary.columns = ['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']
    summary['Avg Correlation'] = summary['Avg Correlation'].round(3)
    summary['Std Dev'] = summary['Std Dev'].round(3)
    
    # Sort by absolute correlation (strongest predictors first)
    summary['Abs Correlation'] = summary['Avg Correlation'].abs()
    summary = summary.sort_values('Abs Correlation', ascending=False)
    
    return summary[['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']]

# ============================================================================
# PROPOSED WEIGHT ADJUSTMENTS (½-step protocol from MLB)
# ============================================================================
def get_proposed_weights(pattern_results):
    """
    Based on pattern analysis, propose conservative weight adjustments.
    Uses the ½-step protocol: move 50% toward the evidence.
    """
    if pattern_results.empty:
        return None
    
    # Current weights (implicit in boom_score calculation)
    current_weights = {
        'target_share': 0.40,
        'yds_per_tgt': 0.30,
        'td_per_tgt': 0.30,
    }
    
    proposed = {}
    
    for _, row in pattern_results.iterrows():
        feature = row['Feature']
        corr = row['Avg Correlation']
        weeks = row['Weeks Sampled']
        
        # Only propose adjustments for features with enough evidence
        if weeks >= 5 and abs(corr) >= 0.10:
            if feature in current_weights:
                current = current_weights[feature]
                # ½-step: move 50% toward the direction of correlation
                # Positive correlation = increase weight, negative = decrease
                adjustment = corr * 0.10  # Scale factor
                new_weight = current + (adjustment * 0.5)  # ½-step
                new_weight = max(0.10, min(0.60, new_weight))  # Clamp
                
                if abs(new_weight - current) >= 0.02:  # Only if meaningful change
                    proposed[feature] = {
                        'current': round(current, 2),
                        'proposed': round(new_weight, 2),
                        'evidence': f"{corr:+.3f} over {weeks} weeks",
                    }
    
    return proposed if proposed else None
