# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine V8
# FIX: get_proposed_weights now derives weights proportionally from evidence
# (can both inflate AND deflate — evidence-based redistribution, not one-sided)

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

# Track both raw and shrunk versions — this directly answers
# "is my shrinkage helping or hurting?"
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
    
    for week in range(1, max_weeks + 1):
        try:
            # Build features (with prior rates for weeks 1-3)
            features = build_features_through(week, season, prior_rates=prior_rates if week <= 3 else None)
            if features.empty:
                continue
            
            # Attach current team AND opponent from week N
            all_data = _load_weekly_raw(season)
            all_data = normalize_columns(all_data)
            week_n = all_data[all_data['week'] == week][
                ['player_id', 'team', 'opponent_team']
            ].drop_duplicates('player_id')
            
            if week_n.empty:
                continue
            
            # Drop stale team from features, merge current team from week N
            if 'team' in features.columns:
                features = features.drop(columns=['team'])
            
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

def get_proposed_weights(pattern_results, min_weeks=5):
    """
    Derive target weights proportionally from evidence, then half-step toward them.
    
    FIX: The old version only proposed UPWARD (features below gate kept their weight).
    This version treats all BOOM_WEIGHTS features — weak features get pulled DOWN,
    strong features get pulled UP. Evidence-based redistribution, not one-sided bump.
    
    Returns dict with current / evidence / target / apply for each feature.
    """
    if pattern_results.empty:
        return None
    
    # Build evidence map: |correlation| for each BOOM_WEIGHTS feature
    ev = {}
    for _, r in pattern_results.iterrows():
        feat = r['Feature']
        if feat in BOOM_WEIGHTS and r['Weeks Sampled'] >= min_weeks:
            ev[feat] = abs(r['Avg Correlation'])
    
    # Need at least 2 features with evidence to redistribute meaningfully
    if len(ev) < 2:
        return None
    
    total_ev = sum(ev.values()) or 1.0
    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    
    proposed = {}
    for feat, cur in BOOM_WEIGHTS.items():
        # Evidence-proportional target weight
        target = (ev.get(feat, 0.0) / total_ev) * total_w
        
        # ½-step toward target (conservative, same as MLB)
        applied = cur + 0.5 * (target - cur)
        
        # Clamp to reasonable range
        applied = max(0.10, min(0.70, applied))
        
        proposed[feat] = {
            'current':  round(cur, 3),
            'evidence': round(ev.get(feat, 0.0), 3),
            'target':   round(target, 3),
            'apply':    round(applied, 3),
        }
    
    return proposed
