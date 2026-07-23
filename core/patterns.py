# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine V8.1
# FIX: abs() bug in get_proposed_weights — negative correlations should get 0 weight, not positive

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
    
    prior_rates = load_prior_rates_from_season(season - 1)
    
    for week in range(1, max_weeks + 1):
        try:
            features = build_features_through(week, season, prior_rates=prior_rates if week <= 3 else None)
            if features.empty:
                continue
            
            all_data = _load_weekly_raw(season)
            all_data = normalize_columns(all_data)
            week_n = all_data[all_data['week'] == week][
                ['player_id', 'team', 'opponent_team']
            ].drop_duplicates('player_id')
            
            if week_n.empty:
                continue
            
            if 'team' in features.columns:
                features = features.drop(columns=['team'])
            
            features = features.merge(week_n, on='player_id', how='inner')
            if features.empty:
                continue
            
            def_features = build_defensive_features_through(week, season)
            if not def_features.empty:
                features = features.merge(
                    def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']]
                        .rename(columns={'team': 'opponent_team'}),
                    on='opponent_team',
                    how='left'
                )
            
            projections = generate_nfl_projections(features, current_week=week)
            if projections.empty:
                continue
            
            actuals = get_weekly_player_stats(week, season)
            if actuals.empty:
                continue
            
            actuals = actuals[['player_id', 'player_name', 'team', 'receiving_tds']].copy()
            actuals = actuals.rename(columns={'receiving_tds': 'actual_tds'}).fillna(0)
            
            test_df = projections.merge(actuals, on=['player_id', 'player_name', 'team'], how='inner')
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            
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
    
    FIX: The abs() bug threw away sign information. A feature with correlation -0.105
    (higher YPT = fewer TDs) was treated as positive evidence. Now:
    - Negative correlations get 0 weight (they're anti-predictors)
    - Only positive correlations contribute to the weight pool
    """
    if pattern_results.empty:
        return None
    
    # Build evidence map: correlation (NOT abs) for each BOOM_WEIGHTS feature
    ev = {}
    for _, r in pattern_results.iterrows():
        feat = r['Feature']
        if feat in BOOM_WEIGHTS and r['Weeks Sampled'] >= min_weeks:
            corr = r['Avg Correlation']
            # FIX: Negative correlations get 0 weight
            if corr < -0.03:
                ev[feat] = 0.0
            else:
                ev[feat] = max(0.0, corr)
    
    if len(ev) < 2:
        return None
    
    total_ev = sum(ev.values()) or 1.0
    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    
    proposed = {}
    for feat, cur in BOOM_WEIGHTS.items():
        target = (ev.get(feat, 0.0) / total_ev) * total_w
        applied = cur + 0.5 * (target - cur)
        applied = max(0.10, min(0.70, applied))
        
        proposed[feat] = {
            'current':  round(cur, 3),
            'evidence': round(ev.get(feat, 0.0), 3),
            'target':   round(target, 3),
            'apply':    round(applied, 3),
        }
    
    return proposed
