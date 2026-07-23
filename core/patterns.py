# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine V7.4
# FIX: Resolve actual season, prevent prior==current leakage, return actual season

import pandas as pd
import numpy as np
import streamlit as st
from data.fetcher import (
    build_features_through, 
    build_defensive_features_through, 
    get_weekly_player_stats,
    load_prior_rates_from_season,
    resolve_season,
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
    """
    Analyze which features correlate with actual TD hits.
    FIX: Resolve actual season and prevent prior==current leakage.
    Returns (summary_df, actual_season) tuple.
    """
    correlations = []
    
    # FIX: Resolve actual season ONCE at the top
    actual = resolve_season(season)
    if actual != season:
        st.info(f"ℹ️ Pattern analysis running on {actual} data (requested {season})")
    
    # FIX: Load priors with leakage check
    prior_rates = load_prior_rates_from_season(actual - 1)
    if resolve_season(actual - 1) == actual:
        prior_rates = pd.DataFrame()
        st.warning("⚠️ Prior-season data unavailable — week 1-3 priors DISABLED")
    
    for week in range(1, max_weeks + 1):
        try:
            # Use actual season throughout
            features = build_features_through(week, actual, prior_rates=prior_rates if week <= 3 else None)
            if features.empty:
                continue
            
            all_data = _load_weekly_raw(actual)
            if all_data.empty:
                continue
                
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
            
            def_features = build_defensive_features_through(week, actual)
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
            
            actuals = get_weekly_player_stats(week, actual)
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
        return pd.DataFrame(), actual
    
    corr_df = pd.DataFrame(correlations)
    
    summary = corr_df.groupby('Feature').agg({
        'Correlation': ['mean', 'std', 'count'],
    }).reset_index()
    summary.columns = ['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']
    summary['Avg Correlation'] = summary['Avg Correlation'].round(3)
    summary['Std Dev'] = summary['Std Dev'].round(3)
    
    summary['Abs Correlation'] = summary['Avg Correlation'].abs()
    summary = summary.sort_values('Abs Correlation', ascending=False)
    
    # FIX: Return tuple with actual season
    return summary[['Feature', 'Avg Correlation', 'Std Dev', 'Weeks Sampled']], actual

def get_proposed_weights(pattern_results, min_weeks=5):
    """
    Derive target weights proportionally from evidence, then half-step toward them.
    Negative correlations get 0 weight (they're anti-predictors).
    """
    if pattern_results.empty:
        return None
    
    ev = {}
    for _, r in pattern_results.iterrows():
        feat = r['Feature']
        if feat in BOOM_WEIGHTS and r['Weeks Sampled'] >= min_weeks:
            corr = r['Avg Correlation']
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
