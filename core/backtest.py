# core/backtest.py
# LaunchCast NFL — Backtesting Engine V5
# FIX: Same defensive merge pattern as fetcher

import pandas as pd
import numpy as np
from data.fetcher import (
    build_features_through, 
    build_defensive_features_through, 
    get_weekly_player_stats,
    _load_weekly_raw,
    normalize_columns
)
from core.scoring import generate_nfl_projections

def run_nfl_backtest(season=2025, max_weeks=18):
    """
    Runs the scoring engine on historical data and grades it.
    Features from weeks 1 to N-1, outcomes from week N.
    """
    results = []
    
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
            
            # Generate projections using historical features
            projections = generate_nfl_projections(features, current_week=week)
            if projections.empty:
                continue
            
            # Get ACTUAL outcomes from week N
            actuals = get_weekly_player_stats(week, season)
            if actuals.empty:
                continue
            
            actuals = actuals[['player_id', 'player_name', 'team', 'receiving_tds', 'receiving_yards', 'receptions']].copy()
            actuals = actuals.rename(columns={
                'receiving_tds': 'actual_tds',
                'receiving_yards': 'actual_yards',
                'receptions': 'actual_rec'
            }).fillna(0)
            
            # Merge projections with actuals on player_id
            test_df = projections.merge(actuals, on=['player_id', 'player_name', 'team'], how='inner', suffixes=('', '_actual'))
            
            # Calculate hits
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            test_df['hit_yards'] = (test_df['actual_yards'] > 45.5).astype(int)
            
            # Calculate Brier scores
            test_df['brier_td'] = (test_df['prob_1plus_td'] - test_df['hit_td']) ** 2
            test_df['brier_yards'] = (test_df['prob_over_45.5_yds'] - test_df['hit_yards']) ** 2
            
            results.append({
                'Week': week,
                'Players': len(test_df),
                'Avg Brier (TD)': round(test_df['brier_td'].mean(), 4),
                'Hit Rate (TD)': round(test_df['hit_td'].mean() * 100, 1),
                'Avg Prob (TD)': round(test_df['prob_1plus_td'].mean() * 100, 1),
                'Avg Brier (Yds)': round(test_df['brier_yards'].mean(), 4),
                'Hit Rate (Yds)': round(test_df['hit_yards'].mean() * 100, 1),
            })
        except Exception as e:
            continue
            
    return pd.DataFrame(results)

def generate_nfl_backtest_copy_text(results_df):
    """Generates a clean, copy-pasteable text report."""
    if results_df.empty:
        return "No backtest data available."
    
    lines = []
    lines.append("🏈 LAUNCHCAST NFL — BACKTEST REPORT")
    lines.append("=" * 40)
    
    avg_brier = results_df['Avg Brier (TD)'].mean()
    avg_hit = results_df['Hit Rate (TD)'].mean()
    avg_prob = results_df['Avg Prob (TD)'].mean()
    
    lines.append(f"Overall Avg Brier (TD): {avg_brier:.4f} (Lower is better)")
    lines.append(f"Overall Hit Rate (TD):  {avg_hit:.1f}%")
    lines.append(f"Overall Avg Prob (TD):  {avg_prob:.1f}%")
    lines.append("")
    lines.append("📊 WEEKLY BREAKDOWN")
    lines.append("-" * 40)
    
    header = f"{'Week':<4} | {'Players':>7} | {'Brier':>5} | {'Hit %':>5} | {'Prob %':>6}"
    lines.append(header)
    lines.append("-" * 40)
    
    for _, row in results_df.iterrows():
        line = f"{int(row['Week']):<4} | {int(row['Players']):>7} | {row['Avg Brier (TD)']:.4f} | {row['Hit Rate (TD)']:>5.1f} | {row['Avg Prob (TD)']:>6.1f}"
        lines.append(line)
        
    lines.append("-" * 40)
    
    if len(results_df) > 0:
        best_week = results_df.loc[results_df['Avg Brier (TD)'].idxmin()]
        worst_week = results_df.loc[results_df['Avg Brier (TD)'].idxmax()]
        
        lines.append("")
        lines.append("🔍 KEY INSIGHTS")
        lines.append(f"• Best Calibrated Week: Week {int(best_week['Week'])} (Brier: {best_week['Avg Brier (TD)']:.4f})")
        lines.append(f"• Worst Calibrated Week: Week {int(worst_week['Week'])} (Brier: {worst_week['Avg Brier (TD)']:.4f})")
    
    return "\n".join(lines)
