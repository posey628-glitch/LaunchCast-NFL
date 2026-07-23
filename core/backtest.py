# core/backtest.py
# LaunchCast NFL — Backtesting Engine V7
# ADDS: Edge metrics (Top20 Hit %, Slate Base %, Edge, Lift)

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
from core.scoring import generate_nfl_projections

def run_nfl_backtest(season=2025, max_weeks=18):
    """
    Runs the scoring engine on historical data and grades it.
    Includes edge metrics: Top20 hit rate vs slate base rate.
    """
    results = []
    
    # Load prior rates once for early weeks
    prior_rates = load_prior_rates_from_season(season - 1)
    
    for week in range(2, max_weeks + 1):
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
            
            actuals = actuals[['player_id', 'player_name', 'team', 'receiving_tds', 'receiving_yards', 'receptions']].copy()
            actuals = actuals.rename(columns={
                'receiving_tds': 'actual_tds',
                'receiving_yards': 'actual_yards',
                'receptions': 'actual_rec'
            }).fillna(0)
            
            test_df = projections.merge(actuals, on=['player_id', 'player_name', 'team'], how='inner', suffixes=('', '_actual'))
            
            # Calculate hits
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            test_df['hit_yards'] = (test_df['actual_yards'] > 45.5).astype(int)
            
            # Calculate Brier scores
            test_df['brier_td'] = (test_df['prob_1plus_td'] - test_df['hit_td']) ** 2
            test_df['brier_yards'] = (test_df['prob_over_45.5_yds'] - test_df['hit_yards']) ** 2
            
            # ADD: Edge metrics
            test_df_sorted = test_df.sort_values('prob_1plus_td', ascending=False)
            top20 = test_df_sorted.head(20)
            base_rate = test_df['hit_td'].mean()
            top20_rate = top20['hit_td'].mean()
            
            results.append({
                'Week': week,
                'Players': len(test_df),
                'Avg Brier (TD)': round(test_df['brier_td'].mean(), 4),
                'Hit Rate (TD)': round(test_df['hit_td'].mean() * 100, 1),
                'Avg Prob (TD)': round(test_df['prob_1plus_td'].mean() * 100, 1),
                'Avg Brier (Yds)': round(test_df['brier_yards'].mean(), 4),
                'Hit Rate (Yds)': round(test_df['hit_yards'].mean() * 100, 1),
                # NEW: Edge metrics
                'Top20 Hit %': round(top20_rate * 100, 1),
                'Slate Base %': round(base_rate * 100, 1),
                'Edge (pp)': round((top20_rate - base_rate) * 100, 1),
                'Lift': round(top20_rate / base_rate, 2) if base_rate > 0 else None,
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
    
    # ADD: Edge metrics summary
    if 'Top20 Hit %' in results_df.columns:
        avg_top20 = results_df['Top20 Hit %'].mean()
        avg_base = results_df['Slate Base %'].mean()
        avg_edge = results_df['Edge (pp)'].mean()
        avg_lift = results_df['Lift'].mean()
        
        lines.append("")
        lines.append("🎯 EDGE METRICS")
        lines.append("-" * 40)
        lines.append(f"Top-20 Hit Rate:      {avg_top20:.1f}%")
        lines.append(f"Slate Base Rate:      {avg_base:.1f}%")
        lines.append(f"Edge:                 {avg_edge:+.1f}pp")
        lines.append(f"Lift:                 {avg_lift:.2f}x")
        lines.append("")
        lines.append("Interpretation:")
        if avg_edge >= 10:
            lines.append("✅ STRONG EDGE — Top picks hit significantly more than baseline")
        elif avg_edge >= 5:
            lines.append("🟡 MODEST EDGE — Some signal, but needs refinement")
        else:
            lines.append("⚠️ WEAK EDGE — Model is calibrated but not discriminating")
    
    lines.append("")
    lines.append("📊 WEEKLY BREAKDOWN")
    lines.append("-" * 40)
    
    header = f"{'Week':<4} | {'Players':>7} | {'Brier':>5} | {'Hit %':>5} | {'Prob %':>6} | {'Top20':>5} | {'Edge':>5}"
    lines.append(header)
    lines.append("-" * 40)
    
    for _, row in results_df.iterrows():
        line = (f"{int(row['Week']):<4} | {int(row['Players']):>7} | "
                f"{row['Avg Brier (TD)']:.4f} | {row['Hit Rate (TD)']:>5.1f} | "
                f"{row['Avg Prob (TD)']:>6.1f} | {row.get('Top20 Hit %', 0):>5.1f} | "
                f"{row.get('Edge (pp)', 0):>+5.1f}")
        lines.append(line)
        
    lines.append("-" * 40)
    
    if len(results_df) > 0:
        best_week = results_df.loc[results_df['Avg Brier (TD)'].idxmin()]
        worst_week = results_df.loc[results_df['Avg Brier (TD)'].idxmax()]
        
        lines.append("")
        lines.append("🔍 KEY INSIGHTS")
        lines.append(f"• Best Calibrated Week: Week {int(best_week['Week'])} (Brier: {best_week['Avg Brier (TD)']:.4f})")
        lines.append(f"• Worst Calibrated Week: Week {int(worst_week['Week'])} (Brier: {worst_week['Avg Brier (TD)']:.4f})")
        
        if 'Edge (pp)' in results_df.columns:
            best_edge_week = results_df.loc[results_df['Edge (pp)'].idxmax()]
            lines.append(f"• Best Edge Week: Week {int(best_edge_week['Week'])} (Edge: {best_edge_week['Edge (pp)']:+.1f}pp)")
    
    return "\n".join(lines)
