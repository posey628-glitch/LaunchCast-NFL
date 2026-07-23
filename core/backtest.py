# core/backtest.py
# LaunchCast NFL — Backtesting Engine

import pandas as pd
import numpy as np
from data.fetcher import get_weekly_player_stats
from core.scoring import generate_nfl_projections

def run_nfl_backtest(season=2025, max_weeks=18):
    """Runs the scoring engine on historical data and grades it."""
    results = []
    
    for week in range(1, max_weeks + 1):
        try:
            raw_data = get_weekly_player_stats(week, year=season)
            if raw_data.empty: continue
                
            projections = generate_nfl_projections(raw_data, current_week=week)
            if projections.empty: continue
                
            actuals = raw_data[['player_name', 'team', 'receiving_tds', 'receiving_yards', 'receptions']].copy()
            actuals = actuals.rename(columns={
                'receiving_tds': 'actual_tds',
                'receiving_yards': 'actual_yards',
                'receptions': 'actual_rec'
            }).fillna(0)
            
            test_df = projections.merge(actuals, on=['player_name', 'team'], how='left', suffixes=('', '_actual')).fillna(0)
            
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            test_df['hit_yards'] = (test_df['actual_yards'] > 45.5).astype(int)
            
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
        except Exception:
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
    
    best_week = results_df.loc[results_df['Avg Brier (TD)'].idxmin()]
    worst_week = results_df.loc[results_df['Avg Brier (TD)'].idxmax()]
    
    lines.append("")
    lines.append("🔍 KEY INSIGHTS")
    lines.append(f"• Best Calibrated Week: Week {int(best_week['Week'])} (Brier: {best_week['Avg Brier (TD)']:.4f})")
    lines.append(f"• Worst Calibrated Week: Week {int(worst_week['Week'])} (Brier: {worst_week['Avg Brier (TD)']:.4f})")
    
    return "\n".join(lines)
