# core/backtest.py
# LaunchCast NFL — Backtesting Engine (Option 3)
# Grades 2024 projections against actual 2024 outcomes.

import pandas as pd
import numpy as np
from data.fetcher import get_weekly_player_stats
from core.scoring import generate_nfl_projections

def run_nfl_backtest(season=2024, max_weeks=18):
    """
    Runs the scoring engine on historical data and grades it against actual outcomes.
    Returns a summary dataframe of weekly performance.
    """
    results = []
    
    # We only test weeks that have data
    available_weeks = list(range(1, max_weeks + 1))
    
    for week in available_weeks:
        try:
            # 1. Fetch raw data for this week
            raw_data = get_weekly_player_stats(week, year=season)
            if raw_data.empty:
                continue
                
            # 2. Generate projections
            projections = generate_nfl_projections(raw_data, current_week=week)
            if projections.empty:
                continue
                
            # 3. Merge with actuals (raw_data has the actual stats)
            # We need player_name and team to merge
            actuals = raw_data[['player_name', 'team', 'receiving_tds', 'receiving_yards', 'receptions']].copy()
            actuals = actuals.rename(columns={
                'receiving_tds': 'actual_tds',
                'receiving_yards': 'actual_yards',
                'receptions': 'actual_rec'
            })
            
            # Fill NaNs with 0 for players who didn't record stats
            actuals = actuals.fillna(0)
            
            # Merge projections with actuals
            test_df = projections.merge(actuals, on=['player_name', 'team'], how='left', suffixes=('', '_actual'))
            test_df = test_df.fillna(0)
            
            # 4. Calculate Outcomes (1 if hit, 0 if miss)
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            test_df['hit_yards'] = (test_df['actual_yards'] > 45.5).astype(int)
            test_df['hit_rec'] = (test_df['actual_rec'] > 3.5).astype(int)
            
            # 5. Calculate Brier Scores (Lower is better)
            # Brier = (Forecast - Actual)^2
            test_df['brier_td'] = (test_df['prob_1plus_td'] - test_df['hit_td']) ** 2
            test_df['brier_yards'] = (test_df['prob_over_45.5_yds'] - test_df['hit_yards']) ** 2
            
            # 6. Aggregate Weekly Stats
            n_players = len(test_df)
            avg_brier_td = test_df['brier_td'].mean()
            hit_rate_td = test_df['hit_td'].mean()
            avg_prob_td = test_df['prob_1plus_td'].mean()
            
            results.append({
                'Week': week,
                'Players': n_players,
                'Avg Brier (TD)': round(avg_brier_td, 4),
                'Hit Rate (TD)': round(hit_rate_td * 100, 1),
                'Avg Prob (TD)': round(avg_prob_td * 100, 1),
                'Avg Brier (Yds)': round(test_df['brier_yards'].mean(), 4),
                'Hit Rate (Yds)': round(test_df['hit_yards'].mean() * 100, 1),
            })
            
        except Exception as e:
            # Skip weeks that fail
            continue
            
    return pd.DataFrame(results)
