# core/backtest.py
# LaunchCast NFL — Backtesting Engine V7.4
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
from core.scoring import generate_nfl_projections

def run_nfl_backtest(season=2025, max_weeks=18):
    """
    Runs the scoring engine on historical data and grades it.
    FIX: Resolve actual season and prevent prior==current leakage.
    Returns (results_df, actual_season) tuple.
    """
    results = []
    
    # FIX: Resolve actual season ONCE at the top
    actual = resolve_season(season)
    if actual != season:
        st.info(f"ℹ️ Backtest running on {actual} data (requested {season})")
    
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
            
            actuals = actuals[['player_id', 'player_name', 'team', 
                              'receiving_tds', 'receiving_yards', 'receptions']].copy()
            actuals = actuals.rename(columns={
                'receiving_tds': 'actual_tds',
                'receiving_yards': 'actual_yards',
                'receptions': 'actual_rec'
            }).fillna(0)
            
            test_df = projections.merge(
                actuals, 
                on=['player_id', 'player_name', 'team'], 
                how='inner', 
                suffixes=('', '_actual')
            )
            
            test_df['hit_td'] = (test_df['actual_tds'] >= 1).astype(int)
            test_df['hit_yards'] = (test_df['actual_yards'] > 45.5).astype(int)
            
            test_df['brier_td'] = (test_df['prob_1plus_td'] - test_df['hit_td']) ** 2
            test_df['brier_yards'] = (test_df['prob_over_45.5_yds'] - test_df['hit_yards']) ** 2
            
            # Edge metrics: TD
            test_df_sorted_td = test_df.sort_values('prob_1plus_td', ascending=False)
            top20_td = test_df_sorted_td.head(20)
            base_rate_td = test_df['hit_td'].mean()
            top20_rate_td = top20_td['hit_td'].mean()
            
            # Edge metrics: Yardage
            test_df_sorted_yds = test_df.sort_values('prob_over_45.5_yds', ascending=False)
            top20_yds = test_df_sorted_yds.head(20)
            base_rate_yds = test_df['hit_yards'].mean()
            top20_rate_yds = top20_yds['hit_yards'].mean()
            
            results.append({
                'Week': week,
                'Players': len(test_df),
                'Avg Brier (TD)': round(test_df['brier_td'].mean(), 4),
                'Hit Rate (TD)': round(test_df['hit_td'].mean() * 100, 1),
                'Avg Prob (TD)': round(test_df['prob_1plus_td'].mean() * 100, 1),
                'Top20 TD Hit %': round(top20_rate_td * 100, 1),
                'TD Edge (pp)': round((top20_rate_td - base_rate_td) * 100, 1),
                'TD Lift': round(top20_rate_td / base_rate_td, 2) if base_rate_td > 0 else None,
                'Avg Brier (Yds)': round(test_df['brier_yards'].mean(), 4),
                'Hit Rate (Yds)': round(test_df['hit_yards'].mean() * 100, 1),
                'Avg Prob (Yds)': round(test_df['prob_over_45.5_yds'].mean() * 100, 1),
                'Top20 Yds Hit %': round(top20_rate_yds * 100, 1),
                'Yds Edge (pp)': round((top20_rate_yds - base_rate_yds) * 100, 1),
                'Yds Lift': round(top20_rate_yds / base_rate_yds, 2) if base_rate_yds > 0 else None,
            })
        except Exception as e:
            continue
    
    # FIX: Return tuple with actual season
    return pd.DataFrame(results), actual

def generate_nfl_backtest_copy_text(results_df, season=2025):
    """
    Generates a clean, copy-pasteable text report.
    FIX: season parameter now receives the ACTUAL resolved season, not the requested one.
    """
    if results_df.empty:
        return "No backtest data available."
    
    lines = []
    lines.append("🏈 LAUNCHCAST NFL — BACKTEST REPORT")
    lines.append("=" * 50)
    
    # FIX: Header shows actual season that ran
    if len(results_df) > 0:
        week_min = int(results_df['Week'].min())
        week_max = int(results_df['Week'].max())
        lines.append(f"Season: {season} | Weeks graded: {len(results_df)} | Range: {week_min}-{week_max}")
    
    # TD SUMMARY
    lines.append("")
    lines.append("🎯 TOUCHDOWN PROPS")
    lines.append("-" * 50)
    
    avg_brier = results_df['Avg Brier (TD)'].mean()
    avg_hit = results_df['Hit Rate (TD)'].mean()
    avg_prob = results_df['Avg Prob (TD)'].mean()
    
    lines.append(f"Overall Avg Brier (TD): {avg_brier:.4f} (Lower is better)")
    lines.append(f"Overall Hit Rate (TD):  {avg_hit:.1f}%")
    lines.append(f"Overall Avg Prob (TD):  {avg_prob:.1f}%")
    
    if 'Top20 TD Hit %' in results_df.columns:
        avg_top20_td = results_df['Top20 TD Hit %'].mean()
        avg_edge_td = results_df['TD Edge (pp)'].mean()
        avg_lift_td = results_df['TD Lift'].mean()
        
        lines.append("")
        lines.append("TD EDGE METRICS")
        lines.append(f"Top-20 Hit Rate:      {avg_top20_td:.1f}%")
        lines.append(f"Slate Base Rate:      {avg_hit:.1f}%")
        lines.append(f"Edge:                 {avg_edge_td:+.1f}pp")
        lines.append(f"Lift:                 {avg_lift_td:.2f}x")
        
        if avg_edge_td >= 10:
            lines.append("✅ STRONG EDGE — Top TD picks hit significantly more than baseline")
        elif avg_edge_td >= 5:
            lines.append("🟡 MODEST EDGE — Some signal, but needs refinement")
        else:
            lines.append("⚠️ WEAK EDGE — TD model is calibrated but not discriminating")
    
    # YARDAGE SUMMARY
    if 'Top20 Yds Hit %' in results_df.columns:
        lines.append("")
        lines.append("📏 YARDAGE PROPS (Over 45.5)")
        lines.append("-" * 50)
        
        avg_brier_yds = results_df['Avg Brier (Yds)'].mean()
        avg_hit_yds = results_df['Hit Rate (Yds)'].mean()
        avg_prob_yds = results_df['Avg Prob (Yds)'].mean()
        
        lines.append(f"Overall Avg Brier (Yds): {avg_brier_yds:.4f}")
        lines.append(f"Overall Hit Rate (Yds):  {avg_hit_yds:.1f}%")
        lines.append(f"Overall Avg Prob (Yds):  {avg_prob_yds:.1f}%")
        
        avg_top20_yds = results_df['Top20 Yds Hit %'].mean()
        avg_edge_yds = results_df['Yds Edge (pp)'].mean()
        avg_lift_yds = results_df['Yds Lift'].mean()
        
        lines.append("")
        lines.append("YARDAGE EDGE METRICS")
        lines.append(f"Top-20 Hit Rate:      {avg_top20_yds:.1f}%")
        lines.append(f"Slate Base Rate:      {avg_hit_yds:.1f}%")
        lines.append(f"Edge:                 {avg_edge_yds:+.1f}pp")
        lines.append(f"Lift:                 {avg_lift_yds:.2f}x")
        
        if avg_edge_yds >= 10:
            lines.append("✅ STRONG EDGE — Top yardage picks hit significantly more than baseline")
        elif avg_edge_yds >= 5:
            lines.append("🟡 MODEST EDGE — Some yardage signal, but needs refinement")
        else:
            lines.append("⚠️ WEAK EDGE — Yardage model is calibrated but not discriminating")
    
    # WEEKLY BREAKDOWN
    lines.append("")
    lines.append("📊 WEEKLY BREAKDOWN")
    lines.append("-" * 50)
    
    header = (f"{'Wk':<3} | {'N':>4} | "
              f"{'BrierTD':>7} | {'Hit%':>5} | {'Prob%':>5} | {'TD Edge':>7} | "
              f"{'BrierYd':>7} | {'YHit%':>5} | {'YProb%':>6} | {'Yd Edge':>7}")
    lines.append(header)
    lines.append("-" * 50)
    
    for _, row in results_df.iterrows():
        line = (f"{int(row['Week']):<3} | {int(row['Players']):>4} | "
                f"{row['Avg Brier (TD)']:.4f} | {row['Hit Rate (TD)']:>5.1f} | "
                f"{row['Avg Prob (TD)']:>5.1f} | {row.get('TD Edge (pp)', 0):>+7.1f} | "
                f"{row['Avg Brier (Yds)']:.4f} | {row['Hit Rate (Yds)']:>5.1f} | "
                f"{row['Avg Prob (Yds)']:>6.1f} | {row.get('Yds Edge (pp)', 0):>+7.1f}")
        lines.append(line)
        
    lines.append("-" * 50)
    
    # KEY INSIGHTS
    if len(results_df) > 0:
        lines.append("")
        lines.append("🔍 KEY INSIGHTS")
        lines.append("-" * 50)
        
        best_td_week = results_df.loc[results_df['Avg Brier (TD)'].idxmin()]
        worst_td_week = results_df.loc[results_df['Avg Brier (TD)'].idxmax()]
        
        lines.append(f"• Best TD Calibration:  Week {int(best_td_week['Week'])} (Brier: {best_td_week['Avg Brier (TD)']:.4f})")
        lines.append(f"• Worst TD Calibration: Week {int(worst_td_week['Week'])} (Brier: {worst_td_week['Avg Brier (TD)']:.4f})")
        
        if 'TD Edge (pp)' in results_df.columns:
            best_td_edge = results_df.loc[results_df['TD Edge (pp)'].idxmax()]
            worst_td_edge = results_df.loc[results_df['TD Edge (pp)'].idxmin()]
            lines.append(f"• Best TD Edge:         Week {int(best_td_edge['Week'])} ({best_td_edge['TD Edge (pp)']:+.1f}pp)")
            lines.append(f"• Worst TD Edge:        Week {int(worst_td_edge['Week'])} ({worst_td_edge['TD Edge (pp)']:+.1f}pp)")
        
        if 'Yds Edge (pp)' in results_df.columns:
            best_yds_edge = results_df.loc[results_df['Yds Edge (pp)'].idxmax()]
            lines.append(f"• Best Yardage Edge:    Week {int(best_yds_edge['Week'])} ({best_yds_edge['Yds Edge (pp)']:+.1f}pp)")
    
    return "\n".join(lines)
