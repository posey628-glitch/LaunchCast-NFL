# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Column Safety & Fallbacks

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Force specific season for testing
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2024
    FALLBACK_SEASON = 2023
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch weekly player stats with safe column handling."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        week_data = nfl.import_weekly_data([year])
        week_data = week_data[week_data['week'] == week].copy()
        
        if week_data.empty:
            return pd.DataFrame()
            
        # SAFELY compute team_dropbacks (sum of routes per team/week)
        # Use 'routes' if available, otherwise fall back to 'receptions' 
        # (since routes ≈ receptions for most non-RB skill positions)
        route_col = 'routes' if 'routes' in week_data.columns else 'receptions'
        
        if 'team' in week_data.columns and route_col in week_data.columns:
            team_dropbacks = week_data.groupby(['team', 'week'])[route_col].sum().reset_index()
            team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
            week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
        else:
            week_data['team_dropbacks'] = 0
            
        # SAFELY compute route participation %
        week_data['route_participation_pct'] = np.where(
            week_data.get('team_dropbacks', 0) > 0,
            (week_data.get(route_col, 0) / week_data['team_dropbacks']) * 100,
            0
        )
        
        # SAFELY compute aDOT
        week_data['adot'] = np.where(
            week_data.get('targets', 0) > 0,
            week_data.get('air_yards', 0) / week_data['targets'],
            0
        )
        
        # SAFELY compute target share
        if 'team_targets' in week_data.columns:
            week_data['target_share'] = np.where(
                week_data['team_targets'] > 0,
                week_data['targets'] / week_data['team_targets'],
                0
            )
        else:
            week_data['target_share'] = 0
            
        return week_data
        
    except Exception as e:
        st.warning(f"Player stats fetch failed for {year}: {e}")
        return pd.DataFrame()

def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        def_data = nfl.import_weekly_data([year])
        
        # Group by opponent_team to get defensive aggregates
        def_agg = def_data.groupby(['opponent_team', 'week']).agg({
            'pressure_rate_when_targeted': 'mean',
            'clean_pocket_pct': 'mean',
            'avg_time_to_throw': 'mean',
            'separation': 'mean',
            'targets': 'sum',
            'receptions': 'sum',
        }).reset_index()
        
        def_agg = def_agg.rename(columns={
            'opponent_team': 'team',
            'pressure_rate_when_targeted': 'def_pressure_rate',
            'clean_pocket_pct': 'def_clean_pocket_rate',
            'avg_time_to_throw': 'def_time_to_throw',
            'separation': 'def_separation_allowed',
        })
        
        def_agg['def_catch_rate_allowed'] = np.where(
            def_agg['targets'] > 0,
            def_agg['receptions'] / def_agg['targets'],
            0.65
        )
        
        def_agg = def_agg[def_agg['week'] == week]
        return def_agg
        
    except Exception as e:
        st.warning(f"Defensive stats fetch failed for {year}: {e}")
        return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix merging offense with defense."""
    if year is None:
        year = PREFERRED_SEASON
        
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("No defensive stats available - proceeding without matchup adjustments")
        return players
    
    def_cols = {
        'def_pressure_rate': 'opp_pressure_rate',
        'def_clean_pocket_rate': 'opp_clean_pocket_rate',
        'def_time_to_throw': 'opp_time_to_throw',
        'def_separation_allowed': 'opp_separation_allowed',
        'def_catch_rate_allowed': 'opp_catch_rate_allowed',
    }
    
    valid_cols = {k: v for k, v in def_cols.items() if k in def_stats.columns}
    def_stats_renamed = def_stats.rename(columns=valid_cols)
    
    matchup_df = players.merge(
        def_stats_renamed[['team'] + list(valid_cols.values())],
        left_on='opponent_team',
        right_on='team',
        how='left',
        suffixes=('', '_def')
    )
    
    if 'team_def' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_def'])
        
    return matchup_df
