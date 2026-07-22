# data/fetcher.py
# LaunchCast NFL — Data Fetcher

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2024
    FALLBACK_SEASON = 2023
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1

def normalize_columns(df):
    rename_map = {}
    if 'team' not in df.columns:
        if 'recent_team' in df.columns:
            rename_map['recent_team'] = 'team'
        elif 'posteam' in df.columns:
            rename_map['posteam'] = 'team'
    if 'opponent_team' not in df.columns:
        if 'defteam' in df.columns:
            rename_map['defteam'] = 'opponent_team'
        elif 'opp' in df.columns:
            rename_map['opp'] = 'opponent_team'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            week_data = normalize_columns(week_data)
            
            if 'team_dropbacks' not in week_data.columns:
                if 'team' in week_data.columns and 'routes' in week_data.columns:
                    team_dropbacks = week_data.groupby(['team', 'week'])['routes'].sum().reset_index()
                    team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
                    week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
                else:
                    week_data['team_dropbacks'] = 0
            
            week_data['route_participation_pct'] = np.where(
                week_data.get('team_dropbacks', 0) > 0,
                (week_data.get('routes', 0) / week_data['team_dropbacks']) * 100,
                0
            )
            
            week_data['adot'] = np.where(
                week_data.get('targets', 0) > 0,
                week_data.get('air_yards', 0) / week_data['targets'],
                0
            )
            
            if 'team_targets' not in week_data.columns:
                if 'team' in week_data.columns and 'targets' in week_data.columns:
                    team_targets = week_data.groupby(['team', 'week'])['targets'].sum().reset_index()
                    team_targets.columns = ['team', 'week', 'team_targets']
                    week_data = week_data.merge(team_targets, on=['team', 'week'], how='left')
                else:
                    week_data['team_targets'] = 0
            
            week_data['target_share'] = np.where(
                week_data.get('team_targets', 0) > 0,
                week_data.get('targets', 0) / week_data['team_targets'],
                0
            )
            
            return week_data
    except Exception as e:
        st.warning(f"Player stats fetch failed: {e}")
    
    return pd.DataFrame()

def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        # FIX: Use s_type='def' instead of stat_type='def'
        def_stats = nfl.import_seasonal_data([year], s_type='def')
        
        if not def_stats.empty:
            available_cols = ['team', 'week', 'season']
            optional_cols = ['pass_epa', 'rush_epa', 'pressure_rate', 'stuff_rate']
            
            for col in optional_cols:
                if col in def_stats.columns:
                    available_cols.append(col)
            
            return def_stats[available_cols]
    except Exception as e:
        st.warning(f"Defensive stats fetch failed: {e}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    if year is None:
        year = PREFERRED_SEASON
    
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("No defensive stats available")
        return players
    
    latest_def = def_stats.sort_values('week').groupby('team').last().reset_index()
    
    def_cols = {
        'pass_epa': 'opp_pass_epa_allowed',
        'rush_epa': 'opp_rush_epa_allowed',
        'pressure_rate': 'opp_pressure_rate',
        'stuff_rate': 'opp_stuff_rate'
    }
    
    valid_cols = {k: v for k, v in def_cols.items() if k in latest_def.columns}
    latest_def = latest_def.rename(columns=valid_cols)
    
    matchup_df = players.merge(
        latest_def[['team'] + list(valid_cols.keys())],
        left_on='opponent_team',
        right_on='team',
        how='left',
        suffixes=('', '_def')
    )
    
    if 'team_def' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_def'])
    
    return matchup_df
