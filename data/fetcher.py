# data/fetcher.py
# LaunchCast NFL — Data Fetcher
# Fully defensive: handles missing columns, normalizes team names, and builds the matchup matrix safely.

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Determine season based on current date
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2024  # Use 2024 for offseason testing
    FALLBACK_SEASON = 2023
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1

def normalize_columns(df):
    """
    Fixes column name mismatches from nfl_data_py.
    Renames 'recent_team' or 'posteam' to 'team'.
    Renames 'defteam' or 'opp' to 'opponent_team'.
    """
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
    """
    Fetches weekly player stats.
    Safely handles missing columns like 'routes', 'team_dropbacks', etc.
    """
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if week_data.empty:
            return pd.DataFrame()
            
        # Normalize team names first
        week_data = normalize_columns(week_data)
        
        # Safely calculate team_dropbacks (sum of routes per team)
        # If 'routes' doesn't exist, we can't calculate it, so default to 0
        if 'team_dropbacks' not in week_data.columns:
            if 'team' in week_data.columns and 'routes' in week_data.columns:
                team_dropbacks = week_data.groupby(['team', 'week'])['routes'].sum().reset_index()
                team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
                week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
            else:
                week_data['team_dropbacks'] = 0
                
        # Calculate route participation safely
        week_data['route_participation_pct'] = np.where(
            week_data.get('team_dropbacks', 0) > 0,
            (week_data.get('routes', 0) / week_data['team_dropbacks']) * 100,
            0
        )
        
        # Calculate aDOT safely
        week_data['adot'] = np.where(
            week_data.get('targets', 0) > 0,
            week_data.get('air_yards', 0) / week_data['targets'],
            0
        )
        
        # Calculate target share safely
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
    """
    Fetches team defensive stats by grouping weekly data by opponent.
    This avoids the import_seasonal_data parameter errors.
    """
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        all_data = nfl.import_weekly_data([year])
        
        # Group by opponent team to get defensive aggregates
        # We use .agg with a dictionary to handle columns that might not exist
        agg_dict = {}
        if 'pass_ypa' in all_data.columns: agg_dict['pass_ypa'] = 'mean'
        if 'pass_epa' in all_data.columns: agg_dict['pass_epa'] = 'mean'
        if 'rush_ypa' in all_data.columns: agg_dict['rush_ypa'] = 'mean'
        if 'rush_epa' in all_data.columns: agg_dict['rush_epa'] = 'mean'
        if 'sacks' in all_data.columns: agg_dict['sacks'] = 'sum'
        if 'qb_hits' in all_data.columns: agg_dict['qb_hits'] = 'sum'
        
        if not agg_dict:
            return pd.DataFrame()
            
        def_agg = all_data.groupby(['opponent_team', 'week']).agg(agg_dict).reset_index()
        
        # Rename for clarity
        rename_map = {'opponent_team': 'team'}
        if 'pass_ypa' in def_agg.columns: rename_map['pass_ypa'] = 'opp_pass_ypa_allowed'
        if 'pass_epa' in def_agg.columns: rename_map['pass_epa'] = 'opp_pass_epa_allowed'
        if 'rush_ypa' in def_agg.columns: rename_map['rush_ypa'] = 'opp_rush_ypa_allowed'
        if 'rush_epa' in def_agg.columns: rename_map['rush_epa'] = 'opp_rush_epa_allowed'
        
        def_agg = def_agg.rename(columns=rename_map)
        
        return def_agg
        
    except Exception as e:
        st.warning(f"Defensive stats fetch failed: {e}")
        return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """
    Builds the matchup matrix by merging player stats with defensive stats.
    """
    if year is None:
        year = PREFERRED_SEASON
        
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
        
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("No defensive stats available - proceeding without matchup adjustments")
        return players
        
    # Merge offense with defense
    # Find common columns to merge on
    merge_cols = ['team']
    if 'week' in players.columns and 'week' in def_stats.columns:
        merge_cols.append('week')
        
    # Rename defensive columns to avoid collision
    def_cols_to_rename = {}
    for col in def_stats.columns:
        if col not in merge_cols and col not in players.columns:
            def_cols_to_rename[col] = f"opp_{col}"
            
    def_stats_renamed = def_stats.rename(columns=def_cols_to_rename)
    
    # Perform the merge
    matchup_df = players.merge(
        def_stats_renamed,
        on=merge_cols,
        how='left'
    )
    
    return matchup_df
