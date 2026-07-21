# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Fixed Defensive Stats

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
    """Fixes column name mismatches from nfl_data_py."""
    rename_map = {}
    if 'team' not in df.columns:
        if 'recent_team' in df.columns: rename_map['recent_team'] = 'team'
        elif 'posteam' in df.columns: rename_map['posteam'] = 'team'
    if 'opponent_team' not in df.columns:
        if 'defteam' in df.columns: rename_map['defteam'] = 'opponent_team'
        elif 'opp' in df.columns: rename_map['opp'] = 'opponent_team'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch weekly player stats."""
    if year is None: year = PREFERRED_SEASON
    try:
        import nfl_data_py as nfl
        st.info(f"🔄 Attempting to load {year} Week {week} data...")
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            week_data = normalize_columns(week_data)
            st.success(f"✅ Successfully loaded {len(week_data)} players from {year} Week {week}")
            
            # Calculate derived metrics safely
            if 'team_dropbacks' not in week_data.columns and 'team' in week_data.columns and 'routes' in week_data.columns:
                team_dropbacks = week_data.groupby(['team', 'week'])['routes'].sum().reset_index()
                team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
                week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
            else:
                week_data['team_dropbacks'] = week_data.get('team_dropbacks', 0)
            
            week_data['route_participation_pct'] = np.where(week_data['team_dropbacks'] > 0, (week_data.get('routes', 0) / week_data['team_dropbacks']) * 100, 0)
            week_data['adot'] = np.where(week_data.get('targets', 0) > 0, week_data.get('air_yards', 0) / week_data['targets'], 0)
            
            if 'team_targets' not in week_data.columns and 'team' in week_data.columns and 'targets' in week_data.columns:
                team_targets = week_data.groupby(['team', 'week'])['targets'].sum().reset_index()
                team_targets.columns = ['team', 'week', 'team_targets']
                week_data = week_data.merge(team_targets, on=['team', 'week'], how='left')
            else:
                week_data['team_targets'] = week_data.get('team_targets', 0)
                
            week_data['target_share'] = np.where(week_data['team_targets'] > 0, week_data['targets'] / week_data['team_targets'], 0)
            return week_data
    except Exception as e:
        st.warning(f"❌ {year} data fetch failed: {str(e)[:150]}")
    return pd.DataFrame()

def get_team_defensive_stats(year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats using the reliable seasonal data endpoint."""
    if year is None: year = PREFERRED_SEASON
    try:
        import nfl_data_py as nfl
        st.info(f"🛡️ Fetching {year} team defensive stats...")
        # import_seasonal_data with stat_type='def' is the standard nflverse way to get team defense
        def_stats = nfl.import_seasonal_data(year, stat_type='def')
        
        if not def_stats.empty:
            # Keep only the most predictive columns
            cols_to_keep = ['team', 'season', 'pass_epa', 'rush_epa', 'pressure_rate', 'stuff_rate']
            existing_cols = [c for c in cols_to_keep if c in def_stats.columns]
            return def_stats[existing_cols]
    except Exception as e:
        st.warning(f"⚠️ Defensive stats fetch failed: {str(e)[:100]}")
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix merging offense with defense."""
    players = get_weekly_player_stats(week, year)
    if players.empty: return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(year)
    if def_stats.empty:
        st.warning("⚠️ No defensive stats available - proceeding with neutral matchups")
        return players
    
    # Merge offense with defense
    def_cols = {'pass_epa': 'opp_pass_epa_allowed', 'rush_epa': 'opp_rush_epa_allowed', 'pressure_rate': 'opp_pressure_rate', 'stuff_rate': 'opp_stuff_rate'}
    valid_cols = {k: v for k, v in def_cols.items() if k in def_stats.columns}
    latest_def = def_stats.rename(columns=valid_cols)
    
    matchup_df = players.merge(latest_def[['team'] + list(valid_cols.values())], left_on='opponent_team', right_on='team', how='left', suffixes=('', '_opp'))
    if 'team_opp' in matchup_df.columns: matchup_df = matchup_df.drop(columns=['team_opp'])
    
    st.success(f"✅ Built matchup matrix: {len(matchup_df)} players with defensive context")
    return matchup_df
