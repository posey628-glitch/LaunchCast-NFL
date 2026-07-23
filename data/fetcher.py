# data/fetcher.py
# LaunchCast NFL — Data Fetcher V3
# FIXES: defensive merge direction, column normalization, caching

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2025  # FIXED: was 2024
    FALLBACK_SEASON = 2024
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1

# ============================================================================
# CACHED RAW DATA LOADER (FIX: download once per hour, not twice per load)
# ============================================================================
@st.cache_data(ttl=3600)
def _load_weekly_raw(year: int) -> pd.DataFrame:
    """Load raw weekly data once and cache it."""
    import nfl_data_py as nfl
    return nfl.import_weekly_data([year])

# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================
def normalize_columns(df):
    """Normalize column names from nflverse to our standard names."""
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

# ============================================================================
# PLAYER STATS
# ============================================================================
def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch weekly player stats with safe column handling."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        all_data = _load_weekly_raw(year)  # Uses cached loader
        week_data = all_data[all_data['week'] == week].copy()
        
        if week_data.empty:
            return pd.DataFrame()
            
        week_data = normalize_columns(week_data)
        
        # Team dropbacks (sum of routes per team)
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
        
        # aDOT: air yards only (NOT total yards)
        week_data['adot'] = np.where(
            week_data.get('targets', 0) > 0,
            week_data.get('air_yards', 0) / week_data['targets'],
            8.0  # league avg fallback
        )
        
        # Yards per target (INCLUDES YAC — this is total yards / targets)
        week_data['yds_per_tgt'] = np.where(
            week_data.get('targets', 0) > 0,
            week_data.get('receiving_yards', 0) / week_data['targets'],
            11.0  # league avg fallback
        )
        
        # TD per target
        week_data['td_per_tgt'] = np.where(
            week_data.get('targets', 0) > 0,
            week_data.get('receiving_tds', 0) / week_data['targets'],
            0.05  # league avg fallback
        )
        
        # Target share
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

# ============================================================================
# DEFENSIVE STATS (FIX: normalize columns, group by defending team correctly)
# ============================================================================
def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    """
    Fetch team defensive stats by aggregating what each defense ALLOWED.
    Returns a DataFrame where each row is a DEFENDING team's stats.
    The 'team' column in the result = the defending team.
    """
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        all_data = _load_weekly_raw(year)  # Uses cached loader
        all_data = normalize_columns(all_data)  # FIX: normalize before grouping
        
        # Group by the DEFENDING team (opponent_team in offensive rows)
        # Sum what they allowed: targets, receptions, yards, TDs
        agg_dict = {}
        if 'targets' in all_data.columns: agg_dict['targets'] = 'sum'
        if 'receptions' in all_data.columns: agg_dict['receptions'] = 'sum'
        if 'receiving_yards' in all_data.columns: agg_dict['receiving_yards'] = 'sum'
        if 'receiving_tds' in all_data.columns: agg_dict['receiving_tds'] = 'sum'
        if 'air_yards' in all_data.columns: agg_dict['air_yards'] = 'sum'
        
        if not agg_dict:
            return pd.DataFrame()
            
        def_agg = all_data.groupby(['opponent_team', 'week']).agg(agg_dict).reset_index()
        
        # Rename: opponent_team (defense) -> team, and prefix stats as "allowed"
        def_agg = def_agg.rename(columns={
            'opponent_team': 'team',  # This is the DEFENDING team
            'targets': 'targets_allowed',
            'receptions': 'receptions_allowed',
            'receiving_yards': 'yards_allowed',
            'receiving_tds': 'tds_allowed',
            'air_yards': 'air_yards_allowed',
        })
        
        # Calculate defensive rates
        def_agg['def_yds_per_tgt'] = np.where(
            def_agg['targets_allowed'] > 0,
            def_agg['yards_allowed'] / def_agg['targets_allowed'],
            11.0
        )
        def_agg['def_td_per_tgt'] = np.where(
            def_agg['targets_allowed'] > 0,
            def_agg['tds_allowed'] / def_agg['targets_allowed'],
            0.05
        )
        
        # Filter to current week
        def_agg = def_agg[def_agg['week'] == week]
        
        return def_agg
    except Exception as e:
        st.warning(f"Defensive stats fetch failed: {e}")
        return pd.DataFrame()

# ============================================================================
# MATCHUP MATRIX (FIX: merge on opponent_team, not team)
# ============================================================================
def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """
    Build matchup matrix. Each offensive player gets the defensive stats
    of the team they are FACING (opponent_team), not their own team.
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
    
    # FIX: Merge on players.opponent_team = def_stats.team
    # This gives each offensive player the defensive stats of the team they face
    def_cols = ['team', 'week', 'targets_allowed', 'receptions_allowed', 
                'yards_allowed', 'tds_allowed', 'def_yds_per_tgt', 'def_td_per_tgt']
    def_cols = [c for c in def_cols if c in def_stats.columns]
    
    matchup_df = players.merge(
        def_stats[def_cols],
        left_on=['opponent_team', 'week'],  # The defense the player faces
        right_on=['team', 'week'],          # The defending team in def_stats
        how='left',
        suffixes=('', '_def')
    )
    
    # Drop the duplicate team column from def_stats
    if 'team_def' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_def'])
    
    return matchup_df
