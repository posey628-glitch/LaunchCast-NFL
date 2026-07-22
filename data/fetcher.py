# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Dynamic Column Detection

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
    """Normalize column names to our standard names."""
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
    """Fetch weekly player stats."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            week_data = normalize_columns(week_data)
            
            # Calculate derived metrics safely
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
    """Fetch team defensive stats with dynamic column detection."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        
        # Try to get defensive stats - try multiple approaches
        def_stats = pd.DataFrame()
        
        # Approach 1: Try import_seasonal_data with stat_type='def'
        try:
            def_stats = nfl.import_seasonal_data(year, stat_type='def')
            if not def_stats.empty:
                st.success(f"✅ Loaded defensive stats via import_seasonal_data")
        except Exception as e1:
            st.warning(f"import_seasonal_data failed: {e1}")
            
            # Approach 2: Try to derive from weekly data
            try:
                weekly = nfl.import_weekly_data([year])
                # Group by opponent team to get defensive aggregates
                def_agg = weekly.groupby(['opponent_team', 'week']).agg({
                    'passing_yards': 'sum',
                    'passing_tds': 'sum',
                    'rushing_yards': 'sum',
                    'rushing_tds': 'sum',
                    'sacks': 'sum',
                }).reset_index()
                
                def_agg = def_agg.rename(columns={
                    'opponent_team': 'team',
                    'passing_yards': 'pass_yards_allowed',
                    'passing_tds': 'pass_tds_allowed',
                    'rushing_yards': 'rush_yards_allowed',
                    'rushing_tds': 'rush_tds_allowed',
                })
                
                def_stats = def_agg
                st.success(f"✅ Loaded defensive stats derived from weekly data")
            except Exception as e2:
                st.warning(f"Weekly derivation failed: {e2}")
        
        if not def_stats.empty:
            # Return whatever columns we have
            return def_stats
    except Exception as e:
        st.warning(f"Defensive stats fetch failed: {e}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix."""
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("No defensive stats available - proceeding without matchup adjustments")
        return players
    
    # Merge offense with defense
    latest_def = def_stats.sort_values('week').groupby('team').last().reset_index()
    
    # Only merge columns that exist
    def_cols_to_merge = {}
    for col in ['pass_yards_allowed', 'rush_yards_allowed', 'pass_tds_allowed', 'sacks']:
        if col in latest_def.columns:
            def_cols_to_merge[col] = f'opp_{col}'
    
    if def_cols_to_merge:
        latest_def = latest_def.rename(columns=def_cols_to_merge)
        players = players.merge(
            latest_def[['team'] + list(def_cols_to_merge.values())],
            left_on='opponent_team',
            right_on='team',
            how='left',
            suffixes=('', '_def')
        )
        
        if 'team_def' in players.columns:
            players = players.drop(columns=['team_def'])
    
    return players
