# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Advanced Metrics
# Pulls player stats, defensive stats, AND advanced metrics
# (tempo, play-action, RPO, red zone, slot rate, separation, etc.)

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Force specific season for testing
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# During offseason (Jan-Aug), use 2024 (2025 not available yet)
# During season (Sept-Dec), use current year
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
    """Fetch weekly player stats with ALL advanced metrics."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            week_data = normalize_columns(week_data)
            
            # === DERIVED METRICS (compute from raw data) ===
            
            # Target efficiency
            week_data['catch_rate'] = np.where(
                week_data.get('targets', 0) > 0,
                week_data.get('receptions', 0) / week_data['targets'],
                0
            )
            
            # YAC efficiency
            week_data['yac_per_rec'] = np.where(
                week_data.get('receptions', 0) > 0,
                week_data.get('yards_after_catch', 0) / week_data['receptions'],
                0
            )
            
            # Air yards per target
            week_data['ay_per_target'] = np.where(
                week_data.get('targets', 0) > 0,
                week_data.get('air_yards', 0) / week_data['targets'],
                0
            )
            
            # Drop rate
            week_data['drop_rate'] = np.where(
                week_data.get('targets', 0) > 0,
                week_data.get('drops', 0) / week_data['targets'],
                0
            )
            
            # Contested catch rate
            week_data['contested_catch_rate'] = np.where(
                week_data.get('contested_targets', 0) > 0,
                week_data.get('contested_catches', 0) / week_data['contested_targets'],
                0
            )
            
            # Route participation
            week_data['route_participation_pct'] = np.where(
                week_data.get('team_dropbacks', 0) > 0,
                (week_data.get('routes', 0) / week_data['team_dropbacks']) * 100,
                0
            )
            
            # Yards per route run
            week_data['yards_per_route'] = np.where(
                week_data.get('routes', 0) > 0,
                week_data.get('receiving_yards', 0) / week_data['routes'],
                0
            )
            
            # Target share
            week_data['target_share'] = np.where(
                week_data.get('team_targets', 0) > 0,
                week_data.get('targets', 0) / week_data['team_targets'],
                0
            )
            
            # Air yards share
            week_data['air_yards_share'] = np.where(
                week_data.get('team_air_yards', 0) > 0,
                week_data.get('air_yards', 0) / week_data['team_air_yards'],
                0
            )
            
            # Red zone target share
            week_data['rz_target_share'] = np.where(
                week_data.get('team_rz_targets', 0) > 0,
                week_data.get('red_zone_targets', 0) / week_data['team_rz_targets'],
                0
            )
            
            # Slot rate
            week_data['slot_rate'] = np.where(
                week_data.get('routes', 0) > 0,
                week_data.get('slot_routes', 0) / week_data['routes'],
                0
            )
            
            # Separation average (feet)
            week_data['separation_avg'] = week_data.get('separation', 0)
            
            # First read target share
            week_data['first_read_share'] = np.where(
                week_data.get('targets', 0) > 0,
                week_data.get('first_read_targets', 0) / week_data['targets'],
                0
            )
            
            # Pressure rate when targeted
            week_data['pressure_rate_targeted'] = week_data.get('pressure_rate_when_targeted', 0)
            
            # Clean pocket rate
            week_data['clean_pocket_rate'] = week_data.get('clean_pocket_pct', 0)
            
            # Time to throw
            week_data['time_to_throw'] = week_data.get('avg_time_to_throw', 0)
            
            return week_data
    except Exception as e:
        st.warning(f"Player stats fetch failed: {e}")
    
    return pd.DataFrame()

def get_team_advanced_metrics(week: int, year: int = None) -> pd.DataFrame:
    """Fetch team-level advanced metrics (tempo, play-action, RPO, etc.)."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        
        # Team situational stats
        team_data = nfl.import_weekly_data([year])
        
        # Aggregate to team level
        team_agg = team_data[team_data['week'] <= week].groupby(['team', 'week']).agg({
            'play_action_pct': 'mean',
            'rpo_pct': 'mean',
            'motion_pct': 'mean',
            'shift_pct': 'mean',
            'tempo_seconds_per_play': 'mean',
            'neutral_script_pass_rate': 'mean',
            'red_zone_targets': 'sum',
            'team_targets': 'sum',
            'team_air_yards': 'sum',
            'team_dropbacks': 'sum',
        }).reset_index()
        
        # Compute derived team metrics
        team_agg['rz_target_share_team'] = np.where(
            team_agg['team_targets'] > 0,
            team_agg['red_zone_targets'] / team_agg['team_targets'],
            0
        )
        
        # Pace factor (inverse of seconds per play — faster = more plays)
        team_agg['pace_factor'] = np.where(
            team_agg['tempo_seconds_per_play'] > 0,
            30 / team_agg['tempo_seconds_per_play'],  # normalized to 30s baseline
            1.0
        )
        
        return team_agg
    except Exception as e:
        st.warning(f"Team advanced metrics fetch failed: {e}")
    
    return pd.DataFrame()

def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats including advanced metrics."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        
        # Get defensive stats
        def_data = nfl.import_weekly_data([year])
        
        # Filter to defensive rows (opponent_team is the defense)
        # Group by opponent_team to get defensive aggregates
        def_agg = def_data.groupby(['opponent_team', 'week']).agg({
            'pressure_rate_when_targeted': 'mean',
            'clean_pocket_pct': 'mean',
            'avg_time_to_throw': 'mean',
            'separation': 'mean',
            'contested_targets': 'sum',
            'contested_catches': 'sum',
            'targets': 'sum',
            'receptions': 'sum',
        }).reset_index()
        
        # Rename for clarity
        def_agg = def_agg.rename(columns={
            'opponent_team': 'team',
            'pressure_rate_when_targeted': 'def_pressure_rate',
            'clean_pocket_pct': 'def_clean_pocket_rate',
            'avg_time_to_throw': 'def_time_to_throw',
            'separation': 'def_separation_allowed',
        })
        
        # Contested catch rate allowed
        def_agg['def_contested_catch_rate'] = np.where(
            def_agg['contested_targets'] > 0,
            def_agg['contested_catches'] / def_agg['contested_targets'],
            0.5  # league average fallback
        )
        
        # Catch rate allowed
        def_agg['def_catch_rate_allowed'] = np.where(
            def_agg['targets'] > 0,
            def_agg['receptions'] / def_agg['targets'],
            0.65  # league average fallback
        )
        
        # Filter to current week
        def_agg = def_agg[def_agg['week'] == week]
        
        return def_agg
    except Exception as e:
        st.warning(f"Defensive stats fetch failed: {e}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build complete matchup matrix with ALL advanced metrics."""
    if year is None:
        year = PREFERRED_SEASON
    
    # Get player stats
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    # Get team advanced metrics
    team_advanced = get_team_advanced_metrics(week, year)
    if not team_advanced.empty:
        # Merge team metrics onto players
        players = players.merge(
            team_advanced[['team', 'week', 'play_action_pct', 'rpo_pct', 
                          'motion_pct', 'shift_pct', 'pace_factor',
                          'neutral_script_pass_rate', 'rz_target_share_team']],
            on=['team', 'week'],
            how='left'
        )
    
    # Get defensive stats
    def_stats = get_team_defensive_stats(week, year)
    if not def_stats.empty:
        # Merge defensive stats
        players = players.merge(
            def_stats[['team', 'week', 'def_pressure_rate', 'def_clean_pocket_rate',
                      'def_time_to_throw', 'def_separation_allowed',
                      'def_contested_catch_rate', 'def_catch_rate_allowed']],
            left_on=['opponent_team', 'week'],
            right_on=['team', 'week'],
            how='left',
            suffixes=('', '_def')
        )
        
        # Drop duplicate team column
        if 'team_def' in players.columns:
            players = players.drop(columns=['team_def'])
    
    return players
