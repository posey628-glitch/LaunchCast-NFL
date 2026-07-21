# data/fetcher.py
# LaunchCast NFL — Data Fetcher (nflverse via nfl_data_py)

import nfl_data_py as nfl
import pandas as pd
import numpy as np

# FORCE 2025 DATA FOR OFFSEASON TESTING
CURRENT_SEASON = 2025

# ============================================================================
# 1. PLAYER OFFENSE
# ============================================================================
def get_weekly_player_stats(week: int, year: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pulls weekly player stats (Targets, Routes, Air Yards, etc.)."""
    try:
        stats = nfl.import_weekly_data([year])
        week_df = stats[stats['week'] == week].copy()
        
        # Calculate derived metrics
        week_df['route_participation_pct'] = np.where(
            week_df['team_dropbacks'] > 0, 
            (week_df['routes'] / week_df['team_dropbacks']) * 100, 0
        )
        week_df['adot'] = np.where(
            week_df['targets'] > 0, 
            week_df['air_yards'] / week_df['targets'], 0
        )
        
        return week_df
    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return pd.DataFrame()

# ============================================================================
# 2. TEAM DEFENSE
# ============================================================================
def get_team_defensive_stats(year: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pulls season-long team defensive stats."""
    try:
        team_stats = nfl.import_team_stats([year])
        def_stats = team_stats[team_stats['side'] == 'def'].copy()
        
        cols_to_keep = [
            'team', 'week', 'season', 
            'pass_epa', 'rush_epa', 'total_epa',
            'pass_success_rate', 'rush_success_rate',
            'pressure_rate', 'blitz_rate',
            'defensive_line_yards', 'stuff_rate'
        ]
        
        existing_cols = [c for c in cols_to_keep if c in def_stats.columns]
        return def_stats[existing_cols]
        
    except Exception as e:
        print(f"Error fetching defensive stats: {e}")
        return pd.DataFrame()

# ============================================================================
# 3. THE MATCHUP MATRIX
# ============================================================================
def build_matchup_matrix(week: int, year: int = CURRENT_SEASON) -> pd.DataFrame:
    """Merges Player Offense with Opponent Defense."""
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
        
    def_stats = get_team_defensive_stats(year)
    if def_stats.empty:
        return players
        
    latest_def = def_stats.sort_values('week').groupby('team').last().reset_index()
    
    def_cols_to_merge = {
        'pass_epa': 'opp_pass_epa_allowed',
        'rush_epa': 'opp_rush_epa_allowed',
        'pressure_rate': 'opp_pressure_rate',
        'stuff_rate': 'opp_stuff_rate'
    }
    
    valid_def_cols = {k: v for k, v in def_cols_to_merge.items() if k in latest_def.columns}
    latest_def = latest_def.rename(columns=valid_def_cols)
    
    matchup_df = players.merge(
        latest_def[['team'] + list(valid_def_cols.keys())], 
        left_on='opponent_team', 
        right_on='team', 
        how='left',
        suffixes=('', '_opp')
    )
    
    if 'team_opp' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_opp'])
        
    return matchup_df
