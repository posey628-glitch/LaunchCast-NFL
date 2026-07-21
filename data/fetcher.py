# data/fetcher.py
# LaunchCast NFL — Data Fetcher (nflverse via nfl_data_py)
# Pulls weekly player stats AND team defensive matchups.

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime

# Force 2025 for testing during offseason
CURRENT_SEASON = 2025  # Changed from datetime.now().year

# ============================================================================
# 1. PLAYER OFFENSE (The "Hitter" Data)
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
# 2. TEAM DEFENSE (The "Pitcher" Data)
# ============================================================================
def get_team_defensive_stats(year: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Pulls season-long team defensive stats. 
    This is the 'Pitcher Season Stats' equivalent.
    Includes EPA/play allowed, Success Rate allowed, and Pressure Rate.
    """
    try:
        # import_team_stats returns both offensive and defensive rows
        team_stats = nfl.import_team_stats([year])
        
        # Filter to only defensive rows
        def_stats = team_stats[team_stats['side'] == 'def'].copy()
        
        # We only need the most predictive columns for our model
        cols_to_keep = [
            'team', 'week', 'season', 
            'pass_epa', 'rush_epa', 'total_epa', # EPA allowed (lower is better for defense)
            'pass_success_rate', 'rush_success_rate',
            'pressure_rate', 'blitz_rate',
            'defensive_line_yards', # Yards allowed at the line of scrimmage
            'stuff_rate' # % of runs stopped at or behind the line
        ]
        
        # Keep only columns that exist (nflverse updates column names occasionally)
        existing_cols = [c for c in cols_to_keep if c in def_stats.columns]
        return def_stats[existing_cols]
        
    except Exception as e:
        print(f"Error fetching defensive stats: {e}")
        return pd.DataFrame()

# ============================================================================
# 3. THE MATCHUP MATRIX (Merging Offense vs Defense)
# ============================================================================
def build_matchup_matrix(week: int, year: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    The Core Engine: Merges Player Offense with Opponent Defense.
    Returns a dataframe where every row is a Player, enriched with the 
    specific defensive stats of the team they are facing that week.
    """
    # 1. Get Player Data
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
        
    # 2. Get Defensive Data (We use the season-long average for the opponent)
    # In NFL, season-long defense is more stable than weekly defense.
    def_stats = get_team_defensive_stats(year)
    if def_stats.empty:
        return players # Return players without defensive context if def fails
        
    # Get the LATEST defensive stats for each team (most predictive)
    latest_def = def_stats.sort_values('week').groupby('team').last().reset_index()
    
    # 3. Merge: Player's 'opponent_team' matches Defense's 'team'
    # We rename defensive columns to avoid collision with player columns
    def_cols_to_merge = {
        'pass_epa': 'opp_pass_epa_allowed',
        'rush_epa': 'opp_rush_epa_allowed',
        'pressure_rate': 'opp_pressure_rate',
        'stuff_rate': 'opp_stuff_rate'
    }
    
    # Only merge columns that actually exist in our defensive dataframe
    valid_def_cols = {k: v for k, v in def_cols_to_merge.items() if k in latest_def.columns}
    latest_def = latest_def.rename(columns=valid_def_cols)
    
    merge_cols = ['team'] + list(valid_def_cols.values())
    
    # The Merge
    matchup_df = players.merge(
        latest_def[['team'] + list(valid_def_cols.keys())], 
        left_on='opponent_team', 
        right_on='team', 
        how='left',
        suffixes=('', '_opp')
    )
    
    # Cleanup: drop the duplicate 'team_opp' column created by the merge
    if 'team_opp' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_opp'])
        
    return matchup_df
