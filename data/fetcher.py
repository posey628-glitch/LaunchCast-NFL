# data/fetcher.py
# LaunchCast NFL — Data Fetcher (Fixed to use REAL nflverse columns)

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
    """Fetch weekly player stats."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        st.info(f"🔄 Attempting to load {year} Week {week} data...")
        
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            week_data = normalize_columns(week_data)
            st.success(f"✅ Successfully loaded {len(week_data)} players from {year} Week {week}")
            
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
        else:
            st.warning(f"No data for {year} Week {week}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Player stats fetch failed: {e}")
        return pd.DataFrame()

def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats using REAL nflverse columns."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        st.info(f"🛡️ Fetching {year} defensive stats...")
        
        # Get all weekly data to calculate defensive aggregates
        all_data = nfl.import_weekly_data([year])
        
        # Group by the DEFENSE (opponent_team) to get defensive stats
        # REAL columns that exist: pass_ypa, pass_epa, rush_ypa, rush_epa, sacks, qb_hits
        def_agg = all_data.groupby(['opponent_team', 'week']).agg({
            'pass_ypa': 'mean',          # Yards per pass attempt allowed
            'pass_epa': 'mean',          # Expected Points Added allowed (pass)
            'rush_ypa': 'mean',          # Yards per rush attempt allowed
            'rush_epa': 'mean',          # Expected Points Added allowed (rush)
            'sacks': 'sum',              # Total sacks
            'qb_hits': 'sum',            # Total QB hits
            'interceptions': 'sum',      # Total interceptions
            'passes_defensed': 'sum'     # Total passes defensed
        }).reset_index()
        
        # Rename for clarity
        def_agg = def_agg.rename(columns={
            'opponent_team': 'team',
            'pass_ypa': 'opp_pass_ypa_allowed',
            'pass_epa': 'opp_pass_epa_allowed',
            'rush_ypa': 'opp_rush_ypa_allowed',
            'rush_epa': 'opp_rush_epa_allowed',
        })
        
        # Calculate pressure rate (sacks + hits) / dropbacks
        # We need dropbacks to calculate this, which we can get from the offensive side
        # For simplicity, we'll use a proxy: (sacks + qb_hits) / (pass attempts)
        # We'll merge this back in the matchup matrix
        
        st.success(f"✅ Loaded defensive stats for {len(def_agg)} team-weeks")
        return def_agg
        
    except Exception as e:
        st.error(f"❌ Defensive stats fetch failed: {e}")
        return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix merging offense with defense."""
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("⚠️ No defensive stats available - proceeding without matchup adjustments")
        return players
    
    # Merge offense with defense
    # We need to calculate pressure rate here since we have both sides
    # First, get team dropbacks (pass attempts) to calculate pressure rate
    if 'team_dropbacks' not in players.columns:
        if 'team' in players.columns and 'routes' in players.columns:
            team_dropbacks = players.groupby(['team', 'week'])['routes'].sum().reset_index()
            team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
            # We'll use routes as a proxy for dropbacks if team_dropbacks isn't there
            # Actually, let's just use the defensive stats we have
    
    # Merge defensive stats onto players
    matchup_df = players.merge(
        def_stats,
        left_on=['opponent_team', 'week'],
        right_on=['team', 'week'],
        how='left',
        suffixes=('', '_def')
    )
    
    # Drop duplicate team column
    if 'team_def' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_def'])
    
    # Calculate pressure rate for the defense
    # We need the opponent's pass attempts to calculate this
    # For now, we'll use a simple metric: sacks per game
    if 'sacks' in matchup_df.columns:
        matchup_df['opp_sacks_per_game'] = matchup_df['sacks']
    
    st.success(f"✅ Built matchup matrix: {len(matchup_df)} players with defensive context")
    return matchup_df
