# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Column Normalization Fix

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

def check_available_seasons():
    """Check which seasons have data available."""
    available = []
    for year in range(2020, CURRENT_YEAR + 1):
        try:
            import nfl_data_py as nfl
            data = nfl.import_weekly_data([year])
            if not data.empty:
                weeks = data['week'].nunique()
                available.append((year, len(data), weeks))
        except:
            pass
    return available

def normalize_columns(df):
    """
    Fixes the 'team' KeyError by renaming nfl_data_py's actual column names 
    (recent_team, posteam, defteam) to our standard names.
    """
    rename_map = {}
    
    # Fix Player Team Column
    if 'team' not in df.columns:
        if 'recent_team' in df.columns:
            rename_map['recent_team'] = 'team'
        elif 'posteam' in df.columns:
            rename_map['posteam'] = 'team'
            
    # Fix Opponent Team Column
    if 'opponent_team' not in df.columns:
        if 'defteam' in df.columns:
            rename_map['defteam'] = 'opponent_team'
        elif 'opp' in df.columns:
            rename_map['opp'] = 'opponent_team'
            
    # Apply renames if any were found
    if rename_map:
        df = df.rename(columns=rename_map)
        
    return df

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch weekly player stats with fallback logic."""
    
    if year is None:
        year = PREFERRED_SEASON
    
    errors = []
    
    # Try preferred season first
    try:
        import nfl_data_py as nfl
        st.info(f"🔄 Attempting to load {year} Week {week} data...")
        
        all_data = nfl.import_weekly_data([year])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            # CRITICAL FIX: Normalize column names immediately
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
            errors.append(f"{year} Week {week} has no data")
            
    except Exception as e:
        errors.append(f"{year} failed: {str(e)[:100]}")
        st.warning(f"❌ {year} data fetch failed: {str(e)[:150]}")
    
    # Try fallback season
    if year != FALLBACK_SEASON:
        st.info(f"🔄 Falling back to {FALLBACK_SEASON}...")
        try:
            import nfl_data_py as nfl
            all_data = nfl.import_weekly_data([FALLBACK_SEASON])
            week_data = all_data[all_data['week'] == week].copy()
            
            if not week_data.empty:
                # CRITICAL FIX: Normalize column names immediately
                week_data = normalize_columns(week_data)
                
                st.success(f"✅ Loaded {len(week_data)} players from {FALLBACK_SEASON} Week {week} (fallback)")
                
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
            errors.append(f"{FALLBACK_SEASON} fallback failed: {str(e)[:100]}")
            st.error(f"❌ Fallback to {FALLBACK_SEASON} also failed: {str(e)[:150]}")
    
    # Show available seasons
    st.info("📊 Checking what seasons are available...")
    available = check_available_seasons()
    if available:
        st.write("**Available seasons:**")
        for year, rows, weeks in available:
            st.write(f"- {year}: {rows:,} player-games across {weeks} weeks")
    else:
        st.error("❌ No seasons available from nfl_data_py")
    
    # All attempts failed
    st.error(f"❌ All data sources failed. Errors: {'; '.join(errors)}")
    return pd.DataFrame()

def get_team_defensive_stats(year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats using import_team_stats with stat_type='def'."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        st.info(f"️ Fetching {year} team defensive stats...")
        # FIX: use import_team_stats with stat_type='def' (not import_seasonal_data)
        def_stats = nfl.import_team_stats(year, stat_type='def')
        
        if not def_stats.empty:
            # Return available columns
            available_cols = ['team', 'week', 'season']
            optional_cols = ['pass_epa', 'rush_epa', 'pressure_rate', 'stuff_rate',
                           'pass_success_rate', 'rush_success_rate', 'blitz_rate']
            
            for col in optional_cols:
                if col in def_stats.columns:
                    available_cols.append(col)
            
            st.success(f"✅ Loaded defensive stats for {len(def_stats)} team-weeks")
            return def_stats[available_cols]
    except Exception as e:
        st.warning(f"⚠️ Defensive stats fetch failed: {str(e)[:100]}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix with error handling."""
    
    players = get_weekly_player_stats(week, year)
    
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(year)
    
    if def_stats.empty:
        st.warning("⚠️ No defensive stats available - proceeding with neutral matchups")
        return players
    
    # Merge offense with defense
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
        suffixes=('', '_opp')
    )
    
    if 'team_opp' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_opp'])
    
    st.success(f"✅ Built matchup matrix: {len(matchup_df)} players with defensive context")
    return matchup_df
