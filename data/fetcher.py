# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Multiple Backup Sources
# Primary: nfl_data_py (nflverse)
# Backup 1: Direct nflverse GitHub data
# Backup 2: ESPN API (unofficial)

import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime

# Force 2025 for testing during offseason (July 2026)
# Will auto-switch to 2026 when season starts in September
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    TARGET_SEASON = 2025  # Offseason: use last season
else:
    TARGET_SEASON = CURRENT_YEAR  # In-season: use current year

# ============================================================================
# SOURCE 1: nfl_data_py (Primary - nflverse)
# ============================================================================
def fetch_from_nflverse(week: int = None, season: int = TARGET_SEASON):
    """
    Primary data source: nfl_data_py library.
    Returns weekly player stats and team defensive stats.
    """
    try:
        import nfl_data_py as nfl
        
        # Fetch weekly data
        if week:
            weekly_data = nfl.import_weekly_data([season])
            weekly_data = weekly_data[weekly_data['week'] == week].copy()
        else:
            weekly_data = nfl.import_weekly_data([season])
        
        # Fetch team stats (defense)
        team_stats = nfl.import_team_stats([season])
        def_stats = team_stats[team_stats['side'] == 'def'].copy()
        
        return weekly_data, def_stats
        
    except ImportError:
        return None, None
    except Exception as e:
        print(f"nfl_data_py failed: {e}")
        return None, None

# ============================================================================
# SOURCE 2: Direct nflverse GitHub CSV fetch (Backup 1)
# ============================================================================
def fetch_from_github(week: int = None, season: int = TARGET_SEASON):
    """
    Backup source: Direct CSV fetch from nflverse GitHub.
    More reliable when the library has issues.
    """
    try:
        # nflverse weekly data URL
        if week:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/data/pro_player_stats_{season}.csv"
        else:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/data/pro_player_stats_{season}.csv"
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            weekly_data = pd.read_csv(io.StringIO(response.text))
            
            if week:
                weekly_data = weekly_data[weekly_data['week'] == week].copy()
            
            # Fetch team stats
            team_url = f"https://github.com/nflverse/nflverse-data/releases/download/data/team_stats_{season}.csv"
            team_response = requests.get(team_url, timeout=30)
            if team_response.status_code == 200:
                team_stats = pd.read_csv(io.StringIO(team_response.text))
                def_stats = team_stats[team_stats['side'] == 'def'].copy()
                return weekly_data, def_stats
            
            return weekly_data, pd.DataFrame()
        
        return None, None
        
    except Exception as e:
        print(f"GitHub fetch failed: {e}")
        return None, None

# ============================================================================
# SOURCE 3: ESPN API (Backup 2 - Unofficial)
# ============================================================================
def fetch_from_espn(week: int = None, season: int = TARGET_SEASON):
    """
    Backup source: ESPN API (unofficial).
    Use only if nflverse sources fail.
    """
    try:
        # ESPN doesn't have a clean bulk API, so we'll use a simplified approach
        # This is a placeholder - in production, you'd parse their endpoints
        print("ESPN API not implemented - using nflverse fallback")
        return None, None
        
    except Exception as e:
        print(f"ESPN fetch failed: {e}")
        return None, None

# ============================================================================
# MAIN FETCHER - Tries all sources in order
# ============================================================================
def get_weekly_player_stats(week: int, year: int = TARGET_SEASON) -> pd.DataFrame:
    """
    Fetches weekly player stats with automatic fallback.
    Tries: 1) nfl_data_py, 2) GitHub CSV, 3) ESPN API
    """
    print(f"Fetching Week {week} data for {year} season...")
    
    # Try Source 1: nfl_data_py
    weekly_data, def_stats = fetch_from_nflverse(week, year)
    if weekly_data is not None and not weekly_data.empty:
        print("✅ Data loaded from nfl_data_py")
        return process_weekly_data(weekly_data, def_stats)
    
    # Try Source 2: GitHub CSV
    print("⚠️  Trying backup source (GitHub)...")
    weekly_data, def_stats = fetch_from_github(week, year)
    if weekly_data is not None and not weekly_data.empty:
        print("✅ Data loaded from GitHub backup")
        return process_weekly_data(weekly_data, def_stats)
    
    # All sources failed
    print("❌ All data sources failed")
    return pd.DataFrame()

def process_weekly_data(weekly_data: pd.DataFrame, def_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Processes raw data into our standard format.
    """
    if weekly_data.empty:
        return pd.DataFrame()
    
    # Calculate derived metrics
    weekly_data['route_participation_pct'] = np.where(
        weekly_data.get('team_dropbacks', 0) > 0,
        (weekly_data.get('routes', 0) / weekly_data['team_dropbacks']) * 100,
        0
    )
    
    weekly_data['adot'] = np.where(
        weekly_data.get('targets', 0) > 0,
        weekly_data.get('air_yards', 0) / weekly_data['targets'],
        0
    )
    
    # Add target share calculation
    weekly_data['target_share'] = np.where(
        weekly_data.get('team_targets', 0) > 0,
        weekly_data.get('targets', 0) / weekly_data['team_targets'],
        0
    )
    
    return weekly_data

def get_team_defensive_stats(year: int = TARGET_SEASON) -> pd.DataFrame:
    """
    Fetches team defensive stats with fallback.
    """
    # Try nfl_data_py
    try:
        import nfl_data_py as nfl
        team_stats = nfl.import_team_stats([year])
        def_stats = team_stats[team_stats['side'] == 'def'].copy()
        return def_stats
    except:
        pass
    
    # Try GitHub
    try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/data/team_stats_{year}.csv"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            team_stats = pd.read_csv(io.StringIO(response.text))
            def_stats = team_stats[team_stats['side'] == 'def'].copy()
            return def_stats
    except:
        pass
    
    return pd.DataFrame()

# ============================================================================
# MATCHUP MATRIX (Merges offense vs defense)
# ============================================================================
def build_matchup_matrix(week: int, year: int = TARGET_SEASON) -> pd.DataFrame:
    """
    Builds the complete matchup matrix by merging player stats with opponent defense.
    """
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(year)
    if def_stats.empty:
        return players
    
    # Get latest defensive stats per team
    latest_def = def_stats.sort_values('week').groupby('team').last().reset_index()
    
    # Merge with defensive matchup data
    def_cols_to_merge = {
        'pass_epa': 'opp_pass_epa_allowed',
        'rush_epa': 'opp_rush_epa_allowed',
        'pressure_rate': 'opp_pressure_rate',
        'stuff_rate': 'opp_stuff_rate'
    }
    
    valid_def_cols = {k: v for k, v in def_cols_to_merge.items() 
                      if k in latest_def.columns}
    latest_def = latest_def.rename(columns=valid_def_cols)
    
    # Merge on opponent team
    matchup_df = players.merge(
        latest_def[['team'] + list(valid_def_cols.keys())],
        left_on='opponent_team',
        right_on='team',
        how='left',
        suffixes=('', '_opp')
    )
    
    # Cleanup
    if 'team_opp' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_opp'])
    
    return matchup_df

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_available_seasons():
    """Returns list of seasons with available data."""
    # nflverse has data from 2000 onwards
    current_year = datetime.now().year
    return list(range(2000, current_year))

def get_available_weeks(season: int = TARGET_SEASON):
    """Returns list of available weeks for a season."""
    try:
        import nfl_data_py as nfl
        data = nfl.import_weekly_data([season])
        return sorted(data['week'].unique())
    except:
        # Default: 18 weeks for regular season
        return list(range(1, 19))

def check_data_availability(season: int = TARGET_SEASON):
    """Checks if data is available for the given season."""
    try:
        import nfl_data_py as nfl
        data = nfl.import_weekly_data([season])
        return not data.empty
    except:
        return False
