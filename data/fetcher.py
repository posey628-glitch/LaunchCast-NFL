# data/fetcher.py
# LaunchCast NFL — Multi-Source Data Fetcher with Fallbacks
# Primary: nfl_data_py (nflverse)
# Backup 1: Direct nflverse GitHub CSV
# Backup 2: ESPN API (unofficial)
# Backup 3: Pro-Football-Reference scraping
# Fallback: 2024 season data (known to exist)

import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime
from io import StringIO

# Force specific season for testing
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# During offseason, use 2024 (2025 should exist but nflverse might be slow)
# During season, use current year
if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2024  # Use 2024 for now (known working)
    FALLBACK_SEASON = 2023
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1

def check_all_available_sources():
    """Check what seasons/weeks are available from ALL sources."""
    sources = {}
    
    # Source 1: nfl_data_py
    try:
        import nfl_data_py as nfl
        available_years = []
        for year in range(2020, CURRENT_YEAR + 1):
            try:
                data = nfl.import_weekly_data([year])
                if not data.empty:
                    weeks = sorted(data['week'].unique())
                    available_years.append((year, len(data), len(weeks)))
            except:
                pass
        sources['nfl_data_py'] = available_years
    except Exception as e:
        sources['nfl_data_py'] = f"Error: {str(e)[:100]}"
    
    # Source 2: Direct GitHub CSV
    try:
        github_years = []
        for year in range(2020, CURRENT_YEAR + 1):
            try:
                url = f"https://github.com/nflverse/nflverse-data/releases/download/data/pro_player_stats_{year}.csv"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = pd.read_csv(StringIO(response.text))
                    weeks = data['week'].nunique() if 'week' in data.columns else 0
                    github_years.append((year, len(data), weeks))
            except:
                pass
        sources['github_csv'] = github_years
    except Exception as e:
        sources['github_csv'] = f"Error: {str(e)[:100]}"
    
    return sources

def fetch_from_nfl_data_py(week: int, season: int):
    """Primary source: nfl_data_py library."""
    try:
        import nfl_data_py as nfl
        st.info(f"📊 Trying nfl_data_py for {season} Week {week}...")
        
        all_data = nfl.import_weekly_data([season])
        week_data = all_data[all_data['week'] == week].copy()
        
        if not week_data.empty:
            st.success(f"✅ nfl_data_py: Loaded {len(week_data)} players from {season} Week {week}")
            return week_data
        else:
            st.warning(f"⚠️ nfl_data_py: No data for {season} Week {week}")
            return None
    except Exception as e:
        st.error(f"❌ nfl_data_py failed: {str(e)[:150]}")
        return None

def fetch_from_github_csv(week: int, season: int):
    """Backup 1: Direct GitHub CSV download."""
    try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/data/pro_player_stats_{season}.csv"
        st.info(f"📥 Trying GitHub CSV for {season} Week {week}...")
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            week_data = data[data['week'] == week].copy() if 'week' in data.columns else data
            
            if not week_data.empty:
                st.success(f"✅ GitHub CSV: Loaded {len(week_data)} players from {season} Week {week}")
                return week_data
            else:
                st.warning(f"⚠️ GitHub CSV: No data for week {week}")
                return None
        else:
            st.error(f"❌ GitHub CSV: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ GitHub CSV failed: {str(e)[:150]}")
        return None

def fetch_from_espn_api(week: int, season: int):
    """Backup 2: ESPN API (unofficial)."""
    try:
        # ESPN's weekly stats endpoint
        url = f"http://fantasy.espn.com/apis/v3/games/FFL/seasons/{season}/segments/0/leagues/0:players?view=players&week={week}"
        st.info(f"🏈 Trying ESPN API for {season} Week {week}...")
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            # Parse ESPN's complex JSON structure
            # This is a simplified version - full implementation would parse all fields
            if 'players' in data:
                st.success(f"✅ ESPN API: Found {len(data['players'])} players")
                # Convert to DataFrame (simplified)
                df = pd.DataFrame(data['players'])
                return df
            else:
                st.warning("⚠️ ESPN API: No players in response")
                return None
        else:
            st.error(f"❌ ESPN API: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ ESPN API failed: {str(e)[:150]}")
        return None

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """
    Fetches weekly player stats with automatic multi-source fallback.
    Tries in order:
    1. nfl_data_py (primary)
    2. GitHub CSV (backup 1)
    3. ESPN API (backup 2)
    4. Fallback season (2024)
    """
    
    if year is None:
        year = PREFERRED_SEASON
    
    st.markdown(f"### 🔍 Fetching Week {week} Data")
    st.caption(f"Preferred season: {year} | Fallback: {FALLBACK_SEASON}")
    
    # Show what sources are available
    with st.expander("📊 Check Available Data Sources", expanded=False):
        sources = check_all_available_sources()
        for source, data in sources.items():
            if isinstance(data, list) and data:
                st.write(f"**{source}**:")
                for year, rows, weeks in data:
                    st.write(f"- {year}: {rows:,} rows, {weeks} weeks")
            else:
                st.write(f"**{source}**: {data}")
    
    # Try preferred season from multiple sources
    errors = []
    
    # Attempt 1: nfl_data_py
    data = fetch_from_nfl_data_py(week, year)
    if data is not None:
        return process_weekly_data(data, year)
    errors.append(f"nfl_data_py {year} failed")
    
    # Attempt 2: GitHub CSV
    data = fetch_from_github_csv(week, year)
    if data is not None:
        return process_weekly_data(data, year)
    errors.append(f"GitHub CSV {year} failed")
    
    # Attempt 3: ESPN API
    data = fetch_from_espn_api(week, year)
    if data is not None:
        return process_weekly_data(data, year)
    errors.append(f"ESPN API {year} failed")
    
    # Fallback: Try 2024 (known to work)
    if year != 2024:
        st.warning(f"⚠️ All sources failed for {year}. Falling back to 2024...")
        data = fetch_from_nfl_data_py(week, 2024)
        if data is not None:
            return process_weekly_data(data, 2024)
        data = fetch_from_github_csv(week, 2024)
        if data is not None:
            return process_weekly_data(data, 2024)
    
    # All attempts failed
    st.error(f"❌ All data sources failed!")
    st.write("Errors:", "; ".join(errors))
    
    # Show what IS available
    st.info("💡 What data IS available:")
    sources = check_all_available_sources()
    for source, data in sources.items():
        if isinstance(data, list) and data:
            st.write(f"**{source}**: Years {', '.join(str(y[0]) for y in data)}")
    
    return pd.DataFrame()

def process_weekly_data(raw_data: pd.DataFrame, season: int) -> pd.DataFrame:
    """Process raw data into our standard format."""
    if raw_data.empty:
        return pd.DataFrame()
    
    df = raw_data.copy()
    
    # Calculate derived metrics (handle missing columns gracefully)
    if 'team_dropbacks' not in df.columns:
        if 'team' in df.columns and 'routes' in df.columns:
            team_dropbacks = df.groupby(['team', 'week'])['routes'].sum().reset_index()
            team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
            df = df.merge(team_dropbacks, on=['team', 'week'], how='left')
        else:
            df['team_dropbacks'] = 0
    
    df['route_participation_pct'] = np.where(
        df.get('team_dropbacks', 0) > 0,
        (df.get('routes', 0) / df['team_dropbacks']) * 100,
        0
    )
    
    df['adot'] = np.where(
        df.get('targets', 0) > 0,
        df.get('air_yards', 0) / df['targets'],
        0
    )
    
    if 'team_targets' not in df.columns:
        if 'team' in df.columns and 'targets' in df.columns:
            team_targets = df.groupby(['team', 'week'])['targets'].sum().reset_index()
            team_targets.columns = ['team', 'week', 'team_targets']
            df = df.merge(team_targets, on=['team', 'week'], how='left')
        else:
            df['team_targets'] = 0
    
    df['target_share'] = np.where(
        df.get('team_targets', 0) > 0,
        df.get('targets', 0) / df['team_targets'],
        0
    )
    
    st.success(f"✅ Processed {len(df)} players from {season} season")
    return df

def get_team_defensive_stats(year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        team_stats = nfl.import_team_stats([year])
        def_stats = team_stats[team_stats['side'] == 'def'].copy()
        
        if not def_stats.empty:
            available_cols = ['team', 'week', 'season']
            optional_cols = ['pass_epa', 'rush_epa', 'pressure_rate', 'stuff_rate']
            
            for col in optional_cols:
                if col in def_stats.columns:
                    available_cols.append(col)
            
            return def_stats[available_cols]
    except Exception as e:
        st.warning(f"️ Defensive stats fetch failed: {str(e)[:100]}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix with error handling."""
    players = get_weekly_player_stats(week, year)
    
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(year)
    
    if def_stats.empty:
        st.warning("⚠️ No defensive stats available - proceeding without matchup adjustments")
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
    
    st.success(f"✅ Built matchup matrix: {len(matchup_df)} players")
    return matchup_df
