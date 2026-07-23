# data/fetcher.py
# LaunchCast NFL — Data Fetcher V4 (Leakage-Free)

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

@st.cache_data(ttl=3600)
def _load_weekly_raw(year: int) -> pd.DataFrame:
    """Load raw weekly data once and cache it."""
    import nfl_data_py as nfl
    return nfl.import_weekly_data([year])

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

def build_features_through(week: int, year: int) -> pd.DataFrame:
    """
    CRITICAL FIX: Build season-to-date player rates using ONLY weeks before `week`.
    This prevents data leakage - features are computed from historical data,
    outcomes are from the week being projected.
    """
    try:
        raw = _load_weekly_raw(year)
        raw = normalize_columns(raw)
        
        # Only use data from weeks BEFORE the week we're projecting
        hist = raw[raw['week'] < week]
        
        if hist.empty:
            return pd.DataFrame()
        
        # Aggregate season-to-date stats
        g = hist.groupby(['player_id', 'player_name', 'team'], as_index=False).agg(
            targets=('targets', 'sum'),
            receiving_yards=('receiving_yards', 'sum'),
            receiving_tds=('receiving_tds', 'sum'),
            air_yards=('air_yards', 'sum'),
            routes=('routes', 'sum'),
            games=('week', 'nunique'),
        )
        
        # Calculate rates from historical data only
        g['yds_per_tgt'] = np.where(g['targets'] > 0, g['receiving_yards'] / g['targets'], 11.0)
        g['td_per_tgt'] = np.where(g['targets'] > 0, g['receiving_tds'] / g['targets'], 0.05)
        g['adot'] = np.where(g['targets'] > 0, g['air_yards'] / g['targets'], 8.0)
        g['target_share'] = np.where(g['targets'] > 0, g['targets'] / g['targets'].sum(), 0)
        
        return g
    except Exception as e:
        st.warning(f"Feature build failed: {e}")
        return pd.DataFrame()

def build_defensive_features_through(week: int, year: int) -> pd.DataFrame:
    """
    Build season-to-date defensive stats using ONLY weeks before `week`.
    """
    try:
        raw = _load_weekly_raw(year)
        raw = normalize_columns(raw)
        
        # Only use data from weeks BEFORE the week we're projecting
        hist = raw[raw['week'] < week]
        
        if hist.empty:
            return pd.DataFrame()
        
        # Aggregate what each defense ALLOWED over the season so far
        def_agg = hist.groupby(['opponent_team'], as_index=False).agg(
            targets_allowed=('targets', 'sum'),
            receptions_allowed=('receptions', 'sum'),
            yards_allowed=('receiving_yards', 'sum'),
            tds_allowed=('receiving_tds', 'sum'),
        )
        
        def_agg = def_agg.rename(columns={'opponent_team': 'team'})
        
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
        
        return def_agg
    except Exception as e:
        st.warning(f"Defensive feature build failed: {e}")
        return pd.DataFrame()

def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch weekly player stats for a specific week (used for outcomes)."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        all_data = _load_weekly_raw(year)
        week_data = all_data[all_data['week'] == week].copy()
        
        if week_data.empty:
            return pd.DataFrame()
            
        week_data = normalize_columns(week_data)
        return week_data
    except Exception as e:
        st.warning(f"Player stats fetch failed: {e}")
        return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """
    Build matchup matrix for LIVE projections.
    Features from weeks 1 to N-1, projecting week N.
    """
    if year is None:
        year = PREFERRED_SEASON
    
    # Build features from historical data (weeks 1 to N-1)
    features = build_features_through(week, year)
    if features.empty:
        return pd.DataFrame()
    
    # Build defensive features from historical data (weeks 1 to N-1)
    def_features = build_defensive_features_through(week, year)
    
    if not def_features.empty:
        # Merge defensive features onto player features
        features = features.merge(
            def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']],
            left_on='team',
            right_on='team',
            how='left'
        )
    
    # Get week N's schedule to know who plays whom
    try:
        all_data = _load_weekly_raw(year)
        all_data = normalize_columns(all_data)
        week_n = all_data[all_data['week'] == week]
        
        if not week_n.empty:
            # Get opponent for each player
            opp_map = week_n[['player_id', 'opponent_team']].drop_duplicates()
            features = features.merge(opp_map, on='player_id', how='left')
    except Exception:
        pass
    
    return features
