# data/fetcher.py
# LaunchCast NFL — Data Fetcher V5
# FIXES: target_share denominator, defensive merge order, defensive agg safety,
# PREFERRED_SEASON back to 2025, column-safe aggregation

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# FIX: Back to 2025 (V3 had this right, V4 regressed)
if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2025
    FALLBACK_SEASON = 2024
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
    Build season-to-date player rates using ONLY weeks before `week`.
    Prevents data leakage.
    """
    try:
        raw = _load_weekly_raw(year)
        raw = normalize_columns(raw)
        
        # Only use data from weeks BEFORE the week we're projecting
        hist = raw[raw['week'] < week]
        
        if hist.empty:
            return pd.DataFrame()
        
        # FIX: Build aggregation dict defensively — only aggregate columns that exist
        agg_dict = {
            'targets': ('targets', 'sum'),
            'receiving_yards': ('receiving_yards', 'sum'),
            'receiving_tds': ('receiving_tds', 'sum'),
        }
        
        # Optional columns — only include if they exist
        if 'air_yards' in hist.columns:
            agg_dict['air_yards'] = ('air_yards', 'sum')
        if 'routes' in hist.columns:
            agg_dict['routes'] = ('routes', 'sum')
        
        # Build the aggregation
        agg_tuples = list(agg_dict.values())
        agg_names = list(agg_dict.keys())
        
        g = hist.groupby(['player_id', 'player_name', 'team'], as_index=False).agg(
            **{name: spec for name, spec in zip(agg_names, agg_tuples)}
        )
        g['games'] = hist.groupby(['player_id', 'player_name', 'team'])['week'].nunique().reset_index()['week']
        
        # Calculate rates from historical data only
        g['yds_per_tgt'] = np.where(g['targets'] > 0, g['receiving_yards'] / g['targets'], 11.0)
        g['td_per_tgt'] = np.where(g['targets'] > 0, g['receiving_tds'] / g['targets'], 0.05)
        
        if 'air_yards' in g.columns:
            g['adot'] = np.where(g['targets'] > 0, g['air_yards'] / g['targets'], 8.0)
        else:
            g['adot'] = 8.0
        
        # FIX: target_share uses TEAM denominator, not league-wide
        g['team_targets'] = g.groupby('team')['targets'].transform('sum')
        g['target_share'] = np.where(
            g['team_targets'] > 0,
            g['targets'] / g['team_targets'],
            0
        )
        
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
        
        hist = raw[raw['week'] < week]
        
        if hist.empty:
            return pd.DataFrame()
        
        # Build defensive aggregation defensively
        agg_dict = {
            'targets_allowed': ('targets', 'sum'),
            'receptions_allowed': ('receptions', 'sum'),
            'yards_allowed': ('receiving_yards', 'sum'),
            'tds_allowed': ('receiving_tds', 'sum'),
        }
        
        def_agg = hist.groupby(['opponent_team'], as_index=False).agg(
            **{name: spec for name, spec in agg_dict.items()}
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
    FIX: Attach opponent FIRST, then merge defense on opponent_team.
    """
    if year is None:
        year = PREFERRED_SEASON
    
    # Build features from historical data (weeks 1 to N-1)
    features = build_features_through(week, year)
    if features.empty:
        return pd.DataFrame()
    
    # FIX STEP 1: Attach this week's opponent FIRST
    try:
        all_data = _load_weekly_raw(year)
        all_data = normalize_columns(all_data)
        week_n = all_data[all_data['week'] == week][['player_id', 'opponent_team']].drop_duplicates('player_id')
        
        if week_n.empty:
            return pd.DataFrame()
        
        # Inner join = only players actually playing this week
        features = features.merge(week_n, on='player_id', how='inner')
    except Exception as e:
        st.warning(f"Opponent attach failed: {e}")
        return pd.DataFrame()
    
    # FIX STEP 2: THEN attach the defense they FACE (on opponent_team)
    def_features = build_defensive_features_through(week, year)
    
    if not def_features.empty:
        features = features.merge(
            def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']]
                .rename(columns={'team': 'opponent_team'}),
            on='opponent_team',
            how='left'
        )
    
    return features
