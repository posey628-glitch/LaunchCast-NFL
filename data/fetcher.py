# data/fetcher.py
# LaunchCast NFL — Data Fetcher V7.1
# FIXES: Week 1 uses priors as base, team_weeks merged for weeks 2-3

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

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

@st.cache_data(ttl=3600)
def load_prior_rates_from_season(season: int) -> pd.DataFrame:
    """
    Load last season's per-player rates to use as priors for early weeks.
    Returns DataFrame with player_id, player_name, team, position, and rates.
    """
    try:
        raw = _load_weekly_raw(season)
        raw = normalize_columns(raw)
        
        if raw.empty:
            return pd.DataFrame()
        
        # Group keys
        groupby_cols = ['player_id', 'player_name', 'team']
        if 'position' in raw.columns:
            groupby_cols.append('position')
        
        # Aggregate full-season stats
        agg_dict = {
            'targets': ('targets', 'sum'),
            'receiving_yards': ('receiving_yards', 'sum'),
            'receiving_tds': ('receiving_tds', 'sum'),
        }
        if 'air_yards' in raw.columns:
            agg_dict['air_yards'] = ('air_yards', 'sum')
        if 'routes' in raw.columns:
            agg_dict['routes'] = ('routes', 'sum')
        
        g = raw.groupby(groupby_cols, as_index=False).agg(**agg_dict)
        
        # Calculate rates
        g['prior_yds_per_tgt'] = np.where(g['targets'] > 0, g['receiving_yards'] / g['targets'], 11.0)
        g['prior_td_per_tgt'] = np.where(g['targets'] > 0, g['receiving_tds'] / g['targets'], 0.05)
        
        # Team-level target share from last season
        g['prior_team_targets'] = g.groupby('team')['targets'].transform('sum')
        g['prior_target_share'] = np.where(
            g['prior_team_targets'] > 0,
            g['targets'] / g['prior_team_targets'],
            0
        )
        
        # Team-level pass volume from last season (targets per game)
        team_games = raw.groupby('team')['week'].nunique().reset_index()
        team_games.columns = ['team', 'team_weeks']
        team_targets = raw.groupby('team')['targets'].sum().reset_index()
        team_targets.columns = ['team', 'team_total_targets']
        team_vol = team_targets.merge(team_games, on='team')
        team_vol['prior_team_pass_att'] = np.where(
            team_vol['team_weeks'] > 0,
            team_vol['team_total_targets'] / team_vol['team_weeks'],
            35.0
        )
        g = g.merge(team_vol[['team', 'prior_team_pass_att']], on='team', how='left')
        g['prior_team_pass_att'] = g['prior_team_pass_att'].fillna(35.0)
        
        return g
    except Exception as e:
        st.warning(f"Prior rates load failed: {e}")
        return pd.DataFrame()

def build_features_through(week: int, year: int, prior_rates: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build season-to-date player rates using ONLY weeks before `week`.
    For Week 1, uses last season's rates as the base.
    For weeks 2-3, blends current season with last season.
    """
    try:
        raw = _load_weekly_raw(year)
        raw = normalize_columns(raw)
        
        hist = raw[raw['week'] < week]
        
        # Build groupby keys defensively
        groupby_cols = ['player_id', 'player_name', 'team']
        if 'position' in raw.columns:
            groupby_cols.append('position')
        
        # FIX 1: Week 1 — use priors as base when history is empty
        if hist.empty:
            if prior_rates is None or prior_rates.empty:
                return pd.DataFrame()
            
            # Use last season as the Week 1 projection base
            g = prior_rates.rename(columns={
                'prior_target_share': 'target_share',
                'prior_yds_per_tgt': 'yds_per_tgt',
                'prior_td_per_tgt': 'td_per_tgt',
                'prior_team_pass_att': 'team_avg_pass_attempts',
            }).copy()
            g['games'] = 0
            g['adot'] = 8.0
            g['routes'] = 0  # No routes data for Week 1
            return g
        
        # Build aggregation dict defensively
        agg_dict = {
            'targets': ('targets', 'sum'),
            'receiving_yards': ('receiving_yards', 'sum'),
            'receiving_tds': ('receiving_tds', 'sum'),
        }
        if 'air_yards' in hist.columns:
            agg_dict['air_yards'] = ('air_yards', 'sum')
        if 'routes' in hist.columns:
            agg_dict['routes'] = ('routes', 'sum')
        
        g = hist.groupby(groupby_cols, as_index=False).agg(**agg_dict)
        
        # Games column via merge
        _gm = hist.groupby(groupby_cols, as_index=False)['week'].nunique()
        _gm = _gm.rename(columns={'week': 'games'})
        g = g.merge(_gm, on=groupby_cols, how='left')
        
        # Team-level pass volume (current season)
        team_games_cur = hist.groupby('team')['week'].nunique().reset_index()
        team_games_cur.columns = ['team', 'team_weeks']
        team_targets_cur = hist.groupby('team')['targets'].sum().reset_index()
        team_targets_cur.columns = ['team', 'team_total_targets']
        team_vol_cur = team_targets_cur.merge(team_games_cur, on='team')
        team_vol_cur['team_avg_pass_attempts'] = np.where(
            team_vol_cur['team_weeks'] > 0,
            team_vol_cur['team_total_targets'] / team_vol_cur['team_weeks'],
            35.0
        )
        
        # FIX 2: Merge team_weeks so the conditional works
        g = g.merge(
            team_vol_cur[['team', 'team_avg_pass_attempts', 'team_weeks']],
            on='team',
            how='left'
        )
        g['team_avg_pass_attempts'] = g['team_avg_pass_attempts'].fillna(35.0)
        g['team_weeks'] = g['team_weeks'].fillna(0)
        
        # Calculate rates from historical data
        g['yds_per_tgt'] = np.where(g['targets'] > 0, g['receiving_yards'] / g['targets'], 11.0)
        g['td_per_tgt'] = np.where(g['targets'] > 0, g['receiving_tds'] / g['targets'], 0.05)
        
        if 'air_yards' in g.columns:
            g['adot'] = np.where(g['targets'] > 0, g['air_yards'] / g['targets'], 8.0)
        else:
            g['adot'] = 8.0
        
        # Target share uses TEAM denominator
        g['team_targets'] = g.groupby('team')['targets'].transform('sum')
        g['target_share'] = np.where(
            g['team_targets'] > 0,
            g['targets'] / g['team_targets'],
            0
        )
        
        # For weeks 1-3, blend with last season's rates as priors
        if prior_rates is not None and not prior_rates.empty and week <= 3:
            # Merge prior rates on player_id
            prior_cols = ['player_id', 'prior_yds_per_tgt', 'prior_td_per_tgt', 
                         'prior_target_share', 'prior_team_pass_att', 'targets']
            prior_cols = [c for c in prior_cols if c in prior_rates.columns]
            
            g = g.merge(
                prior_rates[prior_cols].rename(columns={'targets': 'prior_targets'}),
                on='player_id',
                how='left'
            )
            
            # Bayesian blend: current_targets vs prior_targets (strength of prior)
            prior_strength = 60.0
            
            # Blend target_share
            g['target_share'] = np.where(
                g['prior_target_share'].notna() & (g['prior_targets'].fillna(0) > 0),
                (g['targets'].fillna(0) * g['target_share'] + 
                 prior_strength * g['prior_target_share']) / 
                (g['targets'].fillna(0) + prior_strength),
                g['target_share']
            )
            
            # Blend yds_per_tgt
            g['yds_per_tgt'] = np.where(
                g['prior_yds_per_tgt'].notna() & (g['prior_targets'].fillna(0) > 0),
                (g['targets'].fillna(0) * g['yds_per_tgt'] + 
                 prior_strength * g['prior_yds_per_tgt']) / 
                (g['targets'].fillna(0) + prior_strength),
                g['yds_per_tgt']
            )
            
            # Blend td_per_tgt
            g['td_per_tgt'] = np.where(
                g['prior_td_per_tgt'].notna() & (g['prior_targets'].fillna(0) > 0),
                (g['targets'].fillna(0) * g['td_per_tgt'] + 
                 prior_strength * g['prior_td_per_tgt']) / 
                (g['targets'].fillna(0) + prior_strength),
                g['td_per_tgt']
            )
            
            # Use prior team pass attempts if current season has no data yet (Week 1)
            # FIX 2: Now team_weeks exists, so this conditional works
            g['team_avg_pass_attempts'] = np.where(
                g['team_weeks'] > 0,
                g['team_avg_pass_attempts'],
                g['prior_team_pass_att'].fillna(35.0)
            )
        
        # Deduplicate traded players
        g = g.sort_values('games', ascending=False).drop_duplicates('player_id', keep='first')
        
        return g
    except Exception as e:
        st.warning(f"Feature build failed: {e}")
        return pd.DataFrame()

def build_defensive_features_through(week: int, year: int) -> pd.DataFrame:
    """Build season-to-date defensive stats using ONLY weeks before `week`."""
    try:
        raw = _load_weekly_raw(year)
        raw = normalize_columns(raw)
        
        hist = raw[raw['week'] < week]
        
        if hist.empty:
            return pd.DataFrame()
        
        # Build aggregation dict defensively
        spec = {}
        if 'targets' in hist.columns:
            spec['targets_allowed'] = ('targets', 'sum')
        if 'receptions' in hist.columns:
            spec['receptions_allowed'] = ('receptions', 'sum')
        if 'receiving_yards' in hist.columns:
            spec['yards_allowed'] = ('receiving_yards', 'sum')
        if 'receiving_tds' in hist.columns:
            spec['tds_allowed'] = ('receiving_tds', 'sum')
        
        if 'targets_allowed' not in spec or 'tds_allowed' not in spec:
            st.warning("⚠️ Defensive features unavailable — matchup adjustment is OFF this run")
            return pd.DataFrame()
        
        def_agg = hist.groupby(['opponent_team'], as_index=False).agg(**spec)
        def_agg = def_agg.rename(columns={'opponent_team': 'team'})
        
        _ta = def_agg['targets_allowed']
        
        if 'yards_allowed' in def_agg.columns:
            def_agg['def_yds_per_tgt'] = np.where(_ta > 0, def_agg['yards_allowed'] / _ta, 11.0)
        else:
            def_agg['def_yds_per_tgt'] = 11.0
        
        def_agg['def_td_per_tgt'] = np.where(_ta > 0, def_agg['tds_allowed'] / _ta, 0.05)
        
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
    """Build matchup matrix for LIVE projections."""
    if year is None:
        year = PREFERRED_SEASON
    
    # Load prior rates for early weeks (including Week 1)
    prior_rates = None
    if week <= 3:
        prior_rates = load_prior_rates_from_season(year - 1)
    
    features = build_features_through(week, year, prior_rates=prior_rates)
    if features.empty:
        return pd.DataFrame()
    
    try:
        all_data = _load_weekly_raw(year)
        all_data = normalize_columns(all_data)
        week_n = all_data[all_data['week'] == week][['player_id', 'opponent_team']].drop_duplicates('player_id')
        
        if week_n.empty:
            return pd.DataFrame()
        
        features = features.merge(week_n, on='player_id', how='inner')
    except Exception as e:
        st.warning(f"Opponent attach failed: {e}")
        return pd.DataFrame()
    
    def_features = build_defensive_features_through(week, year)
    
    if not def_features.empty:
        features = features.merge(
            def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']]
                .rename(columns={'team': 'opponent_team'}),
            on='opponent_team',
            how='left'
        )
    
    return features
