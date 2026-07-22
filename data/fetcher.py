# data/fetcher.py
# LaunchCast NFL — Data Fetcher with Advanced Metrics

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
            st.success(f"✅ Successfully loaded {len(week_data)} players from {year} Week {week}")
            
            # Calculate derived metrics safely using .get() consistently
            # First, ensure team_dropbacks exists by computing it if needed
            if 'team_dropbacks' not in week_data.columns:
                if 'team' in week_data.columns and 'routes' in week_data.columns:
                    team_dropbacks = week_data.groupby(['team', 'week'])['routes'].sum().reset_index()
                    team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
                    week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
                else:
                    week_data['team_dropbacks'] = 0
            
            # Now safe to use bracket notation since column exists
            week_data['route_participation_pct'] = np.where(
                week_data['team_dropbacks'] > 0,
                (week_data['routes'] / week_data['team_dropbacks']) * 100,
                0
            )
            
            week_data['adot'] = np.where(
                week_data['targets'] > 0,
                week_data['air_yards'] / week_data['targets'],
                0
            )
            
            # Calculate target share
            if 'team_targets' not in week_data.columns:
                if 'team' in week_data.columns and 'targets' in week_data.columns:
                    team_targets = week_data.groupby(['team', 'week'])['targets'].sum().reset_index()
                    team_targets.columns = ['team', 'week', 'team_targets']
                    week_data = week_data.merge(team_targets, on=['team', 'week'], how='left')
                else:
                    week_data['team_targets'] = 0
            
            week_data['target_share'] = np.where(
                week_data['team_targets'] > 0,
                week_data['targets'] / week_data['team_targets'],
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
                st.success(f"✅ Loaded {len(week_data)} players from {FALLBACK_SEASON} Week {week} (fallback)")
                
                # Same safe calculations for fallback
                if 'team_dropbacks' not in week_data.columns:
                    if 'team' in week_data.columns and 'routes' in week_data.columns:
                        team_dropbacks = week_data.groupby(['team', 'week'])['routes'].sum().reset_index()
                        team_dropbacks.columns = ['team', 'week', 'team_dropbacks']
                        week_data = week_data.merge(team_dropbacks, on=['team', 'week'], how='left')
                    else:
                        week_data['team_dropbacks'] = 0
                
                week_data['route_participation_pct'] = np.where(
                    week_data['team_dropbacks'] > 0,
                    (week_data['routes'] / week_data['team_dropbacks']) * 100,
                    0
                )
                
                week_data['adot'] = np.where(
                    week_data['targets'] > 0,
                    week_data['air_yards'] / week_data['targets'],
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
                    week_data['team_targets'] > 0,
                    week_data['targets'] / week_data['team_targets'],
                    0
                )
                
                return week_data
        except Exception as e:
            errors.append(f"{FALLBACK_SEASON} fallback failed: {str(e)[:100]}")
            st.error(f"❌ Fallback to {FALLBACK_SEASON} also failed: {str(e)[:150]}")
    
    # All attempts failed
    st.error(f"❌ All data sources failed. Errors: {'; '.join(errors)}")
    return pd.DataFrame()

def get_team_defensive_stats(week: int, year: int = None) -> pd.DataFrame:
    """Fetch team defensive stats."""
    if year is None:
        year = PREFERRED_SEASON
    
    try:
        import nfl_data_py as nfl
        st.info(f"🛡️ Fetching {year} team defensive stats...")
        
        # Get defensive stats from weekly data
        all_data = nfl.import_weekly_data([year])
        
        # Group by opponent_team to get defensive aggregates
        def_agg = all_data.groupby(['opponent_team', 'week']).agg({
            'pressure_rate_when_targeted': 'mean',
            'clean_pocket_pct': 'mean',
            'avg_time_to_throw': 'mean',
            'separation': 'mean',
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
        
        # Catch rate allowed
        def_agg['def_catch_rate_allowed'] = np.where(
            def_agg['targets'] > 0,
            def_agg['receptions'] / def_agg['targets'],
            0.65
        )
        
        # Filter to current week
        def_agg = def_agg[def_agg['week'] == week]
        
        st.success(f"✅ Loaded defensive stats for {len(def_agg)} teams")
        return def_agg
    except Exception as e:
        st.warning(f"⚠️ Defensive stats fetch failed: {str(e)[:100]}")
    
    return pd.DataFrame()

def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Build matchup matrix with error handling."""
    players = get_weekly_player_stats(week, year)
    if players.empty:
        return pd.DataFrame()
    
    def_stats = get_team_defensive_stats(week, year)
    if def_stats.empty:
        st.warning("⚠️ No defensive stats available - proceeding without matchup adjustments")
        return players
    
    # Merge offense with defense
    def_cols = {
        'def_pressure_rate': 'opp_pressure_rate',
        'def_clean_pocket_rate': 'opp_clean_pocket_rate',
        'def_time_to_throw': 'opp_time_to_throw',
        'def_separation_allowed': 'opp_separation_allowed',
        'def_catch_rate_allowed': 'opp_catch_rate_allowed',
    }
    
    valid_cols = {k: v for k, v in def_cols.items() if k in def_stats.columns}
    def_stats_renamed = def_stats.rename(columns=valid_cols)
    
    matchup_df = players.merge(
        def_stats_renamed[['team'] + list(valid_cols.values())],
        left_on='opponent_team',
        right_on='team',
        how='left',
        suffixes=('', '_def')
    )
    
    if 'team_def' in matchup_df.columns:
        matchup_df = matchup_df.drop(columns=['team_def'])
    
    st.success(f"✅ Built matchup matrix: {len(matchup_df)} players with defensive context")
    return matchup_df
