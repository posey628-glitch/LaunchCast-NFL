# core/scoring.py
# LaunchCast NFL — The Scoring Engine (Fixed for dynamic columns)

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

def calculate_shrunk_rate(actual_rate, volume, current_week, league_avg_rate, position='WR'):
    """Shrinks a player's raw rate toward the league average."""
    if current_week <= 3:
        prior_volume = 60
        weight_actual = 0.20
    elif current_week <= 10:
        prior_volume = 40
        weight_actual = 0.50
    else:
        prior_volume = 20
        weight_actual = 0.80
        
    if volume <= 0:
        return league_avg_rate
        
    shrinkage_factor = volume / (volume + prior_volume)
    shrunk_rate = (weight_actual * shrinkage_factor * actual_rate) + \
                  ((1 - (weight_actual * shrinkage_factor)) * league_avg_rate)
    return shrunk_rate

def apply_defensive_matchup(base_projection, opp_pass_epa, opp_pressure_rate, position='WR'):
    """Adjusts projection based on defense. (Returns 1.0 for V1 since def stats are skipped)"""
    return 1.0

def calc_td_probability(expected_tds):
    """Calculates P(1+ TD) using Poisson distribution."""
    if expected_tds <= 0:
        return 0.0
    prob_zero_tds = poisson.pmf(0, expected_tds)
    return 1 - prob_zero_tds

def calc_yardage_probability(expected_yards, prop_line, std_dev=22.0):
    """Calculates P(Over prop_line Yards) using Normal distribution."""
    if expected_yards <= 0:
        return 0.0
    z_score = (prop_line - expected_yards) / std_dev
    return 1 - norm.cdf(z_score)

def generate_nfl_projections(matchup_df, current_week):
    """Takes raw matchup_df and outputs projections."""
    df = matchup_df.copy()
    
    LEAGUE_AVG_TARGET_SHARE = 0.20
    LEAGUE_AVG_YAC = 4.5
    LEAGUE_AVG_TD_RATE = 0.05
    TEAM_AVG_PASS_ATTEMPTS = 35.0
    
    # Position-specific TD rates per target
    POSITION_TD_RATES = {
        'WR': 0.025,
        'TE': 0.030,
        'RB': 0.015,
    }
    
    # Step 1: Shrunk Rates
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0), row.get('routes_run', row.get('routes', 0)), 
            current_week, LEAGUE_AVG_TARGET_SHARE, 'WR'
        ), axis=1
    )
    
    # Step 2: Defensive Matchups (Neutral for V1)
    df['matchup_multiplier'] = df.apply(
        lambda row: apply_defensive_matchup(
            1.0, row.get('opp_pass_epa_allowed', 0), 
            row.get('opp_pressure_rate', 0.25), row.get('position', 'WR')
        ), axis=1
    )
    
    # Step 3: Base Projections
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS * df['matchup_multiplier']).round(1)
    df['proj_rec_yards'] = (df['proj_targets'] * (df.get('adot', 8.0) + LEAGUE_AVG_YAC)).round(1)
    
    # FIX: Calculate TDs without .round() inside the apply loop (floats don't have .round())
    df['proj_tds'] = df.apply(
        lambda row: (
            row['proj_targets'] * 
            POSITION_TD_RATES.get(row.get('position', 'WR'), 0.025) * 
            row['matchup_multiplier']
        ),
        axis=1
    )
    # Round the whole column afterwards using pandas
    df['proj_tds'] = df['proj_tds'].round(2)
    
    # Step 4: Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5), axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # CRITICAL FIX: Only return columns that actually exist in the dataframe
    desired_cols = [
        'player_name', 'position', 'team', 'opponent_team', 
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec'
    ]
    
    # Filter to only columns that are present
    valid_cols = [c for c in desired_cols if c in df.columns]
    
    return df[valid_cols]
