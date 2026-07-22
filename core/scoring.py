# core/scoring.py
# LaunchCast NFL — Scoring Engine
# Includes the "Boom Score" composite metric.
# Fully defensive: handles missing columns safely.

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

def calculate_shrunk_rate(actual_rate, volume, current_week, league_avg_rate, position='WR'):
    """
    Shrinks a player's raw rate toward the league average based on sample size.
    """
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
    """
    Takes raw matchup_df and outputs projections.
    Calculates the "Boom Score" composite.
    """
    df = matchup_df.copy()
    
    LEAGUE_AVG_TARGET_SHARE = 0.20
    LEAGUE_AVG_YAC = 4.5
    LEAGUE_AVG_TD_RATE = 0.05
    TEAM_AVG_PASS_ATTEMPTS = 35.0
    
    # Step 1: Shrunk Rates
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0), 
            row.get('routes', 0), 
            current_week, 
            LEAGUE_AVG_TARGET_SHARE, 
            'WR'
        ), axis=1
    )
    
    # Step 2: Base Projections
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS).round(1)
    df['proj_rec_yards'] = (df['proj_targets'] * (df.get('adot', 8.0) + LEAGUE_AVG_YAC)).round(1)
    df['proj_tds'] = (df['proj_targets'] * LEAGUE_AVG_TD_RATE).round(2)
    
    # Step 3: Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5), axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # Step 4: Calculate BOOM SCORE (Composite Metric)
    # Boom Score = Weighted composite of Power, Matchup, and Form
    # We use .get() with defaults to handle missing columns safely
    
    # Power Component (40% weight)
    power_score = df.get('barrel_pct', 0) * 0.4 + df.get('hard_hit', 0) * 0.3 + df.get('iso', 0) * 0.3
    
    # Matchup Component (40% weight)
    matchup_score = df.get('target_share', 0) * 0.5 + df.get('adot', 0) * 0.3 + df.get('route_participation_pct', 0) * 0.2
    
    # Form Component (20% weight) - using recent stats if available
    form_score = df.get('recent_hr', 0) * 0.5 + df.get('recent_iso', 0) * 0.5
    
    # Combine into Boom Score (0-100 scale)
    df['boom_score'] = (power_score * 0.4 + matchup_score * 0.4 + form_score * 0.2).round(1)
    
    # Return only the columns we need
    desired_cols = [
        'player_name', 'position', 'team', 'opponent_team',
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
        'boom_score'
    ]
    
    # Filter to only columns that exist
    valid_cols = [c for c in desired_cols if c in df.columns]
    
    return df[valid_cols]
