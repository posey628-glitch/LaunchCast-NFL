# core/scoring.py
# LaunchCast NFL — Scoring Engine (Bulletproof, using only guaranteed columns)

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
    Calculates Boom Score and TD Spike using ONLY guaranteed columns.
    """
    df = matchup_df.copy()
    
    LEAGUE_AVG_TARGET_SHARE = 0.20
    LEAGUE_AVG_YAC = 4.5
    LEAGUE_AVG_TD_RATE = 0.05
    TEAM_AVG_PASS_ATTEMPTS = 35.0
    
    # Step 1: Calculate basic rates safely
    df['targets'] = df.get('targets', 0)
    df['receiving_yards'] = df.get('receiving_yards', 0)
    df['receiving_tds'] = df.get('receiving_tds', 0)
    df['air_yards'] = df.get('air_yards', 0)
    df['routes'] = df.get('routes', 0)
    
    # Target share (if team_targets exists, else estimate)
    if 'team_targets' in df.columns:
        df['target_share'] = np.where(df['team_targets'] > 0, df['targets'] / df['team_targets'], 0)
    else:
        df['target_share'] = 0.15 # Fallback baseline
        
    # Step 2: Shrunk Rates
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row['target_share'], 
            row['routes'], 
            current_week, 
            LEAGUE_AVG_TARGET_SHARE, 
            row.get('position', 'WR')
        ), axis=1
    )
    
    # Step 3: Base Projections
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS).round(1)
    
    # Yards per target
    df['yds_per_tgt'] = np.where(df['targets'] > 0, df['receiving_yards'] / df['targets'], 8.0)
    df['proj_rec_yards'] = (df['proj_targets'] * (df['yds_per_tgt'] + LEAGUE_AVG_YAC)).round(1)
    
    # TD rate per target
    df['td_per_tgt'] = np.where(df['targets'] > 0, df['receiving_tds'] / df['targets'], LEAGUE_AVG_TD_RATE)
    df['proj_tds'] = (df['proj_targets'] * df['td_per_tgt']).round(2)
    
    # Step 4: Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5), axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # Step 5: TD LIFT (Context Lift)
    # How much better is this week's TD probability compared to their season baseline?
    df['baseline_td_prob'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS * LEAGUE_AVG_TD_RATE).apply(calc_td_probability)
    df['td_lift'] = (df['prob_1plus_td'] - df['baseline_td_prob']).round(3)
    
    # Step 6: BOOM SCORE (0-100 Composite)
    # Uses ONLY guaranteed columns: target_share, yds_per_tgt, td_per_tgt
    def calc_boom_score(row):
        # Volume score (0-40 pts): Target share
        vol_score = min(40, (row['target_share'] / 0.30) * 40) 
        
        # Efficiency score (0-30 pts): Yards per target (league avg ~11)
        eff_score = min(30, (row['yds_per_tgt'] / 15.0) * 30)
        
        # Red zone score (0-30 pts): TD per target (league avg ~0.05)
        rz_score = min(30, (row['td_per_tgt'] / 0.10) * 30)
        
        return round(vol_score + eff_score + rz_score, 1)

    df['boom_score'] = df.apply(calc_boom_score, axis=1)
    
    # Step 7: TD SPIKE (Boom Spot)
    # Identifies elite matchups: High TD prob + High Target Share + High Boom Score
    def calc_td_spike(row):
        if row.get('prob_1plus_td', 0) >= 0.20 and row.get('target_share', 0) >= 0.20 and row.get('boom_score', 0) >= 60:
            return True
        return False

    df['td_spike'] = df.apply(calc_td_spike, axis=1)
    
    # Return only the columns we need
    desired_cols = [
        'player_name', 'position', 'team', 'opponent_team',
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
        'boom_score', 'td_lift', 'td_spike'
    ]
    
    valid_cols = [c for c in desired_cols if c in df.columns]
    return df[valid_cols]
