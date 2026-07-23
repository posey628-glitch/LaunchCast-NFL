# core/scoring.py
# LaunchCast NFL — Scoring Engine with TD Spike & TD Lift

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
    """Takes raw matchup_df and outputs projections with TD Spike and TD Lift."""
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
    
    # Step 4: TD LIFT (Context Lift)
    # How much better is this week's TD probability compared to their season baseline?
    # Baseline = League average TD rate applied to their shrunk target share
    df['baseline_td_prob'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS * LEAGUE_AVG_TD_RATE).apply(calc_td_probability)
    df['td_lift'] = (df['prob_1plus_td'] - df['baseline_td_prob']).round(3)
    
    # Step 5: TD SPIKE (Boom Spot)
    # Identifies elite matchups where multiple factors align
    # We use safe .get() because defensive stats might be missing
    def calculate_td_spike(row):
        # Base requirements: High TD probability and High Target Share
        if row.get('prob_1plus_td', 0) < 0.20: return False
        if row.get('target_share', 0) < 0.20: return False
        
        # If we have defensive data, require a bad matchup
        opp_epa = row.get('opp_pass_epa_allowed', None)
        if pd.notna(opp_epa):
            # EPA > 0.0 means the defense allows more points than average
            if opp_epa > 0.0: return True
        else:
            # Fallback if no defensive data: require very high volume
            if row.get('target_share', 0) > 0.25: return True
            
        return False

    df['td_spike'] = df.apply(calculate_td_spike, axis=1)
    
    # Return only the columns we need
    desired_cols = [
        'player_name', 'position', 'team', 'opponent_team',
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
        'td_lift', 'td_spike'
    ]
    
    valid_cols = [c for c in desired_cols if c in df.columns]
    return df[valid_cols]
