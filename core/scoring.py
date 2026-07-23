# core/scoring.py
# LaunchCast NFL — Scoring Engine V5
# FIXES: BOOM_WEIGHTS math, shrunk_rate volume fallback

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# ============================================================================
# SINGLE SOURCE OF TRUTH: BOOM WEIGHTS
# ============================================================================
BOOM_WEIGHTS = {
    'target_share': 0.40,
    'yds_per_tgt': 0.30,
    'td_per_tgt': 0.30,
}

# League averages
LEAGUE_AVG_TARGET_SHARE = 0.20
LEAGUE_AVG_YDS_PER_TGT = 11.0
LEAGUE_AVG_TD_PER_TGT = 0.05
TEAM_AVG_PASS_ATTEMPTS = 35.0

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

def calc_yardage_probability(expected_yards, prop_line, proj_targets):
    """Calculates P(Over prop_line Yards) using Normal distribution."""
    if expected_yards <= 0:
        return 0.0
    
    std_dev = 15.0 + (proj_targets * 1.5)
    std_dev = max(12.0, min(30.0, std_dev))
    
    z_score = (prop_line - expected_yards) / std_dev
    return 1 - norm.cdf(z_score)

def calc_boom_score(row):
    """
    Composite power/volume metric (0-100 scale).
    FIX: Normalize components 0-1 first, then weight them.
    Scale stays 0-100 no matter what BOOM_WEIGHTS are.
    """
    # Normalize each component to 0-1 range
    vol = min(1.0, row.get('target_share', 0) / 0.30)      # 0-1
    eff = min(1.0, row.get('yds_per_tgt', 11.0) / 15.0)    # 0-1
    rz  = min(1.0, row.get('td_per_tgt', 0.05) / 0.10)     # 0-1
    
    # Weight them
    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    score = (vol * BOOM_WEIGHTS['target_share']
             + eff * BOOM_WEIGHTS['yds_per_tgt']
             + rz  * BOOM_WEIGHTS['td_per_tgt']) / total_w
    
    return round(score * 100, 1)

def calc_ctx_lift(row, season_avg_td_per_tgt):
    """
    Context Lift: how much better is this week's matchup-adjusted TD prob
    compared to the player's OWN season baseline?
    """
    this_week_prob = row.get('prob_1plus_td', 0)
    
    baseline_expected_tds = season_avg_td_per_tgt * TEAM_AVG_PASS_ATTEMPTS * row.get('target_share', 0.20)
    baseline_prob = calc_td_probability(baseline_expected_tds)
    
    return round((this_week_prob - baseline_prob) * 100, 1)

def generate_nfl_projections(matchup_df, current_week):
    """Takes raw matchup_df and outputs projections."""
    df = matchup_df.copy()
    
    # Step 1: Shrunk Rates
    # FIX: Volume fallback — routes may not exist, fall back to targets
    def get_volume(row):
        routes = row.get('routes', 0)
        if pd.notna(routes) and routes > 0:
            return routes
        targets = row.get('targets', 0)
        if pd.notna(targets) and targets > 0:
            return targets
        return 0
    
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0),
            get_volume(row),
            current_week,
            LEAGUE_AVG_TARGET_SHARE,
            row.get('position', 'WR')
        ), axis=1
    )
    
    # Step 2: Base Projections
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS).round(1)
    df['proj_rec_yards'] = (df['proj_targets'] * df.get('yds_per_tgt', LEAGUE_AVG_YDS_PER_TGT)).round(1)
    
    # TD projection using matchup-adjusted rate
    def calc_proj_tds(row):
        player_td_rate = row.get('td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        def_td_rate = row.get('def_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        
        if pd.notna(def_td_rate) and def_td_rate > 0:
            blended_rate = (0.6 * player_td_rate) + (0.4 * def_td_rate)
        else:
            blended_rate = player_td_rate
            
        return row['proj_targets'] * blended_rate
    
    df['proj_tds'] = df.apply(calc_proj_tds, axis=1).round(2)
    
    # Step 3: Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5, row['proj_targets']), 
        axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # Step 4: Boom Score
    df['boom_score'] = df.apply(calc_boom_score, axis=1)
    
    # Step 5: TD Spike
    def calc_td_spike(row):
        if (row.get('prob_1plus_td', 0) >= 0.20 and 
            row.get('target_share', 0) >= 0.20 and 
            row.get('boom_score', 0) >= 60):
            return True
        return False
    df['td_spike'] = df.apply(calc_td_spike, axis=1)
    
    # Step 6: CTX LIFT
    df['ctx_lift_pp'] = df.apply(
        lambda row: calc_ctx_lift(row, row.get('td_per_tgt', LEAGUE_AVG_TD_PER_TGT)),
        axis=1
    )
    
    # Return columns
    desired_cols = [
        'player_id', 'player_name', 'position', 'team', 'opponent_team',
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
        'boom_score', 'td_spike', 'ctx_lift_pp',
        'target_share', 'yds_per_tgt', 'td_per_tgt', 'adot', 'routes',
        'def_yds_per_tgt', 'def_td_per_tgt'
    ]
    
    valid_cols = [c for c in desired_cols if c in df.columns]
    return df[valid_cols]
