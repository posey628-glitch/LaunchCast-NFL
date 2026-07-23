# core/scoring.py
# LaunchCast NFL — Scoring Engine V6.2
# FIXES: yds_per_tgt shrinkage, volume/prior unit consistency, ctx_lift volume hold

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
LEAGUE_AVG_TARGET_SHARE = 0.11  # Real mean (targets spread across ~9-10 players)
LEAGUE_AVG_YDS_PER_TGT = 11.0
LEAGUE_AVG_TD_PER_TGT = 0.05
TEAM_AVG_PASS_ATTEMPTS = 35.0

# ============================================================================
# BAYESIAN SHRINKAGE FUNCTIONS
# ============================================================================
def calculate_shrunk_rate(actual_rate, volume, prior_volume, league_avg_rate):
    """
    Standard Bayesian shrinkage: weight = volume / (volume + prior).
    Prior is calibrated to the unit of volume (targets vs routes).
    """
    if volume <= 0:
        return league_avg_rate
    f = volume / (volume + prior_volume)
    return f * actual_rate + (1 - f) * league_avg_rate

def shrink_td_rate(actual, targets, league_avg=LEAGUE_AVG_TD_PER_TGT, prior=90):
    """
    TD-per-target shrinkage with stronger prior.
    TD rate stabilizes far more slowly than other rates.
    """
    if targets <= 0:
        return league_avg
    f = targets / (targets + prior)
    return f * actual + (1 - f) * league_avg

def shrink_yds_rate(actual, targets, league_avg=LEAGUE_AVG_YDS_PER_TGT, prior=35):
    """
    Yards-per-target shrinkage with lighter prior.
    Yards stabilize faster than TDs but slower than target share.
    """
    if targets <= 0:
        return league_avg
    f = targets / (targets + prior)
    return f * actual + (1 - f) * league_avg

# ============================================================================
# PROBABILITY CALCULATORS
# ============================================================================
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

# ============================================================================
# BOOM SCORE
# ============================================================================
def calc_boom_score(row):
    """
    Composite power/volume metric (0-100 scale).
    Uses SHRUNK rates to avoid rewarding small-sample flukes.
    """
    vol = min(1.0, row.get('target_share', 0) / 0.30)
    # FIX: Use shrunk_yds_per_tgt instead of raw
    eff = min(1.0, row.get('shrunk_yds_per_tgt', 11.0) / 15.0)
    # FIX: Use shrunk_td_per_tgt instead of raw
    rz  = min(1.0, row.get('shrunk_td_per_tgt', 0.05) / 0.10)

    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    score = (vol * BOOM_WEIGHTS['target_share']
             + eff * BOOM_WEIGHTS['yds_per_tgt']
             + rz  * BOOM_WEIGHTS['td_per_tgt']) / total_w

    return round(score * 100, 1)

# ============================================================================
# CTX LIFT (FIX: hold volume constant, vary only defense)
# ============================================================================
def calc_ctx_lift(row):
    """
    Context Lift: how much does tonight's specific matchup move the player
    off his own norm?
    
    FIX: Use proj_targets (already shrunk and renormalized) for BOTH
    baseline and this_week, so we only measure the defensive matchup effect.
    """
    proj_tgt = row.get('proj_targets', 0)
    own_rate = row.get('shrunk_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
    
    # Baseline: league-neutral defense (no defensive adjustment)
    baseline_expected_tds = proj_tgt * own_rate
    baseline_prob = calc_td_probability(baseline_expected_tds)
    
    # This week: actual defense faced (already blended in prob_1plus_td)
    this_week_prob = row.get('prob_1plus_td', 0)
    
    return round((this_week_prob - baseline_prob) * 100, 1)

# ============================================================================
# MAIN PROJECTION FUNCTION
# ============================================================================
def generate_nfl_projections(matchup_df, current_week):
    """Takes raw matchup_df and outputs projections."""
    df = matchup_df.copy()
    
    # Step 1: Shrunk target share
    # FIX: Always use targets (not routes) for target-share shrink
    # because target_share is a target-based metric
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0),
            row.get('targets', 0),  # Always targets, not routes
            40 if current_week <= 10 else 20,  # Prior in target-units
            LEAGUE_AVG_TARGET_SHARE
        ), axis=1
    )
    
    # Renormalize so each team's target shares sum to 1.0
    _team_sum = df.groupby('team')['shrunk_target_share'].transform('sum')
    df['shrunk_target_share'] = np.where(
        _team_sum > 0,
        df['shrunk_target_share'] / _team_sum,
        0
    )
    
    # Step 2: Shrunk rates for efficiency metrics
    # FIX: Shrink yds_per_tgt (same bug as TD rate)
    df['shrunk_yds_per_tgt'] = df.apply(
        lambda row: shrink_yds_rate(
            row.get('yds_per_tgt', LEAGUE_AVG_YDS_PER_TGT),
            row.get('targets', 0)
        ), axis=1
    )
    
    # Shrunk TD rate (already had this)
    df['shrunk_td_per_tgt'] = df.apply(
        lambda row: shrink_td_rate(
            row.get('td_per_tgt', LEAGUE_AVG_TD_PER_TGT),
            row.get('targets', 0)
        ), axis=1
    )
    
    # Step 3: Base Projections
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS).round(1)
    
    # FIX: Use shrunk_yds_per_tgt instead of raw
    df['proj_rec_yards'] = (df['proj_targets'] * df['shrunk_yds_per_tgt']).round(1)
    
    # TD projection using shrunk TD rate and matchup-adjusted defense
    def calc_proj_tds(row):
        player_td_rate = row.get('shrunk_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        def_td_rate = row.get('def_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        
        if pd.notna(def_td_rate) and def_td_rate > 0:
            blended_rate = (0.6 * player_td_rate) + (0.4 * def_td_rate)
        else:
            blended_rate = player_td_rate
        
        return row['proj_targets'] * blended_rate
    
    df['proj_tds'] = df.apply(calc_proj_tds, axis=1).round(2)
    
    # Step 4: Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5, row['proj_targets']),
        axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # Step 5: Boom Score (now uses shrunk rates)
    df['boom_score'] = df.apply(calc_boom_score, axis=1)
    
    # Step 6: TD Spike
    def calc_td_spike(row):
        if (row.get('prob_1plus_td', 0) >= 0.20 and
            row.get('target_share', 0) >= 0.20 and
            row.get('boom_score', 0) >= 60):
            return True
        return False
    df['td_spike'] = df.apply(calc_td_spike, axis=1)
    
    # Step 7: CTX LIFT (now holds volume constant)
    df['ctx_lift_pp'] = df.apply(calc_ctx_lift, axis=1)
    
    # Return columns
    desired_cols = [
        'player_id', 'player_name', 'position', 'team', 'opponent_team',
        'proj_targets', 'proj_rec_yards', 'proj_tds',
        'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
        'boom_score', 'td_spike', 'ctx_lift_pp',
        'target_share', 'shrunk_target_share',
        'yds_per_tgt', 'shrunk_yds_per_tgt',
        'td_per_tgt', 'shrunk_td_per_tgt',
        'adot', 'routes', 'def_yds_per_tgt', 'def_td_per_tgt'
    ]
    
    valid_cols = [c for c in desired_cols if c in df.columns]
    return df[valid_cols]
