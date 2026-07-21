# core/scoring.py
# LaunchCast NFL — The Scoring Engine
# Translates raw stats and matchups into prop probabilities.

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# ============================================================================
# 1. DYNAMIC WEEK-WEIGHTED SHRINKAGE (The "Sample Size" Fix)
# ============================================================================
def calculate_shrunk_rate(actual_rate, volume, current_week, league_avg_rate, position='WR'):
    """
    Shrinks a player's raw rate toward the league average based on sample size 
    and the time of year. 
    - Early season (Weeks 1-3): High shrinkage (we don't trust small samples).
    - Late season (Weeks 11+): Low shrinkage (we trust the actual data).
    
    volume: For WR/TE this is Routes Run. For RB this is Offensive Snaps.
    """
    # Define the "Prior" (how many routes/snaps we need to fully trust the data)
    # This prior shrinks as the season progresses.
    if current_week <= 3:
        prior_volume = 60  # High prior early on
        weight_actual = 0.20
    elif current_week <= 10:
        prior_volume = 40
        weight_actual = 0.50
    else:
        prior_volume = 20
        weight_actual = 0.80
        
    # Bayesian-style shrinkage formula
    # If volume is 0, we return 100% league average.
    if volume <= 0:
        return league_avg_rate
        
    shrinkage_factor = volume / (volume + prior_volume)
    
    # Blend the actual rate with the league average
    shrunk_rate = (weight_actual * shrinkage_factor * actual_rate) + \
                  ((1 - (weight_actual * shrinkage_factor)) * league_avg_rate)
                  
    return shrunk_rate

# ============================================================================
# 2. DEFENSIVE MATCHUP MULTIPLIER (The "Pitcher" Effect)
# ============================================================================
def apply_defensive_matchup(base_projection, opp_pass_epa, opp_pressure_rate, position='WR'):
    """
    Adjusts a player's base projection based on the specific defense they face.
    - opp_pass_epa: Higher (less negative) means a worse pass defense. Boosts WR/TE.
    - opp_pressure_rate: High pressure kills deep routes. Penalizes aDOT, boosts quick game.
    """
    multiplier = 1.0
    
    if position in ['WR', 'TE']:
        # League average pass EPA allowed is roughly 0.0. 
        # Every +0.05 EPA allowed = ~5% boost to receiving projection.
        epa_boost = opp_pass_epa * 1.0 
        multiplier += epa_boost
        
        # High pressure (league avg ~25%) reduces deep threat efficiency.
        # If pressure > 30%, apply a small penalty to yardage (not targets).
        if opp_pressure_rate > 0.30:
            multiplier -= 0.05 
            
    elif position == 'RB':
        # For RBs, we'd use opp_rush_epa (not implemented in fetcher yet, 
        # but the logic is identical).
        pass
        
    # Clamp the multiplier so one bad defense doesn't create a 300% projection.
    return max(0.5, min(1.5, multiplier))

# ============================================================================
# 3. PROP PROBABILITY CALCULATORS
# ============================================================================
def calc_td_probability(expected_tds):
    """
    Calculates P(1+ TD) using a Poisson distribution.
    TDs are discrete, rare events. Poisson is the mathematically correct model.
    P(X >= 1) = 1 - P(X = 0) = 1 - e^(-lambda)
    """
    if expected_tds <= 0:
        return 0.0
    prob_zero_tds = poisson.pmf(0, expected_tds)
    return 1 - prob_zero_tds

def calc_yardage_probability(expected_yards, prop_line, std_dev=22.0):
    """
    Calculates P(Over prop_line Yards) using a Normal distribution.
    Yardage is continuous. We assume a standard deviation of ~22 yards 
    (empirically derived from NFL WR variance).
    """
    if expected_yards <= 0:
        return 0.0
    # Calculate the Z-score
    z_score = (prop_line - expected_yards) / std_dev
    # P(Over) = 1 - CDF(line)
    return 1 - norm.cdf(z_score)

# ============================================================================
# 4. THE MAIN ORCHESTRATOR
# ============================================================================
def generate_nfl_projections(matchup_df, current_week):
    """
    Takes the raw matchup_df from nfl_fetcher.py and outputs a dataframe 
    with projected stats and prop probabilities for every player.
    """
    df = matchup_df.copy()
    
    # --- LEAGUE AVERAGES (Hardcoded for V1, will be dynamic later) ---
    LEAGUE_AVG_TARGET_SHARE = 0.20  # 20% of team targets
    LEAGUE_AVG_YAC = 4.5
    LEAGUE_AVG_TD_RATE = 0.05       # 5% of targets result in a TD
    
    # --- STEP 1: Calculate Shrunk Rates ---
    # For WRs, volume = routes_run. 
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row['target_share'], row['routes_run'], current_week, 
            LEAGUE_AVG_TARGET_SHARE, 'WR'
        ), axis=1
    )
    
    # --- STEP 2: Apply Defensive Matchups ---
    df['matchup_multiplier'] = df.apply(
        lambda row: apply_defensive_matchup(
            1.0, row.get('opp_pass_epa_allowed', 0), 
            row.get('opp_pressure_rate', 0.25), row['position']
        ), axis=1
    )
    
    # --- STEP 3: Generate Base Projections ---
    # We need the team's implied total passes to convert share to actual targets.
    # For V1, we'll use a league-average 35 pass attempts per game.
    TEAM_AVG_PASS_ATTEMPTS = 35.0 
    
    df['proj_targets'] = (df['shrunk_target_share'] * TEAM_AVG_PASS_ATTEMPTS * df['matchup_multiplier']).round(1)
    df['proj_rec_yards'] = (df['proj_targets'] * (df['adot'] + LEAGUE_AVG_YAC)).round(1)
    df['proj_tds'] = (df['proj_targets'] * LEAGUE_AVG_TD_RATE * df['matchup_multiplier']).round(2)
    
    # --- STEP 4: Calculate Prop Probabilities ---
    # P(1+ TD)
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    
    # P(Over 45.5 Rec Yards) - Example prop line
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5), axis=1
    )
    
    # P(Over 3.5 Receptions) - Using Poisson for discrete receptions
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1 # 75% catch rate assumption
    )
    
    return df[['player_name', 'position', 'team', 'opponent_team', 
               'proj_targets', 'proj_rec_yards', 'proj_tds',
               'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec']]
