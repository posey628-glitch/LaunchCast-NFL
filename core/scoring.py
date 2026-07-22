# core/scoring.py
# LaunchCast NFL — Scoring Engine with Advanced Metrics
# Integrates pocket efficiency, tempo, play-action, and more

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# ============================================================================
# POCKET EFFICIENCY SCORE
# ============================================================================
def calculate_pocket_efficiency(time_to_throw, pressure_rate, adot, position='QB'):
    """
    Calculates a pocket efficiency score (0-100) based on the interaction
    between time to throw, pressure rate, and aDOT.
    
    Three scenarios:
    - DEEP SHOT (good): High time + low pressure + high aDOT
    - COLLAPSE (bad): High time + high pressure + low aDOT
    - QUICK GAME (neutral): Low time + any pressure
    
    Returns: (score, scenario_label)
    """
    # League averages
    avg_ttt = 2.70  # seconds
    avg_pressure = 0.25  # 25%
    avg_adot = 8.5  # yards
    
    # Normalize inputs
    ttt_norm = (time_to_throw - avg_ttt) / 0.5  # per 0.5s
    pressure_norm = (pressure_rate - avg_pressure) / 0.10  # per 10%
    adot_norm = (adot - avg_adot) / 2.0  # per 2 yards
    
    # Scenario classification
    if time_to_throw > avg_ttt + 0.3 and pressure_rate < avg_pressure - 0.05:
        scenario = "DEEP_SHOT"
        # Good: QB has time and uses it for deep throws
        score = 70 + (ttt_norm * 5) + (adot_norm * 10) - (pressure_norm * 15)
    elif time_to_throw > avg_ttt + 0.3 and pressure_rate > avg_pressure + 0.05:
        scenario = "COLLAPSE"
        # Bad: QB has time but is under pressure (pocket collapsing)
        score = 30 - (pressure_norm * 20) + (adot_norm * 5)
    elif time_to_throw < avg_ttt - 0.3:
        scenario = "QUICK_GAME"
        # Neutral: Quick release, neutralizes pressure
        score = 50 + (adot_norm * 5) - (pressure_norm * 10)
    else:
        scenario = "AVERAGE"
        score = 50 + (ttt_norm * 3) + (adot_norm * 5) - (pressure_norm * 10)
    
    # Clamp to 0-100
    score = max(0, min(100, score))
    
    return score, scenario

# ============================================================================
# TEMPO FACTOR
# ============================================================================
def calculate_tempo_factor(seconds_per_play):
    """
    Calculates a tempo factor showing volume impact.
    Faster tempo = more plays = more opportunities.
    
    League average: ~30 seconds per play
    Fast: <27 seconds
    Slow: >33 seconds
    """
    avg_tempo = 30.0
    
    if seconds_per_play <= 0:
        return 1.0  # league average fallback
    
    # Inverse relationship: faster = higher factor
    tempo_factor = avg_tempo / seconds_per_play
    
    # Clamp to reasonable range
    tempo_factor = max(0.85, min(1.20, tempo_factor))
    
    return tempo_factor

# ============================================================================
# NEUTRAL SCRIPT PASS RATE FACTOR
# ============================================================================
def calculate_script_factor(neutral_script_pass_rate):
    """
    Calculates a pass volume factor based on neutral script pass rate.
    Teams that pass more in neutral scripts = more pass attempts overall.
    
    League average: ~58%
    Pass-heavy: >65%
    Run-heavy: <50%
    """
    avg_pass_rate = 0.58
    
    if neutral_script_pass_rate <= 0:
        return 1.0  # league average fallback
    
    # Direct relationship: higher pass rate = more volume
    script_factor = neutral_script_pass_rate / avg_pass_rate
    
    # Clamp
    script_factor = max(0.85, min(1.20, script_factor))
    
    return script_factor

# ============================================================================
# RED ZONE TARGET SHARE FACTOR
# ============================================================================
def calculate_rz_factor(rz_target_share, league_avg_rz_share=0.15):
    """
    Calculates a red zone factor for TD projections.
    Higher RZ target share = more TD opportunities.
    """
    if rz_target_share <= 0:
        return 1.0
    
    rz_factor = rz_target_share / league_avg_rz_share
    rz_factor = max(0.70, min(1.40, rz_factor))
    
    return rz_factor

# ============================================================================
# SLOT RATE FACTOR
# ============================================================================
def calculate_slot_factor(slot_rate, position='WR'):
    """
    Calculates a slot factor. Slot receivers typically get more targets
    but fewer air yards. Outside receivers get more air yards but fewer targets.
    """
    avg_slot_rate = 0.35  # 35% of routes from slot
    
    if position == 'TE':
        # TEs are often slot-heavy by default
        return 1.0
    
    slot_factor = slot_rate / avg_slot_rate
    slot_factor = max(0.85, min(1.15, slot_factor))
    
    return slot_factor

# ============================================================================
# SEPARATION FACTOR
# ============================================================================
def calculate_separation_factor(separation_avg, league_avg_sep=1.8):
    """
    Calculates a separation factor. More separation = more completions + YAC.
    League average: ~1.8 yards
    """
    if separation_avg <= 0:
        return 1.0
    
    sep_factor = separation_avg / league_avg_sep
    sep_factor = max(0.85, min(1.20, sep_factor))
    
    return sep_factor

# ============================================================================
# CONTESTED CATCH FACTOR
# ============================================================================
def calculate_contested_factor(contested_catch_rate, league_avg=0.40):
    """
    Calculates a contested catch factor. Important for red zone TDs.
    League average: ~40%
    """
    if contested_catch_rate <= 0:
        return 1.0
    
    contested_factor = contested_catch_rate / league_avg
    contested_factor = max(0.80, min(1.30, contested_factor))
    
    return contested_factor

# ============================================================================
# DROP RATE PENALTY
# ============================================================================
def calculate_drop_penalty(drop_rate, league_avg=0.06):
    """
    Calculates a drop penalty. Higher drop rate = fewer actual receptions.
    League average: ~6%
    """
    if drop_rate <= 0:
        return 1.0
    
    # Inverse: higher drop rate = lower factor
    drop_penalty = 1.0 - (drop_rate - league_avg) * 2
    drop_penalty = max(0.85, min(1.05, drop_penalty))
    
    return drop_penalty

# ============================================================================
# MAIN PROJECTION FUNCTION
# ============================================================================
def generate_nfl_projections(matchup_df, current_week):
    """
    Takes raw matchup_df and outputs projections with ALL advanced metrics.
    """
    df = matchup_df.copy()
    
    # === LEAGUE AVERAGES ===
    LEAGUE_AVG_TARGET_SHARE = 0.20
    LEAGUE_AVG_YAC = 4.5
    LEAGUE_AVG_TD_RATE = 0.05
    TEAM_AVG_PASS_ATTEMPTS = 35.0
    
    # === STEP 1: Calculate base target share ===
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0),
            row.get('routes', 0),
            current_week,
            LEAGUE_AVG_TARGET_SHARE,
            'WR'
        ), axis=1
    )
    
    # === STEP 2: Calculate advanced factors ===
    
    # Pocket efficiency (for QBs, affects all receivers)
    if 'time_to_throw' in df.columns and 'pressure_rate_targeted' in df.columns:
        df['pocket_efficiency'], df['pocket_scenario'] = zip(*df.apply(
            lambda row: calculate_pocket_efficiency(
                row.get('time_to_throw', 2.70),
                row.get('pressure_rate_targeted', 0.25),
                row.get('ay_per_target', 8.5),
                row.get('position', 'WR')
            ), axis=1
        ))
        
        # Pocket efficiency affects completion rate and YAC
        df['pocket_factor'] = df['pocket_efficiency'].apply(
            lambda x: 0.90 + (x / 100) * 0.20  # 0.90 to 1.10 range
        )
    else:
        df['pocket_factor'] = 1.0
        df['pocket_scenario'] = 'AVERAGE'
    
    # Tempo factor
    if 'pace_factor' in df.columns:
        df['tempo_factor'] = df['pace_factor'].apply(calculate_tempo_factor)
    else:
        df['tempo_factor'] = 1.0
    
    # Script factor
    if 'neutral_script_pass_rate' in df.columns:
        df['script_factor'] = df['neutral_script_pass_rate'].apply(calculate_script_factor)
    else:
        df['script_factor'] = 1.0
    
    # Red zone factor
    if 'rz_target_share' in df.columns:
        df['rz_factor'] = df['rz_target_share'].apply(calculate_rz_factor)
    else:
        df['rz_factor'] = 1.0
    
    # Slot factor
    if 'slot_rate' in df.columns:
        df['slot_factor'] = df.apply(
            lambda row: calculate_slot_factor(
                row.get('slot_rate', 0.35),
                row.get('position', 'WR')
            ), axis=1
        )
    else:
        df['slot_factor'] = 1.0
    
    # Separation factor
    if 'separation_avg' in df.columns:
        df['separation_factor'] = df['separation_avg'].apply(calculate_separation_factor)
    else:
        df['separation_factor'] = 1.0
    
    # Contested catch factor
    if 'contested_catch_rate' in df.columns:
        df['contested_factor'] = df['contested_catch_rate'].apply(calculate_contested_factor)
    else:
        df['contested_factor'] = 1.0
    
    # Drop penalty
    if 'drop_rate' in df.columns:
        df['drop_penalty'] = df['drop_rate'].apply(calculate_drop_penalty)
    else:
        df['drop_penalty'] = 1.0
    
    # === STEP 3: Calculate projections ===
    
    # Base targets
    df['proj_targets'] = (
        df['shrunk_target_share'] * 
        TEAM_AVG_PASS_ATTEMPTS * 
        df['tempo_factor'] * 
        df['script_factor']
    ).round(1)
    
    # Adjusted targets (after drop penalty)
    df['adj_targets'] = (df['proj_targets'] * df['drop_penalty']).round(1)
    
    # Receptions (adjusted for catch rate and separation)
    df['proj_receptions'] = (
        df['adj_targets'] * 
        df.get('catch_rate', 0.65) * 
        df['separation_factor']
    ).round(1)
    
    # Receiving yards
    df['proj_rec_yards'] = (
        df['proj_receptions'] * 
        (df.get('ay_per_target', 8.5) + LEAGUE_AVG_YAC) *
        df['pocket_factor']
    ).round(1)
    
    # TDs (adjusted for RZ factor and contested factor)
    df['proj_tds'] = (
        df['proj_targets'] * 
        LEAGUE_AVG_TD_RATE * 
        df['rz_factor'] * 
        df['contested_factor']
    ).round(2)
    
    # === STEP 4: Calculate probabilities ===
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    df['prob_over_45.5_yds'] = df.apply(
        lambda row: calc_yardage_probability(row['proj_rec_yards'], 45.5), axis=1
    )
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_receptions']), axis=1
    )
    
    # === STEP 5: Calculate advanced metrics for display ===
    
    # First read target share (already in data, just label it)
    df['first_read_share_display'] = df.get('first_read_share', 0)
    
    # YAC per reception (already in data)
    df['yac_per_rec_display'] = df.get('yac_per_rec', 0)
    
    # Air yards per target (already in data)
    df['ay_per_target_display'] = df.get('ay_per_target', 0)
    
    return df[['player_name', 'position', 'team', 'opponent_team',
               'proj_targets', 'proj_receptions', 'proj_rec_yards', 'proj_tds',
               'prob_1plus_td', 'prob_over_45.5_yds', 'prob_over_3.5_rec',
               'pocket_efficiency', 'pocket_scenario', 'tempo_factor',
               'script_factor', 'rz_factor', 'slot_factor',
               'separation_factor', 'contested_factor', 'drop_penalty',
               'first_read_share_display', 'yac_per_rec_display',
               'ay_per_target_display']]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
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
