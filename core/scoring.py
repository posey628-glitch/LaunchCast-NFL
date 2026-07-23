# core/scoring.py
# LaunchCast NFL — Scoring Engine
# Updated with reweight and isotonic regression for yardage probabilities

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from sklearn.isotonic import IsotonicRegression

# ============================================================================
# SINGLE SOURCE OF TRUTH: BOOM WEIGHTS (REWEIGHTED)
# ============================================================================
# Reweighted based on 18 weeks of pattern analysis:
# - target_share: +0.283 correlation, std 0.051 (stable, strong evidence)
# - shrunk_yds_per_tgt: -0.105 correlation (anti-predictor, demoted)
# - shrunk_td_per_tgt: +0.085 correlation (modest positive signal)
BOOM_WEIGHTS = {
    'target_share': 0.585,      # was 0.400 → 0.585 (strong evidence)
    'shrunk_yds_per_tgt': 0.150,  # was 0.300 → 0.150 (anti-predictor)
    'shrunk_td_per_tgt': 0.265,   # was 0.300 → 0.265 (modest signal)
}

# League averages
LEAGUE_AVG_TARGET_SHARE = 0.11
LEAGUE_AVG_YDS_PER_TGT = 11.0
LEAGUE_AVG_TD_PER_TGT = 0.05
LEAGUE_AVG_PASS_ATTEMPTS = 35.0

# ============================================================================
# ISOTONIC REGRESSION FOR YARDAGE PROBABILITIES
# ============================================================================
# Fit on historical data only (weeks 1..N-1), predict on current week
# This fixes the distribution problem: receiving yards have mass at zero
# and right skew, not Normal. Isotonic regression learns the mapping
# empirically without assuming a distribution.

@st.cache_resource
def _get_isotonic_model():
    """Get or create the isotonic regression model for yardage probabilities."""
    return IsotonicRegression(out_of_bounds='clip')

def _fit_isotonic_on_history(week, year):
    """
    Fit isotonic regression on historical data (weeks 1..N-1 only).
    Returns True if successful, False if insufficient data.
    """
    from data.fetcher import (
        build_features_through, build_defensive_features_through,
        get_weekly_player_stats, load_prior_rates_from_season,
        _load_weekly_raw, normalize_columns
    )
    
    # Collect training data from all prior weeks
    train_proj_yards = []
    train_hit_yards = []
    
    prior_rates = load_prior_rates_from_season(year - 1)
    
    for train_week in range(1, week):
        try:
            # Build features for this training week
            features = build_features_through(
                train_week, year, 
                prior_rates=prior_rates if train_week <= 3 else None
            )
            if features.empty:
                continue
            
            # Attach opponent
            all_data = _load_weekly_raw(year)
            all_data = normalize_columns(all_data)
            week_n = all_data[all_data['week'] == train_week][
                ['player_id', 'team', 'opponent_team']
            ].drop_duplicates('player_id')
            
            if week_n.empty:
                continue
            
            if 'team' in features.columns:
                features = features.drop(columns=['team'])
            features = features.merge(week_n, on='player_id', how='inner')
            if features.empty:
                continue
            
            # Attach defense
            def_features = build_defensive_features_through(train_week, year)
            if not def_features.empty:
                features = features.merge(
                    def_features[['team', 'def_yds_per_tgt', 'def_td_per_tgt']]
                        .rename(columns={'team': 'opponent_team'}),
                    on='opponent_team',
                    how='left'
                )
            
            # Generate projections
            projections = generate_nfl_projections(features, current_week=train_week)
            if projections.empty:
                continue
            
            # Get actual outcomes
            actuals = get_weekly_player_stats(train_week, year)
            if actuals.empty:
                continue
            
            actuals = actuals[['player_id', 'player_name', 'team', 
                              'receiving_yards']].copy()
            actuals = actuals.rename(columns={'receiving_yards': 'actual_yards'})
            actuals['actual_yards'] = actuals['actual_yards'].fillna(0)
            
            # Merge
            merged = projections.merge(
                actuals, 
                on=['player_id', 'player_name', 'team'], 
                how='inner'
            )
            
            if merged.empty:
                continue
            
            # Extract training data
            train_proj_yards.extend(merged['proj_rec_yards'].values)
            train_hit_yards.extend((merged['actual_yards'] > 45.5).astype(int).values)
            
        except Exception:
            continue
    
    # Fit isotonic regression if we have enough data
    if len(train_proj_yards) >= 50:
        iso = _get_isotonic_model()
        iso.fit(train_proj_yards, train_hit_yards)
        st.session_state['_isotonic_fitted'] = True
        st.session_state['_isotonic_n_samples'] = len(train_proj_yards)
        return True
    else:
        st.session_state['_isotonic_fitted'] = False
        return False

def _predict_yardage_probability_isotonic(proj_yards):
    """
    Predict yardage probability using isotonic regression.
    Falls back to Normal distribution if isotonic not fitted.
    """
    if st.session_state.get('_isotonic_fitted', False):
        iso = _get_isotonic_model()
        return iso.predict(proj_yards)
    else:
        # Fallback to Normal distribution
        return None

# ============================================================================
# BAYESIAN SHRINKAGE FUNCTIONS
# ============================================================================
def calculate_shrunk_rate(actual_rate, volume, current_week, league_avg_rate, position='WR'):
    """Standard Bayesian shrinkage: weight = volume / (volume + prior)."""
    if current_week <= 3:
        prior_volume = 15
    elif current_week <= 10:
        prior_volume = 10
    else:
        prior_volume = 5

    if volume <= 0:
        return league_avg_rate

    f = volume / (volume + prior_volume)
    return f * actual_rate + (1 - f) * league_avg_rate

def shrink_td_rate(actual, targets, league_avg=LEAGUE_AVG_TD_PER_TGT, prior=90):
    """TD-per-target shrinkage with stronger prior."""
    if targets <= 0:
        return league_avg
    f = targets / (targets + prior)
    return f * actual + (1 - f) * league_avg

def shrink_yds_rate(actual, targets, league_avg=LEAGUE_AVG_YDS_PER_TGT, prior=35):
    """Yards-per-target shrinkage with lighter prior."""
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
    """
    Calculates P(Over prop_line Yards) using Normal distribution.
    Note: For more accurate probabilities, use isotonic regression
    via _predict_yardage_probability_isotonic() after fitting.
    """
    if expected_yards <= 0:
        return 0.0

    std_dev = 15.0 + (proj_targets * 1.5)
    std_dev = max(12.0, min(30.0, std_dev))

    z_score = (prop_line - expected_yards) / std_dev
    return 1 - norm.cdf(z_score)

# ============================================================================
# COMPOSITE CALCULATORS
# ============================================================================
def calc_boom_score(row):
    """
    Composite power/volume metric (0-100 scale).
    Uses reweighted BOOM_WEIGHTS.
    """
    vol = min(1.0, row.get('target_share', 0) / 0.30)
    eff = min(1.0, row.get('shrunk_yds_per_tgt', 11.0) / 15.0)
    rz  = min(1.0, row.get('shrunk_td_per_tgt', 0.05) / 0.10)

    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    score = (vol * BOOM_WEIGHTS['target_share']
             + eff * BOOM_WEIGHTS['shrunk_yds_per_tgt']
             + rz  * BOOM_WEIGHTS['shrunk_td_per_tgt']) / total_w

    return round(score * 100, 1)

def calc_ctx_lift(row):
    """Context Lift: how much does tonight's matchup move the player off his norm?"""
    proj_tgt = row.get('proj_targets', 0)
    own_rate = row.get('shrunk_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
    
    baseline_expected_tds = proj_tgt * own_rate
    baseline_prob = calc_td_probability(baseline_expected_tds)
    
    this_week_prob = row.get('prob_1plus_td', 0)
    
    return round((this_week_prob - baseline_prob) * 100, 1)

# ============================================================================
# MAIN PROJECTION FUNCTION
# ============================================================================
def generate_nfl_projections(matchup_df, current_week):
    """Takes raw matchup_df and outputs projections."""
    df = matchup_df.copy()
    
    # Shrunk target share
    df['shrunk_target_share'] = df.apply(
        lambda row: calculate_shrunk_rate(
            row.get('target_share', 0),
            row.get('targets', 0),
            current_week,
            LEAGUE_AVG_TARGET_SHARE,
            row.get('position', 'WR')
        ), axis=1
    )
    
    # Renormalize so each team's target shares sum to 1.0
    _team_sum = df.groupby('team')['shrunk_target_share'].transform('sum')
    df['shrunk_target_share'] = np.where(
        _team_sum > 0,
        df['shrunk_target_share'] / _team_sum,
        0
    )
    
    # Shrunk rates
    df['shrunk_yds_per_tgt'] = df.apply(
        lambda row: shrink_yds_rate(
            row.get('yds_per_tgt', LEAGUE_AVG_YDS_PER_TGT),
            row.get('targets', 0)
        ), axis=1
    )
    
    df['shrunk_td_per_tgt'] = df.apply(
        lambda row: shrink_td_rate(
            row.get('td_per_tgt', LEAGUE_AVG_TD_PER_TGT),
            row.get('targets', 0)
        ), axis=1
    )
    
    # Base Projections
    df['team_avg_pass_attempts'] = df.get('team_avg_pass_attempts', LEAGUE_AVG_PASS_ATTEMPTS)
    if isinstance(df['team_avg_pass_attempts'], (int, float)):
        df['team_avg_pass_attempts'] = LEAGUE_AVG_PASS_ATTEMPTS
    
    df['proj_targets'] = (df['shrunk_target_share'] * df['team_avg_pass_attempts']).round(1)
    df['proj_rec_yards'] = (df['proj_targets'] * df['shrunk_yds_per_tgt']).round(1)
    
    # TD projection
    def calc_proj_tds(row):
        player_td_rate = row.get('shrunk_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        def_td_rate = row.get('def_td_per_tgt', LEAGUE_AVG_TD_PER_TGT)
        
        if pd.notna(def_td_rate) and def_td_rate > 0:
            blended_rate = (0.6 * player_td_rate) + (0.4 * def_td_rate)
        else:
            blended_rate = player_td_rate
        
        return row['proj_targets'] * blended_rate
    
    df['proj_tds'] = df.apply(calc_proj_tds, axis=1).round(2)
    
    # Prop Probabilities
    df['prob_1plus_td'] = df['proj_tds'].apply(calc_td_probability)
    
    # Try isotonic regression for yardage probabilities
    iso_probs = _predict_yardage_probability_isotonic(df['proj_rec_yards'].values)
    if iso_probs is not None:
        df['prob_over_45.5_yds'] = iso_probs
    else:
        # Fallback to Normal distribution
        df['prob_over_45.5_yds'] = df.apply(
            lambda row: calc_yardage_probability(
                row['proj_rec_yards'], 45.5, row['proj_targets']
            ), axis=1
        )
    
    df['prob_over_3.5_rec'] = df.apply(
        lambda row: 1 - poisson.cdf(3, row['proj_targets'] * 0.75), axis=1
    )
    
    # Boom Score (with reweighted BOOM_WEIGHTS)
    df['boom_score'] = df.apply(calc_boom_score, axis=1)
    
    # TD Spike
    def calc_td_spike(row):
        if (row.get('prob_1plus_td', 0) >= 0.20 and
            row.get('target_share', 0) >= 0.20 and
            row.get('boom_score', 0) >= 60):
            return True
        return False
    df['td_spike'] = df.apply(calc_td_spike, axis=1)
    
    # CTX LIFT
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
        'adot', 'routes', 'team_avg_pass_attempts',
        'def_yds_per_tgt', 'def_td_per_tgt'
    ]
    
    valid_cols = [c for c in desired_cols if c in df.columns]
    return df[valid_cols]

# ============================================================================
# FIT ISOTONIC REGRESSION (call from app.py before generating projections)
# ============================================================================
def fit_isotonic_for_week(week, year):
    """
    Fit isotonic regression on historical data for this week.
    Call this once per week before generating projections.
    """
    return _fit_isotonic_on_history(week, year)
