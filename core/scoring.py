# core/scoring.py
# LaunchCast NFL — Scoring Engine
#
# What drives the product's output (and is therefore worth tuning):
#     shrunk_target_share -> proj_targets -> proj_tds -> prob_1plus_td
# BOOM_WEIGHTS does NOT feed that chain — it drives boom_score (a display
# metric) and the td_spike flag. Reweighting it will not move your edge.
# The real knobs are DEF_BLEND and the shrinkage priors below.

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# ============================================================================
# TUNABLE KNOBS — these DO affect predictions. Change one, rerun the backtest,
# keep whichever wins on EDGE (not on Brier).
# ============================================================================

# Weight given to opponent defense in the TD-rate blend.
#
# EXPERIMENT IN PROGRESS. Evidence (2024, 18wk): def_td_per_tgt correlates
# +0.003 with actual scoring — statistically indistinguishable from zero.
# Baseline at 0.4 produced: TD edge +17.2pp, Brier 0.1145.
#
# Set to 0.0 to test removing the matchup layer entirely. Compare EDGE (not
# Brier) against that baseline and keep the winner. Three plausible outcomes:
#   edge HOLDS/RISES -> defense adds nothing; keep 0.0, model is simpler
#   edge FALLS a lot -> the blend was doing useful work after all; revert
#   edge falls SLIGHTLY -> the blend's value was shrinking extreme player
#                          rates toward league average, not matchup signal.
#                          Try 0.2, or shrink td_per_tgt harder instead.
DEF_BLEND = 0.0

# Bayesian shrinkage priors, in TARGET units.
# Evidence: raw target_share (+0.283) still beats shrunk (+0.228), so these
# may want to go lower still. Tune until shrunk >= raw.
PRIOR_TARGET_SHARE_EARLY = 15   # weeks 1-3
PRIOR_TARGET_SHARE_MID = 10     # weeks 4-10
PRIOR_TARGET_SHARE_LATE = 5     # weeks 11+
# EXPERIMENT IN PROGRESS (was 90).
# Removing DEF_BLEND cost a layer of hidden shrinkage: blending 40% toward a
# defense rate near the league mean was quietly pulling elite TD rates down.
# The calibration gap widened +1.0pp -> +1.6pp as a result. This raises the
# prior directly instead, which does the same job honestly.
#   Baseline to beat (DEF_BLEND=0.0, PRIOR_TD_RATE=90):
#       TD edge +17.8pp | lift 2.27x | Brier 0.1138 | calibration gap +1.6pp
#   KEEP 120 only if the gap closes toward +1.0pp AND edge holds near +17.8pp.
#   REVERT to 90 if edge drops — separation is worth more than a centred mean.
# NOTE: this app's own data showed shrinkage DESTROYED 25% of target share's
# signal, so more shrinkage is not automatically better. Let the number decide.
PRIOR_TD_RATE = 120             # TD rate stabilises slowly
PRIOR_YDS_RATE = 35             # yards/target stabilises faster

# ============================================================================
# BOOM_WEIGHTS — display metric only (see header note).
# Keys MUST match the columns calc_boom_score reads, or the weight proposal
# will measure one thing while the score multiplies another.
# ============================================================================
BOOM_WEIGHTS = {
    "target_share": 0.585,        # corr +0.283, std 0.051 (strong, stable)
    "shrunk_yds_per_tgt": 0.150,  # corr -0.105 (anti-predictor, demoted)
    "shrunk_td_per_tgt": 0.265,   # corr +0.085 (modest positive)
}

# League anchors
LEAGUE_AVG_TARGET_SHARE = 0.11   # ~9-10 targeted players per team
LEAGUE_AVG_YDS_PER_TGT = 11.0
LEAGUE_AVG_TD_PER_TGT = 0.05
TEAM_AVG_PASS_ATTEMPTS = 35.0    # fallback only; per-team value preferred

# Prop lines
YARDS_LINE = 45.5
RECEPTIONS_LINE = 3.5


# ============================================================================
# SHRINKAGE — standard Bayesian: weight = volume / (volume + prior)
# ============================================================================
def _shrink(actual, volume, prior, league_avg):
    if volume is None or volume <= 0:
        return league_avg
    f = volume / (volume + prior)
    return f * actual + (1 - f) * league_avg


def calculate_shrunk_rate(actual_rate, volume, current_week, league_avg_rate,
                          position="WR"):
    if current_week <= 3:
        prior = PRIOR_TARGET_SHARE_EARLY
    elif current_week <= 10:
        prior = PRIOR_TARGET_SHARE_MID
    else:
        prior = PRIOR_TARGET_SHARE_LATE
    return _shrink(actual_rate, volume, prior, league_avg_rate)


def shrink_td_rate(actual, targets, league_avg=LEAGUE_AVG_TD_PER_TGT,
                   prior=PRIOR_TD_RATE):
    return _shrink(actual, targets, prior, league_avg)


def shrink_yds_rate(actual, targets, league_avg=LEAGUE_AVG_YDS_PER_TGT,
                    prior=PRIOR_YDS_RATE):
    return _shrink(actual, targets, prior, league_avg)


# ============================================================================
# PROBABILITIES
# ============================================================================
def calc_td_probability(expected_tds):
    """P(1+ TD) via Poisson."""
    try:
        if expected_tds is None or expected_tds <= 0:
            return 0.0
        return float(1 - poisson.pmf(0, expected_tds))
    except (TypeError, ValueError):
        return 0.0


def calc_yardage_probability(expected_yards, prop_line, proj_targets):
    """P(Over line) via Normal, with variance scaled to projected volume.

    NOTE: receiving yards are right-skewed with a large zero-mass, so the
    Normal over-predicts the middle of the distribution. Backtest shows the
    RANKING is excellent (+46pp edge) but the probabilities run ~7pp hot.
    If you need honest probabilities (e.g. to compare with a book line),
    fit an isotonic calibration on weeks 1..N-1 INSIDE the backtest loop —
    never from a shared/cached model, which leaks across weeks."""
    try:
        if expected_yards is None or expected_yards <= 0:
            return 0.0
        sd = 15.0 + (float(proj_targets or 0) * 1.5)
        sd = max(12.0, min(30.0, sd))
        return float(1 - norm.cdf((prop_line - expected_yards) / sd))
    except (TypeError, ValueError):
        return 0.0


# ============================================================================
# COMPOSITES
# ============================================================================
def calc_boom_score(row):
    """0-100 display composite. Components normalised to 0-1 FIRST so the
    scale stays 0-100 whatever the weights are."""
    vol = min(1.0, float(row.get("target_share", 0) or 0) / 0.30)
    eff = min(1.0, float(row.get("shrunk_yds_per_tgt", LEAGUE_AVG_YDS_PER_TGT) or 0) / 15.0)
    rz = min(1.0, float(row.get("shrunk_td_per_tgt", LEAGUE_AVG_TD_PER_TGT) or 0) / 0.10)

    total_w = sum(BOOM_WEIGHTS.values()) or 1.0
    score = (vol * BOOM_WEIGHTS["target_share"]
             + eff * BOOM_WEIGHTS["shrunk_yds_per_tgt"]
             + rz * BOOM_WEIGHTS["shrunk_td_per_tgt"]) / total_w
    return round(score * 100, 1)


def calc_ctx_lift(row):
    """How much does THIS week's matchup move him off his own norm?
    Volume is held constant so only the defensive effect shows."""
    proj_tgt = float(row.get("proj_targets", 0) or 0)
    own = float(row.get("shrunk_td_per_tgt", LEAGUE_AVG_TD_PER_TGT) or 0)
    baseline = calc_td_probability(proj_tgt * own)      # league-neutral D
    this_week = float(row.get("prob_1plus_td", 0) or 0)  # actual D faced
    return round((this_week - baseline) * 100, 1)


# ============================================================================
# MAIN PROJECTION
# ============================================================================
def generate_nfl_projections(matchup_df, current_week):
    if matchup_df is None or matchup_df.empty:
        return pd.DataFrame()
    df = matchup_df.copy()

    # --- shrunk target share, then renormalise: a team's shares MUST sum to 1
    df["shrunk_target_share"] = df.apply(
        lambda r: calculate_shrunk_rate(
            float(r.get("target_share", 0) or 0),
            float(r.get("targets", 0) or 0),
            current_week,
            LEAGUE_AVG_TARGET_SHARE,
            r.get("position", "WR")), axis=1)

    _team_sum = df.groupby("team")["shrunk_target_share"].transform("sum")
    df["shrunk_target_share"] = np.where(
        _team_sum > 0, df["shrunk_target_share"] / _team_sum, 0.0)

    # --- shrunk efficiency rates
    df["shrunk_yds_per_tgt"] = df.apply(
        lambda r: shrink_yds_rate(
            float(r.get("yds_per_tgt", LEAGUE_AVG_YDS_PER_TGT) or LEAGUE_AVG_YDS_PER_TGT),
            float(r.get("targets", 0) or 0)), axis=1)
    df["shrunk_td_per_tgt"] = df.apply(
        lambda r: shrink_td_rate(
            float(r.get("td_per_tgt", LEAGUE_AVG_TD_PER_TGT) or LEAGUE_AVG_TD_PER_TGT),
            float(r.get("targets", 0) or 0)), axis=1)

    # --- volume
    if "team_avg_pass_attempts" not in df.columns:
        df["team_avg_pass_attempts"] = TEAM_AVG_PASS_ATTEMPTS
    df["team_avg_pass_attempts"] = pd.to_numeric(
        df["team_avg_pass_attempts"], errors="coerce").fillna(TEAM_AVG_PASS_ATTEMPTS)

    df["proj_targets"] = (df["shrunk_target_share"]
                          * df["team_avg_pass_attempts"]).round(1)
    df["proj_rec_yards"] = (df["proj_targets"] * df["shrunk_yds_per_tgt"]).round(1)

    # --- TD rate blended with the defense faced (DEF_BLEND is the knob)
    def _proj_tds(r):
        player_rate = float(r.get("shrunk_td_per_tgt", LEAGUE_AVG_TD_PER_TGT) or 0)
        def_rate = r.get("def_td_per_tgt")
        try:
            usable = def_rate is not None and not pd.isna(def_rate) and float(def_rate) > 0
        except (TypeError, ValueError):
            usable = False
        if usable and DEF_BLEND > 0:
            rate = (1 - DEF_BLEND) * player_rate + DEF_BLEND * float(def_rate)
        else:
            rate = player_rate
        return float(r.get("proj_targets", 0) or 0) * rate

    df["proj_tds"] = df.apply(_proj_tds, axis=1).round(2)

    # --- probabilities
    df["prob_1plus_td"] = df["proj_tds"].apply(calc_td_probability)
    df["prob_over_45.5_yds"] = df.apply(
        lambda r: calc_yardage_probability(
            r["proj_rec_yards"], YARDS_LINE, r["proj_targets"]), axis=1)
    df["prob_over_3.5_rec"] = df.apply(
        lambda r: float(1 - poisson.cdf(3, max(0.0, float(r["proj_targets"]) * 0.75))),
        axis=1)

    # --- display composites
    df["boom_score"] = df.apply(calc_boom_score, axis=1)
    df["td_spike"] = df.apply(
        lambda r: bool(float(r.get("prob_1plus_td", 0) or 0) >= 0.20
                       and float(r.get("target_share", 0) or 0) >= 0.20
                       and float(r.get("boom_score", 0) or 0) >= 60), axis=1)
    df["ctx_lift_pp"] = df.apply(calc_ctx_lift, axis=1)

    keep = ["player_id", "player_name", "position", "team", "opponent_team",
            "proj_targets", "proj_rec_yards", "proj_tds",
            "prob_1plus_td", "prob_over_45.5_yds", "prob_over_3.5_rec",
            "boom_score", "td_spike", "ctx_lift_pp",
            "target_share", "shrunk_target_share",
            "yds_per_tgt", "shrunk_yds_per_tgt",
            "td_per_tgt", "shrunk_td_per_tgt",
            "adot", "routes", "targets", "team_avg_pass_attempts",
            "def_yds_per_tgt", "def_td_per_tgt"]
    return df[[c for c in keep if c in df.columns]]
