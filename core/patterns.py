# core/patterns.py
# LaunchCast NFL — Pattern Analysis Engine
#
# Answers "which features actually predict scoring?" using only leakage-free
# features. Two safeguards carried over from the MLB app:
#   1. MODEL OUTPUTS are tracked for display but can never drive weights —
#      a score built from the features must not cite itself as evidence.
#   2. RAW and PROCESSED versions are tracked side by side, so "is our
#      shrinkage helping?" is answered by data rather than by argument.

import pandas as pd
import numpy as np
import streamlit as st

from data.fetcher import (
    assemble_week, get_weekly_player_stats, safe_prior_rates, resolve_season,
)
from core.scoring import generate_nfl_projections, BOOM_WEIGHTS

# Anything derived from the model's own outputs. Displayed, never weighted.
MODEL_OUTPUT_FEATURES = {
    "boom_score",           # built from target_share + the shrunk rates
    "proj_targets",         # shrunk_target_share x team volume
    "proj_tds", "proj_rec_yards",
    "prob_1plus_td", "prob_over_45.5_yds", "prob_over_3.5_rec",
    "ctx_lift_pp", "td_spike",
}

# Raw and processed side by side — the comparison is the point.
TRACKED_FEATURES = [
    # inputs
    "target_share", "yds_per_tgt", "td_per_tgt", "adot", "routes", "targets",
    "team_avg_pass_attempts", "def_yds_per_tgt", "def_td_per_tgt",
    # processed counterparts
    "shrunk_target_share", "shrunk_yds_per_tgt", "shrunk_td_per_tgt",
    # model outputs (display only)
    "boom_score", "proj_targets",
]


def run_pattern_analysis(season=None, max_weeks=18):
    """Returns (summary_df, actual_season). summary_df carries a Model_Output
    flag so the UI can mark which rows are ineligible as evidence."""
    rows = []
    actual = resolve_season(season) if season else resolve_season(2024)
    if season and actual != season:
        st.info(f"ℹ️ Pattern analysis running on {actual} (requested {season})")

    prior_rates = safe_prior_rates(actual)

    for week in range(1, max_weeks + 1):
        try:
            feats = assemble_week(
                week, actual,
                prior_rates=prior_rates if week <= 3 else None)
            if feats.empty:
                continue

            proj = generate_nfl_projections(feats, current_week=week)
            if proj.empty:
                continue

            outcomes = get_weekly_player_stats(week, actual)
            if outcomes.empty:
                continue
            cols = [c for c in ("player_id", "player_name", "team", "receiving_tds")
                    if c in outcomes.columns]
            outcomes = outcomes[cols].rename(
                columns={"receiving_tds": "actual_tds"}).fillna(0)

            test = proj.merge(outcomes,
                              on=[c for c in ("player_id", "player_name", "team")
                                  if c in proj.columns and c in outcomes.columns],
                              how="inner")
            if test.empty:
                continue
            test["hit_td"] = (test["actual_tds"] >= 1).astype(int)

            for feat in TRACKED_FEATURES:
                if feat not in test.columns:
                    continue
                vals = pd.to_numeric(test[feat], errors="coerce")
                if vals.notna().sum() < 10 or float(vals.std(skipna=True) or 0) <= 0:
                    continue
                corr = vals.corr(test["hit_td"])
                if pd.isna(corr):
                    continue
                rows.append({
                    "Week": week,
                    "Feature": feat,
                    "Model_Output": feat in MODEL_OUTPUT_FEATURES,
                    "Correlation": round(float(corr), 3),
                    "N": len(test),
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), actual

    cdf = pd.DataFrame(rows)
    summary = cdf.groupby(["Feature", "Model_Output"]).agg(
        **{"Avg Correlation": ("Correlation", "mean"),
           "Std Dev": ("Correlation", "std"),
           "Weeks Sampled": ("Correlation", "count")}).reset_index()
    summary["Avg Correlation"] = summary["Avg Correlation"].round(3)
    summary["Std Dev"] = summary["Std Dev"].round(3)
    summary["_abs"] = summary["Avg Correlation"].abs()
    summary = summary.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    return summary[["Feature", "Model_Output", "Avg Correlation",
                    "Std Dev", "Weeks Sampled"]], actual


def get_proposed_weights(pattern_results, min_weeks=5):
    """Evidence-proportional target weights, then a ½-step toward them.

    Sign matters: a NEGATIVE correlation means the feature is an anti-predictor,
    so it earns zero weight rather than a large positive one.

    CAVEAT: BOOM_WEIGHTS drives boom_score, which is a DISPLAY metric — it is
    not in the chain that produces prob_1plus_td. Changing these will not move
    your backtest edge. The knobs that do are DEF_BLEND and the shrinkage
    priors in core/scoring.py."""
    if pattern_results is None or pattern_results.empty:
        return None

    ev = {}
    for _, r in pattern_results.iterrows():
        feat = r["Feature"]
        if r.get("Model_Output", False) or feat in MODEL_OUTPUT_FEATURES:
            continue                      # a score may not vouch for itself
        if feat not in BOOM_WEIGHTS or r["Weeks Sampled"] < min_weeks:
            continue
        corr = float(r["Avg Correlation"])
        ev[feat] = 0.0 if corr < -0.03 else max(0.0, corr)

    if len(ev) < 2:
        return None

    total_ev = sum(ev.values()) or 1.0
    total_w = sum(BOOM_WEIGHTS.values()) or 1.0

    proposed = {}
    for feat, cur in BOOM_WEIGHTS.items():
        target = (ev.get(feat, 0.0) / total_ev) * total_w
        applied = cur + 0.5 * (target - cur)            # ½-step
        proposed[feat] = {
            "current": round(cur, 3),
            "evidence": round(ev.get(feat, 0.0), 3),
            "target": round(target, 3),
            "apply": round(max(0.05, min(0.80, applied)), 3),
        }
    return proposed


def pattern_copy_text(pattern_results, season, proposed=None):
    """Plain-text report — everything visible, nothing hidden behind a widget."""
    if pattern_results is None or pattern_results.empty:
        return "No pattern data available."

    weeks = int(pattern_results["Weeks Sampled"].max())
    lines = ["🧠 LAUNCHCAST NFL — PATTERN ANALYSIS", "=" * 62,
             f"Season: {season} | Weeks: {weeks}", "",
             "MODEL rows are outputs of the model itself — shown for context,",
             "never used as evidence for weights.", "",
             f"{'Feature':<26}{'Type':<7}{'Corr':>8}{'StdDev':>9}{'Weeks':>7}",
             "-" * 62]
    for _, r in pattern_results.iterrows():
        lines.append(f"{r['Feature']:<26}"
                     f"{'MODEL' if r['Model_Output'] else 'RAW':<7}"
                     f"{r['Avg Correlation']:>+8.3f}{r['Std Dev']:>9.3f}"
                     f"{int(r['Weeks Sampled']):>7}")

    if proposed:
        lines += ["", "⚖️ PROPOSED BOOM_WEIGHTS (½-step)", "-" * 62,
                  f"{'Feature':<26}{'Current':>9}{'Evidence':>10}"
                  f"{'Target':>8}{'Apply':>8}"]
        for f, d in proposed.items():
            lines.append(f"{f:<26}{d['current']:>9.3f}{d['evidence']:>10.3f}"
                         f"{d['target']:>8.3f}{d['apply']:>8.3f}")
        lines += ["",
                  "NOTE: boom_score is a DISPLAY metric — it does not feed",
                  "prob_1plus_td, so changing these will NOT move backtest edge.",
                  "The knobs that do: DEF_BLEND and the shrinkage priors."]
    return "\n".join(lines)
