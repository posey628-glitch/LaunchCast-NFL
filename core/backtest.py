# core/backtest.py
# LaunchCast NFL — Backtesting Engine
#
# Grades week N using features built ONLY from weeks 1..N-1.
# Reports BOTH calibration (Brier, vs a flat-guess reference) and
# discrimination (top-20 edge) — because Brier alone cannot tell them apart.

import pandas as pd
import numpy as np
import streamlit as st

from data.fetcher import (
    assemble_week, get_weekly_player_stats, safe_prior_rates, resolve_season,
)
from core.scoring import generate_nfl_projections, DEF_BLEND

TOP_N = 20  # size of the "headline picks" slice the edge metric grades


def _edge_block(test_df, prob_col, hit_col, top_n=TOP_N):
    """Top-N hit rate vs the slate base rate. This is the scoreboard."""
    base = float(test_df[hit_col].mean()) if len(test_df) else 0.0
    top = test_df.sort_values(prob_col, ascending=False).head(top_n)
    top_rate = float(top[hit_col].mean()) if len(top) else 0.0
    return base, top_rate


def run_nfl_backtest(season=None, max_weeks=18):
    """Returns (results_df, actual_season)."""
    results = []
    actual = resolve_season(season) if season else resolve_season(2024)
    if season and actual != season:
        st.info(f"ℹ️ Backtest running on {actual} (requested {season})")

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

            cols = [c for c in ("player_id", "player_name", "team",
                                "receiving_tds", "receiving_yards", "receptions")
                    if c in outcomes.columns]
            outcomes = outcomes[cols].rename(columns={
                "receiving_tds": "actual_tds",
                "receiving_yards": "actual_yards",
                "receptions": "actual_rec"}).fillna(0)

            test = proj.merge(outcomes,
                              on=[c for c in ("player_id", "player_name", "team")
                                  if c in proj.columns and c in outcomes.columns],
                              how="inner", suffixes=("", "_actual"))
            if test.empty:
                continue

            test["hit_td"] = (test["actual_tds"] >= 1).astype(int)
            test["hit_yards"] = (test["actual_yards"] > 45.5).astype(int)
            test["brier_td"] = (test["prob_1plus_td"] - test["hit_td"]) ** 2
            test["brier_yards"] = (test["prob_over_45.5_yds"] - test["hit_yards"]) ** 2

            base_td, top_td = _edge_block(test, "prob_1plus_td", "hit_td")
            base_yd, top_yd = _edge_block(test, "prob_over_45.5_yds", "hit_yards")

            results.append({
                "Week": week,
                "Players": len(test),
                # touchdowns
                "Avg Brier (TD)": round(float(test["brier_td"].mean()), 4),
                "Hit Rate (TD)": round(base_td * 100, 1),
                "Avg Prob (TD)": round(float(test["prob_1plus_td"].mean()) * 100, 1),
                "Top20 TD Hit %": round(top_td * 100, 1),
                "TD Edge (pp)": round((top_td - base_td) * 100, 1),
                "TD Lift": round(top_td / base_td, 2) if base_td > 0 else None,
                # yardage
                "Avg Brier (Yds)": round(float(test["brier_yards"].mean()), 4),
                "Hit Rate (Yds)": round(base_yd * 100, 1),
                "Avg Prob (Yds)": round(float(test["prob_over_45.5_yds"].mean()) * 100, 1),
                "Top20 Yds Hit %": round(top_yd * 100, 1),
                "Yds Edge (pp)": round((top_yd - base_yd) * 100, 1),
                "Yds Lift": round(top_yd / base_yd, 2) if base_yd > 0 else None,
            })
        except Exception:
            continue

    return pd.DataFrame(results), actual


def _flat_brier(base_rate_pct):
    """Brier of predicting the base rate for EVERY player: b*(1-b).
    If your model can't beat this, its Brier is measuring calibration only."""
    b = base_rate_pct / 100.0
    return b * (1 - b)


def _prop_summary(lines, label, brier, hit, prob, top, edge, lift):
    lines.append("")
    lines.append(label)
    lines.append("-" * 58)
    lines.append(f"Brier:          {brier:.4f}   (flat-guess reference "
                 f"{_flat_brier(hit):.4f}, gain {_flat_brier(hit) - brier:+.4f})")
    lines.append(f"Hit rate:       {hit:.1f}%")
    lines.append(f"Avg predicted:  {prob:.1f}%   (calibration gap {prob - hit:+.1f}pp)")
    lines.append(f"Top-{TOP_N} hit rate: {top:.1f}%")
    lines.append(f"EDGE:           {edge:+.1f}pp   (lift {lift:.2f}x)")
    if edge >= 10:
        lines.append("✅ STRONG EDGE — top picks clearly beat the field")
    elif edge >= 5:
        lines.append("🟡 MODEST EDGE — some signal, needs refinement")
    else:
        lines.append("⚠️ WEAK EDGE — calibrated but not discriminating")


def generate_nfl_backtest_copy_text(results_df, season=None):
    if results_df is None or results_df.empty:
        return "No backtest data available."

    lines = ["🏈 LAUNCHCAST NFL — BACKTEST REPORT", "=" * 58]
    lines.append(f"Season: {season} | Weeks graded: {len(results_df)} | "
                 f"Range: {int(results_df['Week'].min())}-{int(results_df['Week'].max())} "
                 f"| DEF_BLEND={DEF_BLEND}")
    lines.append("")
    lines.append("Read EDGE, not Brier. Brier is dominated by the ~86% of")
    lines.append("players who don't score, so it mostly measures calibration.")
    lines.append("EDGE measures whether the top picks actually beat the field.")

    _prop_summary(
        lines, "🎯 TOUCHDOWN PROPS",
        results_df["Avg Brier (TD)"].mean(), results_df["Hit Rate (TD)"].mean(),
        results_df["Avg Prob (TD)"].mean(), results_df["Top20 TD Hit %"].mean(),
        results_df["TD Edge (pp)"].mean(), results_df["TD Lift"].mean())

    if "Top20 Yds Hit %" in results_df.columns:
        _prop_summary(
            lines, "📏 YARDAGE PROPS (Over 45.5)",
            results_df["Avg Brier (Yds)"].mean(), results_df["Hit Rate (Yds)"].mean(),
            results_df["Avg Prob (Yds)"].mean(), results_df["Top20 Yds Hit %"].mean(),
            results_df["Yds Edge (pp)"].mean(), results_df["Yds Lift"].mean())

    lines += ["", "📊 WEEKLY BREAKDOWN", "-" * 58,
              f"{'Wk':<3}|{'N':>5}|{'BrierTD':>8}|{'Hit%':>6}|{'Prob%':>6}|"
              f"{'TDEdge':>7}|{'BrierYd':>8}|{'YHit%':>6}|{'YdEdge':>7}", "-" * 58]
    for _, r in results_df.iterrows():
        lines.append(
            f"{int(r['Week']):<3}|{int(r['Players']):>5}|"
            f"{r['Avg Brier (TD)']:>8.4f}|{r['Hit Rate (TD)']:>6.1f}|"
            f"{r['Avg Prob (TD)']:>6.1f}|{r.get('TD Edge (pp)', 0):>+7.1f}|"
            f"{r['Avg Brier (Yds)']:>8.4f}|{r['Hit Rate (Yds)']:>6.1f}|"
            f"{r.get('Yds Edge (pp)', 0):>+7.1f}")
    lines.append("-" * 58)

    lines += ["", "🔍 KEY INSIGHTS", "-" * 58]
    b = results_df.loc[results_df["Avg Brier (TD)"].idxmin()]
    w = results_df.loc[results_df["Avg Brier (TD)"].idxmax()]
    lines.append(f"• Best TD calibration:  Week {int(b['Week'])} ({b['Avg Brier (TD)']:.4f})")
    lines.append(f"• Worst TD calibration: Week {int(w['Week'])} ({w['Avg Brier (TD)']:.4f})")
    be = results_df.loc[results_df["TD Edge (pp)"].idxmax()]
    we = results_df.loc[results_df["TD Edge (pp)"].idxmin()]
    lines.append(f"• Best TD edge:         Week {int(be['Week'])} ({be['TD Edge (pp)']:+.1f}pp)")
    lines.append(f"• Worst TD edge:        Week {int(we['Week'])} ({we['TD Edge (pp)']:+.1f}pp)")
    lines.append("")
    lines.append(f"NOTE: with only {TOP_N} picks/week, one standard error is "
                 f"~10pp. Judge the AVERAGE, not any single week.")
    return "\n".join(lines)
