# data/fetcher.py
# LaunchCast NFL — Data Fetcher
#
# Guarantees:
#   * Features for week N are built ONLY from weeks 1..N-1 (no leakage).
#   * The season actually loaded is stamped and readable (resolve_season),
#     so a silent fallback can never mislabel a report.
#   * Prior-season priors are refused when they'd resolve to the same season
#     as the one being graded (that would leak the outcome into the prior).

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = CURRENT_YEAR - 1
else:
    PREFERRED_SEASON = CURRENT_YEAR


# ============================================================================
# RAW LOADER — cached, with an HONEST fallback
# ============================================================================
@st.cache_data(ttl=3600)
def _load_weekly_raw(year: int) -> pd.DataFrame:
    """Load one season of weekly data. Falls back up to two years, and STAMPS
    which season actually loaded so downstream code can't be fooled."""
    import nfl_data_py as nfl

    for y in [year, year - 1, year - 2]:
        try:
            df = nfl.import_weekly_data([y])
            if df is not None and not df.empty:
                if y != year:
                    st.info(f"ℹ️ {year} data unavailable — using {y}")
                df.attrs["season_used"] = y
                return df
        except Exception:
            continue

    st.error(f"❌ No weekly data for {year} or its fallbacks.")
    return pd.DataFrame()


def resolve_season(year: int) -> int:
    """The season that ACTUALLY loaded (may differ from what was asked for)."""
    df = _load_weekly_raw(year)
    if df is None or df.empty:
        return year
    return df.attrs.get("season_used", year)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """nflverse column names vary by release; map to our canonical names."""
    rename_map = {}
    if "team" not in df.columns:
        if "recent_team" in df.columns:
            rename_map["recent_team"] = "team"
        elif "posteam" in df.columns:
            rename_map["posteam"] = "team"
    if "opponent_team" not in df.columns:
        if "defteam" in df.columns:
            rename_map["defteam"] = "opponent_team"
        elif "opp" in df.columns:
            rename_map["opp"] = "opponent_team"
    return df.rename(columns=rename_map) if rename_map else df


def _group_keys(df: pd.DataFrame) -> list:
    keys = ["player_id", "player_name", "team"]
    if "position" in df.columns:
        keys.append("position")
    return keys


def _agg_spec(df: pd.DataFrame) -> dict:
    """Only aggregate columns that exist — nflverse drops some by season."""
    spec = {}
    for out, src in (("targets", "targets"),
                     ("receiving_yards", "receiving_yards"),
                     ("receiving_tds", "receiving_tds"),
                     ("air_yards", "air_yards"),
                     ("routes", "routes")):
        if src in df.columns:
            spec[out] = (src, "sum")
    return spec


# ============================================================================
# PRIOR-SEASON RATES (used to seed weeks 1-3)
# ============================================================================
@st.cache_data(ttl=3600)
def load_prior_rates_from_season(season: int) -> pd.DataFrame:
    """Full-season per-player rates from a PRIOR season, used as priors."""
    try:
        raw = _load_weekly_raw(season)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = normalize_columns(raw)

        spec = _agg_spec(raw)
        if "targets" not in spec:
            return pd.DataFrame()

        g = raw.groupby(_group_keys(raw), as_index=False).agg(**spec)

        g["prior_yds_per_tgt"] = np.where(
            g["targets"] > 0, g["receiving_yards"] / g["targets"], 11.0)
        g["prior_td_per_tgt"] = np.where(
            g["targets"] > 0, g["receiving_tds"] / g["targets"], 0.05)

        g["_team_tgts"] = g.groupby("team")["targets"].transform("sum")
        g["prior_target_share"] = np.where(
            g["_team_tgts"] > 0, g["targets"] / g["_team_tgts"], 0.0)

        _weeks = raw.groupby("team", as_index=False)["week"].nunique() \
                    .rename(columns={"week": "team_weeks"})
        _tgts = raw.groupby("team", as_index=False)["targets"].sum() \
                   .rename(columns={"targets": "team_total_targets"})
        _vol = _tgts.merge(_weeks, on="team", how="inner")
        _vol["prior_team_pass_att"] = np.where(
            _vol["team_weeks"] > 0,
            _vol["team_total_targets"] / _vol["team_weeks"], 35.0)

        g = g.merge(_vol[["team", "prior_team_pass_att"]], on="team", how="left")
        g["prior_team_pass_att"] = g["prior_team_pass_att"].fillna(35.0)

        g = g.drop_duplicates("player_id", keep="first")
        g.attrs["season_used"] = raw.attrs.get("season_used", season)
        return g
    except Exception as e:
        st.warning(f"Prior rates unavailable: {e}")
        return pd.DataFrame()


def safe_prior_rates(actual_season: int):
    """Priors for weeks 1-3, but ONLY if they come from a different season.
    If the prior season falls back to the season being graded, the 'prior'
    would contain the outcomes we're predicting — refuse it instead."""
    prior_season = actual_season - 1
    if resolve_season(prior_season) == actual_season:
        st.warning("⚠️ Prior-season data unavailable — week 1-3 priors DISABLED "
                   "(using them would leak the graded season into the prior)")
        return pd.DataFrame()
    return load_prior_rates_from_season(prior_season)


# ============================================================================
# PLAYER FEATURES — weeks 1..N-1 only
# ============================================================================
def build_features_through(week: int, year: int,
                           prior_rates: pd.DataFrame = None) -> pd.DataFrame:
    """Season-to-date player rates using ONLY weeks strictly before `week`."""
    try:
        raw = _load_weekly_raw(year)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = normalize_columns(raw)
        hist = raw[raw["week"] < week]

        # ---- Week 1: last season IS the projection base ----
        if hist.empty:
            if prior_rates is None or prior_rates.empty:
                return pd.DataFrame()
            g = prior_rates.rename(columns={
                "prior_target_share": "target_share",
                "prior_yds_per_tgt": "yds_per_tgt",
                "prior_td_per_tgt": "td_per_tgt",
                "prior_team_pass_att": "team_avg_pass_attempts",
            }).copy()
            g["games"] = 0
            g["adot"] = 8.0
            if "routes" not in g.columns:
                g["routes"] = 0
            g["team_weeks"] = 0
            return g.drop_duplicates("player_id", keep="first")

        spec = _agg_spec(hist)
        if "targets" not in spec:
            return pd.DataFrame()

        keys = _group_keys(hist)
        g = hist.groupby(keys, as_index=False).agg(**spec)

        _gm = hist.groupby(keys, as_index=False)["week"].nunique() \
                  .rename(columns={"week": "games"})
        g = g.merge(_gm, on=keys, how="left")

        # team pass volume, season-to-date
        _weeks = hist.groupby("team", as_index=False)["week"].nunique() \
                     .rename(columns={"week": "team_weeks"})
        _tgts = hist.groupby("team", as_index=False)["targets"].sum() \
                    .rename(columns={"targets": "team_total_targets"})
        _vol = _tgts.merge(_weeks, on="team", how="inner")
        _vol["team_avg_pass_attempts"] = np.where(
            _vol["team_weeks"] > 0,
            _vol["team_total_targets"] / _vol["team_weeks"], 35.0)
        g = g.merge(_vol[["team", "team_avg_pass_attempts", "team_weeks"]],
                    on="team", how="left")
        g["team_avg_pass_attempts"] = g["team_avg_pass_attempts"].fillna(35.0)
        g["team_weeks"] = g["team_weeks"].fillna(0)

        g["yds_per_tgt"] = np.where(
            g["targets"] > 0, g["receiving_yards"] / g["targets"], 11.0)
        g["td_per_tgt"] = np.where(
            g["targets"] > 0, g["receiving_tds"] / g["targets"], 0.05)
        if "air_yards" in g.columns:
            g["adot"] = np.where(
                g["targets"] > 0, g["air_yards"] / g["targets"], 8.0)
        else:
            g["adot"] = 8.0

        # target share against the TEAM's pool, not the league's
        g["_team_tgts"] = g.groupby("team")["targets"].transform("sum")
        g["target_share"] = np.where(
            g["_team_tgts"] > 0, g["targets"] / g["_team_tgts"], 0.0)

        # ---- weeks 2-3: blend last season in ----
        if prior_rates is not None and not prior_rates.empty and week <= 3:
            cols = [c for c in ("player_id", "prior_yds_per_tgt",
                                "prior_td_per_tgt", "prior_target_share",
                                "prior_team_pass_att", "targets")
                    if c in prior_rates.columns]
            g = g.merge(prior_rates[cols].rename(columns={"targets": "prior_targets"}),
                        on="player_id", how="left")

            P = 60.0  # prior strength, in target-units
            n = g["targets"].fillna(0)
            has_prior = g["prior_targets"].fillna(0) > 0
            for cur, pri in (("target_share", "prior_target_share"),
                             ("yds_per_tgt", "prior_yds_per_tgt"),
                             ("td_per_tgt", "prior_td_per_tgt")):
                if pri in g.columns:
                    g[cur] = np.where(
                        has_prior & g[pri].notna(),
                        (n * g[cur] + P * g[pri]) / (n + P),
                        g[cur])
            if "prior_team_pass_att" in g.columns:
                g["team_avg_pass_attempts"] = np.where(
                    g["team_weeks"] > 0,
                    g["team_avg_pass_attempts"],
                    g["prior_team_pass_att"].fillna(35.0))

        # one row per player (a traded player appears once per stint)
        g = g.sort_values("games", ascending=False) \
             .drop_duplicates("player_id", keep="first")
        return g
    except Exception as e:
        st.warning(f"Feature build failed: {e}")
        return pd.DataFrame()


# ============================================================================
# DEFENSIVE FEATURES — weeks 1..N-1 only
# ============================================================================
def build_defensive_features_through(week: int, year: int) -> pd.DataFrame:
    """What each DEFENSE has allowed, season-to-date. 'team' = the defense."""
    try:
        raw = _load_weekly_raw(year)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = normalize_columns(raw)
        hist = raw[raw["week"] < week]
        if hist.empty or "opponent_team" not in hist.columns:
            return pd.DataFrame()

        spec = {}
        if "targets" in hist.columns:
            spec["targets_allowed"] = ("targets", "sum")
        if "receptions" in hist.columns:
            spec["receptions_allowed"] = ("receptions", "sum")
        if "receiving_yards" in hist.columns:
            spec["yards_allowed"] = ("receiving_yards", "sum")
        if "receiving_tds" in hist.columns:
            spec["tds_allowed"] = ("receiving_tds", "sum")

        # Fail LOUD rather than silently dropping the matchup adjustment
        if "targets_allowed" not in spec or "tds_allowed" not in spec:
            st.warning("⚠️ Defensive features unavailable — matchup adjustment "
                       "is OFF this run")
            return pd.DataFrame()

        d = hist.groupby(["opponent_team"], as_index=False).agg(**spec) \
                .rename(columns={"opponent_team": "team"})

        ta = d["targets_allowed"]
        d["def_yds_per_tgt"] = (np.where(ta > 0, d["yards_allowed"] / ta, 11.0)
                                if "yards_allowed" in d.columns else 11.0)
        d["def_td_per_tgt"] = np.where(ta > 0, d["tds_allowed"] / ta, 0.05)
        return d
    except Exception as e:
        st.warning(f"Defensive feature build failed: {e}")
        return pd.DataFrame()


def get_weekly_player_stats(week: int, year: int = None) -> pd.DataFrame:
    """Raw rows for a single week — used ONLY as outcomes, never as features."""
    if year is None:
        year = PREFERRED_SEASON
    try:
        all_data = _load_weekly_raw(year)
        if all_data is None or all_data.empty:
            return pd.DataFrame()
        wk = all_data[all_data["week"] == week].copy()
        return normalize_columns(wk) if not wk.empty else pd.DataFrame()
    except Exception as e:
        st.warning(f"Weekly stats fetch failed: {e}")
        return pd.DataFrame()


# ============================================================================
# THE ONE ASSEMBLY PATH — used by live, backtest, and patterns alike
# ============================================================================
def assemble_week(week: int, season: int,
                  prior_rates: pd.DataFrame = None) -> pd.DataFrame:
    """Features (weeks 1..N-1) + this week's team/opponent + the defense faced.

    Every consumer calls THIS so live projections and graded backtests can
    never drift apart."""
    features = build_features_through(week, season, prior_rates=prior_rates)
    if features.empty:
        return pd.DataFrame()

    raw = _load_weekly_raw(season)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = normalize_columns(raw)

    need = [c for c in ("player_id", "team", "opponent_team") if c in raw.columns]
    if "opponent_team" not in need:
        return pd.DataFrame()
    week_n = raw[raw["week"] == week][need].drop_duplicates("player_id")
    if week_n.empty:
        return pd.DataFrame()

    # this week's team supersedes the historical one (trades, free agency)
    if "team" in features.columns and "team" in week_n.columns:
        features = features.drop(columns=["team"])
    features = features.merge(week_n, on="player_id", how="inner")
    if features.empty:
        return pd.DataFrame()

    # attach the defense they FACE, not their own
    dfn = build_defensive_features_through(week, season)
    if not dfn.empty:
        features = features.merge(
            dfn[["team", "def_yds_per_tgt", "def_td_per_tgt"]]
               .rename(columns={"team": "opponent_team"}),
            on="opponent_team", how="left")
    return features


def build_matchup_matrix(week: int, year: int = None) -> pd.DataFrame:
    """Live path. Resolves the real season and refuses leaky priors."""
    if year is None:
        year = PREFERRED_SEASON
    actual = resolve_season(year)
    prior = safe_prior_rates(actual) if week <= 3 else None
    return assemble_week(week, actual, prior_rates=prior)
