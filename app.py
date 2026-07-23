# app.py
# LaunchCast NFL — Main Entry Point
# Updated to fit isotonic regression before generating projections

import streamlit as st
import pandas as pd
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections, fit_isotonic_for_week
from ui.render import render_nfl_dashboard, render_game_browser, render_player_deep_dive

# ============================================================================
# APP CONFIG
# ============================================================================
st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    DISPLAY_YEAR = 2024
    DEFAULT_WEEK = 10
    IS_OFFSEASON = True
else:
    DISPLAY_YEAR = CURRENT_YEAR
    DEFAULT_WEEK = 1
    IS_OFFSEASON = False

st.sidebar.title("🏈 LaunchCast NFL")
if IS_OFFSEASON:
    st.sidebar.warning(f"⚠️ **NFL Offseason**\n\nShowing {DISPLAY_YEAR} season data for testing.")

week_selector = st.sidebar.number_input("Select Week", min_value=1, max_value=18, value=DEFAULT_WEEK)

# ============================================================================
# FIT ISOTONIC REGRESSION (before generating projections)
# ============================================================================
# This fixes yardage probabilities by learning the empirical mapping
# from projected yards to actual hit rates, rather than assuming Normal.
if not st.session_state.get(f'_isotonic_fitted_week_{week_selector}', False):
    with st.spinner("Fitting isotonic regression for yardage probabilities..."):
        success = fit_isotonic_for_week(week_selector, DISPLAY_YEAR)
        st.session_state[f'_isotonic_fitted_week_{week_selector}'] = True
        if success:
            st.sidebar.success(
                f"✅ Isotonic regression fitted on "
                f"{st.session_state.get('_isotonic_n_samples', 0)} samples"
            )
        else:
            st.sidebar.info("ℹ️ Using Normal distribution (insufficient historical data)")

# ============================================================================
# DATA LOAD
# ============================================================================
@st.cache_data(ttl=3600)
def load_and_score_data(week, year):
    try:
        matchup_df = build_matchup_matrix(week=week, year=year)
        if matchup_df.empty: return None, "No data available"
        projections = generate_nfl_projections(matchup_df, current_week=week)
        return projections, None
    except Exception as e:
        return None, f"Error: {str(e)}"

projections, error = load_and_score_data(week_selector, DISPLAY_YEAR)

if error:
    st.error(error)
    st.stop()

# ============================================================================
# MAIN UI
# ============================================================================
tab_main, tab_games = st.tabs(["🎯 Projections", "🎮 Game Browser"])

with tab_main:
    render_nfl_dashboard(projections, IS_OFFSEASON, DISPLAY_YEAR)

with tab_games:
    render_game_browser(projections)
