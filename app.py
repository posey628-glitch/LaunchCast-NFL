# app.py
# LaunchCast NFL — Main Entry Point with TD Spike & Backtest Copy

import streamlit as st
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from ui.render import render_nfl_dashboard, render_backtest_section
from core.backtest import run_nfl_backtest

# App Config
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
    st.sidebar.warning("⚠️ **NFL Offseason**\n\nShowing 2024 season data for testing.")

week_selector = st.sidebar.number_input("Select Week", min_value=1, max_value=18, value=DEFAULT_WEEK)

tab_main, tab_backtest = st.tabs([" Projections", "📈 Backtest (2024)"])

with tab_main:
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
    elif projections is not None and not projections.empty:
        render_nfl_dashboard(None, None, projections, IS_OFFSEASON, DISPLAY_YEAR)
    else:
        st.error("No data available.")

with tab_backtest:
    st.header("📈 2024 Season Backtest")
    st.caption("Grading our 2024 projections against actual 2024 outcomes. Lower Brier = Better Calibration.")
    
    if st.button("Run Full 2024 Backtest", type="primary"):
        with st.spinner("Processing 18 weeks of historical data..."):
            backtest_results = run_nfl_backtest(season=2024, max_weeks=18)
            
            if not backtest_results.empty:
                render_backtest_section(backtest_results)
            else:
                st.error("Backtest failed to generate results.")
