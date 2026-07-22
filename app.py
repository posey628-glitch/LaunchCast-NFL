# app.py
# LaunchCast NFL — Main Entry Point

import streamlit as st
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from ui.render import render_nfl_dashboard

# App Config
st.set_page_config(page_title="LaunchCast NFL", page_icon="", layout="wide")

# Determine current season
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    PREFERRED_SEASON = 2024
    FALLBACK_SEASON = 2023
    IS_OFFSEASON = True
else:
    PREFERRED_SEASON = CURRENT_YEAR
    FALLBACK_SEASON = CURRENT_YEAR - 1
    IS_OFFSEASON = False

# Sidebar
st.sidebar.title("🏈 LaunchCast NFL")

if IS_OFFSEASON:
    st.sidebar.warning("⚠️ **NFL Offseason**\n\nShowing 2024 season data for testing. Live projections begin September 2026.")

week_selector = st.sidebar.number_input("Select Week", min_value=1, max_value=18, value=10)

# Main Tabs
tab_main, tab_advanced, tab_backtest = st.tabs(["🎮 Game-by-Game", "📊 Advanced Metrics", "📈 Backtest"])

with tab_main:
    @st.cache_data(ttl=3600)
    def load_and_score_data(week, year):
        try:
            matchup_df = build_matchup_matrix(week=week, year=year)
            if matchup_df.empty:
                return None, "No data available"
            projections = generate_nfl_projections(matchup_df, current_week=week)
            return projections, None
        except Exception as e:
            return None, f"Error: {str(e)}"

    projections, error = load_and_score_data(week_selector, PREFERRED_SEASON)

    if error:
        st.error(error)
    elif projections is not None and not projections.empty:
        render_nfl_dashboard(None, None, projections, IS_OFFSEASON, PREFERRED_SEASON)
    else:
        st.error("No data available.")

with tab_advanced:
    st.header("📊 Advanced Metrics")
    st.caption("Dinger Scores, Power Composites, and Matchup Scores for every player.")
    
    @st.cache_data(ttl=3600)
    def load_advanced_data(week, year):
        try:
            matchup_df = build_matchup_matrix(week=week, year=year)
            if matchup_df.empty:
                return None
            return matchup_df
        except Exception:
            return None

    adv_data = load_advanced_data(week_selector, PREFERRED_SEASON)
    
    if adv_data is not None and not adv_data.empty:
        st.subheader("💥 Dinger Scores & Power Metrics")
        st.caption("Raw power scores and matchup composites for all players.")
        
        # Select columns to display
        adv_cols = [c for c in [
            "player_name", "team", "position",
            "dinger_score", "power_composite", 
            "barrel_matchup_score", "two_way_matchup_score",
            "hr_game_pct", "hr_score", "pick_score"
        ] if c in adv_data.columns]
        
        st.dataframe(
            adv_data[adv_cols].sort_values("dinger_score", ascending=False),
            hide_index=True, use_container_width=True
        )
    else:
        st.info("No advanced data available for this week.")

with tab_backtest:
    st.header("📈 Backtest & Diagnostics")
    st.caption("Model performance and data quality checks.")
    
    st.subheader(" Pipeline Health")
    st.info("Pipeline health diagnostics would appear here. (Placeholder for future implementation)")
