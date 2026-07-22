# app.py
# LaunchCast NFL — Main Entry Point (Fixed Advanced Metrics Tab)

import streamlit as st
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from ui.render import render_nfl_dashboard

# App Config
st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")

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
    st.sidebar.warning("⚠️ **NFL Offseason**\n\nShowing 2024 season data for testing.")

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
        # FIX: Check which columns exist before trying to use them
        available_cols = [c for c in adv_data.columns if c in [
            "player_name", "team", "position",
            "hr_score", "dinger_score", "power_composite",
            "barrel_matchup_score", "two_way_matchup_score",
            "hr_game_pct", "matchup_opp", "power_score",
            "barrel_pct", "pulled_brl_pct", "iso", "avg_ev",
            "hard_hit", "blast_pct", "env_boost"
        ]]
        
        # FIX: Determine sort column based on what's available
        sort_col = None
        for col in ["dinger_score", "hr_score", "power_composite", "power_score"]:
            if col in adv_data.columns:
                sort_col = col
                break
        
        if sort_col:
            # Sort by the best available composite score
            display_df = adv_data[available_cols].sort_values(sort_col, ascending=False)
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "player_name": st.column_config.TextColumn("Player", width="medium"),
                    "team": st.column_config.TextColumn("Team", width="small"),
                    "position": st.column_config.TextColumn("Pos", width="small"),
                    "hr_score": st.column_config.NumberColumn("HR Score", format="%.0f"),
                    "dinger_score": st.column_config.NumberColumn("Dinger", format="%.0f"),
                    "power_composite": st.column_config.NumberColumn("Combo", format="%.0f"),
                    "barrel_matchup_score": st.column_config.NumberColumn("Brl Match", format="%.0f"),
                    "two_way_matchup_score": st.column_config.NumberColumn("Two-Way", format="%.0f"),
                    "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                    "matchup_opp": st.column_config.NumberColumn("Matchup", format="%.0f"),
                    "power_score": st.column_config.NumberColumn("Power", format="%.0f"),
                    "barrel_pct": st.column_config.NumberColumn("Barrel%", format="%.1f%%"),
                    "pulled_brl_pct": st.column_config.NumberColumn("Pull Brl%", format="%.1f%%"),
                    "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                    "avg_ev": st.column_config.NumberColumn("Avg EV", format="%.1f"),
                    "hard_hit": st.column_config.NumberColumn("Hard Hit%", format="%.1f%%"),
                    "blast_pct": st.column_config.NumberColumn("Blast%", format="%.1f%%"),
                    "env_boost": st.column_config.NumberColumn("Env", format="%.2f"),
                }
            )
        else:
            st.warning("No composite scores available. Showing raw data.")
            st.dataframe(
                adv_data[available_cols],
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No advanced data available for this week.")

with tab_backtest:
    st.header("📈 Backtest & Diagnostics")
    st.caption("Model performance and data quality checks.")
    
    st.subheader(" Pipeline Health")
    st.info("Pipeline health diagnostics would appear here. (Placeholder for future implementation)")
