# app.py
# LaunchCast NFL — Main App with Advanced Metrics

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

# Cache data loading
@st.cache_data(ttl=3600)
def load_and_score_data(week, year):
    """Fetch data and generate projections."""
    try:
        # Build matchup matrix
        matchup_df = build_matchup_matrix(week, year)
        
        if matchup_df.empty:
            return None, "No data available for this week"
        
        # Generate projections
        projections = generate_nfl_projections(matchup_df, week)
        
        return projections, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# Main execution
projections, error = load_and_score_data(week_selector, PREFERRED_SEASON)

if error:
    st.error(error)
elif projections is not None and not projections.empty:
    render_nfl_dashboard(None, None, projections, IS_OFFSEASON, PREFERRED_SEASON)
else:
    st.error("No data available for this week.")
