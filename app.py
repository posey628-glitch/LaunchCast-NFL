# app.py
# LaunchCast NFL — Main Entry Point

import streamlit as st
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from ui.render import render_nfl_dashboard

# App Config
st.set_page_config(page_title="LaunchCast NFL", page_icon="", layout="wide")

# Determine current season dynamically
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# NFL season runs Sept-Dec. If we're in Jan-Aug, use 2025 data for testing.
if CURRENT_MONTH < 9:
    DISPLAY_YEAR = 2025  # Force 2025 data during offseason
    DEFAULT_WEEK = 10    # Default to mid-season week for testing
    IS_OFFSEASON = True
else:
    DISPLAY_YEAR = CURRENT_YEAR
    DEFAULT_WEEK = 1
    IS_OFFSEASON = False

# Sidebar
st.sidebar.title("LaunchCast NFL ")

if IS_OFFSEASON:
    st.sidebar.warning("⚠️ **NFL Offseason**\n\nShowing 2025 season data for testing. Live projections begin September 2026.")

week_selector = st.sidebar.number_input(
    "Select Week", 
    min_value=1, 
    max_value=18, 
    value=DEFAULT_WEEK
)

# Cache data loading
@st.cache_data(ttl=3600)
def load_and_score_data(week, year):
    """Fetches raw data and runs the scoring engine. Cached for 1 hour."""
    try:
        # 1. Get the raw matchup matrix (Players + Defensive Opponents)
        matchup_df = build_matchup_matrix(week=week, year=year)
        
        if matchup_df.empty:
            return None, "No data available for this week"
            
        # 2. Run the math (Shrinkage, Matchups, Poisson/Normal Probs)
        projections = generate_nfl_projections(matchup_df, current_week=week)
        
        return projections, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- Main Execution ---
projections, error = load_and_score_data(week_selector, DISPLAY_YEAR)

if error:
    st.error(error)
    if IS_OFFSEASON:
        st.info("💡 **Tip:** This is expected during the offseason. The app is configured to show 2025 season data for testing.")
elif projections is not None and not projections.empty:
    # Render the UI
    render_nfl_dashboard(
        schedule=None,
        rosters=None, 
        projections=projections,
        is_offseason=IS_OFFSEASON,
        display_year=DISPLAY_YEAR
    )
else:
    st.error("No data available for this week yet.")
    if IS_OFFSEASON:
        st.info("The app is ready for the 2026 season starting in September!")
