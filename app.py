# app.py
# LaunchCast NFL — Main Entry Point

import streamlit as st
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from ui.render import render_nfl_dashboard

# App Config
st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")
CURRENT_WEEK = 5 # Hardcoded for V1 testing, will be dynamic later

@st.cache_data(ttl=3600)
def load_and_score_data(week):
    """Fetches raw data and runs the scoring engine. Cached for 1 hour."""
    # 1. Get the raw matchup matrix (Players + Defensive Opponents)
    matchup_df = build_matchup_matrix(week=week)
    
    if matchup_df.empty:
        return None
        
    # 2. Run the math (Shrinkage, Matchups, Poisson/Normal Probs)
    projections = generate_nfl_projections(matchup_df, current_week=week)
    
    return projections

# --- Main Execution ---
st.sidebar.title("LaunchCast NFL 🏈")
week_selector = st.sidebar.number_input("Select Week", min_value=1, max_value=18, value=CURRENT_WEEK)

projections = load_and_score_data(week_selector)

if projections is not None and not projections.empty:
    # Render the UI (Pass empty schedule/rosters for V1 as we pull them inside fetcher)
    render_nfl_dashboard(schedule=None, rosters=None, projections=projections)
else:
    st.error("No data available for this week yet. Check nflverse data availability.")
