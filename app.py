# app.py
# LaunchCast NFL — V2 UI (Matches MLB App Architecture with Sidebar Owner Login)

import streamlit as st
import pandas as pd
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from core.backtest import run_nfl_backtest, generate_nfl_backtest_copy_text
from ui.render import render_nfl_dashboard, render_game_browser, render_player_deep_dive

# ============================================================================
# 1. CUSTOM CSS (Ported from MLB App)
# ============================================================================
st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")

st.markdown("""
<style>
    /* Dark + Sleek + Amber Accents (Matches MLB App) */
    .stApp { background-color: #081710; color: #F2EDDD; }
    [data-testid="stHeader"] { background-color: #081710; }
    [data-testid="stSidebar"] { background-color: #0C2113; border-right: 1px solid #27492F; }
    
    /* Cards & Containers */
    [data-testid="stContainer"] { background: #10281A; border: 1px solid #27492F; border-radius: 10px; padding: 15px; }
    [data-testid="stDataFrame"] { background: #0C2113 !important; border: 1px solid #27492F !important; border-radius: 10px !important; }
    
    /* Typography & Links */
    h1, h2, h3 { color: #F2EDDD !important; font-family: 'Oswald', sans-serif; }
    .stMarkdown a { color: #F5C518 !important; text-decoration: none; }
    .stMarkdown a:hover { color: #FFD966 !important; text-shadow: 0 0 8px rgba(245, 197, 24, 0.4); }
    
    /* Buttons */
    .stButton > button { background: #10281A; color: #F5C518; border: 1px solid #27492F; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #16301F; border-color: #F5C518; color: #FFD966; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #F5C518 !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricLabel"] { color: #A8B5A0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. OWNER MODE (Sidebar Login - No URL editing required)
# ============================================================================
OWNER_KEY = "Posey628628!"

# Check if already verified in this session
owner_mode = st.session_state.get("_owner_verified", False)

# If not verified, show sidebar login
if not owner_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Owner Access")
    owner_input = st.sidebar.text_input("Enter Secret Key", type="password", key="owner_key_input")
    
    if owner_input == OWNER_KEY:
        st.session_state["_owner_verified"] = True
        owner_mode = True
        st.rerun() # Refresh the app to unlock the tab
    elif owner_input:
        st.sidebar.error("❌ Incorrect key")

# Also support URL parameter as a backup
if not owner_mode:
    try:
        qp = st.query_params
        url_key = qp.get("owner", "")
        if isinstance(url_key, list):
            url_key = url_key[0] if url_key else ""
        if url_key and url_key == OWNER_KEY:
            st.session_state["_owner_verified"] = True
            owner_mode = True
            st.rerun()
    except Exception:
        pass

# ============================================================================
# 3. APP CONFIG & DATA LOAD
# ============================================================================
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

# Cache data loading
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
# 4. MAIN UI TABS
# ============================================================================
tab_main, tab_games, tab_owner = st.tabs(["🎯 Projections", "🎮 Game Browser", "🔒 Owner Tools"])

with tab_main:
    render_nfl_dashboard(projections, IS_OFFSEASON, DISPLAY_YEAR)

with tab_games:
    render_game_browser(projections)

with tab_owner:
    if not owner_mode:
        st.markdown("### 🔒 Owner Access Required")
        st.caption("This section contains backtesting, pattern analysis, and internal diagnostics.")
        st.info("Enter your secret key in the sidebar to unlock this tab.")
    else:
        st.subheader("📈 2024 Season Backtest")
        if st.button("Run Full Backtest", type="primary"):
            with st.spinner("Processing historical data..."):
                backtest_results = run_nfl_backtest(season=DISPLAY_YEAR, max_weeks=18)
                if not backtest_results.empty:
                    st.dataframe(backtest_results, hide_index=True, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_brier = backtest_results['Avg Brier (TD)'].mean()
                        st.metric("Avg Brier Score (TD)", f"{avg_brier:.4f}")
                    with col2:
                        avg_hit = backtest_results['Hit Rate (TD)'].mean()
                        st.metric("Avg Hit Rate (TD)", f"{avg_hit:.1f}%")
                        
                    st.divider()
                    st.subheader("📋 Copy Report")
                    copy_text = generate_nfl_backtest_copy_text(backtest_results)
                    st.code(copy_text, language="text")
                else:
                    st.error("Backtest failed.")
