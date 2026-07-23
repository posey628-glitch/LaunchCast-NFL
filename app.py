# app.py
# LaunchCast NFL — Main Entry Point
# Fixed import: removed fit_isotonic_for_week (isotonic regression removed)

import streamlit as st
import pandas as pd
from datetime import datetime
from data.fetcher import build_matchup_matrix
from core.scoring import generate_nfl_projections
from core.backtest import run_nfl_backtest, generate_nfl_backtest_copy_text
from core.patterns import run_pattern_analysis, get_proposed_weights
from ui.render import render_nfl_dashboard, render_game_browser, render_player_deep_dive

# ============================================================================
# 1. CUSTOM CSS
# ============================================================================
st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #081710; color: #F2EDDD; }
    [data-testid="stHeader"] { background-color: #081710; }
    [data-testid="stSidebar"] { background-color: #0C2113; border-right: 1px solid #27492F; }
    [data-testid="stContainer"] { background: #10281A; border: 1px solid #27492F; border-radius: 10px; padding: 15px; }
    [data-testid="stDataFrame"] { background: #0C2113 !important; border: 1px solid #27492F !important; border-radius: 10px !important; }
    h1, h2, h3 { color: #F2EDDD !important; font-family: 'Oswald', sans-serif; }
    .stMarkdown a { color: #F5C518 !important; text-decoration: none; }
    .stMarkdown a:hover { color: #FFD966 !important; }
    .stButton > button { background: #10281A; color: #F5C518; border: 1px solid #27492F; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #16301F; border-color: #F5C518; color: #FFD966; }
    [data-testid="stMetricValue"] { color: #F5C518 !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricLabel"] { color: #A8B5A0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. OWNER MODE (SECURITY FIX: use st.secrets, not hardcoded key)
# ============================================================================
OWNER_KEY = ""
try:
    OWNER_KEY = st.secrets.get("owner_key", "")
except Exception:
    pass

# Sticky session state
owner_mode = st.session_state.get("_owner_verified", False)

# Sidebar login
if not owner_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Owner Access")
    owner_input = st.sidebar.text_input("Enter Secret Key", type="password", key="owner_key_input")
    
    # Gate on non-empty so missing secret can't let "" == "" through
    if owner_input and OWNER_KEY and owner_input == OWNER_KEY:
        st.session_state["_owner_verified"] = True
        owner_mode = True
        st.rerun()
    elif owner_input:
        st.sidebar.error("❌ Incorrect key")

# URL param backup
if not owner_mode:
    try:
        qp = st.query_params
        url_key = qp.get("owner", "")
        if isinstance(url_key, list):
            url_key = url_key[0] if url_key else ""
        if url_key and OWNER_KEY and url_key == OWNER_KEY:
            st.session_state["_owner_verified"] = True
            owner_mode = True
            st.rerun()
    except Exception:
        pass

# Logout
if owner_mode:
    if st.sidebar.button("Log out", key="_owner_logout"):
        st.session_state["_owner_verified"] = False
        st.rerun()

# ============================================================================
# 3. APP CONFIG (FIX: Offseason default is 2024, not 2025)
# ============================================================================
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# FIX: During offseason, use 2024 (most recent complete season)
# 2025 data won't exist until after the 2025 season ends (Feb 2026)
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
# 4. DATA LOAD
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
# 5. MAIN UI TABS
# ============================================================================
tab_main, tab_games, tab_patterns, tab_owner = st.tabs([
    "🎯 Projections", "🎮 Game Browser", "🧠 Pattern Analysis", "🔒 Owner Tools"
])

with tab_main:
    render_nfl_dashboard(projections, IS_OFFSEASON, DISPLAY_YEAR)

with tab_games:
    render_game_browser(projections)

with tab_patterns:
    st.header("🧠 Pattern Analysis")
    st.caption("Which features actually predict winning props? (Requires accumulated weekly data)")
    
    if st.button("Run Pattern Analysis", type="primary"):
        with st.spinner("Analyzing patterns..."):
            pattern_results = run_pattern_analysis(season=DISPLAY_YEAR, max_weeks=18)
            
            if pattern_results is not None and not pattern_results.empty:
                st.subheader("📊 Feature Correlations with TD Hits")
                st.dataframe(pattern_results, hide_index=True, use_container_width=True)
                
                proposed = get_proposed_weights(pattern_results)
                if proposed:
                    st.subheader("⚖️ Proposed Weight Adjustments")
                    st.caption("Conservative ½-step adjustments based on evidence")
                    st.json(proposed)
            else:
                st.info("Not enough data yet. Pattern analysis requires multiple weeks of accumulated results.")

with tab_owner:
    if not owner_mode:
        st.markdown("### 🔒 Owner Access Required")
        st.caption("Enter your secret key in the sidebar to unlock.")
    else:
        st.subheader("📈 Backtest")
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
                    copy_text = generate_nfl_backtest_copy_text(backtest_results, season=DISPLAY_YEAR)
                    st.code(copy_text, language="text")
                else:
                    st.error("Backtest failed.")
