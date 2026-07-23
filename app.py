# app.py
# LaunchCast NFL — main entry point

import streamlit as st
import pandas as pd
from datetime import datetime

from data.fetcher import build_matchup_matrix, resolve_season
from core.scoring import generate_nfl_projections, DEF_BLEND
from core.backtest import run_nfl_backtest, generate_nfl_backtest_copy_text
from core.patterns import run_pattern_analysis, get_proposed_weights, pattern_copy_text
from ui.render import render_nfl_dashboard, render_game_browser

st.set_page_config(page_title="LaunchCast NFL", page_icon="🏈", layout="wide")

# ============================================================================
# THEME
# ============================================================================
st.markdown("""
<style>
    .stApp { background-color: #081710; color: #F2EDDD; }
    [data-testid="stHeader"] { background-color: #081710; }
    [data-testid="stSidebar"] { background-color: #0C2113; border-right: 1px solid #27492F; }
    [data-testid="stDataFrame"] { background: #0C2113 !important; border: 1px solid #27492F !important; border-radius: 10px !important; }
    h1, h2, h3 { color: #F2EDDD !important; font-family: 'Oswald', sans-serif; }
    .stMarkdown a, .stMarkdown a:link, .stMarkdown a:visited { color: #F5C518 !important; text-decoration: none; }
    .stMarkdown a:hover { color: #FFD966 !important; }
    .stButton > button { background: #10281A; color: #F5C518; border: 1px solid #27492F; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #16301F; border-color: #F5C518; color: #FFD966; }
    [data-testid="stMetricValue"] { color: #F5C518 !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricLabel"] { color: #A8B5A0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# OWNER MODE — key lives in st.secrets, never in source
# ============================================================================
OWNER_KEY = ""
try:
    OWNER_KEY = st.secrets.get("owner_key", "")
except Exception:
    pass

owner_mode = st.session_state.get("_owner_verified", False)

if not owner_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Owner Access")
    pwd = st.sidebar.text_input("Secret key", type="password", key="owner_key_input")
    # `and OWNER_KEY` so a missing secret can't let an empty string through
    if pwd and OWNER_KEY and pwd == OWNER_KEY:
        st.session_state["_owner_verified"] = True
        owner_mode = True
        st.rerun()
    elif pwd:
        st.sidebar.error("❌ Incorrect key")

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

if owner_mode and st.sidebar.button("Log out", key="_owner_logout"):
    st.session_state["_owner_verified"] = False
    st.rerun()

# ============================================================================
# SEASON / WEEK
# ============================================================================
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

if CURRENT_MONTH < 9:
    REQUESTED_YEAR = CURRENT_YEAR - 1     # most recent completed season
    DEFAULT_WEEK = 10
    IS_OFFSEASON = True
else:
    REQUESTED_YEAR = CURRENT_YEAR
    DEFAULT_WEEK = 1
    IS_OFFSEASON = False

# The season that ACTUALLY loaded — never assume the requested one
ACTUAL_SEASON = resolve_season(REQUESTED_YEAR)

st.sidebar.title("🏈 LaunchCast NFL")
if ACTUAL_SEASON != REQUESTED_YEAR:
    st.sidebar.warning(f"⚠️ {REQUESTED_YEAR} unavailable — showing "
                       f"**{ACTUAL_SEASON}** data.")
elif IS_OFFSEASON:
    st.sidebar.info(f"⚠️ **Offseason** — showing {ACTUAL_SEASON} season data.")

week_selector = st.sidebar.number_input(
    "Week", min_value=1, max_value=18, value=DEFAULT_WEEK)
st.sidebar.caption(f"Season in use: **{ACTUAL_SEASON}** · DEF_BLEND={DEF_BLEND}")

# ============================================================================
# DATA
# ============================================================================
@st.cache_data(ttl=3600)
def load_and_score_data(week, year):
    try:
        matchup = build_matchup_matrix(week=week, year=year)
        if matchup.empty:
            return None, ("No data for this week. Early-season weeks need "
                          "prior-season data to project from.")
        return generate_nfl_projections(matchup, current_week=week), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


projections, error = load_and_score_data(week_selector, ACTUAL_SEASON)
if error:
    st.error(error)
    st.stop()

# ============================================================================
# TABS
# ============================================================================
tab_main, tab_games, tab_patterns, tab_owner = st.tabs(
    ["🎯 Projections", "🎮 Game Browser", "🧠 Pattern Analysis", "🔒 Owner Tools"])

with tab_main:
    render_nfl_dashboard(projections, IS_OFFSEASON, ACTUAL_SEASON)

with tab_games:
    render_game_browser(projections)

with tab_patterns:
    st.header("🧠 Pattern Analysis")
    st.caption("Which features actually predict scoring? Rows marked MODEL are "
               "outputs of the model itself — shown for context, never used as "
               "evidence for weights.")

    if st.button("Run Pattern Analysis", type="primary"):
        with st.spinner("Grading every week against its own future-blind features..."):
            results, season_used = run_pattern_analysis(
                season=ACTUAL_SEASON, max_weeks=18)

        if results is not None and not results.empty:
            st.subheader("📊 Feature correlations with TD hits")
            show = results.copy()
            show["Type"] = show["Model_Output"].map(
                {True: "⚠️ MODEL (excluded)", False: "✅ RAW"})
            st.dataframe(
                show[["Feature", "Type", "Avg Correlation", "Std Dev", "Weeks Sampled"]],
                hide_index=True, use_container_width=True)

            proposed = get_proposed_weights(results)
            if proposed:
                st.subheader("⚖️ Proposed BOOM_WEIGHTS (½-step)")
                st.caption("Reminder: boom_score is a display metric. These "
                           "weights do not feed prob_1plus_td, so applying them "
                           "will not move backtest edge.")
                st.dataframe(pd.DataFrame(proposed).T, use_container_width=True)

            st.subheader("📋 Copy report")
            st.code(pattern_copy_text(results, season_used, proposed), language="text")
        else:
            st.info("Not enough graded data yet.")

with tab_owner:
    if not owner_mode:
        st.markdown("### 🔒 Owner access required")
        st.caption("Enter your key in the sidebar to unlock backtesting.")
    else:
        st.subheader("📈 Backtest")
        st.caption(f"Features from weeks 1..N-1, outcomes from week N. "
                   f"Currently DEF_BLEND={DEF_BLEND} — change it in "
                   f"core/scoring.py and rerun to test the matchup layer.")

        if st.button("Run Full Backtest", type="primary"):
            with st.spinner("Grading 18 weeks..."):
                results, season_used = run_nfl_backtest(
                    season=ACTUAL_SEASON, max_weeks=18)

            if results is not None and not results.empty:
                st.caption(f"Season graded: **{season_used}**")
                st.dataframe(results, hide_index=True, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("TD Edge", f"{results['TD Edge (pp)'].mean():+.1f}pp")
                with c2:
                    st.metric("Yds Edge", f"{results['Yds Edge (pp)'].mean():+.1f}pp")
                with c3:
                    st.metric("Brier (TD)", f"{results['Avg Brier (TD)'].mean():.4f}")

                st.divider()
                st.subheader("📋 Copy report")
                st.code(generate_nfl_backtest_copy_text(results, season=season_used),
                        language="text")
            else:
                st.error("Backtest produced no gradable weeks.")
