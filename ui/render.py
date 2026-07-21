# ui/render.py
# LaunchCast NFL — The User Interface
# Focuses on scannability, prop-specific tabs, and hiding complex math behind simple grades.

import streamlit as st
import pandas as pd

# ============================================================================
# 1. HELPER FUNCTIONS (Formatting & Grading)
# ============================================================================
def get_matchup_grade(opp_epa, pressure_rate, position):
    """
    Translates complex defensive EPA/Pressure stats into a simple A-F grade.
    This is the "Kasper-style" scannability win.
    """
    if position in ['WR', 'TE']:
        # Higher EPA allowed = worse pass defense = better grade for WR
        if opp_epa > 0.05: return "A+"
        if opp_epa > 0.00: return "A"
        if opp_epa > -0.05: return "B"
        if opp_epa > -0.10: return "C"
        return "D"
    elif position == 'RB':
        # For RBs, we'd use rush EPA. For V1, we'll just return neutral.
        return "B"
    return "—"

def get_verdict_emoji(prob):
    """Returns an emoji based on the probability threshold."""
    if prob >= 0.60: return "🔥" # High confidence
    if prob >= 0.45: return "✅" # Solid play
    if prob >= 0.30: return "" # Lean
    return "⚠️" # Pass

# ============================================================================
# 2. THE PROPS LEADERBOARD (The Core View)
# ============================================================================
def render_prop_leaderboard(projections_df, prop_type='td'):
    """
    Renders the main scannable table for a specific prop (TDs, Yards, or Receptions).
    """
    if projections_df.empty:
        st.info("Waiting for projections...")
        return

    # Filter out QBs for V1 (we are focusing on skill positions)
    df = projections_df[projections_df['position'].isin(['WR', 'RB', 'TE'])].copy()

    # Sort by the selected probability
    if prop_type == 'td':
        sort_col = 'prob_1plus_td'
        stat_col = 'proj_tds'
        label = "P(1+ TD)"
    elif prop_type == 'yards':
        sort_col = 'prob_over_45.5_yds' # Example line
        stat_col = 'proj_rec_yards'
        label = "P(Ov 45.5 Yds)"
    else:
        sort_col = 'prob_over_3.5_rec'
        stat_col = 'proj_targets'
        label = "P(Ov 3.5 Rec)"

    df = df.sort_values(sort_col, ascending=False)

    # Add visual columns
    df['Grade'] = df.apply(lambda r: get_matchup_grade(r.get('opp_pass_epa_allowed', 0), r.get('opp_pressure_rate', 0.25), r['position']), axis=1)
    df['Verdict'] = df[sort_col].apply(get_verdict_emoji)

    # Select only the columns we want to show (The "Decision View")
    display_cols = ['Verdict', 'player_name', 'position', 'team', 'opponent_team', 'Grade', stat_col, sort_col]
    
    # Rename columns for the UI
    col_config = {
        "Verdict": st.column_config.TextColumn("", width="small"),
        "player_name": st.column_config.TextColumn("PLAYER", width="large"),
        "position": st.column_config.TextColumn("POS", width="small"),
        "team": st.column_config.TextColumn("TM", width="small"),
        "opponent_team": st.column_config.TextColumn("VS", width="small"),
        "Grade": st.column_config.TextColumn("MATCHUP", width="small"),
        stat_col: st.column_config.NumberColumn("PROJ", format="%.1f", width="small"),
        sort_col: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.0f%%", width="medium"),
    }

    st.dataframe(
        df[display_cols], 
        column_config=col_config, 
        hide_index=True, 
        use_container_width=True
    )

# ============================================================================
# 3. THE MAIN DASHBOARD ORCHESTRATOR
# ============================================================================
def render_nfl_dashboard(schedule, rosters, projections, is_offseason=False, display_year=2025):
    """
    The main Streamlit layout. Uses tabs to separate props, keeping the UI clean.
    """
    st.title("🏈 LaunchCast NFL")
    
    if is_offseason:
        st.caption(f"Evidence-based prop projections (2025 season data for testing). Live 2026 season starts September.")
        st.info("📅 **Offseason Mode:** Showing historical 2025 data. Switch weeks in the sidebar to test different matchups.")
    else:
        st.caption("Evidence-based prop projections powered by nflverse & Bayesian shrinkage.")

    # Top-level Prop Tabs
    tab_td, tab_yards, tab_rec = st.tabs(["🎯 Touchdowns", "📏 Receiving Yards", "🎯 Receptions"])

    with tab_td:
        st.subheader("Anytime Touchdown Leaderboard")
        render_prop_leaderboard(projections, prop_type='td')

    with tab_yards:
        st.subheader("Receiving Yards Overs")
        st.caption("Probabilities based on a standard 45.5 yard line. Adjust in settings later.")
        render_prop_leaderboard(projections, prop_type='yards')

    with tab_rec:
        st.subheader("Reception Overs")
        st.caption("Probabilities based on a standard 3.5 reception line.")
        render_prop_leaderboard(projections, prop_type='rec')

    # Sidebar for Data Status
    with st.sidebar:
        st.header("📊 Data Status")
        st.metric("Season", display_year)
        st.metric("Players Tracked", len(projections) if projections is not None else 0)
        st.divider()
        if is_offseason:
            st.info("V1 Engine: Uses nflverse data, EPA matchups, and week-weighted shrinkage.\n\n**Offseason testing mode active.**")
        else:
            st.info("V1 Engine: Uses nflverse data, EPA matchups, and week-weighted shrinkage.")
