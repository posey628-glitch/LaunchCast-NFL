# ui/render.py
# LaunchCast NFL — UI with Dynamic Column Handling

import streamlit as st
import pandas as pd

def get_matchup_grade(prob, position):
    """Get matchup grade based on probability."""
    if prob >= 0.65: return "A+"
    if prob >= 0.50: return "A"
    if prob >= 0.40: return "B"
    if prob >= 0.30: return "C"
    return "D"

def get_verdict_emoji(prob):
    """Get verdict emoji based on probability."""
    if prob >= 0.60: return "🔥"
    if prob >= 0.45: return "✅"
    if prob >= 0.30: return "🟡"
    return "️"

def render_prop_leaderboard(projections_df, prop_type='td'):
    """Render prop leaderboard with dynamic column handling."""
    if projections_df.empty:
        st.info("Waiting for projections...")
        return
    
    # Filter to skill positions
    df = projections_df[projections_df['position'].isin(['WR', 'RB', 'TE'])].copy()
    
    # Sort by probability
    if prop_type == 'td':
        sort_col = 'prob_1plus_td'
        stat_col = 'proj_tds'
        label = "P(1+ TD)"
    elif prop_type == 'yards':
        sort_col = 'prob_over_45.5_yds'
        stat_col = 'proj_rec_yards'
        label = "P(Ov 45.5 Yds)"
    else:
        sort_col = 'prob_over_3.5_rec'
        stat_col = 'proj_targets'
        label = "P(Ov 3.5 Rec)"
    
    df = df.sort_values(sort_col, ascending=False)
    
    # Add visual columns
    df['Grade'] = df.apply(lambda r: get_matchup_grade(r[sort_col], r['position']), axis=1)
    df['Verdict'] = df[sort_col].apply(get_verdict_emoji)
    
    # Display columns - only include columns that exist
    display_cols = ['Verdict', 'player_name', 'position', 'team', 'opponent_team', 'Grade', stat_col, sort_col]
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Column config
    col_config = {
        "Verdict": st.column_config.TextColumn("", width="small"),
        "player_name": st.column_config.TextColumn("PLAYER", width="large"),
        "position": st.column_config.TextColumn("POS", width="small"),
        "team": st.column_config.TextColumn("TM", width="small"),
        "opponent_team": st.column_config.TextColumn("VS", width="small"),
        "Grade": st.column_config.TextColumn("GRADE", width="small"),
        stat_col: st.column_config.NumberColumn("PROJ", format="%.1f", width="small"),
        sort_col: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.0f%%", width="medium"),
    }
    
    # Only use config for columns that exist
    col_config = {k: v for k, v in col_config.items() if k in display_cols}
    
    st.dataframe(
        df[display_cols],
        column_config=col_config,
        hide_index=True,
        use_container_width=True
    )

def render_nfl_dashboard(schedule, rosters, projections, is_offseason=False, display_year=2024):
    """Main dashboard."""
    st.title("🏈 LaunchCast NFL")
    
    if is_offseason:
        st.caption(f"Evidence-based prop projections ({display_year} season data for testing). Live 2026 season starts September.")
    else:
        st.caption("Evidence-based prop projections powered by nflverse & Bayesian shrinkage.")
    
    # Tabs
    tab_td, tab_yards, tab_rec = st.tabs([" Touchdowns", "📏 Receiving Yards", "🎯 Receptions"])
    
    with tab_td:
        st.subheader("Anytime Touchdown Leaderboard")
        render_prop_leaderboard(projections, prop_type='td')
    
    with tab_yards:
        st.subheader("Receiving Yards Overs")
        st.caption("Probabilities based on a standard 45.5 yard line.")
        render_prop_leaderboard(projections, prop_type='yards')
    
    with tab_rec:
        st.subheader("Reception Overs")
        st.caption("Probabilities based on a standard 3.5 reception line.")
        render_prop_leaderboard(projections, prop_type='rec')
