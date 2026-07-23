# ui/render.py
# LaunchCast NFL — UI V3

import streamlit as st
import pandas as pd

def get_matchup_grade(prob, position):
    if prob >= 0.65: return "A+"
    if prob >= 0.50: return "A"
    if prob >= 0.40: return "B"
    if prob >= 0.30: return "C"
    return "D"

def get_verdict_emoji(prob, is_spike=False):
    if is_spike: return "🔥 SPIKE"
    if prob >= 0.60: return "🔥"
    if prob >= 0.45: return "✅"
    if prob >= 0.30: return "🟡"
    return "⚠️"

def render_prop_leaderboard(projections_df, prop_type='td'):
    if projections_df.empty:
        st.info("Waiting for projections...")
        return
    
    df = projections_df[projections_df['position'].isin(['WR', 'RB', 'TE'])].copy()
    
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
    
    df['Grade'] = df.apply(lambda r: get_matchup_grade(r[sort_col], r['position']), axis=1)
    
    is_spike_col = 'td_spike' in df.columns
    if is_spike_col:
        df['Verdict'] = df.apply(lambda r: get_verdict_emoji(r[sort_col], r.get('td_spike', False)), axis=1)
    else:
        df['Verdict'] = df[sort_col].apply(lambda p: get_verdict_emoji(p, False))
    
    display_cols = ['Verdict', 'player_name', 'position', 'team', 'opponent_team', 'Grade', stat_col, sort_col]
    
    # Add ctx_lift if available
    if 'ctx_lift_pp' in df.columns:
        display_cols.append('ctx_lift_pp')
    
    display_cols = [c for c in display_cols if c in df.columns]
    
    col_config = {
        "Verdict": st.column_config.TextColumn("", width="small"),
        "player_name": st.column_config.TextColumn("PLAYER", width="large"),
        "position": st.column_config.TextColumn("POS", width="small"),
        "team": st.column_config.TextColumn("TM", width="small"),
        "opponent_team": st.column_config.TextColumn("VS", width="small"),
        "Grade": st.column_config.TextColumn("GRADE", width="small"),
        stat_col: st.column_config.NumberColumn("PROJ", format="%.1f", width="small"),
        sort_col: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.0f%%", width="medium"),
        "ctx_lift_pp": st.column_config.NumberColumn("Lift", format="%+.1f", help="Context lift vs player's own baseline"),
    }
    
    col_config = {k: v for k, v in col_config.items() if k in display_cols}
    
    st.dataframe(
        df[display_cols],
        column_config=col_config,
        hide_index=True,
        use_container_width=True
    )

def render_player_deep_dive(player_row):
    if player_row.empty:
        return
        
    row = player_row.iloc[0]
    name = row.get('player_name', 'Unknown')
    team = row.get('team', '')
    opp = row.get('opponent_team', '')
    pos = row.get('position', '')
    
    st.markdown(f"### 🔬 {name} ({pos} - {team}) vs {opp}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Proj Targets", f"{row.get('proj_targets', 0):.1f}")
        st.metric("Proj Yards", f"{row.get('proj_rec_yards', 0):.0f}")
    with col2:
        st.metric("Proj TDs", f"{row.get('proj_tds', 0):.2f}")
        st.metric("Boom Score", f"{row.get('boom_score', 0):.0f}")
    with col3:
        st.metric("P(1+ TD)", f"{row.get('prob_1plus_td', 0)*100:.1f}%")
        st.metric("Ctx Lift", f"{row.get('ctx_lift_pp', 0):+.1f}pp")
        
    if row.get('td_spike', False):
        st.success("🔥 **TD SPIKE DETECTED:** Elite matchup conditions align.")

def render_game_browser(projections_df):
    st.subheader("🎮 Game-by-Game Browser")
    
    if projections_df.empty:
        st.info("No data to browse.")
        return
        
    teams = sorted(projections_df['team'].unique())
    selected_team = st.selectbox("Select Team to View Matchup:", teams)
    
    game_data = projections_df[projections_df['team'] == selected_team].copy()
    opp = game_data['opponent_team'].iloc[0] if not game_data.empty else "TBD"
    
    st.markdown(f"#### {selected_team} vs {opp}")
    
    player_names = game_data['player_name'].tolist()
    selected_player = st.selectbox("Select Player for Deep Dive:", ["None"] + player_names)
    
    if selected_player != "None":
        player_row = game_data[game_data['player_name'] == selected_player]
        render_player_deep_dive(player_row)
        st.divider()
        
    render_prop_leaderboard(game_data, prop_type='td')

def render_nfl_dashboard(projections, is_offseason=False, display_year=2025):
    st.title("🏈 LaunchCast NFL")
    
    if is_offseason:
        st.caption(f"Evidence-based prop projections ({display_year} season data for testing). Live 2026 season starts September.")
    else:
        st.caption("Evidence-based prop projections powered by nflverse & Bayesian shrinkage.")
    
    tab_td, tab_yards, tab_rec = st.tabs(["🎯 Touchdowns", "📏 Receiving Yards", "🎯 Receptions"])
    
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
