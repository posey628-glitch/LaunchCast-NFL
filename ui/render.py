# ui/render.py
# LaunchCast NFL — UI with Advanced Metrics Display

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
    return "⚠️"

def render_prop_leaderboard(projections_df, prop_type='td'):
    """Render prop leaderboard with advanced metrics."""
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
        stat_col = 'proj_receptions'
        label = "P(Ov 3.5 Rec)"
    
    df = df.sort_values(sort_col, ascending=False)
    
    # Add visual columns
    df['Grade'] = df.apply(lambda r: get_matchup_grade(r[sort_col], r['position']), axis=1)
    df['Verdict'] = df[sort_col].apply(get_verdict_emoji)
    
    # Display columns
    display_cols = ['Verdict', 'player_name', 'position', 'team', 'opponent_team', 
                    'Grade', stat_col, sort_col]
    
    # Column config
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

def render_advanced_metrics_detail(projections_df):
    """Render detailed view with all advanced metrics."""
    if projections_df.empty:
        return
    
    st.subheader("🔬 Advanced Metrics Detail")
    st.caption("Deep dive into the metrics driving projections")
    
    # Filter to skill positions
    df = projections_df[projections_df['position'].isin(['WR', 'RB', 'TE'])].copy()
    df = df.sort_values('prob_1plus_td', ascending=False)
    
    # Select columns to display
    advanced_cols = [
        'player_name', 'position', 'team',
        'proj_targets', 'proj_receptions', 'proj_rec_yards', 'proj_tds',
        'pocket_efficiency', 'pocket_scenario',
        'tempo_factor', 'script_factor', 'rz_factor',
        'slot_factor', 'separation_factor', 'contested_factor', 'drop_penalty',
        'first_read_share_display', 'yac_per_rec_display', 'ay_per_target_display'
    ]
    
    # Filter to only columns that exist
    available_cols = [c for c in advanced_cols if c in df.columns]
    
    # Column config
    col_config = {
        "player_name": st.column_config.TextColumn("PLAYER", width="large"),
        "position": st.column_config.TextColumn("POS", width="small"),
        "team": st.column_config.TextColumn("TM", width="small"),
        "proj_targets": st.column_config.NumberColumn("Targets", format="%.1f"),
        "proj_receptions": st.column_config.NumberColumn("Rec", format="%.1f"),
        "proj_rec_yards": st.column_config.NumberColumn("Yards", format="%.0f"),
        "proj_tds": st.column_config.NumberColumn("TDs", format="%.2f"),
        "pocket_efficiency": st.column_config.NumberColumn("Pocket Eff", format="%.0f"),
        "pocket_scenario": st.column_config.TextColumn("Pocket", width="small"),
        "tempo_factor": st.column_config.NumberColumn("Tempo", format="%.2f"),
        "script_factor": st.column_config.NumberColumn("Script", format="%.2f"),
        "rz_factor": st.column_config.NumberColumn("RZ", format="%.2f"),
        "slot_factor": st.column_config.NumberColumn("Slot", format="%.2f"),
        "separation_factor": st.column_config.NumberColumn("Sep", format="%.2f"),
        "contested_factor": st.column_config.NumberColumn("Contest", format="%.2f"),
        "drop_penalty": st.column_config.NumberColumn("Drop", format="%.2f"),
        "first_read_share_display": st.column_config.NumberColumn("1st Read", format="%.0%"),
        "yac_per_rec_display": st.column_config.NumberColumn("YAC/Rec", format="%.1f"),
        "ay_per_target_display": st.column_config.NumberColumn("AY/Tgt", format="%.1f"),
    }
    
    # Filter config to available columns
    available_config = {k: v for k, v in col_config.items() if k in available_cols}
    
    st.dataframe(
        df[available_cols].head(50),
        column_config=available_config,
        hide_index=True,
        use_container_width=True
    )

def render_nfl_dashboard(schedule, rosters, projections, is_offseason=False, display_year=2024):
    """Main dashboard with tabs for different views."""
    st.title("🏈 LaunchCast NFL")
    
    if is_offseason:
        st.caption(f"Evidence-based prop projections ({display_year} season data for testing). Live 2026 season starts September.")
    else:
        st.caption("Evidence-based prop projections powered by nflverse & Bayesian shrinkage.")
    
    # Tabs
    tab_props, tab_advanced, tab_detail = st.tabs([
        "🎯 Prop Leaderboards",
        "📊 Advanced Metrics",
        "🔬 Full Detail"
    ])
    
    with tab_props:
        st.subheader("Prop Leaderboards")
        
        # TD props
        st.markdown("### 🎯 Touchdown Props")
        render_prop_leaderboard(projections, prop_type='td')
        
        st.markdown("---")
        
        # Yardage props
        st.markdown("### 📏 Receiving Yards Props")
        render_prop_leaderboard(projections, prop_type='yards')
        
        st.markdown("---")
        
        # Reception props
        st.markdown("### 🎯 Reception Props")
        render_prop_leaderboard(projections, prop_type='rec')
    
    with tab_advanced:
        st.subheader("Advanced Metrics Overview")
        st.caption("Key factors driving projections")
        
        if not projections.empty:
            # Pocket efficiency distribution
            if 'pocket_efficiency' in projections.columns:
                st.markdown("### 🎯 Pocket Efficiency Distribution")
                st.histogram(projections['pocket_efficiency'], bins=20)
                
                # Scenario breakdown
                if 'pocket_scenario' in projections.columns:
                    st.markdown("### Pocket Scenarios")
                    scenario_counts = projections['pocket_scenario'].value_counts()
                    st.bar_chart(scenario_counts)
            
            # Tempo factor distribution
            if 'tempo_factor' in projections.columns:
                st.markdown("### ⚡ Tempo Factor Distribution")
                st.histogram(projections['tempo_factor'], bins=20)
            
            # Red zone factor distribution
            if 'rz_factor' in projections.columns:
                st.markdown("### 🎯 Red Zone Factor Distribution")
                st.histogram(projections['rz_factor'], bins=20)
    
    with tab_detail:
        st.subheader("Full Detail View")
        render_advanced_metrics_detail(projections)
