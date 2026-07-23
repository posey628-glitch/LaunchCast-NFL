# ui/render.py
# LaunchCast NFL — UI rendering

import streamlit as st
import pandas as pd

PROPS = {
    "td": ("prob_1plus_td", "proj_tds", "P(1+ TD)"),
    "yards": ("prob_over_45.5_yds", "proj_rec_yards", "P(Ov 45.5 Yds)"),
    "rec": ("prob_over_3.5_rec", "proj_targets", "P(Ov 3.5 Rec)"),
}


def get_matchup_grade(prob):
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


def render_prop_leaderboard(projections_df, prop_type="td"):
    if projections_df is None or projections_df.empty:
        st.info("Waiting for projections...")
        return

    df = projections_df.copy()
    if "position" in df.columns:
        df = df[df["position"].isin(["WR", "RB", "TE"])]
    if df.empty:
        st.info("No skill-position players in this view.")
        return

    sort_col, stat_col, label = PROPS.get(prop_type, PROPS["td"])
    if sort_col not in df.columns:
        st.info("Projection column unavailable.")
        return

    df = df.sort_values(sort_col, ascending=False)
    df["Grade"] = df[sort_col].apply(get_matchup_grade)
    df["Verdict"] = df.apply(
        lambda r: get_verdict_emoji(r[sort_col], bool(r.get("td_spike", False))),
        axis=1)

    cols = ["Verdict", "player_name", "position", "team", "opponent_team",
            "Grade", stat_col, sort_col]
    if "ctx_lift_pp" in df.columns:
        cols.append("ctx_lift_pp")
    if "boom_score" in df.columns:
        cols.append("boom_score")
    cols = [c for c in cols if c in df.columns]

    cfg = {
        "Verdict": st.column_config.TextColumn("", width="small"),
        "player_name": st.column_config.TextColumn("PLAYER", width="large"),
        "position": st.column_config.TextColumn("POS", width="small"),
        "team": st.column_config.TextColumn("TM", width="small"),
        "opponent_team": st.column_config.TextColumn("VS", width="small"),
        "Grade": st.column_config.TextColumn("GRADE", width="small"),
        stat_col: st.column_config.NumberColumn("PROJ", format="%.1f", width="small"),
        sort_col: st.column_config.ProgressColumn(
            label, min_value=0.0, max_value=1.0, format="%.0f%%", width="medium"),
        "ctx_lift_pp": st.column_config.NumberColumn(
            "MATCHUP", format="%+.1f",
            help="Percentage points this week's defense moves him off his own "
                 "baseline. Positive = favourable spot."),
        "boom_score": st.column_config.NumberColumn(
            "BOOM", format="%.0f",
            help="0-100 volume/efficiency composite. A DISPLAY metric — it "
                 "does not feed the probabilities."),
    }
    st.dataframe(df[cols], column_config={k: v for k, v in cfg.items() if k in cols},
                 hide_index=True, use_container_width=True)


def render_player_deep_dive(player_row):
    if player_row is None or player_row.empty:
        return
    r = player_row.iloc[0]
    st.markdown(f"### 🔬 {r.get('player_name', 'Unknown')} "
                f"({r.get('position', '')} — {r.get('team', '')}) "
                f"vs {r.get('opponent_team', '')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Proj Targets", f"{float(r.get('proj_targets', 0) or 0):.1f}")
        st.metric("Proj Yards", f"{float(r.get('proj_rec_yards', 0) or 0):.0f}")
    with c2:
        st.metric("Proj TDs", f"{float(r.get('proj_tds', 0) or 0):.2f}")
        st.metric("Boom Score", f"{float(r.get('boom_score', 0) or 0):.0f}")
    with c3:
        st.metric("P(1+ TD)", f"{float(r.get('prob_1plus_td', 0) or 0) * 100:.1f}%")
        st.metric("Matchup swing", f"{float(r.get('ctx_lift_pp', 0) or 0):+.1f}pp")

    bits = []
    for lbl, col, fmt in (("Target share", "target_share", "{:.1%}"),
                          ("Yds/target", "shrunk_yds_per_tgt", "{:.1f}"),
                          ("TD/target", "shrunk_td_per_tgt", "{:.3f}"),
                          ("aDOT", "adot", "{:.1f}")):
        v = r.get(col)
        try:
            if v is not None and not pd.isna(v):
                bits.append(f"**{lbl}** {fmt.format(float(v))}")
        except (TypeError, ValueError):
            continue
    if bits:
        st.caption(" · ".join(bits))

    if bool(r.get("td_spike", False)):
        st.success("🔥 **TD SPIKE** — high probability, high target share, "
                   "strong composite.")


def render_game_browser(projections_df):
    st.subheader("🎮 Game-by-Game Browser")
    if projections_df is None or projections_df.empty:
        st.info("No data to browse.")
        return

    teams = sorted(projections_df["team"].dropna().unique())
    if not teams:
        st.info("No teams available.")
        return

    selected = st.selectbox("Team", teams, key="_nfl_team_pick")
    game = projections_df[projections_df["team"] == selected].copy()
    opp = game["opponent_team"].iloc[0] if not game.empty else "TBD"
    st.markdown(f"#### {selected} vs {opp}")

    names = ["— pick a player —"] + sorted(game["player_name"].dropna().unique())
    pick = st.selectbox("Player deep dive", names, key="_nfl_player_pick")
    if pick and pick != "— pick a player —":
        render_player_deep_dive(game[game["player_name"] == pick])
        st.divider()

    render_prop_leaderboard(game, prop_type="td")


def render_nfl_dashboard(projections, is_offseason=False, display_year=2024):
    st.title("🏈 LaunchCast NFL")
    if is_offseason:
        st.caption(f"Evidence-based prop projections — {display_year} season data "
                   f"(offseason testing mode).")
    else:
        st.caption("Evidence-based prop projections powered by nflverse "
                   "with Bayesian shrinkage.")

    t_td, t_yds, t_rec = st.tabs(
        ["🎯 Touchdowns", "📏 Receiving Yards", "🎯 Receptions"])
    with t_td:
        st.subheader("Anytime Touchdown Leaderboard")
        render_prop_leaderboard(projections, "td")
    with t_yds:
        st.subheader("Receiving Yards Overs")
        st.caption("Standard 45.5-yard line. Ranking is strong; the raw "
                   "probabilities run slightly hot (see backtest).")
        render_prop_leaderboard(projections, "yards")
    with t_rec:
        st.subheader("Reception Overs")
        st.caption("Standard 3.5-reception line.")
        render_prop_leaderboard(projections, "rec")
