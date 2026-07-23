# LaunchCast NFL

Evidence-based NFL prop projections — powered by nflverse data, Bayesian
shrinkage, and a backtest that grades itself honestly.

**Validated on the 2024 season (18 weeks):**

| Prop | Top-20 hit rate | Slate base rate | Edge | Lift |
|---|---|---|---|---|
| Anytime TD | 30.6% | 13.9% | **+16.7pp** | 2.19× |
| Receiving yards (o45.5) | 66.4% | 19.9% | **+46.5pp** | 3.35× |

Calibration: predicted 14.5% vs actual 13.9% on touchdowns (0.6pp gap).

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Owner mode** unlocks backtesting and diagnostics. The key lives in
`.streamlit/secrets.toml` locally, or *Manage app → Settings → Secrets* on
Streamlit Cloud — **never in source**:

```toml
owner_key = "your-key-here"
```

`secrets.toml` is gitignored. If the secret is missing, `OWNER_KEY` is empty
and the `and OWNER_KEY` guard keeps owner mode locked rather than letting an
empty password through.

---

## Layout

```
app.py                  entry point, tabs, owner gate
core/scoring.py         projections, shrinkage, probabilities  ← the tunable knobs
core/backtest.py        week-by-week grading, edge metrics
core/patterns.py        feature correlations, weight proposals
data/fetcher.py         nflverse loading, feature assembly     ← the leakage guards
ui/render.py            leaderboards, deep dives, game browser
```

---

## Four invariants — do not break these

These cost several rounds of debugging to establish. Each one has a failure
mode that is silent, which is why they're written down.

### 1. Features for week N come only from weeks 1..N-1

Every consumer calls `assemble_week()` in `data/fetcher.py`. The first version
of this app built projections from the *same week* it was grading, which
produced a Brier of 0.004 and looked like genius. If you add a second assembly
path, the live app and the backtest will drift apart and you won't notice.

**Smell test:** anytime-TD Brier should land around **0.11–0.13**. If you see
0.00-something, you are grading the answer key.

### 2. Read EDGE, not Brier

About 86% of tracked players don't score in a given week, so Brier is dominated
by easy negatives. A predictor that guesses the base rate for *everyone* scores
about **0.120** — barely worse than a real model. Brier tells you the
probabilities are believable; **edge** (top-20 hit rate vs the slate base rate)
tells you whether the tool beats guessing. The backtest report prints the
flat-guess reference next to your Brier so the number can't flatter itself.

With 20 picks per week, one standard error is ~10pp. **Judge the 18-week
average, never a single week.**

### 3. `BOOM_WEIGHTS` does not affect the predictions

```
shrunk_target_share → proj_targets → proj_tds → prob_1plus_td → EDGE
boom_score          → td_spike flag (display only)
```

`boom_score` is not in the prediction chain. Reweighting `BOOM_WEIGHTS`
produces a byte-identical backtest — this was confirmed empirically. The knobs
that actually move results live at the top of `core/scoring.py`:

- **`DEF_BLEND`** — how much opponent defense enters the TD rate. Evidence says
  `def_td_per_tgt` correlates **+0.003** with scoring (nothing), so 40% of the
  rate is currently being diluted by noise. Try 0.2 and 0.0; keep whichever
  wins on edge.
- **shrinkage priors** — raw `target_share` (+0.283) still outperforms
  `shrunk_target_share` (+0.228), so the priors may still be too strong. Tune
  until shrunk beats raw.

### 4. Never let the prior season resolve to the graded season

`nfl_data_py` falls back to earlier years when a season isn't published yet.
If "current" and "prior" both resolve to the same year, the week-1 prior is a
full-season aggregate that *includes* the outcome being predicted.
`safe_prior_rates()` refuses priors in that case and says so. When this leak
was live, Week 1's edge read +19.7pp; with it fixed, the honest number is
**+5.0pp**.

`resolve_season()` returns the season that *actually* loaded — always report
that, never the one that was requested.

---

## How the model works

**Volume drives everything.** Target share correlates **+0.283** with scoring;
per-target efficiency correlates ~+0.03. Receivers score because they get the
ball a lot, not because they're efficient with it.

1. **Target share** — season-to-date, shrunk toward the league mean (~0.11,
   not 0.20: a team spreads targets across 9–10 players), then **renormalized
   so each team's shares sum to 1.0**. Without renormalization the model
   invents ~25 phantom targets per team per week.
2. **Volume** — share × the team's own season-to-date targets per game.
3. **TD rate** — per-target rate shrunk with a strong prior (90 targets; TD
   rate stabilizes slowly), optionally blended with opponent defense.
4. **Probabilities** — Poisson for touchdowns, Normal for yardage.

**Known limitation:** receiving yards are right-skewed with a large zero-mass,
so the Normal over-predicts the middle of the distribution — average predicted
27.3% vs 19.9% actual, while the top-20 hits 66.4%. The *ranking* is excellent;
the raw probabilities run hot. If you need book-comparable numbers, fit an
isotonic calibration on weeks 1..N-1 **inside** the backtest loop — never from
a cached model, which leaks across weeks.

---

## Pattern analysis

Correlates every tracked feature against actual outcomes, week by week.

- **Model outputs are flagged and excluded from weight evidence.** A score
  built from the features cannot be cited as evidence for those features.
  `boom_score` tops the correlation table at +0.296 precisely because it's made
  of the things below it.
- **Raw and processed versions are tracked side by side**, so "is our shrinkage
  helping?" is answered by data rather than argument. That comparison is how we
  discovered the shrinkage was destroying 25% of target share's signal.
- Weight proposals use **evidence-proportional targets with a ½-step**, and
  negatively-correlated features earn **zero** weight rather than a large
  positive one (`shrunk_yds_per_tgt` at −0.105 is an anti-predictor).

---

## Open work

- **Run the `DEF_BLEND` ablation** (0.4 → 0.2 → 0.0). Highest-value experiment
  available.
- **Validate on 2025** once nflverse publishes it. The model was tuned on 2024,
  so 2025 is a clean holdout — run it *once*, untouched. Edge near +15pp means
  it generalizes; a collapse means overfitting.
- **Red-zone data.** Season-long defense-allowed rates measure nothing
  (+0.003). Red-zone target share and red-zone defense are the features that
  should predict touchdowns, and neither is in the model yet.
- **Durable snapshots.** The backtest recomputes from nflverse each run, which
  is fine for history — but in-season you'll want a record of what you actually
  *showed* on Sunday morning, before inactives moved things.
