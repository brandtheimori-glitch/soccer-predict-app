# Soccer Predictions — prototype

A small Flask app that predicts match outcomes with a transparent Poisson
goal model blended with betting-market odds, and — unlike most prediction
apps on the market — publicly logs and grades every prediction it makes.

## Why it's built this way

A quick survey of existing football prediction apps turned up a consistent
pattern:

- **Stat-heavy tools** (SofaScore, FotMob, WhoScored) show great raw data —
  heatmaps, xG timelines, player ratings — but that's a lot to parse before
  you get a straight answer.
- **Table-based statistical sites** (Forebet) give probabilities and
  correct-score forecasts, but not much else — no live scores, no context.
- **"AI" prediction apps** (FootballPredictAI, PredictNext, Rithmm, Dimers,
  and similar) market themselves on unaudited accuracy claims (75%, 87%,
  etc.) without naming their method or publishing a checkable record of
  right vs. wrong picks. A couple of newer entrants (ScoreGPT, SignalOdds)
  do better by naming their models and grading every pick — that
  transparency principle is the main thing this prototype borrows.

So the design goals here are:

1. **Show the work.** Every prediction has a "why?" panel with the actual
   inputs (attack/defense strength vs. league average, recent form, model
   vs. market probabilities) — never just a bare percentage.
2. **Grade yourself in public.** Every prediction is logged before the
   match, then automatically graded against the real result. The running
   accuracy is shown on the page, with the full log visible, not just a
   summary number.
3. **One quick answer, details on demand.** The default view is a single
   pick + score + confidence per match (fast to scan, addressing the
   stat-tool problem above); the underlying stats are one click away
   (addressing the black-box "AI" problem).
4. **Free-tier friendly.** API-Football's free plan (100 requests/day) is
   used instead of football-data.org's, whose free tier only exposes a
   competition list, not actual match data. Live responses are cached for
   6 hours so normal browsing doesn't burn through the daily quota.
5. **Find the actual value.** Beyond "who wins," each match is checked for
   value bets — cases where the model's own probability for an outcome
   sits meaningfully above what the market's odds imply. That's the
   professional framing (a mispriced outcome), not just a pick.

## How the model works

`model.py` implements a standard approach for goal-based sports prediction
(the same family as Maher 1982 / Dixon-Coles):

1. Each team gets an **attack strength** (their average goals scored ÷ the
   league's overall average goals per team per game) and **defense
   strength** (average goals conceded ÷ that same league average).
2. Multiplying those together, plus a single league-wide **home-advantage**
   multiplier applied only to the home side, gives an **expected goals
   (xG)** figure for each side in a specific fixture.
3. Treating each side's goals as independent Poisson-distributed variables
   around those xG values produces a full score matrix (0-0, 1-0, 0-1, …),
   which is summed to get win/draw/loss probabilities, over/under, BTTS,
   and the most likely scorelines.
4. Where bookmaker odds are available, their de-vigged implied
   probabilities are blended in (65% model / 35% market by default) —
   markets price in information (injuries, suspensions, weather) a simple
   stats model can't see, so blending tends to be better calibrated than
   either signal alone.
5. Separately, the **model-only** probability (before blending) is compared
   to the market's implied probability for the same outcome. If the model
   thinks an outcome is at least 5 points more likely than the market's
   price implies — and the expected value of betting it at that price is
   positive — it's flagged as a value bet in the UI, with the actual odds,
   both probabilities, and the expected value shown.

This is a solid, well-understood baseline — it's what most serious
prediction systems use as a foundation before adding machine learning on
top. It is *not* state-of-the-art (a production system would add player-
level data, fatigue/travel, and a model trained on historical outcomes),
but it's honest about exactly what's driving each number, which is the
main gap identified in the research.

## Fitting home advantage and team strength from real results

The first version of this hand-picked a home-advantage constant (1.35, "it
felt about right") and applied it on top of a league-wide home/away goal
split that was *already* baking in a home-scoring boost — double-counting
the effect. `fit_parameters.py` replaces both with numbers estimated from
actual match results:

- **One pooled league average** (goals per team per game, home and away
  combined) instead of separate home/away averages.
- **Home advantage** fitted as the ratio of average home-team goals to that
  pooled average, from real results — not assumed.
- **Each team's attack/defense** computed the same way, from goals scored
  and conceded across every match in the fit sample (home and away
  together), relative to the pooled average.

Free-tier data means small samples — a demo league here has 14-18
historical matches to fit from, and even a live API-Football key's free
plan limits how many past seasons you can pull. Fitting a ratio from a
handful of matches is noisy (one freak scoreline can badly skew a team's
rating), so every fitted number is **shrunk toward a neutral prior** in
proportion to how little data supports it: teams with few games lean
toward league-average attack/defense (1.00), and the league's home
advantage leans toward a documented typical value (1.30) until enough
matches accumulate to trust the fitted number over the prior. The "why?"
panel shows both the raw and shrunk estimate, plus the sample size, so it's
never hiding how much to trust a given number.

A side effect worth knowing: fitting from raw match results this way meant
dropping the per-team `/teams/statistics` API-Football call entirely —
`historical_matches()` pulls one batch of recent results per league, which
covers every team's fit input from that single call. That's fewer requests
against the 100/day free-tier cap than the previous per-team-lookup design.

### Recency weighting

A match from three months ago says less about a team's CURRENT strength
than one from last week — squads change, form swings, managers get sacked.
Every historical match is now weighted by an exponential decay based on
its age: `weight = 0.5 ** (days_ago / 100)`, so a match 100 days old counts
half as much as one today, 200 days old a quarter, and so on. The 100-day
half-life isn't arbitrary — it's the same ballpark Dixon & Coles (1997)
fitted from real results (their xi=0.0065/day implies ~107 days).

This changes what "sample size" means for the shrinkage described above:
it's now an EFFECTIVE sample size (the sum of each match's weight), not a
raw count. 18 matches spread over two seasons carry less real information
than 18 from the last two months, and the fit now reflects that — the
"why?" panel shows both numbers side by side (e.g. "18 historical matches,
14.2 effective after recency weighting").

## Checking whether value bets actually work, and tuning the threshold

Flagging a value bet is a claim ("this price is wrong"), and claims should
get checked. `track_record.value_bet_performance()` (surfaced as its own
section in the UI and at `/api/value-bet-performance`) does that:

1. Every value bet is logged at prediction time with its odds and both
   probabilities — never edited after the fact. Alongside it, the model
   also logs the model-vs-market comparison for ALL THREE outcomes of
   every match, not just the ones that cleared the threshold at the time
   (see `model.compute_all_edges()`) — that full record is what makes the
   threshold sweep below possible.
2. Once a match finishes, each flagged bet (and every logged edge) is
   graded won/lost, with the unit return (odds − 1 if won, −1 if lost)
   recorded alongside it.
3. The running total gives a win rate and ROI, compared against the
   **break-even win rate the odds themselves implied** for those specific
   bets (average of 1/odds) — that's the honest baseline, not 0%.
4. Because a handful of results is mostly noise, nothing is called
   "working" below 20 graded value bets, and past that threshold a simple
   one-proportion z-test checks whether the win rate is actually
   distinguishable from break-even before the UI says so. It's a rough
   gut-check (it assumes bets are independent, which a real season only
   approximately satisfies), not a rigorous backtest — but it's a real
   check instead of an assumed "the edge must be real because I flagged it."

The bundled sample data seeds a small number of already-graded bets so
this section isn't empty on a first run; with that little data it
correctly reports "not enough yet" rather than asserting anything.

### Auto-tuning the edge threshold

`threshold_tuning.py` uses that same full edge record to test whether a
different threshold than the flat 5-point default would actually perform
better — sweeping a fine grid, every point from 2pt to 20pt (19
candidates), against every graded edge in history, not just the ones
flagged at whatever threshold was active when they were made. For each
candidate it computes the same win rate / ROI / significance check as the
live backtest above.

Testing 19 candidates against the same data raises a real statistical
issue on its own: a plain 95%-confidence check (z >= 1.96) on each of 19
candidates will likely find at least one "significant" result by pure
chance, the same way 19 coin flips against a fair-coin null usually turn
up an outlier. So the significance bar is **Bonferroni-corrected** for the
number of candidates tested — the per-candidate significance level is
0.05 / 19 instead of a flat 0.05, which raises the required z-score from
1.96 to roughly 3.0. More candidates tested means a stricter bar to call
any one of them significant, which is the correct trade for a finer grid,
not an afterthought.

**A candidate only ever gets recommended once it clears the same bar the
live backtest holds itself to, at this corrected level** — at least 20
graded bets at that threshold AND a Bonferroni-significant edge over
break-even. Among candidates that clear both, the one with the best ROI
wins. If nothing clears both, `active_threshold()` keeps returning the
documented 5-point default rather than picking whichever candidate looks
best on a handful of noisy results — the same discipline as the rest of
this app, applied to tuning itself instead of just to reporting. The
reasoning behind whichever threshold is currently active is shown directly
in the UI and at `/api/threshold-tuning`, e.g. "5pt (default — no
candidate has both 20+ graded bets and a significant edge once corrected
for testing 19 candidates at once)."

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 — it runs immediately on bundled sample
fixtures across five leagues (Premier League, La Liga, Serie A, Bundesliga,
Ligue 1 — clearly labeled as sample data), no API key required. Use the
League dropdown at the top to switch between them.

### Going live with real data

1. Get a free key at https://dashboard.api-football.com (no card needed,
   100 requests/day).
2. Run with the key set:
   ```bash
   API_FOOTBALL_KEY=your_key_here python app.py
   ```
3. The league picker will then list API-Football's numeric IDs for the
   same five leagues (see `_LIVE_LEAGUES` in `data_source.py` to add more —
   find IDs at https://www.api-football.com/documentation-v3#tag/Leagues).

This is a separate decision from hosting it publicly (next section) — you
can set `API_FOOTBALL_KEY` locally, on a free host, or not at all.

## Deploying it for free

The project is deployment-ready: `app.py` binds to `0.0.0.0` and reads
`$PORT` (what every free host expects), `Procfile` and `requirements.txt`
(gunicorn included) tell a host how to run it, and a local git repo is
already initialized with everything committed except generated files
(`.gitignore` excludes `cache/`, `predictions_log.json`, `__pycache__/`).
Actually creating a live URL needs YOUR accounts (GitHub, and a host) —
that's the one part of this nobody but you can do. Render is the
recommendation below because its free web-service tier needs no credit
card and Flask deploys on it with zero configuration beyond what's already
in this repo; Railway and Fly.io are free-tier alternatives if you'd
rather use one of those (same general steps: connect a repo, point it at
`gunicorn app:app`).

**Steps (Render):**

1. Push this repo to GitHub: create an empty repo on github.com, then from
   this project folder:
   ```bash
   git remote add origin https://github.com/<you>/soccer-predictions.git
   git branch -M main
   git push -u origin main
   ```
2. At https://render.com, sign up free and choose **New > Blueprint**,
   point it at the GitHub repo — it reads `render.yaml` and configures the
   free web service automatically (build command, start command, plan).
   No blueprint? **New > Web Service** works too: build command
   `pip install -r requirements.txt`, start command `gunicorn app:app`.
3. Once deployed, Render gives you a public `https://<name>.onrender.com`
   URL. Optionally add `API_FOOTBALL_KEY` under the service's Environment
   tab to go live with real data instead of the bundled sample.

**Two free-tier things worth knowing before you rely on this:**

- **Cold starts.** Render's free tier spins a service down after periods
  of no traffic and takes 30-60 seconds to wake back up on the next
  request — expected on a free plan, not a bug.
- **Ephemeral disk.** `predictions_log.json` and `cache/` live on local
  disk, which free tiers do NOT persist across restarts or redeploys — the
  track record and value-bet backtest will reset each time the service
  restarts. Fine for demoing; if you want the track record to actually
  accumulate over a real season, the fix is swapping `track_record.py`'s
  flat-file storage for a small free-tier database (Render's free Postgres
  expires after 90 days; a persistent external store like Supabase's free
  tier doesn't) — a bigger change than the free hosting step itself, worth
  doing only once you're confident this is worth running long-term.

## Project layout

```
app.py               Flask routes, wires data_source -> fit_parameters -> threshold_tuning -> model -> track_record
model.py             Poisson goal model + market-odds blending + full edge computation + value-bet detection
fit_parameters.py    Fits home advantage + team attack/defense from historical results, recency-weighted, with shrinkage
threshold_tuning.py  Sweeps candidate value-bet edge thresholds (Bonferroni-corrected) against graded history
data_source.py       SampleDataSource (demo, 5 leagues) and LiveDataSource (API-Football, cached)
track_record.py      Logs every prediction and every model-vs-market edge, grades them, backtests value-bet ROI
templates/, static/   Front end
Procfile, render.yaml, .gitignore   Deployment config for a free host (Render)
predictions_log.json  Created at runtime — the public prediction/result/edge log (gitignored, not in the repo)
cache/               Created at runtime — cached API-Football responses (gitignored, not in the repo)
```

## Next steps worth considering

- The fit still treats every team's attack/defense as static across the
  (recency-weighted) sample rather than modeling within-team home/away
  splits separately — worth adding once there's enough effective data per
  team to support the extra parameters without overfitting.
- Add a date range picker and even more leagues (the live side already
  supports any API-Football league ID — just add it to `_LIVE_LEAGUES` and
  mirror the sample data structure in `_LEAGUES`).
- Swap `track_record.py`'s flat-file storage for a small database (see the
  "Ephemeral disk" note above) if this goes live somewhere the track record
  needs to survive restarts — that's the real prerequisite for the
  value-bet backtest to accumulate enough graded history to say anything
  meaningful, more than any further modeling work.
- If this becomes a real product, budget for a paid data tier — 100
  requests/day covers a personal prototype but not multiple users browsing
  many leagues.
