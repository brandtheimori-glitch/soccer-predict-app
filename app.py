"""
Flask front end tying the pieces together:
  data_source       -> raw fixtures, finished results, odds (live or sample)
  fit_parameters    -> fit home advantage + team attack/defense from real, recency-weighted results
  threshold_tuning  -> pick the live value-bet edge threshold from graded history (or the default)
  model             -> turn a fit + a fixture into a Poisson + market-blended prediction
  track_record      -> log every prediction (and every model-vs-market edge) made, grade once results land

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000

To go live instead of the bundled sample data, sign up for a free key at
https://dashboard.api-football.com (100 requests/day, no card required)
and run with:
    API_FOOTBALL_KEY=xxxx python app.py
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

import threshold_tuning
import track_record
from data_source import get_data_source
from fit_parameters import fit_league_parameters
from model import predict_match

app = Flask(__name__)
DEFAULT_LEAGUE = "Premier League"  # sample-data label; for LiveDataSource this is an API-Football numeric league id


def resolve_league(ds) -> str:
    """Pick the active league from ?league=, falling back to the first one the
    data source offers (or DEFAULT_LEAGUE if it doesn't advertise a list)."""
    leagues = ds.available_leagues()
    requested = request.args.get("league")
    valid_ids = {l["id"] for l in leagues}
    if requested and requested in valid_ids:
        return requested
    if leagues:
        return leagues[0]["id"]
    return DEFAULT_LEAGUE


def fit_meta_dict(fit) -> dict:
    return {
        "sample_size": fit.sample_size,
        "effective_sample_size": fit.effective_sample_size,
        "half_life_days": fit.half_life_days,
        "low_confidence": fit.low_confidence,
        "home_advantage_raw": round(fit.home_advantage_raw, 3),
        "fitted_at": fit.fitted_at,
    }


def seed_and_grade_history(ds, league: str, fit, value_threshold: float):
    """Demo-mode only: predict the bundled 'seed' fixtures (held OUT of the
    fit, so this isn't grading a model on matches it was trained on), log
    them, and immediately grade against the real result — so the track
    record and value-bet backtest aren't empty on a first run. A live
    deployment doesn't need this: real history accumulates naturally as
    genuine advance predictions get graded over time."""
    for fx in ds.history_fixtures(league):
        home = fit.team_strength(fx.home_team)
        away = fit.team_strength(fx.away_team)
        pred = predict_match(home, away, fit.avg_goals, fit.home_advantage,
                              ds.odds(fx.fixture_id), fit_meta_dict(fit), value_threshold)
        track_record.log_prediction(fx.fixture_id, fx.home_team, fx.away_team, pred.to_dict())
        if fx.home_goals is not None:
            track_record.grade(fx.fixture_id, fx.home_goals, fx.away_goals)


def build_predictions(league: str):
    ds = get_data_source()
    fit = fit_league_parameters(ds.historical_matches(league))
    # the threshold used to flag NEW value bets going forward -- tuned from
    # graded history once there's enough of it to trust, else the default
    value_threshold = threshold_tuning.active_threshold()
    seed_and_grade_history(ds, league, fit, value_threshold)

    fixtures = ds.upcoming_fixtures(league, limit=10)
    results = []
    for fx in fixtures:
        home = fit.team_strength(fx.home_team)
        away = fit.team_strength(fx.away_team)
        odds = ds.odds(fx.fixture_id)
        pred = predict_match(home, away, fit.avg_goals, fit.home_advantage, odds,
                              fit_meta_dict(fit), value_threshold)
        pred_dict = pred.to_dict()
        pred_dict["fixture_id"] = fx.fixture_id
        pred_dict["date"] = fx.date
        results.append(pred_dict)
        track_record.log_prediction(fx.fixture_id, fx.home_team, fx.away_team, pred_dict)

    # opportunistically grade any already-logged predictions whose matches have finished
    for fx in ds.finished_fixtures_needing_grade(track_record.ungraded_fixture_ids()):
        track_record.grade(fx.fixture_id, fx.home_goals, fx.away_goals)

    return results, ds.label, ds.available_leagues(), fit


@app.route("/")
def index():
    ds = get_data_source()
    league = resolve_league(ds)
    predictions, source_label, leagues, fit = build_predictions(league)
    record = track_record.summary()
    value_perf = track_record.value_bet_performance()
    tuning = threshold_tuning.evaluate_thresholds()
    return render_template(
        "index.html", predictions=predictions, source_label=source_label,
        record=record, leagues=leagues, active_league=league, fit=fit,
        value_perf=value_perf, tuning=tuning,
    )


@app.route("/api/predictions")
def api_predictions():
    ds = get_data_source()
    league = resolve_league(ds)
    predictions, source_label, leagues, fit = build_predictions(league)
    return jsonify({
        "source": source_label, "league": league, "predictions": predictions,
        "fit": fit_meta_dict(fit) | {"avg_goals": round(fit.avg_goals, 3), "home_advantage": round(fit.home_advantage, 3)},
    })


@app.route("/api/track-record")
def api_track_record():
    return jsonify(track_record.summary())


@app.route("/api/value-bet-performance")
def api_value_bet_performance():
    return jsonify(track_record.value_bet_performance())


@app.route("/api/threshold-tuning")
def api_threshold_tuning():
    return jsonify(threshold_tuning.evaluate_thresholds())


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # 0.0.0.0 + $PORT: what every free host (Render, Railway, Fly.io, Heroku-style
    # buildpacks) expects a web process to bind to. Local runs still default to
    # 5000. In production, gunicorn (see Procfile) serves the app instead of
    # this dev server -- this branch only runs for `python app.py` directly.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
