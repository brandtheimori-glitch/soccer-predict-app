"""
Public, auditable track record.

The research pass on existing prediction apps turned up a consistent gap:
most either don't publish a track record at all, or only surface a rolling
"accuracy" figure with no way to check it. Here, every prediction — and
every value bet flagged inside it — is logged the moment it's made (before
the match is known), and graded against the final score once it's
available. The log is a flat JSON file so it's easy to inspect by hand —
nothing is hidden behind an aggregate stat.

Grading value bets separately from the headline pick matters because
they're a different claim: the pick is "who wins," graded right/wrong.
A value bet is "this specific price is mispriced," graded by whether it
would have made money — win rate AND return matter, and a good value bet
can lose (that's expected some of the time) while still being a correct
call about the price.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

LOG_PATH = Path(__file__).parent / "predictions_log.json"
MIN_VALUE_BETS_FOR_SIGNIFICANCE = 20  # don't claim "beats break-even" on a handful of bets
BTTS_CONFIDENCE_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70]  # "only count picks at least this sure"


def _load() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return json.loads(LOG_PATH.read_text())


def _save(entries: list[dict]) -> None:
    LOG_PATH.write_text(json.dumps(entries, indent=2))


def log_prediction(fixture_id: str, home_team: str, away_team: str, prediction: dict) -> None:
    entries = _load()
    if any(e["fixture_id"] == fixture_id for e in entries):
        return  # already logged; never overwrite a prediction after the fact
    entries.append({
        "fixture_id": fixture_id,
        "home_team": home_team,
        "away_team": away_team,
        "pick": prediction["pick"],
        "p_home": prediction["p_home"],
        "p_draw": prediction["p_draw"],
        "p_away": prediction["p_away"],
        "predicted_score": prediction["top_scorelines"][0]["score"] if prediction["top_scorelines"] else None,
        "btts_pick": prediction.get("btts_pick"),
        "btts_confidence": prediction.get("btts_confidence"),
        "btts_correct": None,
        "date": prediction.get("date"),
        # snapshot of value bets AS FLAGGED AT PREDICTION TIME (under whatever
        # threshold was active then) — odds/probabilities here never change
        # after logging, only "won"/"return" get filled in on grade()
        "value_bets": [dict(vb, won=None, ret=None) for vb in prediction.get("value_bets", [])],
        # the FULL model-vs-market comparison for all three outcomes, regardless
        # of whether they cleared a threshold — this is what makes the edge
        # threshold retroactively tunable (see threshold_tuning.py)
        "all_edges": [dict(e, won=None, ret=None) for e in prediction.get("all_edges", [])],
        "graded": False,
        "correct": None,
        "actual_score": None,
    })
    _save(entries)


def _grade_edge_list(edges: list[dict], actual_outcome: str) -> None:
    for e in edges:
        e["won"] = (e["outcome"] == actual_outcome)
        # profit/loss per 1 unit staked at the odds recorded at prediction time
        e["ret"] = round(e["odds"] - 1, 3) if e["won"] else -1.0


def grade(fixture_id: str, home_goals: int, away_goals: int) -> None:
    entries = _load()
    for e in entries:
        if e["fixture_id"] == fixture_id and not e["graded"]:
            if home_goals > away_goals:
                actual = "Home"
            elif home_goals < away_goals:
                actual = "Away"
            else:
                actual = "Draw"
            e["graded"] = True
            e["correct"] = (e["pick"] == actual)
            e["actual_score"] = f"{home_goals}-{away_goals}"
            if e.get("btts_pick"):
                both_scored = home_goals > 0 and away_goals > 0
                e["btts_correct"] = (e["btts_pick"] == "Yes") == both_scored
            _grade_edge_list(e["value_bets"], actual)
            _grade_edge_list(e.get("all_edges", []), actual)
    _save(entries)


def ungraded_fixture_ids() -> list[str]:
    return [e["fixture_id"] for e in _load() if not e["graded"]]


def all_graded_edges() -> list[dict]:
    """Every graded model-vs-market edge, for every outcome of every graded
    match, regardless of whether it was flagged as a value bet at the time.
    This is the full record threshold_tuning.py sweeps against."""
    out = []
    for e in _load():
        if e["graded"]:
            for edge in e.get("all_edges", []):
                if edge["won"] is not None:
                    out.append({**edge, "fixture": f"{e['home_team']} vs {e['away_team']}"})
    return out


def summary() -> dict:
    entries = _load()
    graded = [e for e in entries if e["graded"]]
    correct = [e for e in graded if e["correct"]]
    return {
        "total_predictions": len(entries),
        "graded": len(graded),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(graded), 3) if graded else None,
        "entries": list(reversed(entries)),
    }


def btts_performance() -> dict:
    """
    Both-teams-to-score hit rate, overall AND restricted to the picks the
    model was most sure about. The point of the breakdown: if picks at 65%+
    confidence really do land more often than picks at 50-55%, sticking to
    the high-confidence ones is how to get the higher win rate -- and this
    shows whether that's actually happening, from real graded results,
    instead of assuming it.
    """
    graded = [e for e in _load() if e["graded"] and e.get("btts_correct") is not None]
    levels = []
    for level in BTTS_CONFIDENCE_LEVELS:
        picks = [e for e in graded if (e.get("btts_confidence") or 0) >= level]
        hits = sum(1 for e in picks if e["btts_correct"])
        levels.append({
            "min_confidence": level,
            "picks": len(picks),
            "hits": hits,
            "hit_rate": round(hits / len(picks), 3) if picks else None,
        })
    hits = sum(1 for e in graded if e["btts_correct"])
    return {
        "graded": len(graded),
        "hits": hits,
        "hit_rate": round(hits / len(graded), 3) if graded else None,
        "by_confidence": levels,
        "enough_data": len(graded) >= MIN_VALUE_BETS_FOR_SIGNIFICANCE,
        "min_for_trust": MIN_VALUE_BETS_FOR_SIGNIFICANCE,
    }


def value_bet_performance() -> dict:
    """
    Aggregate every graded value bet into a running backtest: win rate,
    return on investment (ROI) at 1 unit staked per bet, and a check for
    whether that ROI is distinguishable from noise.

    The "break-even" baseline isn't 0% by assumption — it's the win rate
    the market's own odds implied for each specific bet (average of
    1/odds across all flagged bets). If the model's picks win at close to
    that rate, the flagged "edge" wasn't real; if they win noticeably more
    often, the edge is showing up in practice, not just on paper.

    The significance check is a simple one-proportion z-test against that
    break-even rate — a reasonable gut-check, not a rigorous backtest
    (it assumes independence across bets, which a real season violates to
    some degree via correlated team form). Treat it as "is there enough
    signal here to keep paying attention," not proof of a working edge.
    """
    all_bets = []
    for e in _load():
        if e["graded"]:
            for vb in e["value_bets"]:
                if vb["won"] is not None:
                    all_bets.append({**vb, "fixture": f"{e['home_team']} vs {e['away_team']}"})

    n = len(all_bets)
    if n == 0:
        return {"graded_value_bets": 0, "message": "No value bets graded yet."}

    wins = sum(1 for b in all_bets if b["won"])
    total_return = sum(b["ret"] for b in all_bets)
    win_rate = wins / n
    roi_pct = (total_return / n) * 100
    breakeven_rate = sum(1 / b["odds"] for b in all_bets) / n

    result = {
        "graded_value_bets": n,
        "wins": wins,
        "win_rate": round(win_rate, 3),
        "breakeven_win_rate": round(breakeven_rate, 3),
        "roi_pct": round(roi_pct, 1),
        "total_return_units": round(total_return, 2),
        "bets": list(reversed(all_bets)),
    }

    if n < MIN_VALUE_BETS_FOR_SIGNIFICANCE:
        result["verdict"] = (
            f"Only {n} graded value bets so far — need at least "
            f"{MIN_VALUE_BETS_FOR_SIGNIFICANCE} before ROI is worth reading as anything but noise."
        )
        result["significant"] = False
    else:
        se = math.sqrt(breakeven_rate * (1 - breakeven_rate) / n)
        z = (win_rate - breakeven_rate) / se if se > 0 else 0.0
        result["z_score"] = round(z, 2)
        if z >= 1.96:
            result["verdict"] = "Win rate is significantly above the market's break-even rate (~95% confidence) — the flagged edge is showing up in practice."
            result["significant"] = True
        elif z <= -1.96:
            result["verdict"] = "Win rate is significantly BELOW the market's break-even rate — the value-bet criteria are miscalibrated and need revisiting."
            result["significant"] = True
        else:
            result["verdict"] = "Win rate isn't statistically distinguishable from the market's break-even rate yet — could be a real small edge or could be noise."
            result["significant"] = False

    return result
