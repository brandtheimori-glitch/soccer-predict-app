"""
Prediction engine.

Two blended signals, both computed transparently so every number on screen
can be traced back to an input:

1. A Poisson goal model — each team gets an attack/defense "strength"
   relative to its league's average, and those strengths are combined into
   an expected-goals (xG) figure for each side. A full score-matrix is built
   from independent Poisson distributions around those two xG values, which
   is enough to derive 1X2 probabilities, the most likely scoreline, and
   over/under markets.

2. The betting market's own implied probabilities (de-vigged), when odds
   are available. Markets aggregate a lot of information the simple model
   can't see (injuries, suspensions, weather), so blending in a market
   signal generally improves calibration versus a pure stats model.

The two are blended with a configurable weight (MODEL_WEIGHT). Every
prediction stores the inputs that produced it, so the UI can show its work
instead of just asserting a number.

CHANGELOG NOTE: an earlier version multiplied home expected goals by both
(a) a league-wide home/away goal-average split AND (b) a flat HOME_ADVANTAGE
constant on top — double-counting home advantage. This version fixes that:
attack/defense strengths are now relative to a single pooled league average
(fit_parameters.py's avg_goals), and home advantage is the ONLY thing that
lifts the home side's expected goals, fitted from real results instead of
hand-tuned. See fit_parameters.py for how home_advantage and each team's
attack/defense are estimated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from fit_parameters import TeamStrength, side_multipliers  # TeamStrength re-exported for convenience

MAX_GOALS = 8          # highest scoreline considered per side when building the matrix
MODEL_WEIGHT = 0.65     # weight given to the stats model vs. the market odds (0..1)
VALUE_EDGE_THRESHOLD = 0.05  # min model-vs-market probability gap (5pp) to flag a value bet


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for a Poisson(lam) variable, computed without scipy."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


@dataclass
class OddsSnapshot:
    """Decimal odds for the 1X2 market from one or more bookmakers (already averaged)."""
    home: float
    draw: float
    away: float

    def implied_probabilities(self) -> tuple[float, float, float]:
        """De-vig (remove the bookmaker's margin) via simple normalization."""
        raw = (1 / self.home, 1 / self.draw, 1 / self.away)
        overround = sum(raw)
        return tuple(r / overround for r in raw)  # type: ignore[return-value]


@dataclass
class Prediction:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    p_home: float
    p_draw: float
    p_away: float
    top_scorelines: list[tuple[tuple[int, int], float]]
    over_2_5: float
    btts: float  # both teams to score
    market_probs: Optional[tuple[float, float, float]] = None
    inputs: dict = field(default_factory=dict)
    value_bets: list = field(default_factory=list)
    all_edges: list = field(default_factory=list)

    @property
    def confidence(self) -> str:
        top = max(self.p_home, self.p_draw, self.p_away)
        if top >= 0.55:
            return "High"
        if top >= 0.40:
            return "Medium"
        return "Low"

    @property
    def btts_pick(self) -> str:
        """'Yes' if both teams scoring is more likely than not, else 'No'."""
        return "Yes" if self.btts >= 0.5 else "No"

    @property
    def btts_confidence(self) -> float:
        """Probability of whichever BTTS side is picked (always >= 0.5)."""
        return max(self.btts, 1 - self.btts)

    @property
    def ou_pick(self) -> str:
        """'Over' if 3+ total goals is more likely than not, else 'Under'."""
        return "Over" if self.over_2_5 >= 0.5 else "Under"

    @property
    def ou_confidence(self) -> float:
        """Probability of whichever over/under 2.5 side is picked (always >= 0.5)."""
        return max(self.over_2_5, 1 - self.over_2_5)

    @property
    def pick(self) -> str:
        probs = {"Home": self.p_home, "Draw": self.p_draw, "Away": self.p_away}
        return max(probs, key=probs.get)

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_xg": round(self.home_xg, 2),
            "away_xg": round(self.away_xg, 2),
            "p_home": round(self.p_home, 3),
            "p_draw": round(self.p_draw, 3),
            "p_away": round(self.p_away, 3),
            "pick": self.pick,
            "confidence": self.confidence,
            "top_scorelines": [
                {"score": f"{s[0]}-{s[1]}", "prob": round(p, 3)}
                for s, p in self.top_scorelines
            ],
            "over_2_5": round(self.over_2_5, 3),
            "btts": round(self.btts, 3),
            "btts_pick": self.btts_pick,
            "btts_confidence": round(self.btts_confidence, 3),
            "ou_pick": self.ou_pick,
            "ou_confidence": round(self.ou_confidence, 3),
            "market_probs": (
                {
                    "home": round(self.market_probs[0], 3),
                    "draw": round(self.market_probs[1], 3),
                    "away": round(self.market_probs[2], 3),
                }
                if self.market_probs
                else None
            ),
            "inputs": self.inputs,
            "value_bets": self.value_bets,
            "all_edges": self.all_edges,
        }


def expected_goals(home: TeamStrength, away: TeamStrength, avg_goals: float,
                    home_advantage: float) -> tuple[float, float]:
    """Combine attack/defense strengths into expected goals for each side.
    home_advantage is the home/away goal ratio; it lifts the home side and
    lowers the away side by matching amounts, so the match total stays
    anchored to the league's real scoring rate."""
    home_mult, away_mult = side_multipliers(home_advantage)
    home_xg = avg_goals * home_mult * home.attack * away.defense
    away_xg = avg_goals * away_mult * away.attack * home.defense
    return home_xg, away_xg


def score_matrix(home_xg: float, away_xg: float) -> list[list[float]]:
    matrix = [
        [poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg) for j in range(MAX_GOALS + 1)]
        for i in range(MAX_GOALS + 1)
    ]
    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[cell / total for cell in row] for row in matrix]
    return matrix


def outcomes_from_matrix(matrix: list[list[float]]):
    p_home = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j)
    p_draw = sum(matrix[i][i] for i in range(MAX_GOALS + 1))
    p_away = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i < j)
    over_2_5 = sum(
        matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j > 2.5
    )
    btts = sum(
        matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > 0 and j > 0
    )
    flat = [((i, j), matrix[i][j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return p_home, p_draw, p_away, over_2_5, btts, flat[:5]


def blend(model_probs: tuple[float, float, float], market_probs: Optional[tuple[float, float, float]],
          weight: float = MODEL_WEIGHT) -> tuple[float, float, float]:
    if market_probs is None:
        return model_probs
    blended = tuple(weight * m + (1 - weight) * b for m, b in zip(model_probs, market_probs))
    total = sum(blended)
    return tuple(x / total for x in blended)  # type: ignore[return-value]


def compute_all_edges(model_probs: tuple[float, float, float], odds: Optional[OddsSnapshot]) -> list[dict]:
    """
    Model-vs-market comparison for ALL THREE outcomes, unconditionally — not
    just the ones that clear whatever the current value-bet threshold is.

    This is what makes the threshold tunable later: threshold_tuning.py
    re-examines this full record against candidate thresholds to see which
    one would actually have performed best, which is only possible because
    every outcome's edge got logged, not just the ones flagged at the time.

    Compared against MODEL-ONLY probabilities (not the blended figure shown
    as the headline pick), since blending already pulls the number toward
    the market and would hide the disagreement this is meant to surface.
    """
    if odds is None:
        return []
    market_probs = odds.implied_probabilities()
    outcomes = [
        ("Home", model_probs[0], market_probs[0], odds.home),
        ("Draw", model_probs[1], market_probs[1], odds.draw),
        ("Away", model_probs[2], market_probs[2], odds.away),
    ]
    edges = []
    for name, model_p, market_p, price in outcomes:
        edge = model_p - market_p
        ev = model_p * price - 1
        edges.append({
            "outcome": name,
            "odds": price,
            "model_prob": round(model_p, 4),
            "market_prob": round(market_p, 4),
            "edge": round(edge, 4),
            "edge_pts": round(edge * 100, 1),
            "expected_value_pct": round(ev * 100, 1),
        })
    edges.sort(key=lambda v: v["edge_pts"], reverse=True)
    return edges


def find_value_bets(all_edges: list[dict], threshold: float = VALUE_EDGE_THRESHOLD) -> list[dict]:
    """
    A "value bet" is when the model thinks an outcome is meaningfully more
    likely than the market's odds imply — i.e. the market would be paying
    out more than the outcome's true (model) probability justifies. This is
    the actual thing professional bettors look for: not "who wins" but
    "where does my number disagree with the market's price."

    Expected value per unit staked at decimal odds `o` with true probability
    `p` is `p*o - 1`. A positive EV alone is easy to get from noise on long
    shots, so this also requires the probability gap to clear `threshold`.
    Filters compute_all_edges()'s output — see that function for how the
    underlying numbers are derived.
    """
    return [e for e in all_edges if e["edge"] >= threshold and e["expected_value_pct"] > 0]


def predict_match(home: TeamStrength, away: TeamStrength, avg_goals: float, home_advantage: float,
                   odds: Optional[OddsSnapshot] = None, fit_meta: Optional[dict] = None,
                   value_threshold: float = VALUE_EDGE_THRESHOLD) -> Prediction:
    home_xg, away_xg = expected_goals(home, away, avg_goals, home_advantage)
    matrix = score_matrix(home_xg, away_xg)
    p_home, p_draw, p_away, over_2_5, btts, top5 = outcomes_from_matrix(matrix)

    market_probs = odds.implied_probabilities() if odds else None
    p_home_b, p_draw_b, p_away_b = blend((p_home, p_draw, p_away), market_probs)
    all_edges = compute_all_edges((p_home, p_draw, p_away), odds)
    value_bets = find_value_bets(all_edges, value_threshold)

    inputs = {
        "home_attack_strength": round(home.attack, 2),
        "home_defense_strength": round(home.defense, 2),
        "away_attack_strength": round(away.attack, 2),
        "away_defense_strength": round(away.defense, 2),
        "home_form_games": home.games_played,
        "away_form_games": away.games_played,
        "home_form_effective_games": home.effective_games,
        "away_form_effective_games": away.effective_games,
        "home_advantage_used": round(home_advantage, 3),
        "avg_goals_used": round(avg_goals, 3),
        "model_only_probs": {
            "home": round(p_home, 3), "draw": round(p_draw, 3), "away": round(p_away, 3)
        },
        "blend_weight_model": MODEL_WEIGHT if market_probs else 1.0,
        "value_threshold_used": round(value_threshold, 3),
    }
    if fit_meta:
        inputs["fit"] = fit_meta

    return Prediction(
        home_team=home.team,
        away_team=away.team,
        home_xg=home_xg,
        away_xg=away_xg,
        p_home=p_home_b,
        p_draw=p_draw_b,
        p_away=p_away_b,
        top_scorelines=top5,
        over_2_5=over_2_5,
        btts=btts,
        all_edges=all_edges,
        market_probs=market_probs,
        value_bets=value_bets,
        inputs=inputs,
    )
