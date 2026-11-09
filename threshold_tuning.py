"""
Retroactively tests candidate value-bet edge thresholds against the full
graded history — every model-vs-market edge for every outcome of every
graded match, not just the ones flagged as value bets at the time (see
track_record.all_graded_edges(), fed by model.py's compute_all_edges()).
That's what makes this possible at all: sweeping "what if the threshold
had been 7 points instead of 5" needs the edges that WOULDN'T have cleared
5 points too, not just the ones that did.

Nothing here mutates the default silently. A candidate threshold is only
ever recommended once it clears BOTH a minimum sample size and a
statistical significance check against the break-even rate its own odds
imply. Below that bar, active_threshold() keeps returning the documented
default (model.VALUE_EDGE_THRESHOLD) rather than chasing whichever
candidate happens to look best on a small, noisy sample — the same "don't
overclaim on 12 bets" discipline the rest of this app uses.

The grid is fine (every 1 point from 2 to 20), which raises a real
statistical issue on its own: testing 19 candidates against the same data
means one of them clearing a plain 95%-confidence z-test (z >= 1.96) by
PURE CHANCE is likely, not unlikely — run 19 independent coin flips against
a fair-coin null and expect close to one "significant" result on average.
So the significance bar is Bonferroni-corrected for the number of
candidates actually tested: the per-candidate alpha is 0.05 / (number of
candidates), and the z-score needed to clear that stricter alpha is
computed directly (via a standard inverse-normal approximation) rather than
left at the uncorrected 1.96. More candidates tested -> a higher bar to
call any one of them significant -- which is the correct trade for a
finer grid, not an incidental side effect.
"""
from __future__ import annotations

import math

import track_record
from model import VALUE_EDGE_THRESHOLD

# 2pt to 20pt in 1pt steps -- fine enough to actually distinguish nearby
# thresholds once there's enough graded history, at the cost of needing the
# Bonferroni correction below to keep the significance claim honest.
CANDIDATE_THRESHOLDS = [round(0.02 + 0.01 * i, 2) for i in range(19)]
MIN_BETS_PER_THRESHOLD = track_record.MIN_VALUE_BETS_FOR_SIGNIFICANCE  # same 20-bet bar as the live backtest
BASE_ALPHA = 0.05  # the significance level before correcting for multiple comparisons


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (probit function), via Acklam's rational
    approximation -- accurate to ~1e-9, no scipy dependency needed for a
    single z-threshold lookup."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _required_z(num_candidates: int) -> float:
    """Bonferroni-corrected two-tailed z threshold for testing this many
    candidates at once. More candidates -> smaller per-candidate alpha ->
    a higher z bar, which is what keeps a 19-way sweep from finding a
    'significant' result by chance the way a single test wouldn't."""
    alpha_corrected = BASE_ALPHA / max(num_candidates, 1)
    return _norm_ppf(1 - alpha_corrected / 2)


def _evaluate_one(edges: list[dict], threshold: float, required_z: float) -> dict:
    eligible = [e for e in edges if e["edge"] >= threshold and e["expected_value_pct"] > 0]
    n = len(eligible)
    result = {"threshold": threshold, "n": n, "significant": False, "z_score": None}
    if n == 0:
        return result

    wins = sum(1 for e in eligible if e["won"])
    win_rate = wins / n
    breakeven_rate = sum(1 / e["odds"] for e in eligible) / n
    roi_pct = (sum(e["ret"] for e in eligible) / n) * 100
    result.update({
        "wins": wins,
        "win_rate": round(win_rate, 3),
        "breakeven_win_rate": round(breakeven_rate, 3),
        "roi_pct": round(roi_pct, 1),
    })
    if n >= MIN_BETS_PER_THRESHOLD:
        se = math.sqrt(breakeven_rate * (1 - breakeven_rate) / n)
        z = (win_rate - breakeven_rate) / se if se > 0 else 0.0
        result["z_score"] = round(z, 2)
        result["significant"] = z >= required_z
    return result


def evaluate_thresholds(candidates: list[float] = None) -> dict:
    candidates = candidates or CANDIDATE_THRESHOLDS
    required_z = _required_z(len(candidates))
    edges = track_record.all_graded_edges()
    evaluated = [_evaluate_one(edges, t, required_z) for t in candidates]

    eligible = [r for r in evaluated if r["n"] >= MIN_BETS_PER_THRESHOLD and r["significant"]]
    if eligible:
        best = max(eligible, key=lambda r: r["roi_pct"])
        recommended = best["threshold"]
        is_default = False
        reason = (
            f"{best['threshold'] * 100:.0f}pt clears {best['n']} graded bets with an edge "
            f"significant even after correcting for testing {len(candidates)} candidates "
            f"(z={best['z_score']} >= required {required_z:.2f}) and the best ROI "
            f"({best['roi_pct']}%) among thresholds that qualify — switched from the "
            f"{VALUE_EDGE_THRESHOLD * 100:.0f}pt default."
        )
    else:
        recommended = VALUE_EDGE_THRESHOLD
        is_default = True
        reason = (
            f"No candidate has both {MIN_BETS_PER_THRESHOLD}+ graded bets and a significant edge "
            f"once corrected for testing {len(candidates)} candidates at once (needs z >= "
            f"{required_z:.2f}, vs. 1.96 uncorrected) — keeping the {VALUE_EDGE_THRESHOLD * 100:.0f}pt default."
        )

    return {
        "candidates": evaluated,
        "recommended_threshold": recommended,
        "is_default": is_default,
        "reason": reason,
        "required_z": round(required_z, 2),
        "num_candidates_tested": len(candidates),
        "graded_edges_available": len(edges),
    }


def active_threshold() -> float:
    """The threshold predict_match() should actually use right now — the
    tuned value once the data supports one, else the documented default."""
    return evaluate_thresholds()["recommended_threshold"]
