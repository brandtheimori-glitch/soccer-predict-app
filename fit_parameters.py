"""
Fits the model's parameters (home advantage, each team's attack/defense
strength) from actual historical results, instead of the hand-tuned guesses
`model.py` used before (HOME_ADVANTAGE = 1.35, picked by feel, applied on
top of an already-skewed home/away goal average — see the note in
model.py's changelog comment for the bug that caused).

Method (the standard approach for this class of model, going back to
Maher 1982 / used in Dixon-Coles): every team gets one attack rating and
one defense rating relative to the league's overall average goals per team
per game, and there is a single league-wide home-advantage multiplier —
fitted from the ratio of average home-team goals to the league average,
not assumed.

Two adjustments keep the fit honest given free-tier-sized samples:

1. Recency weighting. A match from 3 months ago says less about a team's
   CURRENT strength than one from last week — squads change, form swings,
   managers get sacked. Every match is weighted by an exponential decay
   based on its age, exactly as Dixon & Coles (1997) did (their fitted
   decay, xi=0.0065/day, implies a ~107-day half-life — used here as
   HALF_LIFE_DAYS=100). A match at the half-life's age counts half as much
   as one today; two half-lives, a quarter; and so on. This also means the
   "sample size" that matters for confidence is an EFFECTIVE size (the sum
   of weights), not a raw count — 20 matches spread over two seasons carry
   less real information than 20 from the last two months.

2. Shrinkage. Fitting a ratio from a dozen (effectively weighted) matches
   is noisy — a team that happens to have scored big in one freak result
   over a small sample would otherwise get a wildly overstated attack
   rating. So every fitted number is shrunk toward a neutral prior in
   proportion to its EFFECTIVE sample size (more/more-recent games -> trust
   the data more; fewer/staler games -> lean on the prior). Plain
   empirical-Bayes shrinkage, not anything exotic, but it's the difference
   between a fitted number that's honest about its own uncertainty and one
   that overreacts to a handful of stale matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timezone
from typing import Optional

DEFAULT_HOME_ADVANTAGE_PRIOR = 1.30   # commonly cited ballpark for top-flight leagues
DEFAULT_AVG_GOALS_PRIOR = 1.35        # goals per team per game, ballpark for top-5-league scoring rates
LEAGUE_PRIOR_GAMES = 40               # how many "effective matches worth" of weight the home-advantage prior gets
TEAM_PRIOR_GAMES = 14                  # how many "effective games worth" of weight each team's neutral prior gets
LOW_CONFIDENCE_THRESHOLD = 30         # effective sample size below this -> flag the fit as low-confidence
HALF_LIFE_DAYS = 100                  # a match this old counts half as much as one today (Dixon & Coles 1997 ballpark)
FIT_ITERATIONS = 25                   # passes of the opponent-adjusted rating fit (converges well before this)


def side_multipliers(home_advantage: float) -> tuple[float, float]:
    """Turn a home/away goal ratio into (home, away) multipliers on the
    pooled league average that average to exactly 1 -- so home advantage
    shifts goals from the away side to the home side without inventing
    extra goals. E.g. 1.30 -> (1.13, 0.87)."""
    return 2 * home_advantage / (1 + home_advantage), 2 / (1 + home_advantage)


def _parse_date(d: str) -> date_cls:
    return datetime.strptime(d, "%Y-%m-%d").date()


def recency_weight(match_date: str, as_of: date_cls, half_life_days: float = HALF_LIFE_DAYS) -> float:
    days_ago = max(0, (as_of - _parse_date(match_date)).days)
    return 0.5 ** (days_ago / half_life_days)


@dataclass
class HistoricalMatch:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    date: str = ""  # "YYYY-MM-DD"; undated matches get full weight (treated as "as of today")


@dataclass
class TeamStrength:
    """A team's fitted attack/defense rating, already relative to the league average."""
    team: str
    games_played: int          # raw count of matches in the fit sample
    effective_games: float     # recency-weighted count — what shrinkage is actually based on
    attack: float
    defense: float


@dataclass
class LeagueFit:
    avg_goals: float               # league-wide average goals per team per game
    home_advantage: float          # fitted, shrunk multiplier applied to the home side's xG
    home_advantage_raw: float      # the unshrunk estimate, for transparency
    teams: dict = field(default_factory=dict)   # team name -> TeamStrength
    sample_size: int = 0             # raw number of matches the fit was built from
    effective_sample_size: float = 0.0  # recency-weighted number — what confidence is actually based on
    half_life_days: float = HALF_LIFE_DAYS
    low_confidence: bool = True
    fitted_at: str = ""

    def team_strength(self, team: str) -> TeamStrength:
        """Look up a fitted team, or fall back to a neutral league-average team
        (attack=defense=1.0) for one with no historical data yet — e.g. a
        newly promoted side."""
        return self.teams.get(team, TeamStrength(team=team, games_played=0, effective_games=0.0, attack=1.0, defense=1.0))


def fit_league_parameters(
    matches: list[HistoricalMatch],
    as_of: Optional[date_cls] = None,
    home_advantage_prior: float = DEFAULT_HOME_ADVANTAGE_PRIOR,
    league_prior_games: int = LEAGUE_PRIOR_GAMES,
    team_prior_games: int = TEAM_PRIOR_GAMES,
    half_life_days: float = HALF_LIFE_DAYS,
) -> LeagueFit:
    as_of = as_of or datetime.now(timezone.utc).date()
    n = len(matches)
    if n == 0:
        return LeagueFit(
            avg_goals=DEFAULT_AVG_GOALS_PRIOR,
            home_advantage=home_advantage_prior,
            home_advantage_raw=home_advantage_prior,
            teams={},
            sample_size=0,
            effective_sample_size=0.0,
            half_life_days=half_life_days,
            low_confidence=True,
            fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    weights = [recency_weight(m.date, as_of, half_life_days) if m.date else 1.0 for m in matches]
    total_weight = sum(weights)

    total_home_goals_w = sum(w * m.home_goals for w, m in zip(weights, matches))
    total_away_goals_w = sum(w * m.away_goals for w, m in zip(weights, matches))
    avg_goals = (total_home_goals_w + total_away_goals_w) / (2 * total_weight)  # per team per game, pooled

    # Home advantage = ratio of home-team goals to away-team goals (the
    # standard definition, ~1.3 in top flights). It is applied as a matched
    # PAIR of multipliers -- home side up, away side down -- so a match's
    # total expected goals stays equal to the league's real average. (An
    # earlier version only lifted the home side, which inflated every
    # match's total and biased both-teams-to-score toward "Yes".)
    raw_home_advantage = (
        total_home_goals_w / total_away_goals_w if total_away_goals_w > 0 else home_advantage_prior
    )
    home_advantage = (
        total_weight * raw_home_advantage + league_prior_games * home_advantage_prior
    ) / (total_weight + league_prior_games)
    home_mult, away_mult = side_multipliers(home_advantage)

    # Team ratings, fitted iteratively so they account for WHO each team
    # played: 2 goals against the best defence is worth more than 2 against
    # the worst. Each pass sets attack = goals scored / goals expected given
    # the opponents' current defence ratings (and vice versa), shrunk toward
    # 1.00 by how much (recency-weighted) data the team has.
    team_names = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
    att = {t: 1.0 for t in team_names}
    dfn = {t: 1.0 for t in team_names}
    weight_sum: dict[str, float] = {t: 0.0 for t in team_names}
    raw_games: dict[str, int] = {t: 0 for t in team_names}
    for w, m in zip(weights, matches):
        for t in (m.home_team, m.away_team):
            weight_sum[t] += w
            raw_games[t] += 1

    for _ in range(FIT_ITERATIONS):
        scored = {t: 0.0 for t in team_names}; exp_scored = {t: 0.0 for t in team_names}
        conceded = {t: 0.0 for t in team_names}; exp_conceded = {t: 0.0 for t in team_names}
        for w, m in zip(weights, matches):
            h, a = m.home_team, m.away_team
            scored[h] += w * m.home_goals; exp_scored[h] += w * avg_goals * home_mult * dfn[a]
            scored[a] += w * m.away_goals; exp_scored[a] += w * avg_goals * away_mult * dfn[h]
            conceded[h] += w * m.away_goals; exp_conceded[h] += w * avg_goals * away_mult * att[a]
            conceded[a] += w * m.home_goals; exp_conceded[a] += w * avg_goals * home_mult * att[h]
        for t in team_names:
            k, g = team_prior_games, weight_sum[t]
            raw_att = scored[t] / exp_scored[t] if exp_scored[t] > 0 else 1.0
            raw_def = conceded[t] / exp_conceded[t] if exp_conceded[t] > 0 else 1.0
            att[t] = (g * raw_att + k) / (g + k)
            dfn[t] = (g * raw_def + k) / (g + k)
        # keep the league average at exactly 1.00 so ratings don't drift
        mean_att = sum(att.values()) / len(att)
        mean_def = sum(dfn.values()) / len(dfn)
        att = {t: v / mean_att for t, v in att.items()}
        dfn = {t: v / mean_def for t, v in dfn.items()}

    teams: dict[str, TeamStrength] = {
        t: TeamStrength(team=t, games_played=raw_games[t], effective_games=round(weight_sum[t], 2),
                        attack=att[t], defense=dfn[t])
        for t in team_names
    }

    return LeagueFit(
        avg_goals=avg_goals,
        home_advantage=home_advantage,
        home_advantage_raw=raw_home_advantage,
        teams=teams,
        sample_size=n,
        effective_sample_size=round(total_weight, 2),
        half_life_days=half_life_days,
        low_confidence=total_weight < LOW_CONFIDENCE_THRESHOLD,
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
