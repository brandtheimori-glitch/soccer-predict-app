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
TEAM_PRIOR_GAMES = 4                  # how many "effective games worth" of weight each team's neutral prior gets
LOW_CONFIDENCE_THRESHOLD = 30         # effective sample size below this -> flag the fit as low-confidence
HALF_LIFE_DAYS = 100                  # a match this old counts half as much as one today (Dixon & Coles 1997 ballpark)


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

    avg_home_goals_per_match = total_home_goals_w / total_weight
    raw_home_advantage = avg_home_goals_per_match / avg_goals if avg_goals > 0 else home_advantage_prior
    # shrink toward the prior in proportion to EFFECTIVE sample size vs. league_prior_games
    home_advantage = (
        total_weight * raw_home_advantage + league_prior_games * home_advantage_prior
    ) / (total_weight + league_prior_games)

    # accumulate each team's (weighted) goals scored/conceded across every match it played
    scored: dict[str, float] = {}
    conceded: dict[str, float] = {}
    weight_sum: dict[str, float] = {}
    raw_games: dict[str, int] = {}
    for w, m in zip(weights, matches):
        scored[m.home_team] = scored.get(m.home_team, 0.0) + w * m.home_goals
        conceded[m.home_team] = conceded.get(m.home_team, 0.0) + w * m.away_goals
        weight_sum[m.home_team] = weight_sum.get(m.home_team, 0.0) + w
        raw_games[m.home_team] = raw_games.get(m.home_team, 0) + 1

        scored[m.away_team] = scored.get(m.away_team, 0.0) + w * m.away_goals
        conceded[m.away_team] = conceded.get(m.away_team, 0.0) + w * m.home_goals
        weight_sum[m.away_team] = weight_sum.get(m.away_team, 0.0) + w
        raw_games[m.away_team] = raw_games.get(m.away_team, 0) + 1

    teams: dict[str, TeamStrength] = {}
    for team, eff_g in weight_sum.items():
        raw_scored_avg = scored[team] / eff_g
        raw_conceded_avg = conceded[team] / eff_g
        # shrink each team's (recency-weighted) per-game average toward the
        # league average, weighted by EFFECTIVE games actually behind it
        shrunk_scored_avg = (eff_g * raw_scored_avg + team_prior_games * avg_goals) / (eff_g + team_prior_games)
        shrunk_conceded_avg = (eff_g * raw_conceded_avg + team_prior_games * avg_goals) / (eff_g + team_prior_games)
        teams[team] = TeamStrength(
            team=team,
            games_played=raw_games[team],
            effective_games=round(eff_g, 2),
            attack=shrunk_scored_avg / avg_goals if avg_goals > 0 else 1.0,
            defense=shrunk_conceded_avg / avg_goals if avg_goals > 0 else 1.0,
        )

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
