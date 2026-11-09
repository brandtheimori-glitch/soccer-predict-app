"""
Data layer. Two interchangeable sources behind the same interface:

- LiveDataSource: talks to API-Football (api-sports.io / RapidAPI). This is
  the free-tier API worth using for a real prototype — unlike
  football-data.org's free tier (which only exposes a competition list),
  API-Football's free plan (100 req/day) includes fixtures, standings,
  head-to-head, odds and stats for every competition, just with limited
  historical seasons. Needs an API key: https://dashboard.api-football.com
- SampleDataSource: a small bundled dataset so the app runs and is
  demonstrable with zero setup. Swap DATA_SOURCE below (or set
  API_FOOTBALL_KEY) to go live.

A data source's job is just to hand back raw facts — fixtures, finished
results, odds. It does NOT compute attack/defense strength or home
advantage anymore; that's fit_parameters.fit_league_parameters(), fed by
historical_matches() below, so the same fitting logic runs identically
whether the results came from the bundled sample or a live API call.
This also means going live no longer needs a per-team statistics call —
fitting from the raw match list one API call already fetched saves quota.

Because the free tier is capped at 100 requests/day, LiveDataSource caches
every response to disk (cache/) for CACHE_TTL_HOURS so a day of casual
browsing doesn't blow the quota re-fetching the same fixtures.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from fit_parameters import HistoricalMatch
from model import OddsSnapshot

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_TTL_HOURS = 6
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_BASE = "https://v3.football.api-sports.io"


@dataclass
class Fixture:
    fixture_id: str
    league: str
    date: str  # ISO date
    home_team: str
    away_team: str
    status: str  # "NS" not started, "FT" full time, etc.
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None


class DataSource:
    """Interface both sources implement."""

    def upcoming_fixtures(self, league: str, limit: int = 10) -> list[Fixture]:
        raise NotImplementedError

    def finished_fixtures_needing_grade(self, fixture_ids: list[str]) -> list[Fixture]:
        raise NotImplementedError

    def odds(self, fixture_id: str) -> Optional[OddsSnapshot]:
        raise NotImplementedError

    def historical_matches(self, league: str) -> list[HistoricalMatch]:
        """Finished results used to FIT model parameters (home advantage,
        team attack/defense). Should not overlap with history_fixtures()
        below, so backtested predictions aren't made by a model that was
        fit on the very match it's predicting."""
        return []

    def history_fixtures(self, league: str) -> list[Fixture]:
        """Already-finished fixtures — held OUT of historical_matches() —
        usable to seed/demo the track record and value-bet backtest by
        predicting them with the fitted model and grading against the real
        result. Optional; a live deployment doesn't need this, since real
        history accumulates naturally as genuine advance predictions get
        graded over time."""
        return []

    def available_leagues(self) -> list[dict]:
        """[{'id': <value passed elsewhere as `league`>, 'name': <display name>}, ...] for a picker."""
        return []


# ---------------------------------------------------------------------------
# Sample data source — zero setup, deterministic, clearly labeled as sample.
# Three leagues so the picker has something real to switch between. Each
# league has three tiers of match data:
#   fit_matches    -> only used to fit home advantage + team attack/defense
#   seed_fixtures  -> finished matches held OUT of fitting, predicted with
#                     the fitted model and graded, to demo the track record
#                     and value-bet backtest without waiting for real time
#                     to pass
#   fixtures       -> upcoming (not started) fixtures shown as predictions
# ---------------------------------------------------------------------------

_LEAGUES = {
    "Premier League": {
        "fit_matches": [
            # (home, away, home_goals, away_goals, date) -- dates spread over ~two months
            # before the "seed" fixtures so recency weighting has something real to bite on
            ("Arsenal", "Man United", 3, 1, "2026-07-05"), ("Man City", "West Ham", 2, 0, "2026-07-05"),
            ("Liverpool", "Chelsea", 2, 1, "2026-07-12"), ("Newcastle", "Brighton", 1, 1, "2026-07-12"),
            ("Aston Villa", "Tottenham", 2, 1, "2026-07-19"), ("Chelsea", "Arsenal", 1, 2, "2026-07-19"),
            ("West Ham", "Liverpool", 0, 2, "2026-07-26"), ("Tottenham", "Newcastle", 2, 2, "2026-07-26"),
            ("Man United", "Man City", 1, 3, "2026-08-02"), ("Brighton", "Aston Villa", 1, 1, "2026-08-02"),
            ("Arsenal", "Newcastle", 2, 0, "2026-08-09"), ("Man City", "Chelsea", 3, 1, "2026-08-09"),
            ("West Ham", "Tottenham", 1, 1, "2026-08-16"), ("Aston Villa", "Brighton", 1, 0, "2026-08-16"),
            ("Chelsea", "Aston Villa", 2, 1, "2026-08-23"), ("Man United", "Arsenal", 0, 1, "2026-08-23"),
            ("Tottenham", "Man City", 1, 3, "2026-08-30"), ("Brighton", "West Ham", 2, 1, "2026-08-30"),
        ],
        "seed_fixtures": [
            ("p0a", "2026-09-06", "Arsenal", "Chelsea", "FT", 2, 1),
            ("p0b", "2026-09-06", "Liverpool", "Man United", "FT", 3, 1),
            ("p0c", "2026-09-07", "Newcastle", "West Ham", "FT", 1, 1),
            ("p0d", "2026-09-07", "Man City", "Tottenham", "FT", 4, 1),
            ("p0e", "2026-09-07", "Brighton", "Chelsea", "FT", 0, 2),
        ],
        "fixtures": [
            ("p1", "2026-09-13", "Arsenal", "Tottenham", "NS", None, None),
            ("p2", "2026-09-13", "Man City", "Chelsea", "NS", None, None),
            ("p3", "2026-09-14", "Liverpool", "Newcastle", "NS", None, None),
            ("p4", "2026-09-14", "West Ham", "Man United", "NS", None, None),
            ("p5", "2026-09-14", "Brighton", "Aston Villa", "NS", None, None),
        ],
        "odds": {
            "p0a": (1.75, 3.80, 4.50), "p0b": (1.45, 4.60, 6.50), "p0c": (2.20, 3.30, 3.40),
            "p0d": (1.30, 5.50, 9.00), "p0e": (3.60, 3.40, 2.00),
            "p1": (1.85, 3.60, 4.20), "p2": (1.55, 4.20, 5.50), "p3": (1.50, 4.40, 6.00),
            "p4": (3.40, 3.30, 2.10), "p5": (2.30, 3.30, 3.00),
        },
    },
    "La Liga": {
        "fit_matches": [
            ("Real Madrid", "Real Sociedad", 3, 0, "2026-07-12"), ("Barcelona", "Girona", 2, 2, "2026-07-12"),
            ("Atletico Madrid", "Sevilla", 2, 0, "2026-07-19"), ("Athletic Bilbao", "Valencia", 1, 1, "2026-07-19"),
            ("Villarreal", "Betis", 1, 2, "2026-07-26"), ("Real Sociedad", "Villarreal", 1, 1, "2026-07-26"),
            ("Sevilla", "Athletic Bilbao", 0, 1, "2026-08-02"), ("Girona", "Atletico Madrid", 1, 2, "2026-08-02"),
            ("Real Madrid", "Barcelona", 2, 1, "2026-08-09"), ("Betis", "Real Sociedad", 1, 0, "2026-08-09"),
            ("Valencia", "Sevilla", 0, 0, "2026-08-16"), ("Atletico Madrid", "Real Madrid", 1, 1, "2026-08-16"),
            ("Barcelona", "Villarreal", 3, 0, "2026-08-23"), ("Athletic Bilbao", "Girona", 2, 1, "2026-08-23"),
        ],
        "seed_fixtures": [
            ("l0a", "2026-09-06", "Real Madrid", "Betis", "FT", 2, 0),
            ("l0b", "2026-09-06", "Barcelona", "Valencia", "FT", 3, 1),
            ("l0c", "2026-09-07", "Atletico Madrid", "Villarreal", "FT", 1, 1),
        ],
        "fixtures": [
            ("l1", "2026-09-13", "Real Madrid", "Villarreal", "NS", None, None),
            ("l2", "2026-09-13", "Barcelona", "Sevilla", "NS", None, None),
            ("l3", "2026-09-14", "Atletico Madrid", "Betis", "NS", None, None),
            ("l4", "2026-09-14", "Athletic Bilbao", "Valencia", "NS", None, None),
        ],
        "odds": {
            "l0a": (1.35, 4.80, 8.00), "l0b": (1.30, 5.50, 9.00), "l0c": (1.75, 3.60, 4.40),
            "l1": (1.40, 4.80, 7.50), "l2": (1.35, 5.20, 8.00),
            "l3": (1.60, 3.90, 5.20), "l4": (2.10, 3.40, 3.40),
        },
    },
    "Serie A": {
        "fit_matches": [
            ("Inter", "Torino", 2, 0, "2026-07-12"), ("Napoli", "Lazio", 1, 1, "2026-07-12"),
            ("AC Milan", "Fiorentina", 2, 1, "2026-07-19"), ("Juventus", "Bologna", 1, 0, "2026-07-19"),
            ("Atalanta", "Roma", 2, 2, "2026-07-26"), ("Torino", "Napoli", 0, 1, "2026-07-26"),
            ("Lazio", "AC Milan", 1, 2, "2026-08-02"), ("Fiorentina", "Juventus", 0, 1, "2026-08-02"),
            ("Bologna", "Atalanta", 1, 1, "2026-08-09"), ("Roma", "Inter", 0, 2, "2026-08-09"),
            ("Inter", "Napoli", 1, 1, "2026-08-16"), ("AC Milan", "Juventus", 2, 0, "2026-08-16"),
            ("Lazio", "Torino", 2, 1, "2026-08-23"), ("Atalanta", "Fiorentina", 3, 1, "2026-08-23"),
        ],
        "seed_fixtures": [
            ("s0ai", "2026-09-06", "Inter", "Atalanta", "FT", 2, 1),
            ("s0bi", "2026-09-06", "Roma", "Napoli", "FT", 0, 1),
            ("s0ci", "2026-09-07", "Juventus", "Fiorentina", "FT", 2, 1),
        ],
        "fixtures": [
            ("s1i", "2026-09-13", "Inter", "Torino", "NS", None, None),
            ("s2i", "2026-09-13", "Napoli", "Lazio", "NS", None, None),
            ("s3i", "2026-09-14", "AC Milan", "Fiorentina", "NS", None, None),
            ("s4i", "2026-09-14", "Juventus", "Bologna", "NS", None, None),
        ],
        "odds": {
            "s0ai": (1.40, 4.60, 7.00), "s0bi": (2.60, 3.20, 2.60), "s0ci": (1.70, 3.70, 4.60),
            "s1i": (1.35, 4.60, 8.50), "s2i": (1.70, 3.70, 4.60),
            "s3i": (1.65, 3.80, 4.80), "s4i": (1.55, 3.90, 5.80),
        },
    },
    "Bundesliga": {
        "fit_matches": [
            ("Bayern Munich", "Werder Bremen", 4, 0, "2026-07-12"), ("Leverkusen", "Wolfsburg", 2, 1, "2026-07-12"),
            ("RB Leipzig", "Union Berlin", 2, 0, "2026-07-19"), ("Dortmund", "Freiburg", 3, 1, "2026-07-19"),
            ("Stuttgart", "Frankfurt", 1, 1, "2026-07-26"), ("Werder Bremen", "Leverkusen", 0, 2, "2026-07-26"),
            ("Wolfsburg", "RB Leipzig", 1, 2, "2026-08-02"), ("Union Berlin", "Dortmund", 0, 2, "2026-08-02"),
            ("Freiburg", "Stuttgart", 1, 1, "2026-08-09"), ("Frankfurt", "Bayern Munich", 1, 3, "2026-08-09"),
            ("Bayern Munich", "Leverkusen", 2, 1, "2026-08-16"), ("RB Leipzig", "Dortmund", 1, 1, "2026-08-16"),
            ("Stuttgart", "Wolfsburg", 2, 0, "2026-08-23"), ("Union Berlin", "Frankfurt", 0, 1, "2026-08-23"),
        ],
        "seed_fixtures": [
            ("b0a", "2026-09-06", "Bayern Munich", "Dortmund", "FT", 3, 1),
            ("b0b", "2026-09-06", "Leverkusen", "RB Leipzig", "FT", 1, 1),
            ("b0c", "2026-09-07", "Frankfurt", "Freiburg", "FT", 2, 0),
        ],
        "fixtures": [
            ("b1", "2026-09-13", "Bayern Munich", "Stuttgart", "NS", None, None),
            ("b2", "2026-09-13", "RB Leipzig", "Frankfurt", "NS", None, None),
            ("b3", "2026-09-14", "Dortmund", "Wolfsburg", "NS", None, None),
            ("b4", "2026-09-14", "Leverkusen", "Union Berlin", "NS", None, None),
        ],
        "odds": {
            "b0a": (1.40, 4.80, 7.00), "b0b": (1.75, 3.70, 4.60), "b0c": (1.65, 3.80, 5.00),
            "b1": (1.30, 5.20, 9.00), "b2": (1.60, 3.90, 5.00),
            "b3": (1.45, 4.40, 6.50), "b4": (1.55, 3.90, 5.80),
        },
    },
    "Ligue 1": {
        "fit_matches": [
            ("PSG", "Toulouse", 4, 0, "2026-07-12"), ("Monaco", "Reims", 2, 0, "2026-07-12"),
            ("Marseille", "Lens", 1, 1, "2026-07-19"), ("Lille", "Nice", 2, 1, "2026-07-19"),
            ("Lyon", "Rennes", 1, 1, "2026-07-26"), ("Toulouse", "Monaco", 0, 2, "2026-07-26"),
            ("Reims", "Marseille", 0, 2, "2026-08-02"), ("Lens", "Lille", 1, 1, "2026-08-02"),
            ("Nice", "Lyon", 1, 0, "2026-08-09"), ("Rennes", "PSG", 0, 3, "2026-08-09"),
            ("PSG", "Monaco", 2, 1, "2026-08-16"), ("Marseille", "Lille", 1, 1, "2026-08-16"),
            ("Lyon", "Lens", 2, 0, "2026-08-23"), ("Nice", "Rennes", 1, 1, "2026-08-23"),
        ],
        "seed_fixtures": [
            ("f0a", "2026-09-06", "PSG", "Marseille", "FT", 3, 1),
            ("f0b", "2026-09-06", "Monaco", "Lyon", "FT", 2, 2),
            ("f0c", "2026-09-07", "Lille", "Nice", "FT", 1, 0),
        ],
        "fixtures": [
            ("f1", "2026-09-13", "PSG", "Lens", "NS", None, None),
            ("f2", "2026-09-13", "Marseille", "Rennes", "NS", None, None),
            ("f3", "2026-09-14", "Monaco", "Toulouse", "NS", None, None),
            ("f4", "2026-09-14", "Lyon", "Reims", "NS", None, None),
        ],
        "odds": {
            "f0a": (1.35, 5.00, 8.00), "f0b": (1.70, 3.80, 4.60), "f0c": (1.60, 3.80, 5.20),
            "f1": (1.25, 5.80, 10.00), "f2": (1.90, 3.50, 4.00),
            "f3": (1.50, 4.00, 6.00), "f4": (1.55, 3.90, 5.60),
        },
    },
}


def _fixtures_from(rows: list, league: str) -> list[Fixture]:
    return [Fixture(fid, league, date, home, away, status, hg, ag) for fid, date, home, away, status, hg, ag in rows]


class SampleDataSource(DataSource):
    label = "sample data (bundled demo, not live)"

    def available_leagues(self) -> list[dict]:
        return [{"id": name, "name": name} for name in _LEAGUES]

    def upcoming_fixtures(self, league: str, limit: int = 10) -> list[Fixture]:
        return _fixtures_from(_LEAGUES[league]["fixtures"], league)[:limit]

    def finished_fixtures_needing_grade(self, fixture_ids: list[str]) -> list[Fixture]:
        all_seed = [f for league in _LEAGUES for f in _fixtures_from(_LEAGUES[league]["seed_fixtures"], league)]
        return [f for f in all_seed if f.fixture_id in fixture_ids and f.status == "FT"]

    def odds(self, fixture_id: str) -> Optional[OddsSnapshot]:
        for league in _LEAGUES.values():
            if fixture_id in league["odds"]:
                h, d, a = league["odds"][fixture_id]
                return OddsSnapshot(home=h, draw=d, away=a)
        return None

    def historical_matches(self, league: str) -> list[HistoricalMatch]:
        return [HistoricalMatch(home, away, hg, ag, date) for home, away, hg, ag, date in _LEAGUES[league]["fit_matches"]]

    def history_fixtures(self, league: str) -> list[Fixture]:
        return _fixtures_from(_LEAGUES[league]["seed_fixtures"], league)


# ---------------------------------------------------------------------------
# Live data source — API-Football, cached to respect the 100 req/day cap.
# ---------------------------------------------------------------------------

# API-Football's numeric league IDs for the handful of competitions covered
# by the free plan's typical usage pattern. Find more at
# https://www.api-football.com/documentation-v3#tag/Leagues
_LIVE_LEAGUES = [
    {"id": "39", "name": "Premier League"},
    {"id": "140", "name": "La Liga"},
    {"id": "135", "name": "Serie A"},
    {"id": "78", "name": "Bundesliga"},
    {"id": "61", "name": "Ligue 1"},
]


class LiveDataSource(DataSource):
    label = "live via API-Football"

    def __init__(self, api_key: str):
        self.api_key = api_key
        CACHE_DIR.mkdir(exist_ok=True)

    def available_leagues(self) -> list[dict]:
        return _LIVE_LEAGUES

    def _get(self, path: str, params: dict) -> dict:
        cache_key = CACHE_DIR / (path.replace("/", "_") + "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items())) + ".json")
        if cache_key.exists():
            age_hours = (time.time() - cache_key.stat().st_mtime) / 3600
            if age_hours < CACHE_TTL_HOURS:
                return json.loads(cache_key.read_text())

        resp = requests.get(
            f"{API_BASE}/{path}",
            headers={"x-apisports-key": self.api_key},
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        cache_key.write_text(json.dumps(data))
        return data

    def upcoming_fixtures(self, league: str, limit: int = 10) -> list[Fixture]:
        data = self._get("fixtures", {"league": league, "next": limit, "season": 2026})
        out = []
        for item in data.get("response", []):
            f = item["fixture"]
            teams = item["teams"]
            out.append(Fixture(
                fixture_id=str(f["id"]), league=league, date=f["date"][:10],
                home_team=teams["home"]["name"], away_team=teams["away"]["name"],
                status=f["status"]["short"],
            ))
        return out

    def finished_fixtures_needing_grade(self, fixture_ids: list[str]) -> list[Fixture]:
        out = []
        for fid in fixture_ids:
            data = self._get("fixtures", {"id": fid})
            for item in data.get("response", []):
                f, teams, goals = item["fixture"], item["teams"], item["goals"]
                if f["status"]["short"] == "FT":
                    out.append(Fixture(
                        fixture_id=str(f["id"]), league="", date=f["date"][:10],
                        home_team=teams["home"]["name"], away_team=teams["away"]["name"],
                        status="FT", home_goals=goals["home"], away_goals=goals["away"],
                    ))
        return out

    def historical_matches(self, league: str) -> list[HistoricalMatch]:
        # API-Football's free plan limits how many past seasons you can pull,
        # but "last N finished fixtures" for the current season is available
        # and is exactly what fit_parameters.py needs — one cached call
        # covers every team in the league at once (no per-team stats calls).
        data = self._get("fixtures", {"league": league, "season": 2026, "last": 40})
        out = []
        for item in data.get("response", []):
            f, teams, goals = item["fixture"], item["teams"], item["goals"]
            if f["status"]["short"] == "FT" and goals["home"] is not None:
                out.append(HistoricalMatch(
                    home_team=teams["home"]["name"], away_team=teams["away"]["name"],
                    home_goals=goals["home"], away_goals=goals["away"], date=f["date"][:10],
                ))
        return out

    def odds(self, fixture_id: str) -> Optional[OddsSnapshot]:
        data = self._get("odds", {"fixture": fixture_id})
        try:
            bookmakers = data["response"][0]["bookmakers"]
            prices = {"Home": [], "Draw": [], "Away": []}
            for bm in bookmakers:
                for bet in bm["bets"]:
                    if bet["name"] == "Match Winner":
                        for val in bet["values"]:
                            prices[val["value"]].append(float(val["odd"]))
            return OddsSnapshot(
                home=sum(prices["Home"]) / len(prices["Home"]),
                draw=sum(prices["Draw"]) / len(prices["Draw"]),
                away=sum(prices["Away"]) / len(prices["Away"]),
            )
        except (KeyError, IndexError, ZeroDivisionError):
            return None


# ---------------------------------------------------------------------------
# Live data source — football-data.org (recommended free option).
# ---------------------------------------------------------------------------
# Free plan: real fixtures + results for the CURRENT season of the top
# competitions, 10 requests/minute, no card. Sign up (email only) at
# https://www.football-data.org/client/register — the key arrives by email.
# It has no betting odds on the free plan, which is fine for the
# both-teams-to-score focus: BTTS comes straight from the goal model.
#
# One call per competition returns the whole season (finished AND upcoming
# matches), so a single cached request covers fitting, upcoming predictions
# and grading for that league.

FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
FOOTBALL_DATA_CACHE_TTL_MINUTES = 30  # results land within the hour; keeps well under 10 req/min

_FD_LEAGUES = [
    {"id": "PL", "name": "Premier League"},
    {"id": "PD", "name": "La Liga"},
    {"id": "SA", "name": "Serie A"},
    {"id": "BL1", "name": "Bundesliga"},
    {"id": "FL1", "name": "Ligue 1"},
    {"id": "DED", "name": "Eredivisie"},
    {"id": "ELC", "name": "Championship"},
    {"id": "PPL", "name": "Primeira Liga"},
    {"id": "BSA", "name": "Brasileirão"},  # plays through the European international breaks
]  # 9 leagues -> at most 9 calls on a cold cache, under the 10/min free cap
UPCOMING_DAYS = 14  # show every fixture in the next two weeks, not just the next round
_FD_UPCOMING = {"SCHEDULED", "TIMED"}


def _fd_team(team: dict) -> str:
    return team.get("shortName") or team.get("name") or "Unknown"


class FootballDataSource(DataSource):
    label = "live via football-data.org"

    def __init__(self, api_key: str):
        self.api_key = api_key
        CACHE_DIR.mkdir(exist_ok=True)

    def available_leagues(self) -> list[dict]:
        return _FD_LEAGUES

    def _matches(self, league: str) -> list[dict]:
        """Every match of the competition's current season (cached)."""
        cache_file = CACHE_DIR / f"fd_{league}_matches.json"
        if cache_file.exists():
            age_min = (time.time() - cache_file.stat().st_mtime) / 60
            if age_min < FOOTBALL_DATA_CACHE_TTL_MINUTES:
                return json.loads(cache_file.read_text()).get("matches", [])
        try:
            resp = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/{league}/matches",
                headers={"X-Auth-Token": self.api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            # rate-limited / network blip: serve the stale cache rather than crash
            if cache_file.exists():
                return json.loads(cache_file.read_text()).get("matches", [])
            raise
        cache_file.write_text(json.dumps(data))
        return data.get("matches", [])

    @staticmethod
    def _full_time(m: dict) -> tuple[Optional[int], Optional[int]]:
        ft = (m.get("score") or {}).get("fullTime") or {}
        return ft.get("home"), ft.get("away")

    def _to_fixture(self, m: dict, league: str) -> Fixture:
        hg, ag = self._full_time(m)
        finished = m.get("status") == "FINISHED"
        return Fixture(
            fixture_id=f"fd{m['id']}", league=league, date=(m.get("utcDate") or "")[:10],
            home_team=_fd_team(m.get("homeTeam", {})), away_team=_fd_team(m.get("awayTeam", {})),
            status="FT" if finished else ("NS" if m.get("status") in _FD_UPCOMING else m.get("status", "")),
            home_goals=hg if finished else None, away_goals=ag if finished else None,
        )

    def upcoming_fixtures(self, league: str, limit: int = 10) -> list[Fixture]:
        horizon = (datetime.now(timezone.utc) + timedelta(days=UPCOMING_DAYS)).strftime("%Y-%m-%d")
        upcoming = [m for m in self._matches(league)
                    if m.get("status") in _FD_UPCOMING and (m.get("utcDate") or "")[:10] <= horizon]
        upcoming.sort(key=lambda m: m.get("utcDate", ""))
        return [self._to_fixture(m, league) for m in upcoming[:limit]]

    def historical_matches(self, league: str) -> list[HistoricalMatch]:
        out = []
        for m in self._matches(league):
            hg, ag = self._full_time(m)
            if m.get("status") == "FINISHED" and hg is not None and ag is not None:
                out.append(HistoricalMatch(
                    home_team=_fd_team(m.get("homeTeam", {})), away_team=_fd_team(m.get("awayTeam", {})),
                    home_goals=hg, away_goals=ag, date=(m.get("utcDate") or "")[:10],
                ))
        return out

    def finished_fixtures_needing_grade(self, fixture_ids: list[str]) -> list[Fixture]:
        wanted = {fid for fid in fixture_ids if fid.startswith("fd")}
        if not wanted:
            return []
        out = []
        for lg in _FD_LEAGUES:
            if not (CACHE_DIR / f"fd_{lg['id']}_matches.json").exists():
                continue  # never predicted anything there yet -> nothing to grade
            try:
                matches = self._matches(lg["id"])
            except requests.RequestException:
                continue
            for m in matches:
                fx = self._to_fixture(m, lg["id"])
                if fx.fixture_id in wanted and fx.status == "FT" and fx.home_goals is not None:
                    out.append(fx)
        return out

    def odds(self, fixture_id: str) -> Optional[OddsSnapshot]:
        return None  # not on football-data.org's free plan


def get_data_source() -> DataSource:
    if FOOTBALL_DATA_KEY:
        return FootballDataSource(FOOTBALL_DATA_KEY)
    if API_FOOTBALL_KEY:
        return LiveDataSource(API_FOOTBALL_KEY)
    return SampleDataSource()
