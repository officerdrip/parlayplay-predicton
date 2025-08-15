# streamlit_app.py
import os
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------------
# Page Config + Theme
# -------------------------
st.set_page_config(
    page_title="ParlayPlays • Player Prop Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Simple CSS polish
# -------------------------
st.markdown(
    """
    <style>
      .metric-good {color:#16c79a; font-weight:700;}
      .metric-bad  {color:#e94560; font-weight:700;}
      .pill {
        display:inline-block; padding:6px 10px; border-radius:999px;
        background:#1f2937; border:1px solid #334155; font-size:12px; color:#e5e7eb;
        margin-right:6px;
      }
      .card {
        background:#0f172a; border:1px solid #1f2937; border-radius:14px; padding:16px;
      }
      .ghost { color:#9ca3af; }
      .btn { background:#16c79a; color:white; border:none; padding:10px 14px; border-radius:8px; font-weight:600;}
      .btn:hover { background:#13a883; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Config / Secrets
# -------------------------
API_KEY = st.secrets.get("SPORTSIO_KEY", os.getenv("SPORTSIO_KEY", ""))
if not API_KEY:
    st.error("Missing API key. Add SPORTSIO_KEY to `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

HEADERS = {"Ocp-Apim-Subscription-Key": API_KEY}

# Base URLs per league
BASE = {
    "NBA": {"scores": "https://api.sportsdata.io/v3/nba/scores/json", "stats": "https://api.sportsdata.io/v3/nba/stats/json"},
    "WNBA": {"scores": "https://api.sportsdata.io/v3/wnba/scores/json", "stats": "https://api.sportsdata.io/v3/wnba/stats/json"},
    "NFL": {"scores": "https://api.sportsdata.io/v3/nfl/scores/json", "stats": "https://api.sportsdata.io/v3/nfl/stats/json"},
    "MLB": {"scores": "https://api.sportsdata.io/v3/mlb/scores/json", "stats": "https://api.sportsdata.io/v3/mlb/stats/json"},
    "CFB": {"scores": "https://api.sportsdata.io/v3/cfb/scores/json", "stats": "https://api.sportsdata.io/v3/cfb/stats/json"},
    "CBB": {"scores": "https://api.sportsdata.io/v3/cbb/scores/json", "stats": "https://api.sportsdata.io/v3/cbb/stats/json"},
}

# Endpoints per league
ENDPOINTS = {
    "NBA": {
        "games_by_date": "/GamesByDate/{date}",
        "players_by_team": "/Players/{team}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
    "WNBA": {
        "games_by_date": "/GamesByDate/{date}",
        "players_by_team": "/Players/{team}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
    "NFL": {
        "games_by_date": "/SchedulesBasic/{season}",
        "players_by_team": "/Players/{team}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
    "MLB": {
        "games_by_date": "/GamesByDate/{date}",
        "players_by_team": "/Players/{team}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
    "CFB": {
        "games_by_date": "/GamesByDate/{date}",
        "players_by_team": "/PlayersByTeam/{teamid}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
    "CBB": {
        "games_by_date": "/GamesByDate/{date}",
        "players_by_team": "/PlayersByTeam/{teamid}",
        "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}",
        "season_param_style": "year"
    },
}

# Stat fields per sport
STAT_FIELDS = {
    "NBA": {
        "Points": "Points",
        "Rebounds": "Rebounds",
        "Assists": "Assists",
        "FG Made": "FieldGoalsMade",
        "3PT Made": "ThreePointersMade",
        "Blocks": "BlockedShots",
        "Steals": "Steals",
    },
    "WNBA": {
        "Points": "Points",
        "Rebounds": "Rebounds",
        "Assists": "Assists",
        "FG Made": "FieldGoalsMade",
        "3PT Made": "ThreePointersMade",
        "Blocks": "BlockedShots",
        "Steals": "Steals",
    },
    "NFL": {
        "Passing Yds": "PassingYards",
        "Rushing Yds": "RushingYards",
        "Receiving Yds": "ReceivingYards",
        "Receptions": "Receptions",
        "Rush Attempts": "RushingAttempts",
    },
    "MLB": {
        "Hits": "Hits",
        "Total Bases": "TotalBases",
        "Home Runs": "HomeRuns",
        "Runs": "Runs",
        "RBIs": "RunsBattedIn",
        "Strikeouts (Batter)": "Strikeouts",
        "Strikeouts (Pitcher)": "PitcherStrikeouts"
    },
    "CFB": {
        "Passing Yds": "PassingYards",
        "Rushing Yds": "RushingYards",
        "Receiving Yds": "ReceivingYards",
        "Receptions": "Receptions",
    },
    "CBB": {
        "Points": "Points",
        "Rebounds": "Rebounds",
        "Assists": "Assists",
        "3PT Made": "ThreePointersMade",
    },
}

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False, ttl=300)
def api_get(url: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            st.warning("Unauthorized: Your API key does not have access to this league.")
        elif r.status_code == 404:
            st.warning("Endpoint not found. Check if WNBA is available in your plan.")
        elif r.status_code == 429:
            st.warning("Rate limit reached. Please try again shortly.")
        else:
            st.error(f"API error {r.status_code}: {r.text[:200]}")
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
    return None

def ymd(date: dt.date) -> str:
    return date.strftime("%Y-%m-%d")

def safe_get(dic: Dict, key: str, default=None):
    return dic.get(key, default) if isinstance(dic, dict) else default

def recent_dates(window_days: int = 30) -> List[str]:
    today = dt.date.today()
    return [ymd(today - dt.timedelta(days=i)) for i in range(window_days)]

def infer_team_keys(sport: str, game: Dict) -> Dict:
    if sport in ("NBA", "WNBA", "NFL", "MLB"):
        return {
            "home_display": safe_get(game, "HomeTeam") or safe_get(game, "HomeTeamName"),
            "away_display": safe_get(game, "AwayTeam") or safe_get(game, "AwayTeamName"),
            "home_key": safe_get(game, "HomeTeam"),
            "away_key": safe_get(game, "AwayTeam"),
        }
    return {
        "home_display": safe_get(game, "HomeTeam") or safe_get(game, "HomeTeamName"),
        "away_display": safe_get(game, "AwayTeam") or safe_get(game, "AwayTeamName"),
        "home_key": safe_get(game, "HomeTeamID") or safe_get(game, "HomeTeam"),
        "away_key": safe_get(game, "AwayTeamID") or safe_get(game, "AwayTeam"),
    }
