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
# CSS polish
# -------------------------
st.markdown(
    """
    <style>
      .metric-good {color:#16c79a; font-weight:700;}
      .metric-bad {color:#e94560; font-weight:700;}
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

# -------------------------
# Base URLs + Endpoints
# -------------------------
BASE = {
    "NBA": {"scores": "https://api.sportsdata.io/v3/nba/scores/json", "stats": "https://api.sportsdata.io/v3/nba/stats/json"},
    "WNBA": {"scores": "https://api.sportsdata.io/v3/wnba/scores/json", "stats": "https://api.sportsdata.io/v3/wnba/stats/json"},
    "NFL": {"scores": "https://api.sportsdata.io/v3/nfl/scores/json", "stats": "https://api.sportsdata.io/v3/nfl/stats/json"},
    "MLB": {"scores": "https://api.sportsdata.io/v3/mlb/scores/json", "stats": "https://api.sportsdata.io/v3/mlb/stats/json"},
    "CFB": {"scores": "https://api.sportsdata.io/v3/cfb/scores/json", "stats": "https://api.sportsdata.io/v3/cfb/stats/json"},
    "CBB": {"scores": "https://api.sportsdata.io/v3/cbb/scores/json", "stats": "https://api.sportsdata.io/v3/cbb/stats/json"},
}

ENDPOINTS = {
    "NBA": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
    "WNBA": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
    "NFL": {"games_by_date": "/SchedulesBasic/{season}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
    "MLB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
    "CFB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/PlayersByTeam/{teamid}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
    "CBB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/PlayersByTeam/{teamid}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}", "season_param_style": "year"},
}

STAT_FIELDS = {
    "NBA": {"Points": "Points","Rebounds": "Rebounds","Assists": "Assists","FG Made": "FieldGoalsMade","3PT Made": "ThreePointersMade","Blocks": "BlockedShots","Steals": "Steals"},
    "WNBA": {"Points": "Points","Rebounds": "Rebounds","Assists": "Assists","FG Made": "FieldGoalsMade","3PT Made": "ThreePointersMade","Blocks": "BlockedShots","Steals": "Steals"},
    "NFL": {"Passing Yds": "PassingYards","Rushing Yds": "RushingYards","Receiving Yds": "ReceivingYards","Receptions": "Receptions","Rush Attempts": "RushingAttempts"},
    "MLB": {"Hits": "Hits","Total Bases": "TotalBases","Home Runs": "HomeRuns","Runs": "Runs","RBIs": "RunsBattedIn","Strikeouts (Batter)": "Strikeouts","Strikeouts (Pitcher)": "PitcherStrikeouts"},
    "CFB": {"Passing Yds": "PassingYards","Rushing Yds": "RushingYards","Receiving Yds": "ReceivingYards","Receptions": "Receptions"},
    "CBB": {"Points": "Points","Rebounds": "Rebounds","Assists": "Assists","3PT Made": "ThreePointersMade"},
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
        return {"home_display": safe_get(game,"HomeTeam") or safe_get(game,"HomeTeamName"), "away_display": safe_get(game,"AwayTeam") or safe_get(game,"AwayTeamName"), "home_key": safe_get(game,"HomeTeam"), "away_key": safe_get(game,"AwayTeam")}
    return {"home_display": safe_get(game,"HomeTeam") or safe_get(game,"HomeTeamName"), "away_display": safe_get(game,"AwayTeam") or safe_get(game,"AwayTeamName"), "home_key": safe_get(game,"HomeTeamID") or safe_get(game,"HomeTeam"), "away_key": safe_get(game,"AwayTeamID") or safe_get(game,"AwayTeam")}

def player_name(rec: Dict) -> str:
    for k in ("Name","FullName","FirstName"):
        if k in rec and rec[k]:
            return rec[k] if k!="FirstName" else f"{rec['FirstName']} {rec.get('LastName','')}".strip()
    return rec.get("Player","Unknown")

def extract_stat(rec: Dict, field: str) -> Optional[float]:
    v = rec.get(field)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def prob_over(sample: List[float], line: float) -> float:
    if not sample: return 0.0
    arr = np.array(sample,dtype=float)
    return float(np.mean(arr > line))

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("### Sport & Date")
sport = st.sidebar.selectbox("Sport", ["NBA","WNBA","NFL","MLB","CFB","CBB"])
target_date = st.sidebar.date_input("Game Date", value=dt.date.today())
window_days = st.sidebar.slider("Recent form window (days)",7,60,30)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Store your SportsDataIO key in `.streamlit/secrets.toml` as `SPORTSIO_KEY`.")

# -------------------------
# Pull games
# -------------------------
st.markdown("## 🎯 ParlayPlays • Player Prop Predictor")
st.markdown("<div class='ghost'>Select a sport, pick a game, choose a player & stat, then enter a line to estimate Over/Under probability.</div>", unsafe_allow_html=True)

with st.spinner("Loading games..."):
    if "{date}" in ENDPOINTS[sport]["games_by_date"]:
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(date=ymd(target_date))
    else:
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(season=dt.date.today().year)
    games = api_get(url) or []

if not games:
    st.info("No games found. Check date or API access.")
    st.stop()

def game_lab(game: Dict) -> str:
    """Formats a game dictionary into a readable label for a selectbox."""
    keys = infer_team_keys(sport, game)
    away_team = keys.get("away_display", "Away")
    home_team = keys.get("home_display", "Home")
    return f"{away_team} @ {home_team}"

# --- The rest of your Streamlit app logic would continue here ---
# Example:
#
# selected_game = st.selectbox("Select a game", games, format_func=game_lab)
# if selected_game:
#     # ... logic to fetch players for the selected game ...
