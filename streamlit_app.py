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
        "Points":
