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
# -------------------------
# Main UI
# -------------------------
st.title("🎯 ParlayPlays • Player Prop Predictor")
st.caption("Predict player prop outcomes using historical performance and trends")
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    sport = st.selectbox("Sport", list(BASE.keys()), index=0)
    selected_date = st.date_input("Game Date", dt.date.today())
    stat_options = list(STAT_FIELDS[sport].keys())
    stat_type = st.selectbox("Stat to Predict", stat_options)

# Main content
if not sport or not selected_date:
    st.info("Please select a sport and date to get started.")
    st.stop()

# Fetch games for the selected date
date_str = ymd(selected_date)
games_endpoint = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(date=date_str)
if ENDPOINTS[sport]["season_param_style"] == "year":
    season = selected_date.year
    games_endpoint = games_endpoint.replace("{season}", str(season))

games = api_get(games_endpoint)

if not games:
    st.warning(f"No games found for {sport} on {selected_date}.")
    st.stop()

# Filter completed games only (optional)
completed_games = [g for g in games if safe_get(g, "Status") in ("Final", "Completed", "Closed")]
if not completed_games:
    st.warning("No completed games found for this date.")
else:
    games = completed_games

# Game selection
st.subheader("📍 Select Game")
game_options = {
    f"{safe_get(g, 'AwayTeam')} @ {safe_get(g, 'HomeTeam')} ({safe_get(g, 'StadiumName', 'Unknown Venue')})": g
    for g in games
}
selected_game_label = st.radio("Game", list(game_options.keys()), horizontal=True)
game = game_options[selected_game_label]

# Extract team keys
teams = infer_team_keys(sport, game)
home_team = teams["home_key"]
away_team = teams["home_display"]
home_display = teams["home_display"]
away_display = teams["away_display"]

st.markdown(f"<br><div class='card'>", unsafe_allow_html=True)
st.markdown(f"### 🏁 {away_display} @ {home_display}")
st.markdown(f"<small class='ghost'>Date: {selected_date} | Venue: {safe_get(game, 'StadiumName', 'Unknown')}</small>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Team selection for player
st.subheader("🎽 Select Team")
team_choice = st.radio("Team", [away_display, home_display], horizontal=True, key="team_choice")
team_key = away_team if team_choice == away_display else home_team

# Fetch players for the team
players_endpoint = BASE[sport]["stats"] + ENDPOINTS[sport]["players_by_team"].format(team=team_key)
players = api_get(players_endpoint)

if not players:
    st.warning(f"No players found for {team_choice}.")
    st.stop()

# Player selection
player_names = [f"{p.get('FirstName', '')} {p.get('LastName', '')}".strip() for p in players]
player_dict = {name: player for name, player in zip(player_names, players)}
selected_player_name = st.selectbox("Select Player", player_names)
player = player_dict[selected_player_name]

# Display player header
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class='card'>
        <h4>👤 {selected_player_name}</h4>
        <p><span class='pill'>{team_choice}</span> <span class='pill'>{safe_get(player, 'Position', 'N/A')}</span></p>
        <p class='ghost'>Jersey #{safe_get(player, 'Jersey', 'N/A')} | {safe_get(player, 'Height', 'N/A')} | {safe_get(player, 'Weight', 'N/A')} lbs</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch gamelogs for the player over recent dates
st.subheader("📊 Recent Performance")

field_name = STAT_FIELDS[sport][stat_type]
recent_days = recent_dates(window_days=30)
gamelogs = []

for log_date in recent_days:
    logs_url = BASE[sport]["stats"] + ENDPOINTS[sport]["player_gamelogs_by_date"].format(date=log_date)
    logs = api_get(logs_url)
    if logs:
        for log in logs:
            if safe_get(log, "PlayerID") == safe_get(player, "PlayerID"):
                val = safe_get(log, field_name)
                if val is not None and val >= 0:
                    gamelogs.append({
                        "Date": log_date,
                        "Opponent": f"{safe_get(log, 'Opponent')}",
                        "Home": safe_get(log, "IsHome") == True,
                        "StatValue": float(val)
                    })

if not gamelogs:
    st.warning(f"No recent data found for {selected_player_name} in '{stat_type}'.")
else:
    # Convert to DataFrame
    df = pd.DataFrame(gamelogs)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    # Show table
    st.dataframe(
        df[["Date", "Opponent", "StatValue"]]
        .rename(columns={"StatValue": stat_type})
        .style.format({"Date": lambda x: x.strftime("%b %d")}),
        use_container_width=True
    )

    # Show stats
    avg = df["StatValue"].mean()
    median = df["StatValue"].median()
    std = df["StatValue"].std()
    last_5 = df["StatValue"].head(5).mean() if len(df) >= 5 else df["StatValue"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg", f"{avg:.1f}")
    col2.metric("Median", f"{median:.1f}")
    col3.metric("Std Dev", f"{std:.1f}")
    col4.metric("Last 5 Games", f"{last_5:.1f}")

    # Simple "prediction"
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔮 Prediction")
    pred = (avg * 0.6) + (last_5 * 0.4)  # Weighted average
    st.markdown(f"### Expected {stat_type}: <span class='metric-good'>{pred:.1f}</span>", unsafe_allow_html=True)

    # Add prop comparison
    st.markdown("#### Compare to a Prop Line")
    prop_line = st.number_input("Enter prop line (e.g., 22.5)", value=22.5, step=0.5)
    if pred > prop_line:
        verdict = "✅ Over"
        color = "metric-good"
    else:
        verdict = "❌ Under"
        color = "metric-bad"
    st.markdown(f"**Recommendation: {verdict}** — <span class={color}>{pred:.1f} vs {prop_line}</span>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("ParlayPlays • Use data wisely. Not for gambling advice.")
