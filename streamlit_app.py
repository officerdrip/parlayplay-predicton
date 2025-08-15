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
      .stButton>button {
          background:#16c79a;
          color:white;
          border:none;
          padding:10px 14px;
          border-radius:8px;
          font-weight:600;
          width:100%;
      }
      .stButton>button:hover {
          background:#13a883;
          color:white;
          border:none;
      }
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
    "NBA": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
    "WNBA": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
    "NFL": {"games_by_date": "/SchedulesBasic/{season}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
    "MLB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/Players/{team}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
    "CFB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/PlayersByTeam/{teamid}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
    "CBB": {"games_by_date": "/GamesByDate/{date}", "players_by_team": "/PlayersByTeam/{teamid}", "player_gamelogs_by_date": "/PlayerGameStatsByDate/{date}"},
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
            st.warning("Endpoint not found. Check if this sport is available in your plan.")
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

def recent_dates(window_days: int) -> List[str]:
    today = dt.date.today()
    return [ymd(today - dt.timedelta(days=i)) for i in range(window_days)]

def infer_team_keys(sport: str, game: Dict) -> Dict:
    if sport in ("NBA", "WNBA", "NFL", "MLB"):
        return {"home_display": safe_get(game,"HomeTeam") or safe_get(game,"HomeTeamName"), "away_display": safe_get(game,"AwayTeam") or safe_get(game,"AwayTeamName"), "home_key": safe_get(game,"HomeTeam"), "away_key": safe_get(game,"AwayTeam")}
    return {"home_display": safe_get(game,"HomeTeam") or safe_get(game,"HomeTeamName"), "away_display": safe_get(game,"AwayTeam") or safe_get(game,"AwayTeamName"), "home_key": safe_get(game,"HomeTeamID") or safe_get(game,"HomeTeam"), "away_key": safe_get(game,"AwayTeamID") or safe_get(game,"AwayTeam")}

def player_name(rec: Dict) -> str:
    for k in ("Name","FullName","FirstName"):
        if k in rec and rec[k]:
            name = rec[k] if k != "FirstName" else f"{rec['FirstName']} {rec.get('LastName','')}".strip()
            pos = f" ({rec.get('Position')})" if rec.get('Position') else ""
            return f"{name}{pos}"
    return rec.get("Player","Unknown")

def extract_stat(rec: Dict, field: str) -> Optional[float]:
    v = rec.get(field)
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def prob_over(sample: List[float], line: float) -> float:
    if not sample: return 0.0
    arr = np.array(sample, dtype=float)
    return float(np.mean(arr > line))

@st.cache_data(show_spinner=False, ttl=300)
def get_player_gamelogs(sport: str, player_id: int, window_days: int) -> List[Dict]:
    """Fetches all game logs for a player over a given window of days."""
    logs = []
    dates_to_check = recent_dates(window_days)
    base = BASE[sport]["stats"]
    endpoint = ENDPOINTS[sport]["player_gamelogs_by_date"]

    for date_str in dates_to_check:
        url = base + endpoint.format(date=date_str)
        daily_stats = api_get(url)
        if daily_stats:
            player_stats = [s for s in daily_stats if s.get("PlayerID") == player_id]
            logs.extend(player_stats)
    return logs

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("### ⚙️ Controls")
sport = st.sidebar.selectbox("Sport", ["NBA", "WNBA", "NFL", "MLB", "CFB", "CBB"])
target_date = st.sidebar.date_input("Game Date", value=dt.date.today())
window_days = st.sidebar.slider("Recent Form Window (Days)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Store your SportsDataIO API key in `.streamlit/secrets.toml` as `SPORTSIO_KEY` to avoid entering it.")

# -------------------------
# Main App
# -------------------------
st.markdown("## 🎯 ParlayPlays • Player Prop Predictor")
st.markdown("<div class='ghost'>Select a sport, pick a game, choose a player & stat, then enter a line to estimate Over/Under probability.</div><br>", unsafe_allow_html=True)

# --- Step 1: Fetch and select a game ---
with st.spinner("Loading games..."):
    if "{date}" in ENDPOINTS[sport]["games_by_date"]:
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(date=ymd(target_date))
    else: # Handle season-based endpoints like NFL
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(season=dt.date.today().year)
    games = api_get(url) or []

if not games:
    st.info(f"No {sport} games found for {target_date.strftime('%b %d, %Y')}. Check the date or your API access.")
    st.stop()

def game_lab(game: Dict) -> str:
    keys = infer_team_keys(sport, game)
    return f"{keys.get('away_display', 'Away')} @ {keys.get('home_display', 'Home')}"

c1, c2, c3, c4 = st.columns(4)
selected_game = c1.selectbox("Select Game", games, format_func=game_lab, label_visibility="collapsed")

if selected_game:
    # --- Step 2: Fetch and select a player ---
    with st.spinner(f"Loading players for {game_lab(selected_game)}..."):
        team_keys = infer_team_keys(sport, selected_game)
        players = []
        for team_key in [team_keys["home_key"], team_keys["away_key"]]:
            if team_key:
                endpoint = ENDPOINTS[sport]["players_by_team"]
                url = BASE[sport]["scores"] + endpoint.format(team=team_key, teamid=team_key)
                team_players = api_get(url)
                if team_players:
                    players.extend(team_players)
    
    # Sort players by name for easier selection
    players = sorted(players, key=lambda p: player_name(p))
    selected_player = c2.selectbox("Select Player", players, format_func=player_name, label_visibility="collapsed", index=None, placeholder="Select Player")

    if selected_player:
        # --- Step 3: Select a stat and enter a line ---
        stat_options = list(STAT_FIELDS.get(sport, {}).keys())
        selected_stat_display = c3.selectbox("Select Stat", stat_options, label_visibility="collapsed", index=None, placeholder="Select Stat")
        
        line = c4.number_input("Enter Line", min_value=0.0, step=0.5, value=0.5, label_visibility="collapsed")
        
        # --- Step 4: Run Analysis ---
        st.write("") # Spacer
        run_analysis = st.button("📈 Run Analysis")

        if run_analysis and selected_stat_display and selected_player:
            player_id = selected_player.get("PlayerID")
            stat_field = STAT_FIELDS[sport][selected_stat_display]
            
            with st.spinner(f"Analyzing {player_name(selected_player)}'s recent performance..."):
                game_logs = get_player_gamelogs(sport, player_id, window_days)
                
                if not game_logs:
                    st.warning(f"No recent game logs found for {player_name(selected_player)} in the last {window_days} days.")
                    st.stop()

                # Process logs into a DataFrame
                log_data = []
                for log in game_logs:
                    stat_val = extract_stat(log, stat_field)
                    if stat_val is not None:
                        log_data.append({
                            "Date": log.get("Day", "").split("T")[0],
                            "Opponent": log.get("Opponent", "N/A"),
                            selected_stat_display: stat_val
                        })
                
                if not log_data:
                    st.warning(f"No data found for the stat '{selected_stat_display}'. The player may not have recorded this stat recently.")
                    st.stop()
                
                df = pd.DataFrame(log_data)
                df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)
                
                # --- Step 5: Display Results ---
                st.markdown(f"#### Analysis for {player_name(selected_player)} - **{selected_stat_display}**")
                
                res1, res2, res3 = st.columns(3)
                
                recent_stats = df[selected_stat_display].dropna().tolist()
                avg_stat = np.mean(recent_stats) if recent_stats else 0
                prob = prob_over(recent_stats, line)
                hit_rate_pct = f"{prob:.0%}"
                
                # Color code the probability metric
                prob_color = "good" if prob >= 0.55 else "bad" if prob <= 0.45 else "neutral"

                res1.metric(f"Avg (Last {len(recent_stats)} Games)", f"{avg_stat:.2f}")
                res2.markdown(f"<div class='card'><div class='ghost'>Prob. Over {line}</div> <div class='metric-{prob_color}'>{hit_rate_pct}</div></div>", unsafe_allow_html=True)
                res3.metric("Games Over Line", f"{sum(s > line for s in recent_stats)} of {len(recent_stats)}")
                
                st.markdown("---")
                st.dataframe(df, use_container_width=True)
