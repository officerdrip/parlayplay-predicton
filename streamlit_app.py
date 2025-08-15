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

# Endpoints
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

# Stat field mapping
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
    "WNBA": {  # same as NBA
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
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            st.warning("Rate limit reached. Please try again in a moment.")
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

def player_name(rec: Dict) -> str:
    for k in ("Name", "FullName", "FirstName"):
        if k in rec and rec[k]:
            return rec[k] if k != "FirstName" else f"{rec['FirstName']} {rec.get('LastName','')}".strip()
    return rec.get("Player", "Unknown")

def extract_stat(rec: Dict, field: str) -> Optional[float]:
    v = rec.get(field)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def prob_over(sample: List[float], line: float) -> float:
    if not sample:
        return 0.0
    arr = np.array(sample, dtype=float)
    return float(np.mean(arr > line))

# -------------------------
# Sidebar: Sport / Date
# -------------------------
st.sidebar.markdown("### Sport & Date")
sport = st.sidebar.selectbox("Sport", ["NBA", "WNBA", "NFL", "MLB", "CFB", "CBB"])
target_date = st.sidebar.date_input("Game Date", value=dt.date.today())
window_days = st.sidebar.slider("Recent form window (days)", 7, 60, 30, help="We’ll pull player game logs across this window.")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Store your SportsDataIO key in `.streamlit/secrets.toml` as `SPORTSIO_KEY`.")

# -------------------------
# Pull Games
# -------------------------
st.markdown("## 🎯 ParlayPlays • Player Prop Predictor")
st.markdown(
    "<div class='ghost'>Select a sport, pick a game, choose a player & stat, then enter a line to estimate Over/Under probability.</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading games..."):
    if "{date}" in ENDPOINTS[sport]["games_by_date"]:
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(date=ymd(target_date))
    else:
        url = BASE[sport]["scores"] + ENDPOINTS[sport]["games_by_date"].format(season=dt.date.today().year)
    games = api_get(url) or []

if not games:
    st.info("No games found for this selection. Try another date.")
    st.stop()

def game_label(g: Dict) -> str:
    home = safe_get(g, "HomeTeam") or safe_get(g, "HomeTeamName") or "HOME"
    away = safe_get(g, "AwayTeam") or safe_get(g, "AwayTeamName") or "AWAY"
    ts = safe_get(g, "DateTime") or safe_get(g, "Day") or safe_get(g, "Date")
    ts = str(ts)[:16].replace("T", " ")
    return f"{away} @ {home}  •  {ts}"

game_idx = st.selectbox("Upcoming Games", list(range(len(games))), format_func=lambda i: game_label(games[i]))
game = games[game_idx]
keys = infer_team_keys(sport, game)

# -------------------------
# Team → Players
# -------------------------
st.markdown("### Pick Team & Player")

col_team, col_player = st.columns(2)

with col_team:
    team_choice = st.radio(
        "Team",
        [f"{keys['away_display']} (Away)", f"{keys['home_display']} (Home)"]
    )
team_key = keys["away_key"] if "Away" in team_choice else keys["home_key"]

players_endpoint = ENDPOINTS[sport]["players_by_team"]
if "{teamid}" in players_endpoint:
    url_players = BASE[sport]["scores"] + players_endpoint.format(teamid=team_key)
else:
    url_players = BASE[sport]["scores"] + players_endpoint.format(team=team_key)

with st.spinner("Loading roster..."):
    roster = api_get(url_players) or []

if not roster:
    st.warning("No roster found for this team.")
    st.stop()

with col_player:
    player_idx = st.selectbox(
        "Player",
        list(range(len(roster))),
        format_func=lambda i: player_name(roster[i]),
        index=0,
    )
player = roster[player_idx]
player_id = player.get("PlayerID") or player.get("GlobalPlayerID") or player.get("ID")

# -------------------------
# Stat & Line
# -------------------------
st.markdown("### Choose Stat & Line")
valid_stats = STAT_FIELDS[sport]
stat_choice = st.selectbox("Stat", list(valid_stats.keys()))
stat_field = valid_stats[stat_choice]

line_value = st.number_input("Prop Line", value=20.5 if sport in ("NBA", "WNBA", "CBB") else 60.5, step=0.5)

# -------------------------
# Gather recent game logs
# -------------------------
st.markdown("### Results")

def collect_player_logs_by_date(sport: str, days: int) -> pd.DataFrame:
    rows = []
    for d in recent_dates(days):
        url_logs = BASE[sport]["stats"] + ENDPOINTS[sport]["player_gamelogs_by_date"].format(date=d)
        payload = api_get(url_logs) or []
        for rec in payload:
            pid = rec.get("PlayerID") or rec.get("GlobalPlayerID") or rec.get("ID")
            if pid == player_id:
                rows.append(rec)
    return pd.DataFrame(rows)

with st.spinner("Pulling recent game logs..."):
    df_logs = collect_player_logs_by_date(sport, window_days)

if df_logs.empty:
    st.warning("No recent logs found for this player in the selected window. Try expanding the window or pick another player/stat.")
    st.stop()

values = [extract_stat(r.to_dict(), stat_field) for _, r in df_logs.iterrows() if extract_stat(r.to_dict(), stat_field) is not None]

if not values:
    st.warning("This stat isn’t available for the selected player. Try another stat.")
    st.stop()

arr = np.array(values, dtype=float)
p_over = prob_over(values, line_value)
p_under = 1.0 - p_over

mean_v = float(np.mean(arr))
median_v = float(np.median(arr))
stdev_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

# -------------------------
# Display
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sample Size", len(arr))
c2.markdown(f"**Avg {stat_choice}**<br><span class='metric-good'>{mean_v:.2f}</span>", unsafe_allow_html=True)
c3.markdown(f"**Median**<br><span class='metric-good'>{median_v:.2f}</span>", unsafe_allow_html=True)
c4.markdown(f"**Std Dev**<br><span class='ghost'>{stdev_v:.2f}</span>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='card'>
      <div class='pill'>Prop: {stat_choice}</div>
      <div class='pill'>Line: {line_value:g}</div>
      <div class='pill'>Window: last {window_days} days</div>
      <h3 style='margin-top:10px;'>Prediction</h3>
      <div>Over Probability: <span class='metric-{"good" if p_over>=0.5 else "bad"}'>{p_over*100:.1f}%</span></div>
      <div>Under Probability: <span class='metric-{"good" if p_under>0.5 else "bad"}'>{p_under*100:.1f}%</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

hist_counts, bin_edges = np.histogram(arr, bins=min(12, max(6, int(np.sqrt(len(arr))))))
hist_lines = []
max_count = hist_counts.max() if hist_counts.size else 1
for c, (b0, b1) in zip(hist_counts, zip(bin_edges[:-1], bin_edges[1:])):
    bar = "█" * int((c / max_count) * 25)
    hist_lines.append(f"{b0:6.1f}–{b1:6.1f} | {bar} {c}")

with st.expander("Recent Distribution (text histogram)"):
    st.code("\n".join(hist_lines))

st.caption("Note: This quick model uses recent game logs only—no opponent/pace/odds adjustments. Extend the feature set as needed.")

# -------------------------
# Action row
# -------------------------
colA, colB = st.columns([1,1])
with colA:
    if st.button("🔁 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with colB:
    st.download_button(
        label="⬇️ Export Logs (CSV)",
        data=df_logs.to_csv(index=False),
        file_name=f"{sport}_{player_name(player)}_{stat_choice}_logs.csv",
        mime="text/csv",
        use_container_width=True
    )
