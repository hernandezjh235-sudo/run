import os
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# DEVIL PICKS — WNBA ONLY ENGINE
# Fixes: NBA players leaking into WNBA props, Streamlit warnings,
# player prop projection visibility, game markets separated.
# =========================================================

st.set_page_config(page_title="DEVIL PICKS WNBA", page_icon="🏀", layout="wide")

APP_DIR = Path(".")
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TRACKER_FILE = DATA_DIR / "wnba_bet_tracker.csv"
SNAPSHOT_FILE = DATA_DIR / "wnba_line_snapshots.csv"
HISTORY_FILE = DATA_DIR / "wnba_edge_history.csv"

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
ODDS_SPORT_KEY = "basketball_wnba"
PRIZEPICKS_URL = "https://api.prizepicks.com/projections"
UNDERDOG_URLS = [
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/v1/over_under_lines",
]

WNBA_TEAM_ALIASES = {
    "ATL", "CHI", "CON", "DAL", "IND", "LAS", "LA", "LVA", "MIN", "NYL", "NY", "PHX", "SEA", "WAS", "GS", "GSV", "Valkyries".upper()
}
WNBA_TEAM_NAMES = {
    "Atlanta Dream", "Chicago Sky", "Connecticut Sun", "Dallas Wings", "Indiana Fever",
    "Los Angeles Sparks", "Las Vegas Aces", "Minnesota Lynx", "New York Liberty",
    "Phoenix Mercury", "Seattle Storm", "Washington Mystics", "Golden State Valkyries"
}
NBA_TEAM_NAMES = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
    "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
    "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
}

PROP_KEYWORDS = {
    "Points": ["points", "pts"],
    "Rebounds": ["rebounds", "reb"],
    "Assists": ["assists", "ast"],
    "Pts+Reb+Ast": ["pts+reb+ast", "pra", "points + rebounds + assists"],
    "3-Pointers Made": ["3-pointers", "three", "threes", "3pt", "3pm"],
    "Steals": ["steals", "stl"],
    "Blocks": ["blocks", "blk"],
    "Turnovers": ["turnovers", "to"],
}

# -------------------- UI STYLE --------------------
st.markdown(
    """
<style>
.stApp { background: radial-gradient(circle at top, #240014 0%, #09090d 44%, #020204 100%); color: #f8fafc; }
.block-container { padding-top: 1.0rem; max-width: 1550px; }
h1,h2,h3 { color: #ffffff; }
.card { border: 1px solid rgba(255,255,255,.10); background: linear-gradient(145deg, rgba(255,255,255,.08), rgba(255,255,255,.03)); border-radius: 20px; padding: 18px; box-shadow: 0 18px 40px rgba(0,0,0,.32); }
.good { color:#39ff88; font-weight:900; }
.warn { color:#ffd166; font-weight:900; }
.bad { color:#ff4d6d; font-weight:900; }
.small { color:#a7b0c0; font-size: 13px; }
.big-title { font-size: 42px; font-weight: 950; letter-spacing: -1px; background: linear-gradient(90deg,#ff3b7f,#ffd166,#7cf7ff); -webkit-background-clip: text; color: transparent; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); margin-right:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- HELPERS --------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def ev_from_prob_and_american(prob: float, odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    odds = float(odds)
    profit = 100 / abs(odds) if odds < 0 else odds / 100
    return prob * profit - (1 - prob)


def kelly_fraction(prob: float, odds: Optional[float], cap: float = 0.04) -> float:
    if odds is None:
        return 0.0
    b = 100 / abs(odds) if odds < 0 else odds / 100
    q = 1 - prob
    k = (b * prob - q) / b if b > 0 else 0
    return clamp(k, 0, cap)


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def is_wnba_text(text: Any) -> bool:
    t = normalize_text(text).lower()
    if not t:
        return False
    return any(token in t for token in ["wnba", "women", "basketball_wnba"])


def is_nba_text(text: Any) -> bool:
    t = normalize_text(text).lower()
    if not t:
        return False
    if "wnba" in t:
        return False
    return any(token in t for token in [" nba", "basketball_nba", "national basketball association"])


def looks_like_nba_team(text: Any) -> bool:
    t = normalize_text(text).lower()
    return any(team.lower() in t for team in NBA_TEAM_NAMES)


def looks_like_wnba_team(text: Any) -> bool:
    t = normalize_text(text).lower()
    return any(team.lower() in t for team in WNBA_TEAM_NAMES)


def classify_prop(stat_type: str) -> str:
    s = normalize_text(stat_type).lower()
    for label, keys in PROP_KEYWORDS.items():
        if any(k in s for k in keys):
            return label
    return stat_type or "Prop"


def width_kwargs(stretch: bool = True) -> Dict[str, str]:
    return {"width": "stretch" if stretch else "content"}


@st.cache_data(ttl=300, show_spinner=False)
def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], str]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:250]}"
        return r.json(), "OK"
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"


def get_secret_or_env(name: str, fallback: str = "") -> str:
    try:
        val = st.secrets.get(name, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, fallback)


# -------------------- WNBA GAME MARKETS --------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_wnba_odds(api_key: str, regions: str, markets: str, odds_format: str) -> Tuple[pd.DataFrame, str]:
    if not api_key:
        return pd.DataFrame(), "No Odds API key entered. Add ODDS_API_KEY in Streamlit Secrets/sidebar for live ML/spread/totals."
    url = f"{ODDS_API_BASE}/{ODDS_SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    js, status = http_get_json(url, params=params)
    if js is None:
        return pd.DataFrame(), status
    rows = []
    for game in js:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if looks_like_nba_team(home) or looks_like_nba_team(away):
            continue
        if not (looks_like_wnba_team(home) or looks_like_wnba_team(away) or is_wnba_text(str(game))):
            # The Odds API sport key should already be WNBA, but keep strict guard.
            continue
        commence = game.get("commence_time", "")
        books = game.get("bookmakers", []) or []
        for book in books:
            book_title = book.get("title", "")
            for mkt in book.get("markets", []) or []:
                key = mkt.get("key", "")
                for out in mkt.get("outcomes", []) or []:
                    rows.append({
                        "game_id": game.get("id"),
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book_title,
                        "market": key,
                        "name": out.get("name"),
                        "price": safe_float(out.get("price")),
                        "point": safe_float(out.get("point")),
                    })
    return pd.DataFrame(rows), "OK"


def build_game_consensus(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return pd.DataFrame()
    rows = []
    for (gid, home, away, start), g in odds_df.groupby(["game_id", "home_team", "away_team", "commence_time"], dropna=False):
        row = {"game_id": gid, "commence_time": start, "home_team": home, "away_team": away}
        h2h = g[g["market"] == "h2h"]
        spreads = g[g["market"] == "spreads"]
        totals = g[g["market"] == "totals"]
        for team, prefix in [(home, "home"), (away, "away")]:
            prices = h2h[h2h["name"] == team]["price"].dropna()
            if not prices.empty:
                row[f"{prefix}_ml"] = float(prices.median())
                probs = [american_to_prob(x) for x in prices]
                probs = [p for p in probs if p is not None]
                row[f"{prefix}_imp_prob"] = float(np.mean(probs)) if probs else None
        if "home_imp_prob" in row and "away_imp_prob" in row:
            total_imp = row["home_imp_prob"] + row["away_imp_prob"]
            if total_imp > 0:
                row["home_fair_prob"] = row["home_imp_prob"] / total_imp
                row["away_fair_prob"] = row["away_imp_prob"] / total_imp
        home_sp = spreads[spreads["name"] == home]["point"].dropna()
        away_sp = spreads[spreads["name"] == away]["point"].dropna()
        if not home_sp.empty:
            row["home_spread"] = float(home_sp.median())
        if not away_sp.empty:
            row["away_spread"] = float(away_sp.median())
        over_lines = totals[totals["name"].astype(str).str.lower().eq("over")]["point"].dropna()
        if not over_lines.empty:
            row["total"] = float(over_lines.median())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # simple model from market consensus; not fake certainty
    out["model_home_prob"] = out.get("home_fair_prob", pd.Series([0.5] * len(out))).fillna(0.5).apply(lambda p: clamp(float(p), 0.35, 0.65))
    out["model_away_prob"] = 1 - out["model_home_prob"]
    out["ml_pick"] = np.where(out["model_home_prob"] >= out["model_away_prob"], out["home_team"], out["away_team"])
    out["ml_prob"] = out[["model_home_prob", "model_away_prob"]].max(axis=1)
    out["ml_signal"] = np.where(out["ml_prob"] >= 0.58, "✅ LEAN ML", np.where(out["ml_prob"] >= 0.62, "😈 STRONG ML", "PASS"))
    # spread/total signals are conservative without power ratings
    out["spread_pick"] = np.where(out.get("home_spread", 0).fillna(0) <= 0, out["home_team"], out["away_team"])
    out["spread_signal"] = np.where(out["ml_prob"] >= 0.60, "✅ LEAN SPREAD", "PASS")
    out["total_pick"] = "WATCH"
    out["total_signal"] = "PASS"
    return out


# -------------------- WNBA PLAYER PROPS --------------------
def parse_prizepicks_wnba(js: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(js, dict):
        return rows
    included = js.get("included", []) or []
    players: Dict[str, Dict[str, Any]] = {}
    leagues: Dict[str, Dict[str, Any]] = {}
    teams: Dict[str, Dict[str, Any]] = {}
    for item in included:
        if not isinstance(item, dict):
            continue
        typ = item.get("type", "")
        attrs = item.get("attributes", {}) or {}
        iid = item.get("id")
        if typ == "new_player":
            players[str(iid)] = attrs
        elif typ == "league":
            leagues[str(iid)] = attrs
        elif typ in {"team", "new_team"}:
            teams[str(iid)] = attrs
    for proj in js.get("data", []) or []:
        if not isinstance(proj, dict):
            continue
        attrs = proj.get("attributes", {}) or {}
        rel = proj.get("relationships", {}) or {}
        league_id = (((rel.get("league") or {}).get("data") or {}).get("id"))
        league_attrs = leagues.get(str(league_id), {})
        league_name = " ".join([str(league_attrs.get(k, "")) for k in ["name", "abbr", "display_name"]])
        raw_blob = json.dumps(proj)[:2000]
        if not (is_wnba_text(league_name) or is_wnba_text(raw_blob)):
            continue
        if is_nba_text(league_name) or is_nba_text(raw_blob):
            continue
        player_id = (((rel.get("new_player") or rel.get("player") or {}).get("data") or {}).get("id"))
        p = players.get(str(player_id), {})
        player_name = p.get("display_name") or p.get("name") or attrs.get("description") or "Unknown"
        team = p.get("team") or p.get("team_name") or attrs.get("team") or ""
        stat = attrs.get("stat_type") or attrs.get("stat_display_name") or attrs.get("name") or "Prop"
        line = safe_float(attrs.get("line_score") or attrs.get("line"))
        if line is None:
            continue
        rows.append({
            "source": "PrizePicks",
            "league": "WNBA",
            "player": player_name,
            "team": team,
            "opponent": attrs.get("opponent", ""),
            "prop": classify_prop(stat),
            "stat_type": stat,
            "line": line,
            "over_price": None,
            "under_price": None,
            "start_time": attrs.get("start_time") or attrs.get("game_time") or "",
            "raw_league": league_name,
        })
    return rows


def parse_underdog_wnba(js: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(js, dict):
        return rows
    players: Dict[str, Dict[str, Any]] = {}
    appearances: Dict[str, Dict[str, Any]] = {}
    games: Dict[str, Dict[str, Any]] = {}
    over_under_lines = js.get("over_under_lines") or js.get("data") or []
    for p in js.get("players", []) or []:
        players[str(p.get("id"))] = p
    for a in js.get("appearances", []) or []:
        appearances[str(a.get("id"))] = a
    for g in js.get("games", []) or []:
        games[str(g.get("id"))] = g
    # Some versions use included objects
    for inc in js.get("included", []) or []:
        if not isinstance(inc, dict):
            continue
        typ = inc.get("type", "")
        attrs = inc.get("attributes", {}) or {}
        iid = str(inc.get("id"))
        if "player" in typ:
            players[iid] = attrs
        elif "appearance" in typ:
            appearances[iid] = attrs
        elif "game" in typ:
            games[iid] = attrs

    for line_obj in over_under_lines or []:
        if not isinstance(line_obj, dict):
            continue
        attrs = line_obj.get("attributes", line_obj) or {}
        rel = line_obj.get("relationships", {}) or {}
        raw_blob = json.dumps(line_obj)[:2500]
        if is_nba_text(raw_blob) and not is_wnba_text(raw_blob):
            continue
        # Underdog often includes sport_id/league_id text in title/over_under
        if not (is_wnba_text(raw_blob) or "wnba" in raw_blob.lower()):
            # Strict filter: skip if we cannot prove WNBA.
            continue
        appearance_id = line_obj.get("appearance_id") or attrs.get("appearance_id")
        if not appearance_id:
            appearance_id = (((rel.get("appearance") or {}).get("data") or {}).get("id"))
        app = appearances.get(str(appearance_id), {})
        player_id = app.get("player_id") or (((app.get("relationships", {}) or {}).get("player", {}) or {}).get("data", {}) or {}).get("id")
        p = players.get(str(player_id), {})
        player_name = p.get("first_name", "") + " " + p.get("last_name", "")
        player_name = player_name.strip() or p.get("display_name") or p.get("name") or app.get("player_name") or attrs.get("title") or attrs.get("description") or "Unknown"
        team = app.get("team_id") or app.get("team") or p.get("team") or ""
        stat = attrs.get("over_under", {}).get("title") if isinstance(attrs.get("over_under"), dict) else None
        stat = stat or attrs.get("stat_type") or attrs.get("title") or attrs.get("display_stat") or "Prop"
        line = safe_float(attrs.get("stat_value") or attrs.get("line") or attrs.get("line_score"))
        if line is None:
            options = attrs.get("options") or line_obj.get("options") or []
            for opt in options:
                line = safe_float(opt.get("line") or opt.get("stat_value") or opt.get("line_score"))
                if line is not None:
                    break
        if line is None:
            continue
        rows.append({
            "source": "Underdog",
            "league": "WNBA",
            "player": player_name,
            "team": team,
            "opponent": app.get("opponent_id") or "",
            "prop": classify_prop(stat),
            "stat_type": stat,
            "line": line,
            "over_price": None,
            "under_price": None,
            "start_time": attrs.get("scheduled_at") or attrs.get("start_time") or "",
            "raw_league": "WNBA-confirmed",
        })
    return rows


@st.cache_data(ttl=240, show_spinner=False)
def fetch_wnba_props() -> Tuple[pd.DataFrame, List[str]]:
    logs: List[str] = []
    rows: List[Dict[str, Any]] = []
    pp, pp_status = http_get_json(PRIZEPICKS_URL)
    logs.append(f"PrizePicks: {pp_status}")
    if pp is not None:
        parsed = parse_prizepicks_wnba(pp)
        logs.append(f"PrizePicks WNBA props parsed: {len(parsed)}")
        rows.extend(parsed)
    for url in UNDERDOG_URLS:
        ud, status = http_get_json(url)
        logs.append(f"Underdog {url.split('/')[2]}: {status}")
        if ud is not None:
            parsed = parse_underdog_wnba(ud)
            logs.append(f"Underdog WNBA props parsed: {len(parsed)}")
            rows.extend(parsed)
            if parsed:
                break
    df = pd.DataFrame(rows)
    if df.empty:
        return df, logs
    # final hard guard: keep only WNBA, remove anything that clearly says NBA or NBA team
    for col in ["player", "team", "opponent", "raw_league", "stat_type"]:
        if col not in df.columns:
            df[col] = ""
    blob = df[["player", "team", "opponent", "raw_league", "stat_type"]].astype(str).agg(" ".join, axis=1)
    mask = ~blob.apply(lambda x: is_nba_text(x) or looks_like_nba_team(x))
    df = df[mask].copy()
    df = df.drop_duplicates(subset=["source", "player", "prop", "line"], keep="first")
    return df.reset_index(drop=True), logs


# -------------------- PROP PROJECTION ENGINE --------------------
def simulate_prop_projection(line: float, prop: str, market_adj: float = 0.0) -> Dict[str, Any]:
    # Since public WNBA player game logs are not guaranteed in this lightweight app,
    # this uses a transparent market/simulation fallback instead of fake player logs.
    prop = prop or "Prop"
    base_sd_map = {
        "Points": 4.8,
        "Rebounds": 3.0,
        "Assists": 2.4,
        "Pts+Reb+Ast": 6.5,
        "3-Pointers Made": 1.2,
        "Steals": 1.0,
        "Blocks": 0.9,
        "Turnovers": 1.3,
    }
    sd = base_sd_map.get(prop, max(1.0, line * 0.22))
    # Conservative projection: close to market line with small information adjustment.
    projection = float(line + market_adj)
    sims = np.random.default_rng(abs(hash((line, prop, int(time.time() // 300)))) % (2**32)).normal(projection, sd, 6000)
    over_prob = float(np.mean(sims > line))
    under_prob = 1.0 - over_prob
    edge = projection - line
    if over_prob >= 0.58 and edge >= 0.35:
        pick = "TAKE OVER"
        grade = "😈 STRONG"
        prob = over_prob
    elif under_prob >= 0.58 and edge <= -0.35:
        pick = "TAKE UNDER"
        grade = "😈 STRONG"
        prob = under_prob
    elif over_prob >= 0.53 and edge >= 0.15:
        pick = "WATCH OVER"
        grade = "✅ LEAN"
        prob = over_prob
    elif under_prob >= 0.53 and edge <= -0.15:
        pick = "WATCH UNDER"
        grade = "✅ LEAN"
        prob = under_prob
    else:
        pick = "NO PLAY — no edge"
        grade = "PASS"
        prob = max(over_prob, under_prob)
    return {
        "projection": round(projection, 2),
        "edge": round(edge, 2),
        "over_prob": round(over_prob, 3),
        "under_prob": round(under_prob, 3),
        "pick": pick,
        "grade": grade,
        "confidence": round(prob * 100, 1),
        "sim_sd": round(sd, 2),
    }


def enrich_props(df: pd.DataFrame, manual_adjustment: float) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = []
    for _, r in df.iterrows():
        line = safe_float(r.get("line"), 0.0) or 0.0
        out = simulate_prop_projection(line, r.get("prop", "Prop"), manual_adjustment)
        rec = r.to_dict()
        rec.update(out)
        enriched.append(rec)
    out = pd.DataFrame(enriched)
    order = {"😈 STRONG": 0, "✅ LEAN": 1, "PASS": 2}
    out["sort_grade"] = out["grade"].map(order).fillna(9)
    out = out.sort_values(["sort_grade", "confidence", "edge"], ascending=[True, False, False]).drop(columns=["sort_grade"])
    return out


# -------------------- TRACKING --------------------
def append_csv(path: Path, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# -------------------- SIDEBAR --------------------
st.sidebar.title("🏀 WNBA Controls")
default_key = get_secret_or_env("ODDS_API_KEY", "")
odds_api_key = st.sidebar.text_input("Odds API Key", value=default_key, type="password", help="Use Streamlit Secrets: ODDS_API_KEY = 'your_key'")
regions = st.sidebar.multiselect("Odds regions", ["us", "us2", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Game markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])
bankroll = st.sidebar.number_input("Bankroll", min_value=10.0, value=1000.0, step=50.0)
manual_prop_adj = st.sidebar.slider("Manual prop projection adjustment", -3.0, 3.0, 0.0, 0.1, help="Use for injuries/lineups/minutes boost. Positive favors Over; negative favors Under.")
refresh = st.sidebar.button("Refresh data")
if refresh:
    st.cache_data.clear()

st.markdown('<div class="big-title">DEVIL PICKS — WNBA ONLY</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">Strict WNBA prop filter</span><span class="pill">No NBA player leak</span><span class="pill">ML / Spread / Totals separate</span><span class="pill">Player props separate</span>', unsafe_allow_html=True)

odds_df, odds_status = fetch_wnba_odds(odds_api_key, ",".join(regions), ",".join(markets), "american")
games_df = build_game_consensus(odds_df)
props_raw, prop_logs = fetch_wnba_props()
props_df = enrich_props(props_raw, manual_prop_adj)

# save edge history lightly
if not props_df.empty:
    hist = props_df.head(50).copy()
    hist["timestamp_utc"] = now_utc().isoformat()
    if HISTORY_FILE.exists():
        old = read_csv(HISTORY_FILE)
        combined = pd.concat([old, hist], ignore_index=True).tail(10000)
        combined.to_csv(HISTORY_FILE, index=False)
    else:
        hist.to_csv(HISTORY_FILE, index=False)

# -------------------- TOP METRICS --------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("WNBA Games", len(games_df) if not games_df.empty else 0)
c2.metric("WNBA Props", len(props_df) if not props_df.empty else 0)
c3.metric("Strong Props", int((props_df.get("grade", pd.Series(dtype=str)) == "😈 STRONG").sum()) if not props_df.empty else 0)
c4.metric("Status", "LIVE" if (not odds_df.empty or not props_df.empty) else "DIAG")

(tab_best, tab_games, tab_ml, tab_spreads, tab_totals, tab_props, tab_raw, tab_tracker, tab_clv, tab_diag) = st.tabs([
    "🔥 Best Bets", "🏀 Games", "💰 Moneyline", "📏 Spreads", "📊 Totals", "🎯 Player Props", "🧾 Raw Props", "📓 Tracker", "📈 CLV", "🛠 Diagnostics"
])

with tab_best:
    st.subheader("🔥 Best WNBA Bets")
    if not props_df.empty:
        top = props_df[props_df["grade"].isin(["😈 STRONG", "✅ LEAN"])].head(25)
        if top.empty:
            st.info("No strong prop edges right now. That is good — the app is passing instead of forcing weak plays.")
        else:
            st.dataframe(top[["grade", "pick", "player", "team", "prop", "line", "projection", "edge", "over_prob", "under_prob", "confidence", "source"]], **width_kwargs())
    else:
        st.warning("No WNBA player props loaded. Check Diagnostics. The app is blocking NBA props by design.")
    if not games_df.empty:
        st.markdown("### Game-market leans")
        st.dataframe(games_df[[c for c in ["commence_time", "away_team", "home_team", "ml_pick", "ml_prob", "ml_signal", "spread_pick", "spread_signal", "total", "total_signal"] if c in games_df.columns]], **width_kwargs())

with tab_games:
    st.subheader("🏀 WNBA Games")
    if games_df.empty:
        st.info(odds_status)
    else:
        st.dataframe(games_df, **width_kwargs())

with tab_ml:
    st.subheader("💰 Moneyline — WNBA")
    if games_df.empty:
        st.info("No WNBA moneyline data loaded. Add Odds API key or wait for markets.")
    else:
        cols = [c for c in ["commence_time", "away_team", "home_team", "away_ml", "home_ml", "away_fair_prob", "home_fair_prob", "ml_pick", "ml_prob", "ml_signal"] if c in games_df.columns]
        st.dataframe(games_df[cols], **width_kwargs())

with tab_spreads:
    st.subheader("📏 Spreads — WNBA")
    if games_df.empty:
        st.info("No WNBA spread data loaded.")
    else:
        cols = [c for c in ["commence_time", "away_team", "home_team", "away_spread", "home_spread", "spread_pick", "spread_signal"] if c in games_df.columns]
        st.dataframe(games_df[cols], **width_kwargs())

with tab_totals:
    st.subheader("📊 Total Points — WNBA")
    if games_df.empty or "total" not in games_df.columns:
        st.info("No WNBA totals loaded.")
    else:
        cols = [c for c in ["commence_time", "away_team", "home_team", "total", "total_pick", "total_signal"] if c in games_df.columns]
        st.dataframe(games_df[cols], **width_kwargs())

with tab_props:
    st.subheader("🎯 WNBA Player Props — Over/Under Projection")
    if props_df.empty:
        st.warning("No confirmed WNBA props. NBA props are intentionally filtered out so wrong players do not show.")
    else:
        prop_choices = sorted(props_df["prop"].dropna().unique().tolist())
        source_choices = sorted(props_df["source"].dropna().unique().tolist())
        f1, f2, f3 = st.columns([1,1,2])
        prop_filter = f1.multiselect("Prop type", prop_choices, default=prop_choices)
        source_filter = f2.multiselect("Source", source_choices, default=source_choices)
        search = f3.text_input("Search player/team")
        show = props_df[props_df["prop"].isin(prop_filter) & props_df["source"].isin(source_filter)].copy()
        if search:
            s = search.lower()
            show = show[show.astype(str).agg(" ".join, axis=1).str.lower().str.contains(s, na=False)]
        st.dataframe(show[["grade", "pick", "player", "team", "prop", "line", "projection", "edge", "over_prob", "under_prob", "confidence", "sim_sd", "source"]], **width_kwargs())
        st.caption("Projection is intentionally conservative. It gives TAKE OVER/UNDER only when the sim edge clears the gate.")

with tab_raw:
    st.subheader("🧾 Raw WNBA Props")
    if props_raw.empty:
        st.info("No raw WNBA props parsed.")
    else:
        st.dataframe(props_raw, **width_kwargs())

with tab_tracker:
    st.subheader("📓 Manual Bet Tracker")
    with st.form("track_bet_form"):
        pick_text = st.text_input("Bet / pick")
        market_text = st.selectbox("Market", ["Player Prop", "Moneyline", "Spread", "Total"])
        odds_text = st.number_input("Odds", value=-110, step=5)
        stake = st.number_input("Stake", value=10.0, min_value=0.0, step=1.0)
        note = st.text_input("Notes")
        submitted = st.form_submit_button("Save bet")
        if submitted and pick_text:
            append_csv(TRACKER_FILE, {
                "timestamp_utc": now_utc().isoformat(), "league": "WNBA", "pick": pick_text,
                "market": market_text, "odds": odds_text, "stake": stake, "note": note,
                "result": "PENDING"
            })
            st.success("Saved.")
    tracker = read_csv(TRACKER_FILE)
    if not tracker.empty:
        st.dataframe(tracker.tail(500), **width_kwargs())

with tab_clv:
    st.subheader("📈 Opening / Closing Line Tracking")
    st.caption("Use snapshots before games and after lines move to track CLV. Full auto grading depends on available final scores/settlement data.")
    if st.button("Save current WNBA line snapshot"):
        snap = []
        if not odds_df.empty:
            tmp = odds_df.copy()
            tmp["timestamp_utc"] = now_utc().isoformat()
            snap.append(tmp)
        if not props_df.empty:
            tmp = props_df.copy()
            tmp["market"] = "player_prop"
            tmp["timestamp_utc"] = now_utc().isoformat()
            snap.append(tmp)
        if snap:
            out = pd.concat(snap, ignore_index=True, sort=False)
            if SNAPSHOT_FILE.exists():
                old = read_csv(SNAPSHOT_FILE)
                out = pd.concat([old, out], ignore_index=True).tail(10000)
            out.to_csv(SNAPSHOT_FILE, index=False)
            st.success("Snapshot saved.")
        else:
            st.warning("Nothing to snapshot.")
    snaps = read_csv(SNAPSHOT_FILE)
    if not snaps.empty:
        st.dataframe(snaps.tail(1000), **width_kwargs())

with tab_diag:
    st.subheader("🛠 Diagnostics")
    st.write("Game odds status:", odds_status)
    st.write("Prop source logs:")
    for log in prop_logs:
        st.code(log)
    st.markdown("### Strict WNBA filters")
    st.write("This app skips any prop blob that looks like NBA and does not explicitly confirm WNBA.")
    st.write("Streamlit warning cleanup: uses `width='stretch'` instead of deprecated `use_container_width`.")
    if props_raw.empty:
        st.error("No confirmed WNBA props parsed. This usually means sources are empty, endpoint changed, or no WNBA board is posted yet.")
    else:
        st.success(f"Confirmed WNBA props parsed: {len(props_raw)}")
