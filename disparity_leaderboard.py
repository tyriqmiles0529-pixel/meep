import pandas as pd
import numpy as np
import requests
import http.client
import json
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams

# -----------------------------
# CONFIG
# -----------------------------
API_KEY_SPORTS = "3ee00eb314b80853c6c77920c5bf74f7"  # SportsGameOdds API key
RAPIDAPI_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"  # NBA schedule API key
LEAGUE_ID = "NBA"
PLAYER_STATS = ["points_line", "rebounds_line", "assists_line"]
SEASON = "2025-26"
MIN_MINUTES = 15
RECENT_GAMES = 10
OPPONENT_ADJUST = 1.05
TOP_N_PLAYERS = 7
TOP_PER_CATEGORY = 5
DAYS_TO_FETCH = 3  # today + next 2 days

# -----------------------------
# STEP 1: FETCH UPCOMING NBA GAMES (API)
# -----------------------------
def fetch_upcoming_games(days=DAYS_TO_FETCH):
    today = datetime.today()
    all_games = []

    conn = http.client.HTTPSConnection("nba-schedule.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "nba-schedule.p.rapidapi.com"
    }

    for i in range(days):
        date_str = (today + timedelta(days=i)).strftime("%d-%m-%Y")
        conn.request("GET", f"/schedule?date={date_str}", headers=headers)
        res = conn.getresponse()
        try:
            data = json.loads(res.read())
        except:
            print(f"❌ Error parsing schedule for {date_str}")
            continue

        for game in data:
            all_games.append({
                "Date": date_str,
                "Home Team": game.get('home_team', ''),
                "Away Team": game.get('away_team', ''),
                "Time": game.get('time', '')
            })

    if not all_games:
        print("❌ No upcoming games found.")
    return all_games

# -----------------------------
# STEP 2: PREDICT PLAYER STATS
# -----------------------------
def predict_player_stats(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON)
        df = gamelog.get_data_frames()[0]
    except:
        return {"PTS":0,"REB":0,"AST":0,"MIN":0}
    
    df = df[['GAME_DATE','MATCHUP','MIN','PTS','REB','AST']]
    df = df[df['MIN'] >= MIN_MINUTES].head(RECENT_GAMES)
    if df.empty:
        return {"PTS":0,"REB":0,"AST":0,"MIN":0}
    
    weights = np.arange(1, len(df)+1)/np.sum(np.arange(1, len(df)+1))
    preds = {stat: np.dot(df[stat], weights)*OPPONENT_ADJUST for stat in ['PTS','REB','AST']}
    preds['MIN'] = np.mean(df['MIN'])
    return preds

# -----------------------------
# STEP 3: GET TOP PLAYERS PER TEAM
# -----------------------------
def get_team_id(team_name):
    all_teams = teams.get_teams()
    team_map = {team['full_name']: team['id'] for team in all_teams}
    return team_map.get(team_name, None)

def get_top_players(teams_list):
    all_players = players.get_active_players()
    top_players = []
    for team_name in teams_list:
        team_id = get_team_id(team_name)
        if not team_id:
            continue
        team_players = [p for p in all_players if p['team_id']==team_id]
        for p in team_players:
            stats = predict_player_stats(p['id'])
            p.update(stats)
        top_players.extend(sorted(team_players, key=lambda x: x['MIN'], reverse=True)[:TOP_N_PLAYERS])
    return top_players

# -----------------------------
# STEP 4: FETCH PLAYER PROPS
# -----------------------------
def fetch_player_props():
    url = f"https://api.sportsgameodds.com/v2/sports/player_props?league={LEAGUE_ID}"
    headers = {'X-Api-Key': API_KEY_SPORTS}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("❌ Error fetching player props")
        return pd.DataFrame()
    data = resp.json().get("players", [])
    return pd.DataFrame([{**{"player_name": p["name"]}, **{stat: p.get(stat,0) for stat in PLAYER_STATS}} for p in data])

# -----------------------------
# STEP 5: FETCH GAME LINES
# -----------------------------
def fetch_game_lines():
    url = f"https://api.sportsgameodds.com/v2/sports/game_lines?league={LEAGUE_ID}"
    headers = {'X-Api-Key': API_KEY_SPORTS}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("❌ Error fetching game lines")
        return pd.DataFrame()
    data = resp.json().get("games", [])
    return pd.DataFrame([{
        "home_team": g["home_team"]["name"],
        "away_team": g["away_team"]["name"],
        "moneyline_home": g["moneyline_home"],
        "moneyline_away": g["moneyline_away"],
        "spread_home": g["spread_home"],
        "spread_away": g["spread_away"],
        "total_points": g["over_under_total"]
    } for g in data])

# -----------------------------
# STEP 6: CALCULATE PLAYER DISPARITIES
# -----------------------------
def calculate_player_disparities(predictions, props):
    df_pred = pd.DataFrame(predictions)
    merged = pd.merge(df_pred, props, left_on="full_name", right_on="player_name", how="inner")
    for stat in PLAYER_STATS:
        disp_col = stat.replace("_line","_disparity")
        merged[disp_col] = abs(merged.get(stat.replace("_line","_pred"),0) - merged[stat])
    merged["total_disparity"] = merged[[s.replace("_line","_disparity") for s in PLAYER_STATS]].sum(axis=1)
    return merged

# -----------------------------
# STEP 7: CALCULATE GAME DISPARITIES
# -----------------------------
def calculate_game_disparities(predictions_df, games_df):
    disparities = []
    for _, game in games_df.iterrows():
        home_players = predictions_df[predictions_df['team_name']==game['home_team']].nlargest(TOP_N_PLAYERS,'PTS')
        away_players = predictions_df[predictions_df['team_name']==game['away_team']].nlargest(TOP_N_PLAYERS,'PTS')
        home_pts = home_players['PTS'].sum()
        away_pts = away_players['PTS'].sum()

        # Total points
        total_disp = abs((home_pts+away_pts) - game['total_points'])

        # Spread
        spread_disp = abs((home_pts-away_pts) - game['spread_home'])

        # Moneyline
        pred_prob = 1/(1+np.exp(-0.1*(home_pts-away_pts)))
        implied_prob = game['moneyline_home']/(game['moneyline_home']+game['moneyline_away']) if (game['moneyline_home']+game['moneyline_away']) !=0 else 0.5
        moneyline_disp = abs(pred_prob - implied_prob)

        disparities.append({
            "home_team": game['home_team'],
            "away_team": game['away_team'],
            "predicted_total_points": home_pts+away_pts,
            "total_points_disparity": total_disp,
            "predicted_point_diff": home_pts-away_pts,
            "spread_disparity": spread_disp,
            "predicted_home_win_prob": pred_prob,
            "implied_home_win_prob": implied_prob,
            "moneyline_disparity": moneyline_disp
        })
    return pd.DataFrame(disparities)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    # 1️⃣ Fetch upcoming games dynamically (today + next 2 days)
    games_list = fetch_upcoming_games()
    if not games_list:
        exit()
    teams_today = {team for game in games_list for team in [game['Home Team'], game['Away Team']]}
    print(f"Teams playing in the next {DAYS_TO_FETCH} days: {teams_today}")

    # 2️⃣ Get top players per team
    top_players = get_top_players(list(teams_today))
    for p in top_players:
        team_id = p.get('team_id')
        p['team_name'] = next((t['full_name'] for t in teams.get_teams() if t['id']==team_id), None)

    # 3️⃣ Player disparities
    player_props = fetch_player_props()
    player_disparities = calculate_player_disparities(top_players, player_props)

    # 4️⃣ Game disparities
    game_lines = fetch_game_lines()
    game_disparities = calculate_game_disparities(pd.DataFrame(top_players), game_lines)

    # 5️⃣ Combine top 5 per category
    categories = [
        'points_disparity','rebounds_disparity','assists_disparity','total_disparity',
        'total_points_disparity','spread_disparity','moneyline_disparity'
    ]
    top_list = []

    # Player categories
    for cat in categories[:4]:
        if cat in player_disparities.columns:
            top_list.append(
                player_disparities[['full_name','team_name',cat]].rename(columns={cat:'disparity'}).assign(category=cat).nlargest(TOP_PER_CATEGORY,'disparity')
            )

    # Game categories
    for cat in categories[4:]:
        top_list.append(
            game_disparities[['home_team','away_team',cat]].rename(columns={cat:'disparity'}).assign(category=cat).nlargest(TOP_PER_CATEGORY,'disparity')
        )

    combined_top = pd.concat(top_list, ignore_index=True)
    combined_top.to_excel("top5_disparities_per_category.xlsx", index=False)
    print("✅ Top 5 disparities per category saved to top5_disparities_per_category.xlsx")
