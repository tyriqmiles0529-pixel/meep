# ==============================================
# NBA Betting Model v11 - Robust Version
# Top 20 Bets including Player Props, Spreads, Totals
# Weighted Recent Games, Kelly Fraction, Composite Score
# Suggested Stake included
# ==============================================
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog
import requests

# -----------------------------
# CONFIGURATION
# -----------------------------
ROLLING_WINDOW = 10
FRACTIONAL_KELLY = 0.5
BANKROLL = 100
SEASON = "2024-25"  # Current NBA season

# -----------------------------
# SPORTSBOOK CONFIG
# -----------------------------
SPORTSBOOK_URL = 'https://api.sportsgameodds.com/v2/sports/'
SPORTSBOOK_API_KEY = "3ee00eb314b80853c6c77920c5bf74f7"
HEADERS = {'X-Api-Key': SPORTSBOOK_API_KEY}

# -----------------------------
# DATA FETCH FUNCTIONS
# -----------------------------
def fetch_active_players():
    return players.get_active_players()

def fetch_active_teams():
    return teams.get_teams()

def fetch_player_last_games(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0].head(ROLLING_WINDOW)
        return df
    except:
        return pd.DataFrame()

def fetch_team_last_games(team_id):
    try:
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=SEASON, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0].head(ROLLING_WINDOW)
        return df
    except:
        return pd.DataFrame()

# -----------------------------
# SPORTSBOOK ODDS FETCH
# -----------------------------
def fetch_sportsbook_odds():
    try:
        response = requests.get(SPORTSBOOK_URL, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        nba_odds = []
        for event in data.get('events', []):
            if event.get('league') != "NBA":
                continue
            nba_odds.append({
                "game_id": event.get('id'),
                "home_team": event.get('home_team'),
                "away_team": event.get('away_team'),
                "home_spread": event.get('points-home-game-sp-home'),
                "home_spread_odds": event.get('points-home-game-sp-home-odds'),
                "total_line": event.get('points-all-game-ou-line'),
                "over_odds": event.get('points-all-game-ou-over'),
                "under_odds": event.get('points-all-game-ou-under'),
                "player_props": event.get('player_props', []),
                "date": event.get('start_time')
            })
        return nba_odds
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sportsbook odds: {e}")
        return []

# -----------------------------
# PROBABILITY FUNCTIONS
# -----------------------------
def estimate_player_probability(player_stats, line, stat='PTS'):
    if player_stats.empty or line is None or stat not in player_stats.columns:
        return 0.5
    weights = np.linspace(1, len(player_stats), len(player_stats))
    avg_minutes = player_stats['MIN'].mean()
    avg_usage = player_stats.get('USG_PCT', pd.Series([1]*len(player_stats))).mean()
    adjusted_stats = []
    for i, row in enumerate(player_stats.iloc[::-1].itertuples()):
        minutes_factor = row.MIN / avg_minutes if avg_minutes > 0 else 1
        usage_factor = getattr(row, 'USG_PCT', avg_usage) / avg_usage if avg_usage > 0 else 1
        adjusted_stat = getattr(row, stat) * minutes_factor * usage_factor
        adjusted_stats.append(adjusted_stat * weights[i])
    weighted_avg = sum(adjusted_stats) / sum(weights)
    prob = min(max(weighted_avg / line, 0), 1)
    return prob

def estimate_team_probability(team_stats, matchup_context):
    if team_stats.empty:
        return 0.5
    weights = np.linspace(1, len(team_stats), len(team_stats))
    off_ratings = team_stats['OFF_RATING'].iloc[::-1].values
    def_ratings = team_stats['DEF_RATING'].iloc[::-1].values
    weighted_off = np.average(off_ratings, weights=weights)
    weighted_def = np.average(def_ratings, weights=weights)
    base_prob = 0.5 + ((weighted_off - weighted_def) * 0.001)
    home_adv = 0.05 if matchup_context.get('home', True) else -0.05
    prob = base_prob + home_adv
    return min(max(prob, 0), 1)

def estimate_spread_probability(team_stats, opponent_stats, spread):
    if team_stats.empty or opponent_stats.empty or spread is None:
        return 0.5
    weights = np.linspace(1, len(team_stats), len(team_stats))
    team_off = np.average(team_stats['OFF_RATING'].iloc[::-1], weights=weights)
    team_pace = np.average(team_stats.get('PACE', pd.Series([1]*len(team_stats))).iloc[::-1], weights=weights)
    opp_def = np.average(opponent_stats['DEF_RATING'].iloc[::-1], weights=weights)
    opp_pace = np.average(opponent_stats.get('PACE', pd.Series([1]*len(opponent_stats))).iloc[::-1], weights=weights)
    expected_margin = (team_off * team_pace / 100) - (opp_def * opp_pace / 100) - spread
    prob = 1 / (1 + np.exp(-0.1 * expected_margin))
    return prob

def estimate_total_probability(home_stats, away_stats, total_line, over=True):
    if home_stats.empty or away_stats.empty or total_line is None:
        return 0.5
    weights = np.linspace(1, len(home_stats), len(home_stats))
    home_off = np.average(home_stats['OFF_RATING'].iloc[::-1], weights=weights)
    home_def = np.average(home_stats['DEF_RATING'].iloc[::-1], weights=weights)
    home_pace = np.average(home_stats.get('PACE', pd.Series([1]*len(home_stats))).iloc[::-1], weights=weights)
    away_off = np.average(away_stats['OFF_RATING'].iloc[::-1], weights=weights)
    away_def = np.average(away_stats['DEF_RATING'].iloc[::-1], weights=weights)
    away_pace = np.average(away_stats.get('PACE', pd.Series([1]*len(away_stats))).iloc[::-1], weights=weights)
    home_expected = home_off * home_pace / 100 - away_def * away_pace / 100
    away_expected = away_off * away_pace / 100 - home_def * home_pace / 100
    expected_total = home_expected + away_expected
    if over:
        prob = 1 / (1 + np.exp(-0.1 * (expected_total - total_line)))
    else:
        prob = 1 - (1 / (1 + np.exp(-0.1 * (expected_total - total_line))))
    return prob

# -----------------------------
# KELLY FRACTION
# -----------------------------
def kelly_fraction(decimal_odds, predicted_prob):
    b = decimal_odds - 1
    q = 1 - predicted_prob
    f = (b * predicted_prob - q) / b
    return max(f * FRACTIONAL_KELLY, 0)

# -----------------------------
# COMPOSITE SCORE
# -----------------------------
def calculate_composite_score(pred_prob, decimal_odds, kelly, edge, variance=0.1, consistency=1.0, context=1.0):
    return (kelly * pred_prob * edge * consistency * context) / (1 + variance)

# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    active_players = fetch_active_players()
    active_teams = fetch_active_teams()
    odds_data = fetch_sportsbook_odds()
    bets = []

    for game in odds_data:
        home_team = game['home_team']
        away_team = game['away_team']
        matchup_context = {'home': True}

        home_team_id = next((t['id'] for t in active_teams if t['full_name'] == home_team), None)
        away_team_id = next((t['id'] for t in active_teams if t['full_name'] == away_team), None)
        home_team_stats = fetch_team_last_games(home_team_id) if home_team_id else pd.DataFrame()
        away_team_stats = fetch_team_last_games(away_team_id) if away_team_id else pd.DataFrame()

        # ----------------- Spread -----------------
        if game.get('home_spread') and game.get('home_spread_odds'):
            spread_prob = estimate_spread_probability(home_team_stats, away_team_stats, game['home_spread'])
            kelly = kelly_fraction(game['home_spread_odds'], spread_prob)
            implied_prob = 1 / game['home_spread_odds']
            edge = spread_prob - implied_prob
            score = calculate_composite_score(spread_prob, game['home_spread_odds'], kelly, edge)
            suggested_stake = round(kelly * BANKROLL,2)
            bets.append({
                "Bet Type": f"Spread {home_team} {game['home_spread']:+}",
                "Player/Team": home_team,
                "Team": home_team,
                "Opponent": away_team,
                "Game Date": game.get('date'),
                "Line": game['home_spread'],
                "Score": score,
                "Suggested Stake": suggested_stake
            })

        # ----------------- Totals -----------------
        if game.get('total_line'):
            if game.get('over_odds'):
                over_prob = estimate_total_probability(home_team_stats, away_team_stats, game['total_line'], over=True)
                kelly = kelly_fraction(game['over_odds'], over_prob)
                implied_prob = 1 / game['over_odds']
                edge = over_prob - implied_prob
                score = calculate_composite_score(over_prob, game['over_odds'], kelly, edge)
                suggested_stake = round(kelly * BANKROLL,2)
                bets.append({
                    "Bet Type": f"Total Over {game['total_line']}",
                    "Player/Team": f"{home_team} vs {away_team}",
                    "Team": None,
                    "Opponent": None,
                    "Game Date": game.get('date'),
                    "Line": game['total_line'],
                    "Score": score,
                    "Suggested Stake": suggested_stake
                })
            if game.get('under_odds'):
                under_prob = estimate_total_probability(home_team_stats, away_team_stats, game['total_line'], over=False)
                kelly = kelly_fraction(game['under_odds'], under_prob)
                implied_prob = 1 / game['under_odds']
                edge = under_prob - implied_prob
                score = calculate_composite_score(under_prob, game['under_odds'], kelly, edge)
                suggested_stake = round(kelly * BANKROLL,2)
                bets.append({
                    "Bet Type": f"Total Under {game['total_line']}",
                    "Player/Team": f"{home_team} vs {away_team}",
                    "Team": None,
                    "Opponent": None,
                    "Game Date": game.get('date'),
                    "Line": game['total_line'],
                    "Score": score,
                    "Suggested Stake": suggested_stake
                })

        # ----------------- Player Props -----------------
        for prop in game.get('player_props', []):
            player_name = prop.get('player')
            stat = prop.get('stat')
            line = prop.get('line')
            decimal_odds = prop.get('odds')
            player_id = next((p['id'] for p in active_players if p['full_name'] == player_name), None)
            player_stats = fetch_player_last_games(player_id) if player_id else pd.DataFrame()
            prob = estimate_player_probability(player_stats, line, stat)
            kelly = kelly_fraction(decimal_odds, prob)
            implied_prob = 1 / decimal_odds
            edge = prob - implied_prob
            score = calculate_composite_score(prob, decimal_odds, kelly, edge)
            suggested_stake = round(kelly * BANKROLL,2)
            bets.append({
                "Bet Type": f"{stat} Over",
                "Player/Team": player_name,
                "Team": None,
                "Opponent": prop.get('opponent'),
                "Game Date": game.get('date'),
                "Line": line,
                "Score": score,
                "Suggested Stake": suggested_stake
            })

    # ----------------- FINAL OUTPUT -----------------
    # Ensure all bets have 'Score'
    for b in bets:
        if 'Score' not in b or b['Score'] is None:
            b['Score'] = 0

    df = pd.DataFrame(bets)
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)

    # Sort top 20
    df = df.sort_values(by='Score', ascending=False).head(20)

    output_columns = ["Bet Type", "Player/Team", "Team", "Opponent", "Game Date", "Line", "Score", "Suggested Stake"]
    df[output_columns].to_csv("nba_bets_top20.csv", index=False)
    print(df[output_columns])

if __name__ == "__main__":
    main()
