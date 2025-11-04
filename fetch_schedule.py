from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

chrome_driver_path = "C:/Users/tmiles11/chromedriver/chromedriver.exe"

def get_today_games():
    """
    Returns a list of today's games with home and away team names.
    """
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service)

    url = "https://www.nba.com/schedule?cal=all&region=1"
    driver.get(url)

    # Wait until the games are loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.ScheduleGame_sgTeam__TEPZa a.Anchor_anchor__cSc3P"))
    )

    game_elements = driver.find_elements(By.CSS_SELECTOR, "div.ScheduleGame_sgTeam__TEPZa a.Anchor_anchor__cSc3P")
    games = []

    for i in range(0, len(game_elements), 2):
        try:
            away_team = game_elements[i].text
            home_team = game_elements[i+1].text
            games.append({"home_team": home_team, "away_team": away_team})
        except IndexError:
            continue

    driver.quit()
    return games
