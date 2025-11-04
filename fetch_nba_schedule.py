import pandas as pd
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def fetch_next_3_days_schedule():
    url = "https://www.nba.com/schedule?cal=all&region=1"

    # --- Chrome Driver Setup ---
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        # Wait for games to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "section.ScheduleDay_scheduleDay__body__S1Y4A"))
        )

        # Define target dates (today + next 2 days)
        today = datetime.date.today()
        target_dates = [(today + datetime.timedelta(days=i)).strftime("%A, %B %-d") for i in range(3)]
        # Windows fix (use %#d for day number without zero)
        if not any("%-d" in d for d in target_dates):
            target_dates = [(today + datetime.timedelta(days=i)).strftime("%A, %B %#d") for i in range(3)]

        games_data = []
        days = driver.find_elements(By.CSS_SELECTOR, "section.ScheduleDay_scheduleDay__body__S1Y4A")

        for day in days:
            try:
                date_element = day.find_element(By.CSS_SELECTOR, "h2.ScheduleDay_scheduleDay__date__w4Jz3")
                date_text = date_element.text.strip()

                # Only collect if date matches our target range
                if any(t_date in date_text for t_date in target_dates):
                    games = day.find_elements(By.CSS_SELECTOR, "div.ScheduleGame_scheduleGame__wrapper__lY_Zl")

                    for game in games:
                        try:
                            teams = game.find_elements(By.CSS_SELECTOR, "p.ScheduleGame_scheduleGame__teamName__rF7Vx")
                            away_team = teams[0].text if len(teams) > 0 else ""
                            home_team = teams[1].text if len(teams) > 1 else ""

                            time_element = game.find_element(By.CSS_SELECTOR, "span.ScheduleGame_scheduleGame__time__WcIvx")
                            game_time = time_element.text.strip()
                        except:
                            away_team = home_team = game_time = ""

                        games_data.append({
                            "Date": date_text,
                            "Away Team": away_team,
                            "Home Team": home_team,
                            "Time": game_time
                        })
            except Exception:
                continue

        if games_data:
            df = pd.DataFrame(games_data)
            df.to_excel("nba_schedule_next3.xlsx", index=False)
            print(f"✅ Successfully exported {len(games_data)} games from the next 3 days to nba_schedule_next3.xlsx")
        else:
            print("❌ No games found for the next 3 days. (The NBA season may not have started yet.)")

    except Exception as e:
        print("⚠️ Error scraping:", e)
    finally:
        driver.quit()


if __name__ == "__main__":
    fetch_next_3_days_schedule()
