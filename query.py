import sqlite3
def show():
    conn = sqlite3.connect('meep_terminal/data/meep_production.db')
    cursor = conn.cursor()
    cursor.execute("SELECT player_name, prop_type, line, metadata_json FROM picks WHERE game_date >= '2026-03-08'")
    rows = cursor.fetchall()
    print("TOTAL:", len(rows))
    for r in rows:
        print(f"Player: {r[0]}, Prop: {r[1]}, Line: {r[2]}, Meta: {r[3]}")
    conn.close()
show()
