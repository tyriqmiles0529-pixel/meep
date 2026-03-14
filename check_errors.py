import sqlite3
conn=sqlite3.connect('meep_terminal/data/meep_production.db')
c=conn.cursor()
c.execute("SELECT timestamp, category, message FROM system_events WHERE level = 'error' ORDER BY timestamp DESC LIMIT 10")
rows = c.fetchall()
if not rows:
    print("No level=error events found.")
for r in rows:
    print(f"[{r[0]}] {r[1].upper()}: {r[2]}")
conn.close()
