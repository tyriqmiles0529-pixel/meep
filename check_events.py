import sqlite3
conn=sqlite3.connect('meep_terminal/data/meep_production.db')
c=conn.cursor()
c.execute("SELECT timestamp, category, level, message FROM system_events ORDER BY timestamp DESC LIMIT 50")
for r in c.fetchall():
    print(f"[{r[0]}] {r[1].upper()} | {r[2].upper()}: {r[3]}")
conn.close()
