import sys
from datetime import datetime
from meep_terminal.data.models import DatabaseManager, Pick
import json

db = DatabaseManager()
sess = db.get_session()
today_str = datetime.now().strftime('%Y-%m-%d')
picks = sess.query(Pick).filter(Pick.game_date >= today_str).all()
for p in picks:
    print(f"[{p.id}] {p.player_name} {p.prop_type} {p.line} | meta={p.metadata_json}")
