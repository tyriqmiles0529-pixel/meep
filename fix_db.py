from sqlalchemy import create_engine, text
import os

db_path = "meep_terminal/data/meep_production.db"
engine = create_engine(f"sqlite:///{db_path}")

def migrate():
    print(f"Checking database at {db_path}...")
    with engine.connect() as conn:
        # 1. Check users table for preferences
        try:
            conn.execute(text("SELECT preferences FROM users LIMIT 1"))
            print("   [OK] Column 'preferences' already exists in 'users'.")
        except Exception:
            print("   [FIX] Adding 'preferences' column to 'users'...")
            conn.execute(text("ALTER TABLE users ADD COLUMN preferences JSON DEFAULT '{}'"))
            conn.commit()

        # 2. Check picks table for metadata_json
        try:
            conn.execute(text("SELECT metadata_json FROM picks LIMIT 1"))
            print("   [OK] Column 'metadata_json' already exists in 'picks'.")
        except Exception:
            print("   [FIX] Adding 'metadata_json' column to 'picks'...")
            conn.execute(text("ALTER TABLE picks ADD COLUMN metadata_json JSON DEFAULT '{}'"))
            conn.commit()

    print("\n[SUCCESS] Critical database schema fixes applied.")

if __name__ == "__main__":
    migrate()
