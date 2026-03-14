import sys
import os

file_path = 'meep/nba_predictor/data_processor.py'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    sys.exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# 1. Update sorting logic
old_sort = "if 'game_number_in_season' in self.df.columns: sort_cols.append('game_number_in_season')"
new_sort = "if 'date' in self.df.columns: sort_cols.append('date')\n        elif 'game_number_in_season' in self.df.columns: sort_cols.append('game_number_in_season')"

if old_sort in content:
    content = content.replace(old_sort, new_sort)
    print("Patched sorting logic.")
else:
    print("Sorting logic not found or already patched.")

# 2. Update load_data to parse date
old_load = "self.df = pd.read_csv(self.file_path, nrows=nrows)"
new_load = "self.df = pd.read_csv(self.file_path, nrows=nrows)\n        if 'date' in self.df.columns:\n            self.df['date'] = pd.to_datetime(self.df['date'])"

if old_load in content:
    content = content.replace(old_load, new_load)
    print("Patched load_data logic.")
else:
    print("load_data logic not found or already patched.")

with open(file_path, 'w') as f:
    f.write(content)

print("Finished updating data_processor.py")
