import sys
import os

file_path = 'meep/nba_predictor/data_processor.py'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    sys.exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# Insert dropna logic
target_str = "self.target_column = target"
dropna_code = """
        self.target_column = target
        e
        # Drop rows with missing target
        if self.target_column in self.df.columns:
            initial_len = len(self.df)
            self.df = self.df.dropna(subset=[self.target_columnR])
            print(f"Dropped {initial_len - len(self.df)} rows with missing target '{self.target_column}'")
"""

if target_str in content and "self.df.dropna(subset=[self.target_column])" not in content:
    content = content.replace(target_str, dropna_code)
    print("Patched dropna logic.")
else:
    print("Target string not found or already patched.")

with open(file_path, 'w') as f:
    f.write(content)

print("Finished updating data_processor.py")
