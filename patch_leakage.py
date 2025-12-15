import sys
import os

file_path = 'meep/nba_predictor/data_processor.py'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    sys.exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# We want to append the long names to the leakage_cols list
# The list ends with 'gmsc', '+/-'\n        ]
# We can replace the closing bracket with the new columns

new_cols = """,
            'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalsPercentage',
            'threePointersAttempted', 'threePointersMade', 'threePointersPercentage',
            'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage',
            'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
            'foulsPersonal', 'turnovers', 'plusMinusPoints'
        ]"""

if "'gmsc', '+/-'\n        ]" in content:
    content = content.replace("'gmsc', '+/-'\n        ]", "'gmsc', '+/-'" + new_cols)
    print("Patched leakage_cols.")
elif "'gmsc', '+/-']" in content:
    content = content.replace("'gmsc', '+/-']", "'gmsc', '+/-'" + new_cols)
    print("Patched leakage_cols (inline).")
else:
    print("Could not find leakage_cols end pattern. Please check file content.")
    # Fallback: Try to find the list definition and replace the whole thing if needed, 
    # but for now let's rely on the pattern we saw in `cat` output earlier.
    # The output showed:
    # 'mp', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc', '+/-'
    #         ]

with open(file_path, 'w') as f:
    f.write(content)

print("Finished updating data_processor.py")
