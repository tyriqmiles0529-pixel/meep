import re

with open('run_phase_i.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Direct string replace
content = content.replace('**Odds Source:**', '**Primary Book:** FanDuel')
content = content.replace('Timestamp:', 'Run Time:')
content = content.replace('{odds_source} | ', '')

with open('run_phase_i.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
