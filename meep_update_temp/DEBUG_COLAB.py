# ============================================================
# DEBUG - Find where files are being saved
# ============================================================

import os
import glob
import sys

print("="*70)
print("🔍 DEBUGGING FILE LOCATIONS")
print("="*70)

print("\n1️⃣ Current Working Directory:")
print(f"   {os.getcwd()}")

print("\n2️⃣ Contents of current directory:")
for item in sorted(os.listdir('.')):
    path = os.path.join('.', item)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1024
        print(f"   📄 {item} ({size:.1f} KB)")
    else:
        print(f"   📁 {item}/")

print("\n3️⃣ Searching entire /content for prediction files...")
all_pkl = glob.glob('/content/**/bets_ledger.pkl', recursive=True)
all_json = glob.glob('/content/**/prop_analysis_*.json', recursive=True)

if all_pkl:
    print(f"\n✅ Found {len(all_pkl)} ledger file(s):")
    for f in all_pkl:
        size = os.path.getsize(f) / 1024
        print(f"   {f} ({size:.1f} KB)")
else:
    print("\n❌ No bets_ledger.pkl found anywhere in /content")

if all_json:
    print(f"\n✅ Found {len(all_json)} analysis file(s):")
    for f in all_json:
        size = os.path.getsize(f) / 1024
        print(f"   {f} ({size:.1f} KB)")
else:
    print("\n❌ No prop_analysis files found anywhere in /content")

print("\n4️⃣ Checking if riq_analyzer.py exists:")
if os.path.exists('riq_analyzer.py'):
    print("   ✅ riq_analyzer.py found")
    # Check where it saves files
    with open('riq_analyzer.py', 'r') as f:
        content = f.read()
        if 'bets_ledger.pkl' in content:
            print("   ✅ Code contains 'bets_ledger.pkl'")
            # Find the save location
            for line in content.split('\n'):
                if 'bets_ledger' in line and ('dump' in line or 'save' in line):
                    print(f"   📝 Save line: {line.strip()[:80]}")
        else:
            print("   ⚠️  Code doesn't mention 'bets_ledger.pkl'")
else:
    print("   ❌ riq_analyzer.py NOT found")

print("\n5️⃣ Python sys.path (where Python looks for files):")
for p in sys.path[:5]:
    print(f"   • {p}")

print("\n" + "="*70)
print("🔍 DIAGNOSIS")
print("="*70)

if not all_pkl and not all_json:
    print("\n❌ PROBLEM: No prediction files created!")
    print("\nPossible causes:")
    print("  1. Analysis ran but hit an error (check output above)")
    print("  2. Files being saved to unexpected location")
    print("  3. riq_analyzer.py not configured correctly")
    print("\n💡 Next step: Check the FULL output of the analysis cell")
    print("   Look for error messages or warnings")
else:
    print("\n✅ Files exist! Path issue in download cell")
    print(f"\nUpdate download cell to use correct path:")
    if all_pkl:
        print(f"   files.download('{all_pkl[0]}')")
    if all_json:
        print(f"   files.download('{all_json[-1]}')")

print("\n" + "="*70)
