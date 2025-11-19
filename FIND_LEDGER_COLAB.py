# ============================================================
# FIND LEDGER IN COLAB - Run this to locate your predictions
# ============================================================
# Copy and paste this entire cell into Colab
# ============================================================

import os
import glob
from google.colab import files

print("="*70)
print("🔍 SEARCHING FOR PREDICTION FILES")
print("="*70)

# Search locations
search_dirs = [
    '/content',
    '/content/meep',
    os.getcwd(),
]

print("\n1️⃣ Searching for bets_ledger.pkl...\n")

found_ledger = False
for search_dir in search_dirs:
    if os.path.exists(search_dir):
        ledger_files = glob.glob(f"{search_dir}/**/bets_ledger.pkl", recursive=True)
        if ledger_files:
            for ledger in ledger_files:
                print(f"✅ FOUND: {ledger}")
                size_kb = os.path.getsize(ledger) / 1024
                print(f"   Size: {size_kb:.2f} KB")
                found_ledger = True
                
                # Try to download it
                try:
                    print(f"   Downloading...")
                    files.download(ledger)
                    print(f"   ✅ Downloaded successfully!")
                except Exception as e:
                    print(f"   ❌ Download failed: {e}")

if not found_ledger:
    print("❌ bets_ledger.pkl not found in any location")
    print("\nSearched:")
    for d in search_dirs:
        print(f"  • {d}")

print("\n2️⃣ Searching for prop_analysis files...\n")

found_analysis = False
for search_dir in search_dirs:
    if os.path.exists(search_dir):
        analysis_files = glob.glob(f"{search_dir}/**/prop_analysis_*.json", recursive=True)
        if analysis_files:
            # Get the latest one
            latest = sorted(analysis_files)[-1]
            print(f"✅ FOUND: {latest}")
            size_kb = os.path.getsize(latest) / 1024
            print(f"   Size: {size_kb:.2f} KB")
            found_analysis = True
            
            # Try to download it
            try:
                print(f"   Downloading...")
                files.download(latest)
                print(f"   ✅ Downloaded successfully!")
            except Exception as e:
                print(f"   ❌ Download failed: {e}")

if not found_analysis:
    print("❌ prop_analysis files not found")

print("\n3️⃣ Checking current directory contents...\n")
print(f"Current directory: {os.getcwd()}")
print("\nFiles in current directory:")
for item in sorted(os.listdir('.')):
    if os.path.isfile(item):
        size_kb = os.path.getsize(item) / 1024
        print(f"  📄 {item} ({size_kb:.2f} KB)")
    else:
        print(f"  📁 {item}/")

print("\n" + "="*70)
print("🔍 DIAGNOSIS")
print("="*70)

if not found_ledger and not found_analysis:
    print("\n❌ PROBLEM: No prediction files found!")
    print("\nPossible reasons:")
    print("  1. Analysis wasn't run (predictions never generated)")
    print("  2. Files saved to unexpected location")
    print("  3. Colab session was restarted (files lost)")
    print("\n💡 SOLUTION:")
    print("  • Re-run the analysis cell in Riq_Machine.ipynb")
    print("  • Then immediately run this script again")
    print("  • Download files before session expires!")
elif found_ledger or found_analysis:
    print("\n✅ SUCCESS: Files found and downloaded!")
    print("\nWhat to do next:")
    print("  1. Save downloaded files to your computer")
    print("  2. Copy bets_ledger.pkl to C:\\Users\\tmiles11\\nba_predictor\\")
    print("  3. Upload to Evaluate_Predictions.ipynb for calibration")

print("\n" + "="*70)
