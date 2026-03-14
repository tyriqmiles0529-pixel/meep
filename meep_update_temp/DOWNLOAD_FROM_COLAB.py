# ============================================================
# DOWNLOAD RESULTS - Add this cell to Riq_Machine.ipynb
# ============================================================
# 
# Copy this code into a NEW cell at the END of your notebook
# Run it after analysis completes to download files
# ============================================================

from google.colab import files
import os
import glob

print("="*70)
print("📥 DOWNLOADING PREDICTION FILES")
print("="*70)

os.chdir("/content/meep")

# 1. Download bets ledger (all predictions)
if os.path.exists("bets_ledger.pkl"):
    print("\n1️⃣ Downloading bets_ledger.pkl...")
    files.download("bets_ledger.pkl")
    print("   ✅ Downloaded: bets_ledger.pkl")
    print("   → Save this to your project folder!")
else:
    print("\n❌ No bets_ledger.pkl found")

# 2. Download latest prop analysis
analysis_files = sorted(glob.glob("prop_analysis_*.json"))
if analysis_files:
    latest = analysis_files[-1]
    print(f"\n2️⃣ Downloading {latest}...")
    files.download(latest)
    print(f"   ✅ Downloaded: {latest}")
    print("   → This has your latest predictions")
else:
    print("\n❌ No prop_analysis files found")

print("\n" + "="*70)
print("✅ DOWNLOAD COMPLETE!")
print("="*70)
print("\nWhat to do with these files:")
print("  1. bets_ledger.pkl → Copy to project root")
print("  2. prop_analysis_*.json → View predictions in text editor")
print("  3. Upload bets_ledger.pkl to Evaluate_Predictions.ipynb")
print("     to settle bets and generate calibration!")
