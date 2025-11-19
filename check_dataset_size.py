"""Quick check if dataset is sufficient for TabNet+LightGBM and H2O AutoML"""
import pandas as pd
import os

print("="*70)
print("DATASET SIZE ANALYSIS FOR ML MODELS")
print("="*70)

# Load dataset
print("\nLoading PlayerStatistics.csv...")
df = pd.read_csv('PlayerStatistics.csv', low_memory=False)

print(f"\n📊 TOTAL DATASET:")
print(f"  Rows: {len(df):,}")
print(f"  Unique games: {df['gameId'].nunique():,}")
print(f"  Unique players: {df['personId'].nunique():,}")
print(f"  File size: {os.path.getsize('PlayerStatistics.csv') / (1024**2):.1f} MB")

# Parse dates
df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', errors='coerce')
df['year'] = df['gameDate'].dt.year

print(f"\n📅 ERA BREAKDOWN:")
print(f"  1946-1999: {len(df[df['year'] < 2000]):,} rows")
print(f"  2000-2009: {len(df[(df['year'] >= 2000) & (df['year'] < 2010)]):,} rows")
print(f"  2010-2019: {len(df[(df['year'] >= 2010) & (df['year'] < 2020)]):,} rows")
print(f"  2020-2025: {len(df[df['year'] >= 2020]):,} rows")

recent = df[df['year'] >= 2015]
print(f"\n🔥 MODERN ERA (2015-2025):")
print(f"  Rows: {len(recent):,}")
print(f"  Unique games: {recent['gameId'].nunique():,}")
print(f"  Unique players: {recent['personId'].nunique():,}")

print(f"\n{'='*70}")
print("MODEL REQUIREMENTS ANALYSIS")
print(f"{'='*70}")

print("\n1️⃣  TABNET + LIGHTGBM HYBRID")
print("   Minimum recommended: 10,000 samples")
print("   Ideal: 100,000+ samples")
print(f"   Your dataset: {len(df):,} samples")
if len(df) >= 100000:
    print("   ✅ EXCELLENT - More than enough data!")
elif len(df) >= 10000:
    print("   ✅ GOOD - Sufficient data")
else:
    print("   ⚠️  MARGINAL - May need data augmentation")

print("\n   📝 Per Prop Type (after filtering):")
prop_estimates = {
    'Points': int(len(df) * 0.8),  # Most players score
    'Assists': int(len(df) * 0.6),  # Guards/wings
    'Rebounds': int(len(df) * 0.7),  # Most positions
    'Threes': int(len(df) * 0.5),   # 3-pt shooters
    'Minutes': int(len(df) * 0.9),  # Almost all
}
for prop, count in prop_estimates.items():
    status = "✅" if count >= 100000 else "⚠️"
    print(f"     {status} {prop}: ~{count:,} samples")

print("\n2️⃣  H2O AUTOML")
print("   Minimum recommended: 10,000 samples")
print("   Ideal: 50,000+ samples")
print("   Max efficient: ~1-2M samples (memory limits)")
print(f"   Your dataset: {len(df):,} samples")
if len(df) > 2000000:
    print("   ⚠️  VERY LARGE - May need sampling or distributed training")
elif len(df) >= 50000:
    print("   ✅ EXCELLENT - Perfect for AutoML")
elif len(df) >= 10000:
    print("   ✅ GOOD - Sufficient for AutoML")
else:
    print("   ⚠️  SMALL - AutoML may overfit")

print("\n3️⃣  TRAINING SPLIT ESTIMATES")
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = int(len(df) * 0.15)
print(f"   70% Train: {train_size:,} samples")
print(f"   15% Val:   {val_size:,} samples")
print(f"   15% Test:  {test_size:,} samples")

print("\n4️⃣  FEATURE COUNT ESTIMATE")
print("   After aggregation: ~178 features")
print("   Phase 7 engineered: +40 features")
print("   Total: ~218 features")
print(f"   Samples-to-features ratio: {len(df) / 218:.0f}:1")
if len(df) / 218 >= 1000:
    print("   ✅ EXCELLENT ratio (>1000:1)")
elif len(df) / 218 >= 100:
    print("   ✅ GOOD ratio (>100:1)")
else:
    print("   ⚠️  LOW ratio - risk of overfitting")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print(f"{'='*70}")

print("\n✅ YOUR DATASET IS SUFFICIENT FOR:")
print("   • TabNet + LightGBM hybrid")
print("   • H2O AutoML")
print("   • Deep learning models (TabNet)")
print("   • Tree ensembles (LightGBM, XGBoost)")
print("   • Meta-learners / stacking")

print("\n⚡ OPTIMIZATIONS TO CONSIDER:")
print("   1. Train on recent era only (2015+): {recent:,} samples")
print("      - More relevant to modern NBA")
print("      - Faster training")
print(f"   2. Use full dataset ({len(df):,}): ")
print("      - More robust models")
print("      - Better rare event detection")
print("      - Longer training time")

print("\n🎯 COLAB GPU CONSIDERATIONS:")
print(f"   • {len(df):,} rows × ~220 features = ~{len(df) * 220 * 8 / (1024**3):.2f} GB RAM")
print("   • Colab free tier: 12-16 GB RAM")
print("   • Colab Pro: 25-52 GB RAM")
if len(df) * 220 * 8 / (1024**3) < 10:
    print("   ✅ Fits comfortably in free tier!")
elif len(df) * 220 * 8 / (1024**3) < 20:
    print("   ⚠️  May need Colab Pro or sampling")
else:
    print("   ❌ Need Colab Pro or distributed training")

print("\n" + "="*70)
