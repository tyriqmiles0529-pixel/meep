#!/usr/bin/env python
"""Remove redundant dtype optimization that causes memory spike"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The chunked loading already optimizes dtypes per chunk
# This final pass is redundant and causes memory spikes
old_code = '''        # Final dtype optimization pass
        print(f"- Final dtype optimization...")

        # Convert object columns to category if they have low cardinality
        for col in agg_df.select_dtypes(include=['object']).columns:
            num_unique = agg_df[col].nunique()
            num_total = len(agg_df)
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                agg_df[col] = agg_df[col].astype('category')

        # Downcast numeric types
        for col in agg_df.select_dtypes(include=['float']).columns:
            agg_df[col] = pd.to_numeric(agg_df[col], downcast='float')

        for col in agg_df.select_dtypes(include=['integer']).columns:
            agg_df[col] = pd.to_numeric(agg_df[col], downcast='integer')

        # Force garbage collection
        gc.collect()

        print(f"- Memory optimized. Current usage: {agg_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")'''

new_code = '''        # Skip redundant dtype optimization - already done during chunked loading
        print(f"- Skipping redundant dtype optimization (already optimized per chunk)")
        gc.collect()
        print(f"- Current memory usage: {agg_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Removed redundant dtype optimization")
else:
    print("[SKIP] Could not find optimization code")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)
