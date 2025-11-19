#!/usr/bin/env python
"""
Clear all cached models from Modal volume.

This ensures training starts completely fresh.
"""

import modal

app = modal.App("clear-cache")

model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

@app.function(
    volumes={"/models": model_volume}
)
def clear_all_models():
    """Remove all cached models from Modal volume"""
    import os
    import glob

    print("=" * 70)
    print("CLEARING MODEL CACHE ON MODAL")
    print("=" * 70)

    # Find all model files
    model_files = glob.glob("/models/*.pkl") + glob.glob("/models/*.json")

    if not model_files:
        print("No cached models found - cache is already empty")
        return {"cleared": 0}

    print(f"\nFound {len(model_files)} cached files:")
    for f in model_files[:10]:
        print(f"  - {os.path.basename(f)}")
    if len(model_files) > 10:
        print(f"  ... and {len(model_files) - 10} more")

    # Remove all files
    print(f"\nRemoving {len(model_files)} files...")
    for f in model_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"  Warning: Could not remove {f}: {e}")

    # Commit changes
    model_volume.commit()

    print("\n✅ Model cache cleared!")
    print("=" * 70)

    return {"cleared": len(model_files)}


@app.local_entrypoint()
def main():
    """Clear cache - runs from your laptop"""
    print("=" * 70)
    print("CLEARING MODAL VOLUME CACHE")
    print("=" * 70)
    print("This will remove all cached models from Modal volume")
    print("Training will start fresh for all 27 windows")
    print("=" * 70)

    result = clear_all_models.remote()

    print(f"\n✅ Complete! Cleared {result['cleared']} files")
    print("\nNow ready to train from scratch:")
    print("  py -3.12 -m modal run modal_train.py")


if __name__ == "__main__":
    pass
