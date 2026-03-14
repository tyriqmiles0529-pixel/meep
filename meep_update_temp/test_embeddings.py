"""
Test script to verify TabNet embeddings are being extracted correctly
"""
import numpy as np
import pandas as pd

def test_embedding_extraction():
    """Test that embeddings are extracted properly from NeuralHybridPredictor"""

    print("=" * 70)
    print("TESTING TABNET EMBEDDING EXTRACTION")
    print("=" * 70)

    # Import after printing to see any import errors
    try:
        from neural_hybrid import NeuralHybridPredictor
        print("[OK] Successfully imported NeuralHybridPredictor")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    # Create dummy data (similar to your NBA stats)
    n_samples = 500
    n_features = 150  # Similar to your 150+ features

    X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
    y_train = pd.Series(np.random.randn(n_samples) * 5 + 15)  # pandas Series

    X_val = pd.DataFrame(np.random.randn(100, n_features))
    y_val = pd.Series(np.random.randn(100) * 5 + 15)  # pandas Series

    X_test = pd.DataFrame(np.random.randn(20, n_features))

    print(f"\n[OK] Created test data:")
    print(f"  - Train: {n_samples} samples, {n_features} features")
    print(f"  - Val: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")

    # Initialize model
    model = NeuralHybridPredictor(prop_name='test_points', use_gpu=False)

    print(f"\n[OK] Initialized NeuralHybridPredictor")
    print(f"  - n_d (dimension): {model.tabnet_params['n_d']}")
    print(f"  - n_a (attention): {model.tabnet_params['n_a']}")
    print(f"  - n_steps: {model.tabnet_params['n_steps']}")
    print(f"  - Expected embedding size: n_d + n_a = {model.tabnet_params['n_d'] + model.tabnet_params['n_a']}")

    # Train model with just 1 epoch for quick testing
    print(f"\n[TRAINING] Training model (1 epoch for quick test)...")
    model.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=256)

    print(f"\n[OK] Model trained")

    # Get embeddings
    print(f"\n[TESTING] Extracting embeddings from test data...")
    print(f"  (Watch for [DEBUG] messages to see which extraction method is used)")

    try:
        embeddings = model._get_embeddings(X_test.values.astype(np.float32))

        print(f"\n" + "=" * 70)
        print(f"RESULTS:")
        print(f"=" * 70)
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Expected shape: (20, {model.tabnet_params['n_d'] + model.tabnet_params['n_a']})")
        print(f"Actual dimensions: {embeddings.shape[1]}")

        expected_dim = model.tabnet_params['n_d'] + model.tabnet_params['n_a']

        # Check if embeddings are correct
        if embeddings.shape[1] >= expected_dim:
            print(f"\n[SUCCESS] Embeddings are {embeddings.shape[1]}-dimensional!")
            print(f"\nSample embedding (first row):")
            print(embeddings[0])
            print(f"\nEmbedding statistics:")
            print(f"  - Mean: {embeddings.mean():.4f}")
            print(f"  - Std: {embeddings.std():.4f}")
            print(f"  - Min: {embeddings.min():.4f}")
            print(f"  - Max: {embeddings.max():.4f}")
            return True
        else:
            print(f"\n[WARNING] Embeddings are {embeddings.shape[1]}-dimensional")
            print(f"   Expected: {expected_dim}-dimensional")
            print(f"\nThis means fallback code is being used.")
            print(f"Check neural_hybrid.py:639-724 _get_embeddings() method")
            return False

    except Exception as e:
        print(f"\n[ERROR] Error extracting embeddings:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_extraction()

    if success:
        print("\n" + "=" * 70)
        print("[SUCCESS] All tests passed! Embeddings working correctly.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("[FAILED] Tests failed! Embeddings need fixing.")
        print("=" * 70)
        exit(1)
