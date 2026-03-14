"""
FIXED TabNet Embedding Extraction Method

Replace the _get_embeddings() method in neural_hybrid.py (line 639-724)
with this simpler, more reliable version that uses TabNet's official API.
"""

def _get_embeddings(self, X):
    """
    Extract embeddings from TabNet using official API.

    This uses TabNet's built-in return_embeddings=True parameter,
    which is more reliable than manually accessing internal layers.

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
                   where embedding_dim should be n_d + n_a (typically 48)
    """
    if self.tabnet is None:
        raise ValueError("TabNet not trained yet")

    try:
        # Use TabNet's official embedding extraction API
        # This is the same approach used in GameNeuralHybrid (line 180-181)
        _, embeddings = self.tabnet.predict(X, return_embeddings=True)

        print(f"  [INFO] Extracted {embeddings.shape[1]}-dim embeddings using return_embeddings=True")

        return embeddings

    except Exception as e:
        print(f"  [WARNING] TabNet embedding extraction failed: {str(e)}")
        print(f"  [FALLBACK] Using predictions as 1-dim embeddings")

        # Fallback: just use predictions
        predictions = self.tabnet.predict(X)
        return predictions.reshape(-1, 1)


# USAGE:
# 1. Open neural_hybrid.py
# 2. Find line 639: def _get_embeddings(self, X):
# 3. Replace lines 639-724 with the function above
# 4. Re-upload to Colab
# 5. Re-run your test

# EXPECTED RESULT:
# [INFO] Extracted 48-dim embeddings using return_embeddings=True
#   (or possibly 24-dim, 32-dim, depending on TabNet config)

# This should give you the ACTUAL encoder embeddings (n_d + n_a dimensions)
# instead of the 150-dim embedder output you're currently getting.
