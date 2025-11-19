# TabNet Embedding Dimension Test

## Quick Test (Run in Colab)

Upload `test_embeddings.py` to your Colab and run:

```python
!python test_embeddings.py
```

**This will:**
1. Train TabNet for 1 epoch (~30-60 seconds on GPU)
2. Extract embeddings
3. Print the actual dimension

---

## What to Look For

### ✅ GOOD (Embeddings working correctly):

```
Step 2: Extracting Deep Feature Embeddings
  - Embedding dimension: 48
  - Train embeddings: (500, 48)
  - Val embeddings: (100, 48)

[DEBUG] Using 48-dim embeddings from encoder (3D)
```

OR

```
  - Embedding dimension: 24
  - Train embeddings: (500, 24)

[DEBUG] Using 24-dim embeddings from encoder (2D)
```

**Expected:** 24-48 dimensions (n_d=24, n_a=24)

---

### ❌ BAD (Fallback being used):

```
  - Embedding dimension: 2
  - Train embeddings: (500, 2)

[WARNING] Cannot access encoder, falling back to 2-dim embeddings
[DEBUG] Using 2-dim embeddings: predictions + attention weights
```

OR

```
  - Embedding dimension: 1
  - Train embeddings: (500, 1)

Warning: Using predictions as embeddings
```

---

## Current TabNet Configuration

From `neural_hybrid.py:378-391`:

```python
self.tabnet_params = {
    'n_d': 24,           # Feature dimension
    'n_a': 24,           # Attention dimension
    'n_steps': 4,        # Number of decision steps
    'n_independent': 2,  # Independent GLU layers
    'n_shared': 2,       # Shared GLU layers
}
```

**Expected embedding:** `n_d + n_a = 48 dimensions`

---

## How Embeddings Are Extracted

`neural_hybrid.py:639-724` - `_get_embeddings()` method tries 3 approaches:

### Method 1: Access encoder directly (BEST)
```python
if hasattr(self.tabnet.network, 'encoder'):
    steps_output, _ = self.tabnet.network.encoder(x)
    # Returns: (batch, n_steps, n_d+n_a) or (batch, n_d+n_a)
    embeddings = steps_output[:, -1, :]  # Last step = 48-dim
```

### Method 2: Use forward_masks
```python
elif hasattr(self.tabnet.network, 'forward_masks'):
    output, _ = self.tabnet.network.forward_masks(x)
    embeddings = output  # Should be 48-dim
```

### Method 3: Fallback (BAD - only 2-dim)
```python
else:
    predictions = self.tabnet.predict(batch).reshape(-1, 1)  # 1-dim
    mean_attention = M_explain.mean(axis=1, keepdims=True)  # 1-dim
    embeddings = np.hstack([predictions, mean_attention])    # 2-dim total
```

---

## If You Get 2-Dim or 1-Dim Embeddings

The issue is that TabNet's internal architecture isn't accessible via the methods above.

**Solution:** Use the simpler `return_embeddings=True` approach that's already in `GameNeuralHybrid`:

```python
# From neural_hybrid.py:180-181
_, train_embeddings = self.tabnet.predict(X_train_np, return_embeddings=True)
_, val_embeddings = self.tabnet.predict(X_val_np, return_embeddings=True)
```

This is PyTorch-TabNet's official API and should return proper multi-dim embeddings.

---

## Next Steps After Testing

1. **If embeddings are 24-48 dim:** ✅ Your system is already working! No changes needed.

2. **If embeddings are 2-dim or 1-dim:** Replace `_get_embeddings()` method (neural_hybrid.py:639) with the simpler `return_embeddings=True` approach.

3. **After fixing:** Consider adding H2O AutoML to discover additional feature interactions (separate from TabNet embeddings).

---

## Test in Colab

```python
# Upload test_embeddings.py to Colab, then:
!pip install pytorch-tabnet lightgbm scikit-learn pandas numpy torch

!python test_embeddings.py
```

Look for the line:
```
  - Embedding dimension: XX
```

Then report back what `XX` is!
