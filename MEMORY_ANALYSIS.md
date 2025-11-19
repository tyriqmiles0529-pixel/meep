# Memory Analysis - Training Peak RAM

## Question: Does training stay under 25 GB RAM?

**Answer**: Yes, comfortably. Peak RAM is ~12-15 GB.

---

## Memory Breakdown

### 1. Dataset Loading (Largest Component)

**aggregated_nba_data.csv.gzip:**
- Rows: ~125,000 player-games
- Columns: 235 features (float32)
- Size per row: 235 features × 4 bytes = 940 bytes
- Total dataset size: 125,000 × 940 bytes = **~117 MB**

**In memory (pandas DataFrame):**
- DataFrame overhead: ~50% more than raw data
- Total: 117 MB × 1.5 = **~175 MB**

### 2. Train/Val Split

**80/20 split:**
- Training: 100,000 rows × 235 cols × 4 bytes = **~94 MB**
- Validation: 25,000 rows × 235 cols × 4 bytes = **~23 MB**
- Total: **~117 MB** (same as original, just split)

### 3. TabNet Training (GPU Memory, not RAM)

**TabNet model on GPU:**
- Model parameters: n_d=24, n_a=24, n_steps=4
- Estimated parameters: ~50K-100K
- Size: ~400 KB (tiny!)
- **GPU VRAM usage: 2-4 GB** (batch processing on GPU)
- **RAM usage: ~500 MB** (data transfer buffers)

**Batch processing:**
- Batch size: 2048 samples
- Batch memory: 2048 × 235 × 4 bytes = **~1.9 MB per batch**
- Virtual batch: 256 samples = **~240 KB**

### 4. Embedding Generation

**After TabNet training, generate embeddings:**
- Input: 100,000 rows × 235 features = 94 MB
- Output: 100,000 rows × 24 embeddings × 4 bytes = **~9.6 MB**
- Peak during generation: **~104 MB** (input + output simultaneously)

### 5. LightGBM Training (Largest RAM Usage)

**Combined feature matrix:**
- Raw features: 100,000 × 235 = 94 MB
- Embeddings: 100,000 × 24 = 9.6 MB
- Total: 100,000 × 259 features = **~104 MB**

**LightGBM internal structures:**
- Histogram bins: ~259 features × 256 bins × 500 trees = **~33 MB**
- Tree structure: 500 trees × ~31 leaves × 8 bytes = **~0.12 MB** (negligible)
- Gradient/hessian arrays: 100,000 × 8 bytes × 2 = **~1.6 MB**
- Working memory: **~50-100 MB** (temporary allocations)

**LightGBM peak**: 104 MB (data) + 100 MB (working) = **~200 MB**

### 6. Sigma Model Training

**Same process as LightGBM:**
- Input: 259 features × 100,000 rows = 104 MB
- Working memory: ~100 MB
- Peak: **~200 MB**

### 7. Python Overhead

**Python interpreter + libraries:**
- Python runtime: ~100 MB
- NumPy: ~50 MB
- Pandas: ~100 MB
- PyTorch: ~500 MB (even when using GPU)
- LightGBM: ~50 MB
- Scikit-learn: ~50 MB
- Total: **~850 MB**

---

## Total Peak RAM Usage

### During TabNet Training
```
Dataset: 175 MB
Train/val split: 117 MB
TabNet buffers: 500 MB
Python overhead: 850 MB
---------------------------------
Peak: ~1.6 GB
```

### During LightGBM Training (HIGHEST PEAK)
```
Dataset: 175 MB
Combined features: 104 MB
Embeddings: 10 MB
LightGBM working memory: 100 MB
TabNet model (kept in RAM): 500 MB
Python overhead: 850 MB
---------------------------------
Peak: ~1.7 GB
```

### Training 5 Props Sequentially

Each prop trains independently (one at a time):
- Peak per prop: **~1.7 GB**
- Models are saved and unloaded between props
- **Total peak never exceeds 2 GB per prop**

### Kaggle/Colab Available RAM

**Kaggle:**
- Standard: 16 GB RAM
- GPU sessions: 13 GB RAM (some reserved for GPU)

**Colab:**
- Free: 12.7 GB RAM
- Pro: 25 GB RAM
- Pro+: 51 GB RAM

---

## Answer: Will it Stay Under 25 GB?

**Yes, easily!**

- **Peak RAM usage: ~2 GB per prop**
- **Total for all 5 props: ~2 GB** (sequential training)
- **Safety margin: 23 GB unused**

### Memory Usage Over Time

```
Time    Phase                RAM Usage
------  -------------------  ---------
0:00    Load dataset         0.2 GB
0:05    Train Minutes        1.7 GB
0:25    Save Minutes         0.2 GB
0:26    Train Points         1.7 GB
0:46    Save Points          0.2 GB
0:47    Train Rebounds       1.7 GB
1:07    Save Rebounds        0.2 GB
1:08    Train Assists        1.7 GB
1:28    Save Assists         0.2 GB
1:29    Train Threes         1.7 GB
1:49    Save Threes          0.2 GB
1:50    Done                 0.2 GB
```

**Pattern**: Peak during training (~1.7 GB), drops to ~0.2 GB between models.

---

## GPU Memory (VRAM) Usage

Separate from RAM:

### TabNet Training
- Model parameters: ~0.4 MB
- Batch processing: 2048 samples × 235 features × 4 bytes = **~1.9 MB**
- Gradients: ~0.8 MB
- Optimizer states (AdamW): ~1.2 MB
- **Peak VRAM: ~4 GB per prop**

### Kaggle/Colab Available VRAM

**Kaggle:**
- P100: 16 GB VRAM ✅
- T4: 16 GB VRAM ✅

**Colab:**
- Free T4: 15 GB VRAM ✅
- Pro A100: 40 GB VRAM ✅

**All have plenty of VRAM headroom.**

---

## Why Memory Usage is So Low

### 1. Small Dataset Size
- 125K rows is tiny by ML standards
- Modern models often train on 10M+ rows
- Your data: 117 MB uncompressed

### 2. Efficient Data Types
- Using float32 (4 bytes) not float64 (8 bytes)
- Saves 50% memory

### 3. Sequential Training
- Train one prop at a time
- Previous model unloaded before next
- No cumulative memory buildup

### 4. GPU Offloading
- TabNet runs on GPU (uses VRAM, not RAM)
- Only data transfer uses RAM
- RAM mostly idle during TabNet training

### 5. Batch Processing
- Don't load entire dataset into GPU at once
- Process 2048 samples at a time
- Minimal memory footprint

---

## Comparison to Other ML Tasks

### Your Training (NBA Props)
- **RAM: 2 GB peak**
- **VRAM: 4 GB peak**
- **Dataset: 125K rows, 235 features**

### Typical Deep Learning
- **RAM: 20-50 GB** (large datasets)
- **VRAM: 24-80 GB** (huge models)
- **Dataset: 10M+ samples**

### Large Language Models
- **RAM: 100-500 GB**
- **VRAM: 80-800 GB** (A100 clusters)
- **Dataset: Billions of tokens**

**Your training is extremely lightweight!**

---

## Potential Memory Issues (Unlikely)

### Scenario 1: Memory Leak
If Python doesn't release memory between props:
- First prop: 1.7 GB
- Second prop: 3.4 GB
- Third prop: 5.1 GB
- ...
- Fifth prop: 8.5 GB

**Still under 25 GB!** And this is unlikely because:
- Models are explicitly saved and deleted
- Garbage collection runs between props
- Modern Python is good at cleanup

### Scenario 2: Parallel Training
If you accidentally train multiple props simultaneously:
- 5 props × 1.7 GB = 8.5 GB peak

**Still well under 25 GB!**

### Scenario 3: Huge Aggregated Dataset
If your aggregated CSV is much larger than expected:
- Expected: 125K rows = 117 MB
- If 10x larger: 1.25M rows = **1.17 GB**
- Peak RAM: 1.17 GB + 1.7 GB working = **~3 GB**

**Still safe!**

---

## Recommendations

### Kaggle (Your Choice)
✅ **Perfect for your use case**
- 13 GB RAM available
- Peak usage: ~2 GB
- **11 GB headroom** (5.5x safety margin)
- No memory concerns at all

### Colab Free
✅ **Also works fine**
- 12.7 GB RAM available
- Peak usage: ~2 GB
- **10.7 GB headroom**

### Colab Pro
✅ **Overkill but safe**
- 25 GB RAM available
- Peak usage: ~2 GB
- **23 GB headroom** (11x safety margin)

---

## Monitoring Memory During Training

### Check RAM usage in Kaggle:
```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"RAM usage: {mem_gb:.2f} GB")

# Add to training loop
print_memory_usage()  # Check periodically
```

### Expected output:
```
Before training: 0.18 GB
During TabNet: 1.65 GB
During LightGBM: 1.72 GB
After save: 0.19 GB
```

---

## Final Answer

**Q: Does training stay under 25 GB RAM?**

**A: Yes, easily. Peak RAM is only ~2 GB.**

- **Kaggle (13 GB)**: ✅ Perfect fit, 11 GB headroom
- **Colab Free (12.7 GB)**: ✅ Works fine, 10.7 GB headroom
- **Colab Pro (25 GB)**: ✅ Massive overkill

**You could train 6 props simultaneously and still stay under 25 GB.**

Memory is not a concern for this training at all. The bottleneck is GPU compute time (3-4 hours), not memory.

---

## TL;DR

**Peak RAM: ~2 GB**
**Kaggle RAM: 13 GB**
**Safety margin: 11 GB (5.5x overhead)**

✅ **You're good to go!**
