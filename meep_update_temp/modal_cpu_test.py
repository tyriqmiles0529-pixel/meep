#!/usr/bin/env python
"""
Minimal Modal CPU Test
"""

import modal

app = modal.App("nba-cpu-test")

image = modal.Image.debian_slim().pip_install([
    "pandas", "numpy", "scikit-learn"
])

@app.function(image=image, gpu=None, memory=8192)
def test_cpu():
    """Test basic CPU functionality"""
    import pandas as pd
    import numpy as np
    print("✅ Modal CPU working!")
    return pd.DataFrame({'test': [1, 2, 3]})

@app.local_entrypoint()
def main():
    result = test_cpu.remote()
    print(f"Result shape: {result.shape}")
    print("✅ CPU test successful!")

if __name__ == "__main__":
    main()
