import os
import shutil
import sys
from pathlib import Path

def hard_reload():
    print("=== MEEP HARD RELOAD SYSTEM ===")
    
    # 1. Clean __pycache__
    print("Deleting all __pycache__ directories...")
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"))
            print(f"  Cleaned: {root}")

    # 2. Check for stale .pyc files in the root
    for f in os.listdir("."):
        if f.endswith(".pyc"):
            os.remove(f)
            print(f"  Removed stale bytecode: {f}")

    print("\n[SUCCESS] Bytecode purged.")
    print("Now restart your terminal with: python run_terminal.py")

if __name__ == "__main__":
    hard_reload()
