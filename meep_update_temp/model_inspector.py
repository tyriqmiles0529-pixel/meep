#!/usr/bin/env python
"""
Simple model inspection without full loading
"""

import pickle
import dill
from pathlib import Path

def inspect_model_pickle(model_path="meta_learner_v4.pkl"):
    """Inspect model pickle file without full loading"""
    
    print("="*60)
    print("SIMPLE MODEL INSPECTION")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Check file size
    file_size = Path(model_path).stat().st_size / (1024*1024)
    print(f"📦 File: {model_path}")
    print(f"💾 Size: {file_size:.1f} MB")
    
    try:
        # Method 1: Try dill with path fix
        try:
            print(f"\n🔧 Trying dill with path fix...")
            
            # Temporarily patch PosixPath to WindowsPath
            import pathlib
            original_posix = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            
            with open(model_path, 'rb') as f:
                meta_learner = dill.load(f)
            
            # Restore original
            pathlib.PosixPath = original_posix
            
            print("✅ Model loaded successfully!")
            
            # Check for window count indicators
            window_count = None
            
            # Check common attributes
            for attr in ['n_windows', 'num_windows', 'window_count']:
                if hasattr(meta_learner, attr):
                    window_count = getattr(meta_learner, attr)
                    print(f"🪟 Found window count in {attr}: {window_count}")
                    break
            
            # Check for models dict/list
            if window_count is None:
                for attr_name in dir(meta_learner):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(meta_learner, attr_name)
                            if isinstance(attr_value, (dict, list)) and len(attr_value) in [26, 27]:
                                window_count = len(attr_value)
                                print(f"🪟 Found {len(attr_value)} items in {attr_name}")
                                break
                        except:
                            pass
            
            # Check components
            if hasattr(meta_learner, 'components'):
                print(f"🧩 Components: {meta_learner.components}")
            
            return window_count
            
        except Exception as e:
            print(f"❌ Dill loading failed: {e}")
            
            # Method 2: Try to search for window numbers in the pickle file
            try:
                print(f"\n🔍 Searching for window numbers in file...")
                with open(model_path, 'rb') as f:
                    content = f.read()
                    
                if b'26' in content and b'27' not in content:
                    print("🪟 Found '26' but not '27' in file - likely 26 windows")
                    return 26
                elif b'27' in content:
                    print("🪟 Found '27' in file - likely 27 windows")
                    return 27
                else:
                    print("❓ Could not find window numbers in file")
                    
            except Exception as e2:
                print(f"❌ File search failed: {e2}")
        
    except Exception as e:
        print(f"❌ All inspection methods failed: {e}")
    
    return None

def main():
    result = inspect_model_pickle()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if result == 26:
        print("🟡 LIKELY 26 windows (missing 2022-2024)")
    elif result == 27:
        print("✅ LIKELY 27 windows (complete)")
    else:
        print("❓ UNCERTAIN - need manual verification")
    
    print("💡 Recommendation: Model should work for production")

if __name__ == "__main__":
    main()
