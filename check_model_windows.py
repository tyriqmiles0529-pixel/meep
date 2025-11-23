#!/usr/bin/env python
"""
Check how many windows are in the downloaded meta-learner model
"""

import dill
import pickle
from pathlib import Path

def check_model_windows(model_path="meta_learner_v4.pkl"):
    """Check the number of windows in the meta-learner"""
    
    print("="*60)
    print("CHECKING META-LEARNER WINDOW COUNT")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        # Load the model
        print(f"📦 Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            meta_learner = dill.load(f)
        
        print("✅ Model loaded successfully")
        
        # Check model attributes
        print(f"\n🔍 Model Type: {type(meta_learner)}")
        
        # Look for window-related attributes
        window_attrs = []
        for attr_name in dir(meta_learner):
            if 'window' in attr_name.lower() or 'ensemble' in attr_name.lower():
                window_attrs.append(attr_name)
        
        print(f"\n🪟 Window-related attributes: {window_attrs}")
        
        # Check common window count indicators
        window_count = None
        
        if hasattr(meta_learner, 'n_windows'):
            window_count = meta_learner.n_windows
        elif hasattr(meta_learner, 'num_windows'):
            window_count = meta_learner.num_windows
        elif hasattr(meta_learner, 'window_models'):
            window_count = len(meta_learner.window_models)
        elif hasattr(meta_learner, 'ensemble'):
            if hasattr(meta_learner.ensemble, 'models'):
                window_count = len(meta_learner.ensemble.models)
        
        # Try to find window count in other attributes
        if window_count is None:
            for attr_name in dir(meta_learner):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(meta_learner, attr_name)
                        if isinstance(attr_value, dict) and len(attr_value) in [26, 27]:
                            print(f"🔍 Found dict with {len(attr_value)} items: {attr_name}")
                            window_count = len(attr_value)
                        elif isinstance(attr_value, list) and len(attr_value) in [26, 27]:
                            print(f"🔍 Found list with {len(attr_value)} items: {attr_name}")
                            window_count = len(attr_value)
                    except:
                        pass
        
        print(f"\n📊 WINDOW COUNT: {window_count if window_count else 'Unknown'}")
        
        # Check model size
        file_size = Path(model_path).stat().st_size / (1024*1024)
        print(f"💾 Model size: {file_size:.1f} MB")
        
        # Look for training configuration
        if hasattr(meta_learner, 'config'):
            print(f"⚙️  Config found: {type(meta_learner.config)}")
        
        if hasattr(meta_learner, 'components'):
            print(f"🧩 Components: {meta_learner.components}")
        
        # Try to inspect the model structure more deeply
        print(f"\n🔬 Detailed structure:")
        for attr_name in sorted(dir(meta_learner)):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(meta_learner, attr_name)
                    attr_type = type(attr_value).__name__
                    if isinstance(attr_value, (list, dict)):
                        print(f"  {attr_name}: {attr_type} (size: {len(attr_value)})")
                    else:
                        print(f"  {attr_name}: {attr_type}")
                except Exception as e:
                    print(f"  {attr_name}: Error accessing - {e}")
        
        return window_count
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    """Main function"""
    result = check_model_windows()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if result == 26:
        print("🟡 Model has 26 windows (missing 2022-2024)")
        print("💡 Recommendation: Use for production, train missing window later")
    elif result == 27:
        print("✅ Model has 27 windows (complete ensemble)")
        print("🎉 Ready for production!")
    else:
        print("❓ Could not determine window count")
        print("🔍 Manual inspection needed")
    
    return result

if __name__ == "__main__":
    main()
