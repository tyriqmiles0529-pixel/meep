#!/usr/bin/env python
"""
Simple Model Verification (No torch dependency)
Checks model files and provides recommendations for GCE deployment.
"""

import pickle
from pathlib import Path

def check_model_file(model_path):
    """Basic model file check without torch dependency"""
    print(f"[*] Checking {model_path.name}...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Basic checks
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        data_type = type(model_data).__name__
        
        print(f"  ✅ File size: {file_size_mb:.1f} MB")
        print(f"  ✅ Data type: {data_type}")
        
        # Check if it's a dictionary (common format)
        if isinstance(model_data, dict):
            print(f"  ✅ Dictionary format with {len(model_data)} keys")
            
            # Look for obvious GPU indicators in keys/values
            gpu_indicators = []
            for key, value in model_data.items():
                key_str = str(key).lower()
                if 'cuda' in key_str or 'gpu' in key_str:
                    gpu_indicators.append(f"    - Key '{key}' contains GPU indicator")
            
            if gpu_indicators:
                print(f"  ⚠️  Potential GPU indicators found:")
                for indicator in gpu_indicators:
                    print(indicator)
            else:
                print(f"  ✅ No obvious GPU indicators in keys")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return False

def main():
    """Check all downloaded models"""
    print("="*70)
    print("SIMPLE MODEL VERIFICATION")
    print("="*70)
    print("Note: This is a basic check without torch dependency")
    print("For full GPU/CPU verification, run on GCE with torch installed")
    print("="*70)
    
    player_models_dir = Path("player_models")
    if not player_models_dir.exists():
        print("❌ No player_models directory found")
        return False
    
    # Find all .pkl files
    model_files = list(player_models_dir.glob("*.pkl"))
    if not model_files:
        print("❌ No .pkl files found in player_models/")
        return False
    
    print(f"[*] Found {len(model_files)} model files\n")
    
    # Categorize models
    window_models = []
    meta_models = []
    
    for model_file in sorted(model_files):
        if model_file.name.startswith('player_models_'):
            window_models.append(model_file)
        elif model_file.name.startswith('meta_learner_'):
            meta_models.append(model_file)
    
    print(f"📊 Model Categories:")
    print(f"  Window models: {len(window_models)}")
    print(f"  Meta-learners: {len(meta_models)}")
    print()
    
    # Check a few sample models
    print("🔍 Checking sample models...")
    sample_models = window_models[:3] + meta_models[:1]
    
    for model_file in sample_models:
        check_model_file(model_file)
        print()
    
    # Analyze year coverage
    print("="*70)
    print("YEAR COVERAGE ANALYSIS")
    print("="*70)
    
    years_covered = set()
    for model_file in window_models:
        # Extract years from filename like player_models_2004_2006.pkl
        parts = model_file.stem.replace('player_models_', '').split('_')
        if len(parts) == 2:
            try:
                start_year = int(parts[0])
                end_year = int(parts[1])
                years_covered.update(range(start_year, end_year + 1))
            except ValueError:
                pass
    
    if years_covered:
        min_year = min(years_covered)
        max_year = max(years_covered)
        print(f"✅ Years covered: {min_year}-{max_year}")
        print(f"✅ Total years: {len(years_covered)}")
        
        # Check for missing 2004-2022 range
        missing_years = set(range(2004, 2023)) - years_covered
        if missing_years:
            print(f"❌ Missing years for 2004-2022: {sorted(missing_years)}")
            print(f"   Need to train windows: 2004-2006, 2007-2009, 2010-2012, 2013-2015, 2016-2018, 2019-2021, 2022-2024")
        else:
            print(f"✅ All 2004-2022 years covered!")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"1. Upload these models to GCE")
    print(f"2. Train missing 2004-2022 windows on GCE")
    print(f"3. Run full GPU verification on GCE with torch installed")
    
    return True

if __name__ == "__main__":
    main()
