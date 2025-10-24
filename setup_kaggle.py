#!/usr/bin/env python3
"""
Interactive Kaggle setup helper
"""
import os
import json
from pathlib import Path

print("=" * 70)
print("KAGGLE AUTHENTICATION SETUP")
print("=" * 70)

kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

print(f"\n📂 Looking for credentials at: {kaggle_json}")

# Check if already set up
if kaggle_json.exists():
    print("✅ Found kaggle.json!")

    # Verify it's valid JSON
    try:
        with open(kaggle_json, 'r') as f:
            creds = json.load(f)

        if 'username' in creds and 'key' in creds:
            print(f"✅ Username: {creds['username']}")
            print(f"✅ API Key: {'*' * 20}{creds['key'][-4:]}")

            # Check permissions (should be 600)
            perms = oct(os.stat(kaggle_json).st_mode)[-3:]
            if perms == '600':
                print("✅ Permissions: 600 (secure)")
            else:
                print(f"⚠️  Permissions: {perms} (should be 600)")
                print("   Fixing permissions...")
                os.chmod(kaggle_json, 0o600)
                print("   ✅ Fixed!")

            # Test connection
            print("\n🔌 Testing Kaggle API connection...")
            try:
                import kagglehub
                # Try a simple API call
                kagglehub.login()
                print("✅ Connection successful!\n")

                print("=" * 70)
                print("🎉 All set! You can now run:")
                print("   python explore_dataset.py")
                print("=" * 70)

            except Exception as e:
                print(f"❌ Connection failed: {e}")
                print("\nTry running: kaggle datasets list")
        else:
            print("❌ Invalid kaggle.json (missing username or key)")

    except json.JSONDecodeError:
        print("❌ kaggle.json is not valid JSON")

else:
    print("❌ kaggle.json not found\n")
    print("=" * 70)
    print("SETUP INSTRUCTIONS")
    print("=" * 70)

    print("\n📋 Step 1: Get your Kaggle API token")
    print("   1. Go to: https://www.kaggle.com/settings")
    print("   2. Scroll to 'API' section")
    print("   3. Click 'Create New Token'")
    print("   4. This downloads 'kaggle.json' to your Downloads folder\n")

    print("📋 Step 2: Upload kaggle.json to this environment")
    print(f"   The file should be placed at: {kaggle_json}")
    print(f"   Directory: {kaggle_dir}\n")

    print("📋 Step 3: Set permissions")
    print("   Run: chmod 600 ~/.kaggle/kaggle.json\n")

    print("=" * 70)
    print("ALTERNATIVE: Manual Setup")
    print("=" * 70)

    print("\nIf you have your Kaggle credentials, I can create the file for you:")
    print("(Leave blank to skip)")

    username = input("\nKaggle username: ").strip()

    if username:
        key = input("Kaggle API key: ").strip()

        if key:
            # Create directory if needed
            kaggle_dir.mkdir(exist_ok=True)

            # Create kaggle.json
            creds = {
                "username": username,
                "key": key
            }

            with open(kaggle_json, 'w') as f:
                json.dump(creds, f, indent=2)

            # Set secure permissions
            os.chmod(kaggle_json, 0o600)

            print(f"\n✅ Created: {kaggle_json}")
            print("✅ Permissions: 600")

            # Test connection
            print("\n🔌 Testing connection...")
            try:
                import kagglehub
                kagglehub.login()
                print("✅ Connection successful!\n")

                print("=" * 70)
                print("🎉 All set! You can now run:")
                print("   python explore_dataset.py")
                print("=" * 70)

            except Exception as e:
                print(f"❌ Connection test failed: {e}")
        else:
            print("\n⏭️  Skipped - no API key provided")
    else:
        print("\n⏭️  Skipped - no username provided")
        print("\n💡 When you're ready:")
        print("   1. Download kaggle.json from https://www.kaggle.com/settings")
        print(f"   2. Upload it to: {kaggle_json}")
        print("   3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("   4. Run this script again to verify")

print()
