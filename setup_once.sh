#!/bin/bash
# ONE-TIME SETUP: Inject API keys into scripts (permanently, locally)
# Run this ONCE after cloning the repo, then never again!

echo "=========================================="
echo "RIQ MEEPING MACHINE - One-Time Setup"
echo "=========================================="
echo ""

API_SPORTS_KEY="4979ac5e1f7ae10b1d6b58f1bba01140"
KAGGLE_KEY="f005fb2c580e2cbfd2b6b4b931e10dfc"

echo "🔧 Injecting your API keys into scripts..."
echo ""

# Inject into NBA analyzer
if grep -q "YOUR_KEY_HERE" nba_prop_analyzer_fixed.py 2>/dev/null; then
    sed -i "s/YOUR_KEY_HERE/$API_SPORTS_KEY/g" nba_prop_analyzer_fixed.py
    echo "✅ Updated nba_prop_analyzer_fixed.py"
elif grep -q "$API_SPORTS_KEY" nba_prop_analyzer_fixed.py; then
    echo "✅ nba_prop_analyzer_fixed.py already has your key"
else
    echo "⚠️  nba_prop_analyzer_fixed.py might need manual edit"
fi

# Inject into trainer
if grep -q "YOUR_KEY_HERE" train_auto.py 2>/dev/null; then
    sed -i "s/YOUR_KEY_HERE/$KAGGLE_KEY/g" train_auto.py
    echo "✅ Updated train_auto.py"
elif grep -q "$KAGGLE_KEY" train_auto.py; then
    echo "✅ train_auto.py already has your key"
else
    echo "⚠️  train_auto.py might need manual edit"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Your keys are now permanently in the scripts."
echo "You can run them anytime without setup:"
echo ""
echo "  python nba_prop_analyzer_fixed.py"
echo "  python train_auto.py"
echo ""
echo "⚠️  IMPORTANT: Don't commit these files to git!"
echo "   (They have your API keys hardcoded)"
echo ""
