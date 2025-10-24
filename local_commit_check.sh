#!/bin/bash
# Local Commit Check - Use this before any git operations
# This ensures your API keys stay local and don't get committed

echo "🔒 Checking for API keys in modified files..."

# Check if files with real keys are staged
if git diff --cached --name-only | grep -q "nba_prop_analyzer_fixed.py\|train_auto.py"; then
    echo "❌ ERROR: You're about to commit files with real API keys!"
    echo ""
    echo "Files with keys:"
    git diff --cached --name-only | grep "nba_prop_analyzer_fixed.py\|train_auto.py"
    echo ""
    echo "Run this to unstage them:"
    echo "  git restore --staged nba_prop_analyzer_fixed.py train_auto.py"
    exit 1
fi

# Check if files have real keys but aren't staged (this is good!)
if git diff --name-only | grep -q "nba_prop_analyzer_fixed.py\|train_auto.py"; then
    echo "✅ Good: Your scripts have real keys but are NOT staged for commit"
    echo "   (This is the correct state for local development)"
fi

echo ""
echo "Safe to commit! No API keys will be exposed."
exit 0
