---
description: Daily routine for NBA Paper Trading & GUI Analytics
---

# NBA Elite Predictor: Daily Workflow (GUI Edition)

Follow this routine every day to maintain data integrity and generate high-EV picks.

## 1. Launch the Terminal
Launch the sleek GUI to manage your slate and analytics.

```powershell
python run_terminal.py
```

## 2. Daily Data Sync (Morning)
Once the app opens, navigate to the sidebar and click **"🔄 REFRESH DATA"** (or "Sync NBA Data").
- This fetches the latest box scores from last night.
- It automatically optimizes your master file (keeping 2021+ data) to prevent MemoryErrors.
- It recalculates all player streaks and rolling averages.

## 3. Generate & Review Picks
Go to the **"🎯 COMMAND"** tab (or "Value Picks" tab):
- Click **"🚀 GENERATE PICKS"**.
- Review the **Heat Score** and **Tier** for each projection.
- Use the **Expected Value (EV)** slider to filter for the best edges.

## 4. Deep Dive Analytics
Before locking in a high-stake bet, use the **"🔬 ANALYTICS PLATFORM"** tab:
- Search for the player (filtered to active 2025/26 players only).
- Check the **"🧬 ARCHETYPE DNA"** tab to see how they compare to their style-peers.
- Switch the graph axes to **Impact (BPM)** to see if their value is defensive or box-score driven.

## 5. Execution
- Add your favorite picks to the **Bet Slip**.
- View your historical wins/losses in **"📊 PERFORMANCE"**.

## Troubleshooting
- **Memory Errors**: If you see "Unable to allocate", the system will auto-retry in Ultra-Low memory mode. Ensure you've clicked "Refresh Data" at least once to trim the file.
- **Missing Player**: If a player isn't in the list, they might not have played enough minutes this season yet. Check the "Analyzing Profile... 🧪" status.
