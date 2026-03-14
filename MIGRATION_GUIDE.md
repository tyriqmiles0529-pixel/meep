# 🏀 Project Migration Guide: NBA Prediction System (meep V4)

## 📋 Project Overview
This is a production-grade NBA Player Prop prediction system. It uses an **Ensemble Model (V4)** which combines XGBoost, TabNet, and LightGBM to predict PTS, AST, REB, 3PM, and MIN. The system calculates **Expected Value (EV)** and **Kelly Criterion** stake sizes by comparing projections against live market lines from "The Odds API."

---

## 🏗️ Core Architecture
1.  **Feature Matrix**: 235 features per player-game, including advanced rolling metrics, opponent defensive context, and embeddings.
2.  **Integrity Guard**: A semantic layer that checkpoints data freshness and strips non-numeric metadata before prediction to prevent XGBoost schema crashes.
3.  **Betting Logic**: Calculates win probability via normal distribution (mu/sigma) and applies a "Safe Mode" margin.

---

## 🗂️ Critical Files (The "Must-Haves")
| File | Purpose |
| :--- | :--- |
| `predict_live_FINAL.py` | **The Brain.** Entry point for predictions, EV calculation, and betting integration. |
| `daily_refresh.py` | **Automation.** Single script to sync NBA API logs and update the Feature Matrix. |
| `models/production_v4/` | **Artifacts.** Contains the `.joblib` model files and the canonical `features.joblib` schema. |
| `final_feature_matrix_...csv` | **Database.** The primary 800MB+ historical dataset for both inference and future training. |
| `integrity_guard.py` | **Safety.** Detects "stale" data (3+ days old) and adjusts prediction confidence automatically. |
| `FEATURE_INVENTORY.md` | **Documentation.** Complete list of all 235 stats used by the models. |

---

## 🛠️ Immediate Next Steps for the New Agent
### 1. Renew "The Odds API" Key (URGENT)
The current key (`feb98...`) is **expired/deactivated**. 
*   **Action**: New agent must help user input a new `THE_ODDS_API_KEY` into the environment variables or `run_phase_i.py`.
*   **Status**: Models are safe, but live pick generation will fail until this is fixed.

### 2. Perform a "Daily Refresh"
Data is currently synced up to Feb 9th/10th.
*   **Action**: Run `python daily_refresh.py`.
*   **Goal**: Fetch all player logs from the last few days to ensure tonight's predictions are based on the latest performance stats.

### 3. GUI Development (Phase S3 Plan)
The user wants a visual dashboard to replace the terminal output.
*   **Recommendation**: Use **Streamlit** or **Next.js** to create a "Betting Dashboard."
*   **Features**:
    *   Sortable table of Top +EV Picks.
    *   Confidence Meters (based on `integrity_guard.py`).
    *   Parlay Builder interface.

---

## 🚀 Daily Workflow for New Agent
Every day, the project should follow this sequence:
1.  `python daily_refresh.py` (Syncs data).
2.  `python predict_live_FINAL.py --refresh --betting` (Generates picks).
3.  Review `[Today's Date] - Riq's Paper Picks.md` for the final betting card.

---

## 📝 Note for the Incoming AI
The system was recently patched for **Schema Integrity**. Do NOT modify `get_player_features` in `predict_live_FINAL.py` without referring to the `models/production_v4/features.joblib` schema, or XGBoost will throw "KeyError: 'object'" or "Feature name mismatch" errors.
