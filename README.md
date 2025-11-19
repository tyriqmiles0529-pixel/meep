# NBA Player Props Predictor

**Advanced NBA player performance prediction system** using ensemble learning, context-aware meta-learning, and 27 window models trained on complete NBA history (1947-2026).

**Latest Update (Nov 19, 2025):**
- ✅ **Player Props Only** - Points, Rebounds, Assists, Threes (no game betting)
- ✅ **27-Window Ensemble** - Historical models from different NBA eras (1947-2026)
- ✅ **Context-Aware Meta-Learner** - LightGBM stacking with player/game context
- ✅ **Modal Cloud Training** - 16 CPU cores, 32GB RAM for meta-learner training
- ✅ **Dual API Integration** - TheOdds API + API-Sports for player props
- ✅ **Kaggle Training Data** - PlayerStatistics.csv (1.6M+ box scores)

## 🎯 Overview

**A production-ready NBA analytics platform** that predicts player performance using:

### 🤖 Advanced Ensemble Architecture

- **27 Window Models**: Hybrid multi-task TabNet models trained on 3-year windows
  - Windows span complete NBA history (1947-2026)
  - Each window specializes in its era's playing style
  - ~150 features per window (rolling stats, advanced metrics, priors)

- **Context-Aware Meta-Learner**: Intelligent stacking ensemble
  - LightGBM learns optimal weights for 27 window predictions
  - Player context features (position, usage rate, minutes, home/away)
  - Out-of-fold training prevents data leakage
  - Expected: 10-15% RMSE improvement over simple averaging

- **Player Props Focus**:
  - Points, Rebounds, Assists, Three-Pointers
  - No game-level betting (moneyline, spread, totals removed)
  - Fetches props from TheOdds API + API-Sports

### 📊 Training Data

- **Kaggle Dataset**: PlayerStatistics.csv (1.6M+ player box scores, 1947-2026)
- **27 Window Models**: Pre-trained on Modal (available in `nba-models` volume)
- **Meta-Learner Training**: 2024-2025 season (trains on Modal, ~30 min)
- **Live Integration**: Real-time props from TheOdds API + API-Sports

## 🚀 Quick Start - Modal Cloud Workflow

### Prerequisites

```bash
# Install Modal
pip install modal

# Setup Modal credentials
modal setup

# Clone repository
git clone https://github.com/tyriqmiles0529-pixel/meep.git
cd meep
```

### Workflow

#### 1️⃣ **Upload Training Data to Modal**

```bash
# Upload Kaggle PlayerStatistics.csv to Modal volume
modal volume put nba-data PlayerStatistics.csv PlayerStatistics.csv
```

**Get PlayerStatistics.csv:**
- Download from [Kaggle](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/)
- File: `PlayerStatistics.csv` (~1.6M rows, all NBA box scores 1947-2026)

#### 2️⃣ **Train Meta-Learner on Modal**

```bash
# Train context-aware meta-learner (uses 27 pre-trained window models)
# Resources: 16 CPU cores, 32GB RAM, ~30-60 minutes
modal run modal_train_meta.py
```

**What this does:**
- Loads 27 window models from Modal volume (`nba-models`)
- Loads 2024-2025 season from PlayerStatistics.csv
- Gets predictions from all 27 windows for each game
- Trains LightGBM meta-learner with player context
- Saves `meta_learner_2025_2026.pkl` to Modal volume

**Expected Output:**
```
======================================================================
META-LEARNER TRAINING COMPLETE
======================================================================
  Saved: /models/meta_learner_2025_2026.pkl
  Props trained: 4

Results:
  points      : 15,234 samples, +12.3% improvement
  rebounds    : 14,891 samples, +10.8% improvement
  assists     : 15,102 samples, +11.5% improvement
  threes      : 14,567 samples, +9.7% improvement
======================================================================
```

#### 3️⃣ **Run Analyzer on Modal**

```bash
# Get today's player prop predictions
# Resources: 8 CPU cores, 16GB RAM
modal run modal_analyzer.py
```

**What this does:**
- Loads 27 window models + meta-learner from Modal volume
- Fetches today's games from API-Sports
- Fetches player props from TheOdds API + API-Sports
- Generates predictions using ensemble + meta-learner
- Outputs: Recommended player props with confidence scores

**Expected Output:**
```
======================================================================
TODAY'S PLAYER PROP PREDICTIONS
======================================================================

Game: Lakers vs Warriors (7:30 PM ET)

OVER PICKS (High Confidence):
  LeBron James - Points OVER 24.5 (Proj: 27.3, Edge: +2.8, Conf: 62%)
  Stephen Curry - Threes OVER 3.5 (Proj: 4.8, Edge: +1.3, Conf: 58%)

UNDER PICKS (High Confidence):
  Anthony Davis - Rebounds UNDER 11.5 (Proj: 9.2, Edge: -2.3, Conf: 59%)

======================================================================
```

### Configure API Secrets

**Required:** Set up API keys in Modal before running analyzer:

```bash
# Set TheOdds API key (for player props)
modal secret create theodds-api-key THEODDS_API_KEY=your_key_here

# Set API-Sports key (for games and player props)
modal secret create api-sports-key API_SPORTS_KEY=your_key_here
```

**Get API Keys:**
- TheOdds API: https://the-odds-api.com/ (player props)
- API-Sports: https://api-sports.io/documentation/nba/v2 (games + player props)

## 🔧 Technical Architecture

### Key Components

#### 1. Ensemble Predictor (`ensemble_predictor.py`)
- **Purpose**: Load and use 27 window models with meta-learner
- **Windows**: 27 overlapping 3-year windows (1947-2026)
- **Architecture**: Hybrid multi-task TabNet + LightGBM per window
- **Features**: ~150 per window (rolling stats, advanced metrics, priors)
- **Inference**: Collects predictions from all windows, feeds to meta-learner

#### 2. Meta-Learner (`meta_learner_ensemble.py`)
- **Purpose**: Context-aware stacking to learn optimal window weights
- **Architecture**: LightGBM regressor
- **Input Features**:
  - 27 window predictions (base models)
  - Prediction statistics (mean, std, min, max, CV)
  - Player context (position, usage rate, minutes, home/away)
  - Interaction features (position × pred_mean, usage × pred_std)
- **Training**: Out-of-fold (5-fold CV) on 2024-2025 season
- **Expected Improvement**: +10-15% RMSE over simple averaging

#### 3. Modal Training (`modal_train_meta.py`)
- **Purpose**: Train meta-learner on Modal cloud
- **Resources**: 16 CPU cores, 32GB RAM, 2-hour timeout
- **Data**: PlayerStatistics.csv (filters to 2024-2025 season)
- **Season Logic**: Handles NBA season spanning two calendar years (Oct-June)
- **Output**: `meta_learner_2025_2026.pkl` saved to Modal volume

#### 4. Modal Analyzer (`modal_analyzer.py`)
- **Purpose**: Run daily predictions on Modal
- **Resources**: 8 CPU cores, 16GB RAM, 1-hour timeout
- **APIs**: TheOdds API + API-Sports (both enabled for player props)
- **Filtering**: NBA games only (league_id == 12), no college basketball
- **Output**: High-confidence player prop recommendations

#### 5. RIQ Analyzer (`riq_analyzer.py`)
- **Purpose**: Core prediction engine
- **Props**: Player props only (points, rebounds, assists, threes)
- **Data Sources**:
  - TheOdds API: Player props (points, rebounds, assists, threes)
  - API-Sports: Games + player props
- **Filtering**: NBA-only, team name normalization for matching
- **Ensemble Mode**: Uses `EnsemblePredictor` with meta-learner if available

### Meta-Learner Training Details

**Input to Meta-Learner (per player-game):**
```python
# Base predictions from 27 windows
window_predictions = [25.3, 24.8, 26.1, ..., 25.9]  # Shape: (27,)

# Player context features
player_context = {
    'position_encoded': 2,      # SF=2
    'usage_rate': 0.28,          # 28% usage
    'minutes_avg': 34.5,         # 34.5 mpg
    'is_home': 1                 # Home game
}

# Meta-learner creates ~35 features:
# - 27 window predictions
# - 5 stats (mean, std, min, max, CV)
# - 4 context features
# - 3 interactions (position×mean, usage×std, minutes×mean)
```

**Training Process:**
1. Load 27 window models from Modal volume
2. Load 2024-2025 season from PlayerStatistics.csv (~15k games)
3. Sample 5,000 games for speed (or use all for max accuracy)
4. For each game:
   - Get predictions from all 27 windows
   - Extract player context (position, usage, minutes, home/away)
   - Record actual outcome
5. Train LightGBM with 5-fold Out-of-Fold CV
6. Save `meta_learner_2025_2026.pkl`

**Expected Performance:**
- Baseline (simple average): RMSE = 5.12
- Meta-learner ensemble: RMSE = 4.65
- **Improvement: +9-12% RMSE reduction**

### Model Performance Validation

The system includes comprehensive validation:

1. **Out-of-Fold Training**: Meta-learner trained with 5-fold CV (prevents overfitting)
2. **Temporal Separation**: Meta-learner trained on 2024-2025, used for 2025-2026
3. **Window Independence**: 27 windows never see future data (trained up to their end year)
4. **Feature Alignment**: Each window uses only features available in its era

## 📁 Project Structure

```
nba_predictor/
├── Modal Deployment/
│   ├── modal_train_meta.py        # Train meta-learner on Modal (16 cores, 32GB)
│   ├── modal_analyzer.py          # Run analyzer on Modal (8 cores, 16GB)
│   └── .modal.toml                # Modal configuration
│
├── Ensemble System/
│   ├── ensemble_predictor.py      # Load 27 windows + meta-learner
│   ├── meta_learner_ensemble.py   # Context-aware stacking meta-learner
│   └── hybrid_multi_task.py       # Multi-task TabNet architecture
│
├── Core Prediction/
│   ├── riq_analyzer.py            # Daily predictions (player props only)
│   └── train_meta_learner.py      # Local training script (alternative)
│
├── Feature Engineering/
│   ├── optimization_features.py   # Phase 6: Momentum, trend detection
│   ├── phase7_features.py         # Phase 7: Basketball Reference priors
│   └── rolling_features.py        # Rolling stats (L3, L5, L7, L10, L15)
│
├── Data/
│   ├── shared/                    # Shared utilities (data loading, CSV aggregation)
│   ├── priors_data/               # Basketball Reference priors (68 features)
│   └── PlayerStatistics.csv       # Kaggle dataset (1.6M box scores) - NOT IN GIT
│
└── Documentation/
    ├── README.md                  # This file
    ├── META_LEARNER_PLAN.md       # Meta-learner implementation details
    └── MODAL_SETUP_GUIDE.md       # Modal cloud setup guide
```

## 📈 Current Status

### System Status (Nov 19, 2025)

| Component | Status | Notes |
|-----------|--------|-------|
| **27 Window Models** | ✅ Pre-trained | Available in Modal `nba-models` volume |
| **Meta-Learner Training** | ✅ Working | Trains on 2024-2025 season via Modal |
| **Modal Analyzer** | ✅ Production | Fetches props from both APIs, NBA-only filtering |
| **Player Props Only** | ✅ Implemented | Game-level betting removed |
| **API Integration** | ✅ Dual APIs | TheOdds + API-Sports for player props |

### Roadmap

1. **Monitor Meta-Learner Performance** (Target: +10-15% improvement)
   - Track predictions vs actuals for 2025-2026 season
   - Compare meta-learner vs simple averaging
   - Retrain monthly with new data

2. **Expand Training Data**
   - Currently: 2024-2025 season (~15k games)
   - Future: Use multiple seasons for meta-learner (2022-2025)
   - Expected: More robust learned weights

3. **Feature Engineering**
   - Add more player context (recent form, opponent defense)
   - Include game situation (playoff vs regular season)
   - Matchup-specific adjustments

## 💡 Usage Examples

### Train Meta-Learner on Modal

```bash
# Upload training data
modal volume put nba-data PlayerStatistics.csv PlayerStatistics.csv

# Train meta-learner (saves to Modal volume)
modal run modal_train_meta.py

# Optional: Train for specific season
modal run modal_train_meta.py --season "2023-2024"
```

### Run Analyzer on Modal

```bash
# Get today's predictions (uses ensemble + meta-learner)
modal run modal_analyzer.py

# Run without ensemble (simple averaging)
modal run modal_analyzer.py --use-ensemble=false
```

### Local Testing (without Modal)

```bash
# Install dependencies
pip install -r requirements.txt

# Download models from Modal
python download_all_models.py

# Train meta-learner locally
python train_meta_learner.py

# Run analyzer locally (requires models + priors)
python riq_analyzer.py --use-ensemble
```

## 🔒 Research Use Only

**This project is for educational, research, and portfolio demonstration purposes only.**

- Demonstrates advanced ensemble learning and meta-learning
- Showcases cloud-based ML infrastructure (Modal)
- Explores NBA analytics and predictive modeling
- **Not intended for any wagering activities**

Focus areas: Ensemble learning, stacking, context-aware ML, sports analytics.

## 📧 Contact

**Tyriq Miles**
- GitHub: [@tyriqmiles0529-pixel](https://github.com/tyriqmiles0529-pixel)
- Email: tyriqmiles0529@gmail.com

---

## 🆕 Recent Updates

### Nov 19, 2025 - Meta-Learner + Player Props Only

**Context-Aware Meta-Learner:**
- Trains on Modal (16 cores, 32GB) using PlayerStatistics.csv
- Learns optimal weights for 27 window predictions
- Uses player context (position, usage, minutes, home/away)
- Out-of-fold training prevents overfitting
- Expected: +10-15% RMSE improvement

**Player Props Only:**
- Removed all game-level betting (moneyline, spread, totals)
- Enabled TheOdds API + API-Sports for player props
- NBA-only filtering (league_id == 12)
- Team name normalization for better matching

**Kaggle Data Integration:**
- Updated `modal_train_meta.py` to load from PlayerStatistics.csv
- Handles mixed datetime formats (ISO8601 + "YYYY-MM-DD HH:MM:SS")
- Extracts season from gameDate (handles Oct-June NBA season logic)
- Maps to Kaggle column names (points, assists, reboundsTotal, threePointersMade)

**Git Cleanup:**
- Added large data files to .gitignore (*.csv, *.parquet, *.gzip, *.zip)
- Prevents 700MB+ files from being pushed to GitHub

---

*Last Updated: November 19, 2025*
*Version: 9.0 (Meta-Learner + Player Props Only)*
*Status: Production-Ready on Modal*
