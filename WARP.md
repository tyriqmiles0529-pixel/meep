# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project focus: NBA game and player prop prediction with ML models, ensemble methods, and an ELG (Expected Log Growth) betting framework. Two main flows coexist: a full ML training/inference pipeline and a fast standalone analyzer under meep/.

Commands you’ll use often

- Environment (PowerShell on Windows)
  - Optional venv
    - python -m venv .venv
    - .\.venv\Scripts\Activate.ps1
  - Install core deps (minimal for analyzer and training)
    - pip install numpy pandas scikit-learn lightgbm requests kagglehub nba_api scipy

- Train models (end-to-end, pulls Kaggle data)
  - python .\train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose --skip-rest --fresh
  - Artifacts written under models/:
    - Player: points_model.pkl, assists_model.pkl, rebounds_model.pkl, threes_model.pkl, minutes_model.pkl
    - Game: moneyline_model.pkl, moneyline_calibrator.pkl, spread_model.pkl
    - Metadata: training_metadata.json, spread_sigma.json

- Train enhanced ensemble (adds Ridge, Elo, Four Factors, meta-learner)
  - python .\train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --fresh --verbose
  - Enhanced artifacts (if integrated): ridge_model_enhanced.pkl, elo_model_enhanced.pkl, four_factors_model_enhanced.pkl, ensemble_meta_learner_enhanced.pkl

- Run analyzers
  - Full ML-integrated analyzer: python .\riq_analyzer.py
  - Fast standalone analyzer (heuristics + ELG): python .\nba_prop_analyzer_fixed.py
  - Example ensemble usage: python .\example_ensemble_usage.py
  - Daily predictions helper: python .\run_daily_predictions.py

- “Tests” (script-based checks)
  - Verify model artifacts load: python .\test_model_loading.py
  - API diagnostics (manual):
    - python .\test_sgo_api.py
    - python .\meep\test_api_services.py
  - Run a single test/script: python .\test_model_loading.py (there is no pytest config here)

High-level architecture and flow

- Data sources and caching
  - Historical data via Kaggle (kagglehub). train_auto.py discovers TeamStatistics and PlayerStatistics CSVs in the selected dataset and constructs modeling frames.
  - Optional live augmentation via nba_api for current season games/players.
  - Historical odds and player props via The Odds API (with local CSV caches to control costs).

- Training pipeline (train_auto.py)
  - Build games from TeamStatistics (build_games_from_teamstats):
    - Resolves CSV variants, derives home/away pairing, constructs season features (season_end_year/season_decade), recent pace/off/def strength and matchup edges, optional rest/b2b features, and merges historical priors when available.
  - Fit game models (_fit_game_models):
    - Moneyline classifier with optional isotonic calibration; spread regressor; time-ordered folds for OOF predictions (oof_ml_prob, oof_spread_pred) used downstream; metrics/logloss, RMSE/MAE, spread sigma estimated from residuals.
  - Build player frames (build_players_from_playerstats):
    - Robust column resolution for varied PlayerStatistics CSVs; constructs player rolling trends, home/away splits, per-minute rates, rest and back-to-back flags, and joins game context by teamId or home/away side. Injects OOF game signals to avoid leakage. Supports optional Basketball Reference priors. Produces training sets for minutes, points, rebounds, assists, threes.
  - Player model training and registry
    - Trains regression models for the five player targets, writes pickles + training_metadata.json and spread_sigma.json; ensures feature schema consistency across train/infer.
  - Enhanced ensemble integration (optional; see INTEGRATION_GUIDE_ENHANCED.md)
    - train_all_ensemble_components trains Ridge (score diff), Dynamic Elo, Four Factors (rolling priors), and a Logistic meta-learner with polynomial interactions and refit cadence (typically 20 games), emitting enhanced model artifacts and coefficient_evolution.csv/ensemble_analysis.txt.

- Inference and analysis
  - riq_analyzer.py
    - Loads player and game models and training metadata; computes player prop probabilities and ranks opportunities with ELG using dynamic fractional Kelly; game models are loaded and staged for integration in bet analysis (moneyline/spread) with inverse-variance blending against market when enabled.
  - meep/ (standalone path)
    - nba_prop_analyzer_fixed.py implements a self-contained fast analyzer: heuristic projections → prop distributions (Normal/Negative Binomial) → Beta posterior sampling → dynamic Kelly → ELG ranking; designed for quick runs without the full training pipeline.

What matters from the repository docs

- QUICKSTART_ENSEMBLE.md
  - One-command training expands your pipeline to add Ridge + Elo + Four Factors + meta-learner; expect four new model files in models/ and small additional training time.

- INTEGRATION_GUIDE_ENHANCED.md
  - Use train_all_ensemble_components in train_auto.py after base LGB training; artifacts saved as ridge_model_enhanced.pkl, elo_model_enhanced.pkl, four_factors_model_enhanced.pkl, ensemble_meta_learner_enhanced.pkl, with coefficient_evolution.csv and ensemble_analysis.txt for interpretability; refit frequency of ~20 games is a good default.

- MODEL_INTEGRATION_SUMMARY.md
  - riq_analyzer.py fixes prior feature mismatches, loads both player and game models, and exposes predict_moneyline/predict_spread; ensures model files and RMSEs are read from training_metadata.json; spread uncertainty comes from spread_sigma.json.

- meep/README.md and QUICK_START.md
  - For a quick ELG-driven analysis without ML training, run python nba_prop_analyzer_fixed.py after installing requests/pandas/numpy/lightgbm/scikit-learn; FAST_MODE targets sub-minute runtimes and outputs top props plus a JSON dump of details.

Conventions and notes

- No dedicated linter/formatter config is present.
- Many scripts hit external APIs. Prefer running local training and artifact validation scripts when iterating; only call API-backed scripts intentionally.
- Models and caches live in the repo root (e.g., models/, *.pkl, cached CSVs). Clean up or .gitignore locally as needed.
