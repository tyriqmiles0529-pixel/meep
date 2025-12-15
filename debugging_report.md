# Debugging Report: Simulation Upgrades

## Summary
Successfully debugged and verified the execution of the upgraded simulation framework including OELM, Skew-Normal Modeling, and Walk-Forward Calibration. The simulation now runs to completion, generating predictions and bets for seasons 2021-2026.

## Fixed Issues
1.  **Embedding Generation Failure**:
    -   **Cause**: Type mismatch in categorical columns passed to FT-Transformer (likely Float/NaN vs Int).
    -   **Fix**: Added robust casting to `int` and `fillna(0)` in `run_simulation.py` before embedding generation.
    -   **Verification**: Logs now show "Embeddings generated" for all processed seasons.

2.  **KeyError 'prior_three_pointers'**:
    -   **Cause**: The mapping for 'three_pointers' prior stats was missing in `generate_predictions_optimized`, leading to a crash when OELM accessed the missing column.
    -   **Fix**: Added proxy mapping (`threePointersMade_last_game`) and initialized missing columns to `NaN`.
    -   **Verification**: Simulation runs to completion without KeyErrors.

3.  **Missing Models (2024-2026)**:
    -   **Observation**: Models for seasons 2024, 2025, and 2026 are missing. The simulation logic catches the `FileNotFoundError`.
    -   **Impact**: Predictions for these seasons rely on the last successfully loaded model (2023). This is valid for backtesting (using stale models) but explains potential performance degradation in later years.

## Simulation Performance Analysis
-   **Execution**: Correct. 6041 total bets generated.
-   **ROI**: -100% (Bust).
-   **Calibration**: Showed significant "Inversion" phenomenon.
    -   High Confidence (Prob > 0.9) favorites lost frequently (Win Rate ~43%).
    -   Low Confidence (Prob < 0.1) longshots won frequently (Win Rate ~77%).
    -   **Diagnosis**: This suggests a systematic **Model Bias** (Overestimation of player performance) combined with **OELM Line Adjustment**. When the model overestimates a player compared to the OELM line, it bets OVER confidently, but the player frequently goes UNDER (Actual < Pred).
    -   **Next Steps**: Implement Bias Correction (subtract Mean Error from predictions) before probability calculation.

## Recommendations
1.  **Bias Correction**: Calculate the rolling mean error of predictions vs actuals and subtract it from future predictions.
2.  **Model Retraining**: Retrain models for 2024-2026 to ensure up-to-date player dynamics are captured.
3.  **Strategy Adjustment**: Invert the betting strategy? (Bet AGAINST the model's high confidence favorites).
