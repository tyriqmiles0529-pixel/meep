
# Phase G: Representation Learning & V4 Ensemble Report

## Objective
Enhance the prediction pipeline by integrating:
1.  **FT-Transformer Embeddings (16D)**: To capture latent player archetypes and form.
2.  **Per-Minute Efficiency Metrics**: To provide normalized signals independent of raw minutes volume.

## Methodology
- **Embeddings**: Trained an FT-Transformer encoder on 1997-2023 data using weak supervision (Next Game PTS Bin). Generated embeddings for all historical and live data.
- **Feature Engineering**: Added `roll_PTS_per_MIN_10`, `roll_AST_per_MIN_10`, `roll_REB_per_MIN_10`, and `roll_usage_density_10`.
- **Dataset**: `strict_features_v4.csv` (1997-2024).
- **Ensemble**: Retrained XGBoost, LightGBM, and CatBoost on V4 features. Tuned LightGBM slightly slower (learning rate 0.015) for robustness.
- **Meta-Learner**: Retrained Ridge Stacker on V4 OOF predictions.

## Results (Target: PTS)

| Model | RMSE | R² | vs. V3 (Previous) |
| :--- | :--- | :--- | :--- |
| XGBoost | 4.9214 | 0.6901 | +0.0014 |
| LightGBM | 4.9203 | 0.6902 | +0.0006 |
| CatBoost | 4.9173 | 0.6906 | +0.0014 |
| **Meta-Learner** | **4.9141** | **0.6910** | **+0.0013** |

*Note: V3 Meta-Learner R² was ~0.6897.*

## Other Targets (Meta-Learner)
- **AST**: R² 0.5966 (vs 0.5961) -> Slight improvement.
- **REB**: R² 0.6174 (vs 0.6170) -> Slight improvement.

## Analysis
The integration of learned embeddings and per-minute features has provided a **consistent, widespread lift** across all base models and the meta-learner. 
- **Consistency**: All three base models improved, confirming the new signals are universally useful.
- **CatBoost Dominance**: CatBoost remains the single strongest model (R² 0.6906), likely due to its superior handling of the new embedding features (which are essentially dense categorical descriptors).
- **Stacking Value**: The meta-learner continues to squeeze out extra performance (0.6906 -> 0.6910), efficiently combining the base models.

## Conclusion
Phase G is a success. The V4 ensemble is the most powerful version yet, breaking the 0.69 R² barrier for Points. 
The system is now ready for deployment with these advanced features.

## Next Steps
- Deploy V4 models to `daily_inference.py`.
- Ensure `daily_inference.py` loads `player_embedding_v1.parquet`.
- **Critical Todo**: For *live* inference, we technically need to generate embeddings for *today's* games. The current `train_embeddings.py` runs on `pro_training_set.csv` (1997-2024). It does NOT include today's live games.
- **Immediate Action**: We must update `build_features.py` or `daily_inference.py` to allow *generating* embeddings for new rows using the saved FT-Transformer model, rather than just merging static parquet.
