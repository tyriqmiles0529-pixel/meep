
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from predict_live_FINAL import LivePredictionEngine

try:
    engine = LivePredictionEngine(models_dir="./models", aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv")
    print("Models loaded:", list(engine.models.keys()))
except Exception as e:
    import traceback
    traceback.print_exc()
