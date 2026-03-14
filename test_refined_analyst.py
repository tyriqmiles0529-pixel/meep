
import json
from unittest.mock import MagicMock
from meep_terminal.core.ai_analyst import MEEPAnalyst

def test_refined_optimizations():
    # Mock Engine
    mock_engine = MagicMock()
    mock_engine.get_stats.return_value = {"status": "Healthy", "bankroll": 1000}
    mock_engine.get_portfolio_slate.return_value = {} # Empty slate
    mock_engine.get_player_correlation.return_value = None # Simulate missing data

    analyst = MEEPAnalyst(mock_engine, api_key="fake_key")
    
    # 1. Test Batch Intelligence
    print("\n[BATCH TEST]")
    res_batch = analyst.call_tool("get_batch_intelligence", {"player_names": ["LeBron", "Curry"]})
    print(f"Batch Output: {res_batch}")
    assert "analytics_unavailable" in res_batch
    
    # 2. Test Error Standardization
    print("\n[ERROR STANDARDIZATION TEST]")
    res_slate = analyst.call_tool("get_slate", {"date": "2026-03-08"})
    print(f"Slate (Empty) Output: {res_slate}")
    assert "analytics_unavailable" in res_slate
    
    # 3. Test Cache (Re-run batch)
    mock_engine.get_player_correlation.reset_mock()
    analyst.call_tool("get_batch_intelligence", {"player_names": ["LeBron", "Curry"]})
    # Should NOT have called engine again
    mock_engine.get_player_correlation.assert_not_called()
    print("✅ Cache hit confirmed for batch.")

    print("\n✅ Refined optimizations verified!")

if __name__ == "__main__":
    test_refined_optimizations()
