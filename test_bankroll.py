
from meep_terminal.core.engine import TerminalEngine
from meep_terminal.data.models import DatabaseManager

def test_bankroll_persistence():
    engine = TerminalEngine()
    
    # 1. Reset/Clear preferences for clean test
    engine.update_user_preferences({"bankroll": None})
    
    # 2. Get initial stats
    initial_stats = engine.get_stats()
    initial_bankroll = initial_stats.get('bankroll')
    print(f"Initial Bankroll: ${initial_bankroll}")
    
    # 3. Update bankroll
    test_value = 5500.0
    print(f"Setting manual bankroll to ${test_value}...")
    engine.update_user_preferences({"bankroll": test_value})
    
    # 4. Verify updated stats
    updated_stats = engine.get_stats()
    updated_bankroll = updated_stats.get('bankroll')
    print(f"Updated Bankroll: ${updated_bankroll}")
    
    assert updated_bankroll == test_value, f"Expected {test_value}, got {updated_bankroll}"
    print("✅ Bankroll persistence verified!")
    
    # 5. Cleanup
    engine.update_user_preferences({"bankroll": None})

if __name__ == "__main__":
    try:
        test_bankroll_persistence()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
