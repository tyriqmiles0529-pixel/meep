
import unittest
from unittest.mock import MagicMock, patch
from meep_terminal.core.engine import TerminalEngine

class TestBankrollLogic(unittest.TestCase):
    @patch('meep_terminal.core.engine.LivePredictionEngine')
    @patch('meep_terminal.core.engine.DatabaseManager')
    @patch('meep_terminal.core.engine.sqlalchemy')
    def test_bankroll_override(self, mock_sql, mock_db, mock_predict):
        # Setup mocks
        engine = TerminalEngine()
        
        # Mock get_user_preferences to return a manual bankroll
        engine.get_user_preferences = MagicMock(return_value={'bankroll': 5000.0})
        
        # Mock session and queries to prevent hitting 'except'
        mock_session = MagicMock()
        engine.db.get_session.return_value = mock_session
        mock_session.query.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.count.return_value = 0
        
        # Mock the engine's internal data for date detection
        engine.engine.aggregated_data = MagicMock()
        engine.engine.aggregated_data.columns = ['date']
        
        # Run get_stats
        stats = engine.get_stats()
        
        # Verify
        self.assertEqual(stats['bankroll'], 5000.0)
        print("Bankroll override logic verified (Mock)!")

    @patch('meep_terminal.core.engine.LivePredictionEngine')
    @patch('meep_terminal.core.engine.DatabaseManager')
    @patch('meep_terminal.core.engine.sqlalchemy')
    def test_bankroll_fallback(self, mock_sql, mock_db, mock_predict):
        # Setup mocks
        engine = TerminalEngine()
        
        # Mock get_user_preferences to return NO manual bankroll
        engine.get_user_preferences = MagicMock(return_value={})
        
        # Mock session queries for get_stats to return 0 staked
        mock_session = MagicMock()
        engine.db.get_session.return_value = mock_session
        mock_session.query.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.count.return_value = 0
        
        # Mock the engine's internal data for date detection
        engine.engine.aggregated_data = MagicMock()
        engine.engine.aggregated_data.columns = ['date']
        
        # Run get_stats
        stats = engine.get_stats()
        
        # Verify (Fallback should be 1000.0 + 0*0.024)
        self.assertEqual(stats['bankroll'], 1000.0)
        print("Bankroll fallback logic verified (Mock)!")

if __name__ == "__main__":
    unittest.main()
