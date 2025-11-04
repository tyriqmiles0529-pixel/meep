"""
Weighted Team Context Model
Based on correlation analysis of team factors to player stats.

Impact weights scale from -3 to +3:
  +3 = strong positive correlation
  -3 = strong negative correlation
   0 = little to no direct effect
"""

import numpy as np
from typing import Dict
from collections import defaultdict


class WeightedTeamContext:
    """
    Team context with empirically-derived weights for each stat type.

    Factors:
    - Pace (tempo)
    - Offensive scheme (motion/iso)
    - Usage distribution
    - Defensive scheme
    - Team efficiency (ORTG/DRTG)
    - Lineup composition
    - Role & rotation
    - Rebounding environment
    - Transition frequency
    - Opponent pace/defense
    """

    # Impact weights for each stat (-3 to +3)
    IMPACT_WEIGHTS = {
        'points': {
            'pace': 3,
            'offensive_scheme': 2,
            'usage_distribution': 3,
            'defensive_scheme': 0,
            'team_efficiency': 2,
            'lineup_quality': 1,
            'role_rotation': 2,
            'rebounding_env': 0,
            'transition_freq': 2,
            'opponent_defense': 2
        },
        'assists': {
            'pace': 2,
            'offensive_scheme': 3,
            'usage_distribution': -2,  # High usage = fewer assists
            'defensive_scheme': 0,
            'team_efficiency': 2,
            'lineup_quality': 3,
            'role_rotation': 1,
            'rebounding_env': 0,
            'transition_freq': 1,
            'opponent_defense': 1
        },
        'rebounds': {
            'pace': 2,
            'offensive_scheme': 0,
            'usage_distribution': 0,
            'defensive_scheme': 1,
            'team_efficiency': 1,
            'lineup_quality': 1,
            'role_rotation': 1,
            'rebounding_env': 3,  # Shot mix & spacing matter most
            'transition_freq': 1,
            'opponent_defense': 1
        },
        'threes': {
            'pace': 2,
            'offensive_scheme': 3,  # Spacing & motion offense
            'usage_distribution': 2,
            'defensive_scheme': 0,
            'team_efficiency': 2,
            'lineup_quality': 2,
            'role_rotation': 1,
            'rebounding_env': 1,  # 3PT-heavy = long rebounds
            'transition_freq': 1,
            'opponent_defense': 1
        },
        'minutes': {
            'pace': 1,
            'offensive_scheme': 0,
            'usage_distribution': 1,
            'defensive_scheme': 0,
            'team_efficiency': 1,
            'lineup_quality': 0,
            'role_rotation': 3,  # Rotation depth matters most
            'rebounding_env': 0,
            'transition_freq': 0,
            'opponent_defense': 0  # Blowouts handled separately
        }
    }

    def __init__(self):
        # Rolling team stats storage
        self.team_stats = defaultdict(lambda: {
            'pace': [],
            'ortg': [],
            'drtg': [],
            'ast_rate': [],
            '3pa_rate': [],
            'transition_freq': [],
            'usage_concentration': []
        })

    def update_team_stats(self, team: str, game_stats: Dict):
        """Update rolling team statistics."""
        stats = self.team_stats[team]

        # Extract game-level team stats
        stats['pace'].append(game_stats.get('pace', 100.0))
        stats['ortg'].append(game_stats.get('ortg', 110.0))
        stats['drtg'].append(game_stats.get('drtg', 110.0))
        stats['ast_rate'].append(game_stats.get('ast_rate', 0.6))
        stats['3pa_rate'].append(game_stats.get('3pa_rate', 0.35))
        stats['transition_freq'].append(game_stats.get('transition_pct', 0.15))
        stats['usage_concentration'].append(game_stats.get('usage_gini', 0.3))

        # Keep last 20 games
        for key in stats:
            if len(stats[key]) > 20:
                stats[key] = stats[key][-20:]

    def calculate_context_features(self,
                                   player_team: str,
                                   opponent_team: str,
                                   stat_name: str) -> Dict[str, float]:
        """
        Calculate weighted team context features for a specific stat.

        Returns dict of features with values normalized to [0, 1] range.
        """
        features = {}
        weights = self.IMPACT_WEIGHTS.get(stat_name, {})

        team_stats = self.team_stats[player_team]
        opp_stats = self.team_stats[opponent_team]

        # 1. Pace (normalized to ~100 possessions/game)
        team_pace = np.mean(team_stats['pace'][-10:]) if team_stats['pace'] else 100.0
        opp_pace = np.mean(opp_stats['pace'][-10:]) if opp_stats['pace'] else 100.0
        combined_pace = (team_pace + opp_pace) / 2
        features['pace'] = (combined_pace - 95) / 10  # Normalize around 95-105

        # 2. Offensive scheme (assist rate as proxy for motion offense)
        team_ast_rate = np.mean(team_stats['ast_rate'][-10:]) if team_stats['ast_rate'] else 0.6
        features['offensive_scheme'] = (team_ast_rate - 0.5) / 0.2  # Normalize around 0.5-0.7

        # 3. Usage distribution (Gini coefficient - higher = more concentrated)
        usage_conc = np.mean(team_stats['usage_concentration'][-10:]) if team_stats['usage_concentration'] else 0.3
        features['usage_distribution'] = (usage_conc - 0.25) / 0.15  # Normalize around 0.25-0.4

        # 4. Defensive scheme (opponent DRTG)
        opp_drtg = np.mean(opp_stats['drtg'][-10:]) if opp_stats['drtg'] else 110.0
        features['defensive_scheme'] = (115 - opp_drtg) / 10  # Good defense = lower DRTG

        # 5. Team efficiency
        team_ortg = np.mean(team_stats['ortg'][-10:]) if team_stats['ortg'] else 110.0
        features['team_efficiency'] = (team_ortg - 105) / 10  # Normalize around 105-115

        # 6. Lineup quality (approximated by team ORTG relative to league)
        league_avg_ortg = 112.0
        features['lineup_quality'] = (team_ortg - league_avg_ortg) / 5

        # 7. Role & rotation (placeholder - would need player-specific data)
        features['role_rotation'] = 0.5  # Neutral default

        # 8. Rebounding environment (3PT rate)
        team_3pa_rate = np.mean(team_stats['3pa_rate'][-10:]) if team_stats['3pa_rate'] else 0.35
        features['rebounding_env'] = (team_3pa_rate - 0.3) / 0.15  # Normalize around 0.3-0.45

        # 9. Transition frequency
        trans_freq = np.mean(team_stats['transition_freq'][-10:]) if team_stats['transition_freq'] else 0.15
        features['transition_freq'] = (trans_freq - 0.1) / 0.1  # Normalize around 0.1-0.2

        # 10. Opponent defense quality
        features['opponent_defense'] = (115 - opp_drtg) / 10

        return features

    def get_context_adjustment(self,
                               player_team: str,
                               opponent_team: str,
                               stat_name: str,
                               baseline_prediction: float) -> float:
        """
        Calculate context-adjusted prediction using weighted factors.

        Adjusted = Baseline × (1 + Σ(weight_i × feature_i) / 30)

        Division by 30 scales the adjustment (max weight = 3, max features = 10)
        to keep adjustments reasonable (±10-20% range).
        """
        features = self.calculate_context_features(player_team, opponent_team, stat_name)
        weights = self.IMPACT_WEIGHTS.get(stat_name, {})

        # Calculate weighted sum
        weighted_sum = 0.0
        for factor_name, feature_value in features.items():
            weight = weights.get(factor_name, 0)
            weighted_sum += weight * feature_value

        # Apply adjustment (scale by 30 to keep adjustments modest)
        adjustment_factor = 1.0 + (weighted_sum / 30.0)

        # Clamp adjustment to reasonable range (0.8 to 1.2)
        adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)

        return baseline_prediction * adjustment_factor

    def get_feature_vector(self,
                          player_team: str,
                          opponent_team: str,
                          stat_name: str) -> np.ndarray:
        """
        Get feature vector for meta-learner input.

        Returns: Array of weighted context features.
        """
        features = self.calculate_context_features(player_team, opponent_team, stat_name)
        weights = self.IMPACT_WEIGHTS.get(stat_name, {})

        # Create weighted feature vector
        feature_vector = []
        for factor_name in sorted(features.keys()):
            weight = weights.get(factor_name, 0)
            feature_value = features[factor_name]
            feature_vector.append(weight * feature_value)

        return np.array(feature_vector)


def test_context_model():
    """Test the weighted context model."""
    context = WeightedTeamContext()

    # Simulate team stats
    context.update_team_stats('LAL', {
        'pace': 102.0,
        'ortg': 115.0,
        'drtg': 108.0,
        'ast_rate': 0.65,
        '3pa_rate': 0.38,
        'transition_pct': 0.18,
        'usage_gini': 0.35
    })

    context.update_team_stats('GSW', {
        'pace': 99.0,
        'ortg': 118.0,
        'drtg': 112.0,
        'ast_rate': 0.68,
        '3pa_rate': 0.42,
        'transition_pct': 0.16,
        'usage_gini': 0.32
    })

    # Test adjustments
    baseline_points = 20.0
    baseline_assists = 5.0

    print("Weighted Team Context Model Test")
    print("=" * 50)
    print(f"\nBaseline predictions:")
    print(f"  Points: {baseline_points}")
    print(f"  Assists: {baseline_assists}")

    adj_points = context.get_context_adjustment('LAL', 'GSW', 'points', baseline_points)
    adj_assists = context.get_context_adjustment('LAL', 'GSW', 'assists', baseline_assists)

    print(f"\nContext-adjusted predictions (LAL vs GSW):")
    print(f"  Points: {adj_points:.2f} ({(adj_points/baseline_points - 1)*100:+.1f}%)")
    print(f"  Assists: {adj_assists:.2f} ({(adj_assists/baseline_assists - 1)*100:+.1f}%)")

    # Show feature breakdown
    print(f"\nPoints context features:")
    features_pts = context.calculate_context_features('LAL', 'GSW', 'points')
    for name, value in sorted(features_pts.items()):
        weight = context.IMPACT_WEIGHTS['points'].get(name, 0)
        print(f"  {name:20s}: {value:+.3f} (weight: {weight:+d})")

    print(f"\nAssists context features:")
    features_ast = context.calculate_context_features('LAL', 'GSW', 'assists')
    for name, value in sorted(features_ast.items()):
        weight = context.IMPACT_WEIGHTS['assists'].get(name, 0)
        print(f"  {name:20s}: {value:+.3f} (weight: {weight:+d})")


if __name__ == "__main__":
    test_context_model()
