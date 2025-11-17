#!/usr/bin/env python
"""
Historical Player Archetypes for Cold Start Problem

Uses full NBA history (1947-2026) to:
1. Cluster players into archetypes based on playing style
2. Match new/unknown players to historical archetypes
3. Provide prior distributions for predictions

This solves:
- Rookie predictions (no history)
- Players returning from injury
- Traded players in new systems
- Players with limited sample size
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PlayerArchetypeSystem:
    """
    Cluster historical players into archetypes for cold start predictions.

    Example archetypes:
    - Volume Scorer (high USG%, high PPG)
    - 3PT Specialist (high 3PAr, high 3P%)
    - Rim Protector (high BLK%, low 3PAr)
    - Playmaker (high AST%, low USG%)
    - Two-Way Wing (balanced offensive/defensive metrics)
    """

    def __init__(self, n_archetypes: int = 12):
        self.n_archetypes = n_archetypes
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.archetype_profiles = {}
        self.archetype_names = {}
        self.feature_cols = []
        self.fitted = False

    def _select_clustering_features(self, df: pd.DataFrame) -> List[str]:
        """Select features that define playing style."""

        # Priority features for clustering (playing style indicators)
        style_features = [
            # Usage and volume
            'adv_usg_percent',  # Usage rate
            'adv_ts_percent',   # True shooting %

            # Scoring profile
            'adv_x3p_ar',       # 3PT attempt rate
            'shoot_avg_dist_fga',  # Average shot distance
            'shoot_percent_fga_from_x3p_range',  # % shots from 3
            'shoot_percent_fga_from_x0_3_range',  # % shots at rim

            # Playmaking
            'adv_ast_percent',  # Assist %
            'adv_tov_percent',  # Turnover %

            # Rebounding
            'adv_orb_percent',  # Offensive rebound %
            'adv_drb_percent',  # Defensive rebound %
            'adv_trb_percent',  # Total rebound %

            # Defense
            'adv_stl_percent',  # Steal %
            'adv_blk_percent',  # Block %

            # Impact
            'adv_bpm',          # Box Plus/Minus
            'adv_vorp',         # Value over replacement
            'adv_ws_48',        # Win shares per 48

            # Position indicators
            'pbp_pg_percent',   # % time at PG
            'pbp_sg_percent',   # % time at SG
            'pbp_sf_percent',   # % time at SF
            'pbp_pf_percent',   # % time at PF
            'pbp_c_percent',    # % time at C
        ]

        # Filter to features that exist in data
        available = [f for f in style_features if f in df.columns]

        if len(available) < 5:
            # Fallback to basic stats if advanced not available
            basic_features = ['points', 'assists', 'reboundsTotal', 'steals', 'blocks',
                            'threePointersMade', 'fieldGoalsPercentage', 'numMinutes']
            available = [f for f in basic_features if f in df.columns]

        return available

    def fit(self, player_seasons_df: pd.DataFrame, min_minutes: int = 500,
            min_games: int = 20, verbose: bool = True) -> 'PlayerArchetypeSystem':
        """
        Fit archetypes on historical player-season data.

        Args:
            player_seasons_df: DataFrame with season-level stats per player
            min_minutes: Minimum minutes played to include
            min_games: Minimum games played to include
            verbose: Print progress
        """
        if verbose:
            print("=" * 60)
            print("BUILDING HISTORICAL PLAYER ARCHETYPES")
            print("=" * 60)

        # Aggregate to player-season level (average stats per season)
        if verbose:
            print(f"Input data: {len(player_seasons_df):,} player-game rows")

        # Group by player and season
        id_col = 'personId' if 'personId' in player_seasons_df.columns else 'player_id'
        season_col = None
        for col in ['season', 'game_year', 'season_end_year']:
            if col in player_seasons_df.columns:
                season_col = col
                break

        if season_col is None:
            raise ValueError("No season column found")

        # Aggregate stats by player-season
        numeric_cols = player_seasons_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID columns from aggregation
        agg_cols = [c for c in numeric_cols if c not in [id_col, season_col, 'gameId']]

        if verbose:
            print(f"Aggregating {len(agg_cols)} numeric features by player-season...")

        player_seasons = player_seasons_df.groupby([id_col, season_col])[agg_cols].mean().reset_index()

        if verbose:
            print(f"Player-seasons: {len(player_seasons):,}")

        # Filter by minutes and games
        if 'numMinutes' in player_seasons.columns:
            # Estimate total minutes (avg * games)
            if 'adv_g' in player_seasons.columns:
                player_seasons['total_minutes'] = player_seasons['numMinutes'] * player_seasons['adv_g']
                player_seasons = player_seasons[player_seasons['total_minutes'] >= min_minutes]
                player_seasons = player_seasons[player_seasons['adv_g'] >= min_games]
            else:
                # Just use average minutes threshold
                player_seasons = player_seasons[player_seasons['numMinutes'] >= min_minutes / 82]

        if verbose:
            print(f"After filtering (min {min_minutes} mins, {min_games} games): {len(player_seasons):,}")

        # Select clustering features
        self.feature_cols = self._select_clustering_features(player_seasons)

        if verbose:
            print(f"Clustering on {len(self.feature_cols)} features:")
            for f in self.feature_cols[:10]:
                print(f"  - {f}")
            if len(self.feature_cols) > 10:
                print(f"  ... and {len(self.feature_cols) - 10} more")

        # Prepare data for clustering
        X = player_seasons[self.feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        if verbose:
            print(f"Feature matrix: {X.shape}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Optional: PCA for visualization
        if X_scaled.shape[1] > 3:
            self.pca = PCA(n_components=min(3, X_scaled.shape[1]))
            X_pca = self.pca.fit_transform(X_scaled)
            if verbose:
                explained = sum(self.pca.explained_variance_ratio_) * 100
                print(f"PCA: {explained:.1f}% variance explained by 3 components")

        # Fit KMeans
        if verbose:
            print(f"\nClustering into {self.n_archetypes} archetypes...")

        self.kmeans = KMeans(n_clusters=self.n_archetypes, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        # Build archetype profiles
        player_seasons['archetype'] = clusters

        for arch_id in range(self.n_archetypes):
            arch_data = player_seasons[player_seasons['archetype'] == arch_id]

            # Store profile (mean and std of each feature)
            profile = {
                'count': len(arch_data),
                'features': {},
                'target_priors': {}
            }

            for col in self.feature_cols:
                profile['features'][col] = {
                    'mean': arch_data[col].mean(),
                    'std': arch_data[col].std(),
                    'median': arch_data[col].median()
                }

            # Store target variable priors (what we predict)
            for target in ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'numMinutes']:
                if target in arch_data.columns:
                    profile['target_priors'][target] = {
                        'mean': arch_data[target].mean(),
                        'std': arch_data[target].std(),
                        'median': arch_data[target].median(),
                        'p25': arch_data[target].quantile(0.25),
                        'p75': arch_data[target].quantile(0.75)
                    }

            self.archetype_profiles[arch_id] = profile

            # Auto-name archetypes based on dominant features
            self.archetype_names[arch_id] = self._name_archetype(profile)

        if verbose:
            print("\nArchetype Summary:")
            print("-" * 60)
            for arch_id in range(self.n_archetypes):
                name = self.archetype_names[arch_id]
                count = self.archetype_profiles[arch_id]['count']

                # Get key stats
                pts = self.archetype_profiles[arch_id]['target_priors'].get('points', {}).get('mean', 0)
                ast = self.archetype_profiles[arch_id]['target_priors'].get('assists', {}).get('mean', 0)
                reb = self.archetype_profiles[arch_id]['target_priors'].get('reboundsTotal', {}).get('mean', 0)

                print(f"  {arch_id}: {name}")
                print(f"      {count:,} player-seasons | Avg: {pts:.1f} pts, {ast:.1f} ast, {reb:.1f} reb")

        self.fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("ARCHETYPE SYSTEM READY")
            print("=" * 60)

        return self

    def _name_archetype(self, profile: dict) -> str:
        """Auto-generate archetype name based on dominant features."""

        features = profile['features']

        # Check key indicators
        high_usg = features.get('adv_usg_percent', {}).get('mean', 0) > 25
        high_3par = features.get('adv_x3p_ar', {}).get('mean', 0) > 0.4
        high_ast = features.get('adv_ast_percent', {}).get('mean', 0) > 25
        high_blk = features.get('adv_blk_percent', {}).get('mean', 0) > 3
        high_reb = features.get('adv_trb_percent', {}).get('mean', 0) > 15

        # Position indicators
        pg_pct = features.get('pbp_pg_percent', {}).get('mean', 0)
        c_pct = features.get('pbp_c_percent', {}).get('mean', 0)

        # Determine archetype name
        if high_usg and high_3par:
            return "Volume 3PT Scorer"
        elif high_usg and not high_3par:
            return "Interior Scorer"
        elif high_ast and pg_pct > 50:
            return "Primary Playmaker"
        elif high_ast:
            return "Secondary Playmaker"
        elif high_blk and c_pct > 50:
            return "Rim Protector"
        elif high_reb and c_pct > 30:
            return "Rebounding Big"
        elif high_3par and not high_usg:
            return "3PT Specialist"
        elif high_reb:
            return "Versatile Forward"
        else:
            return "Role Player"

    def get_archetype(self, player_stats: pd.Series) -> Tuple[int, str, float]:
        """
        Match a player to their closest archetype.

        Args:
            player_stats: Series with player's current stats

        Returns:
            (archetype_id, archetype_name, confidence_score)
        """
        if not self.fitted:
            raise ValueError("System not fitted. Call fit() first.")

        # Extract features
        X = np.array([[player_stats.get(f, 0) for f in self.feature_cols]])

        # Handle missing
        X = np.nan_to_num(X, nan=0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict cluster
        arch_id = self.kmeans.predict(X_scaled)[0]

        # Calculate confidence (inverse distance to cluster center)
        distances = self.kmeans.transform(X_scaled)[0]
        min_dist = distances[arch_id]
        confidence = 1.0 / (1.0 + min_dist)  # Higher = more confident

        return arch_id, self.archetype_names[arch_id], confidence

    def get_prior_distribution(self, archetype_id: int, target: str = 'points') -> Dict:
        """
        Get the prior distribution for a target variable based on archetype.

        Args:
            archetype_id: The archetype cluster ID
            target: Target variable (points, assists, rebounds, etc.)

        Returns:
            Dictionary with mean, std, median, p25, p75
        """
        if archetype_id not in self.archetype_profiles:
            raise ValueError(f"Unknown archetype: {archetype_id}")

        priors = self.archetype_profiles[archetype_id]['target_priors']

        if target not in priors:
            return {'mean': 0, 'std': 1, 'median': 0, 'p25': 0, 'p75': 0}

        return priors[target]

    def save(self, path: str = 'models/archetype_system.pkl'):
        """Save the fitted archetype system."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'n_archetypes': self.n_archetypes,
                'scaler': self.scaler,
                'kmeans': self.kmeans,
                'pca': self.pca,
                'archetype_profiles': self.archetype_profiles,
                'archetype_names': self.archetype_names,
                'feature_cols': self.feature_cols,
                'fitted': self.fitted
            }, f)

        print(f"Saved archetype system to {path}")

    @classmethod
    def load(cls, path: str = 'models/archetype_system.pkl') -> 'PlayerArchetypeSystem':
        """Load a saved archetype system."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        system = cls(n_archetypes=data['n_archetypes'])
        system.scaler = data['scaler']
        system.kmeans = data['kmeans']
        system.pca = data['pca']
        system.archetype_profiles = data['archetype_profiles']
        system.archetype_names = data['archetype_names']
        system.feature_cols = data['feature_cols']
        system.fitted = data['fitted']

        return system


def add_archetype_features(df: pd.DataFrame, archetype_system: PlayerArchetypeSystem) -> pd.DataFrame:
    """
    Add archetype-based features to a prediction dataframe.

    Args:
        df: DataFrame with player stats
        archetype_system: Fitted PlayerArchetypeSystem

    Returns:
        DataFrame with additional archetype features
    """
    if not archetype_system.fitted:
        raise ValueError("Archetype system not fitted")

    # Get archetype for each row
    archetype_ids = []
    archetype_confidences = []

    for idx, row in df.iterrows():
        arch_id, _, confidence = archetype_system.get_archetype(row)
        archetype_ids.append(arch_id)
        archetype_confidences.append(confidence)

    df['archetype_id'] = archetype_ids
    df['archetype_confidence'] = archetype_confidences

    # Add prior features for each target
    for target in ['points', 'assists', 'reboundsTotal', 'threePointersMade']:
        df[f'{target}_archetype_mean'] = df['archetype_id'].apply(
            lambda x: archetype_system.get_prior_distribution(x, target)['mean']
        )
        df[f'{target}_archetype_std'] = df['archetype_id'].apply(
            lambda x: archetype_system.get_prior_distribution(x, target)['std']
        )

    return df


if __name__ == "__main__":
    # Example usage
    print("Loading aggregated player data...")

    # This would be your actual data
    # df = pd.read_parquet('aggregated_nba_data.parquet')

    # For testing, create dummy data
    np.random.seed(42)
    n_samples = 10000

    dummy_data = pd.DataFrame({
        'personId': np.random.randint(1, 500, n_samples),
        'season': np.random.randint(1990, 2025, n_samples),
        'points': np.random.normal(12, 8, n_samples).clip(0, 40),
        'assists': np.random.normal(3, 3, n_samples).clip(0, 15),
        'reboundsTotal': np.random.normal(5, 4, n_samples).clip(0, 20),
        'threePointersMade': np.random.normal(1, 1.5, n_samples).clip(0, 8),
        'numMinutes': np.random.normal(25, 10, n_samples).clip(0, 48),
        'adv_usg_percent': np.random.normal(20, 5, n_samples).clip(10, 40),
        'adv_ts_percent': np.random.normal(0.55, 0.05, n_samples).clip(0.4, 0.7),
        'adv_x3p_ar': np.random.normal(0.3, 0.15, n_samples).clip(0, 0.8),
        'adv_ast_percent': np.random.normal(15, 8, n_samples).clip(0, 50),
        'adv_blk_percent': np.random.normal(1.5, 1.5, n_samples).clip(0, 10),
        'adv_trb_percent': np.random.normal(10, 5, n_samples).clip(0, 30),
        'adv_g': np.random.randint(20, 82, n_samples),
    })

    # Fit archetype system
    system = PlayerArchetypeSystem(n_archetypes=8)
    system.fit(dummy_data, min_minutes=300, min_games=10, verbose=True)

    # Test matching a player
    test_player = dummy_data.iloc[0]
    arch_id, arch_name, confidence = system.get_archetype(test_player)
    print(f"\nTest player matched to: {arch_name} (ID: {arch_id}, confidence: {confidence:.3f})")

    # Get prior for points
    pts_prior = system.get_prior_distribution(arch_id, 'points')
    print(f"Points prior: {pts_prior['mean']:.1f} +/- {pts_prior['std']:.1f}")

    # Save system
    system.save('models/archetype_system.pkl')

    # Load and verify
    loaded_system = PlayerArchetypeSystem.load('models/archetype_system.pkl')
    print(f"\nLoaded system with {loaded_system.n_archetypes} archetypes")
