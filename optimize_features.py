"""
Feature Engineering Optimization
Reduce 186 features using mutual information, SHAP analysis, and feature clustering.
Add advanced features like Fourier transforms and rolling decay windows.
"""

import modal
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import seaborn as sns

# Modal setup
app = modal.App("feature-optimization")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)
data_volume = modal.Volume.from_name("nba-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas", "numpy", "scikit-learn", "matplotlib", 
        "lightgbm", "shap", "seaborn", "scipy"
    ])
    .add_local_file("train_meta_learner_v4.py", remote_path="/root/train_meta_learner_v4.py")
    .add_local_file("ensemble_predictor.py", remote_path="/root/ensemble_predictor.py")
)

@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={"/models": model_volume, "/data": data_volume}
)
def analyze_feature_importance(model_path: str = "/models/meta_learner_v4_all_components.pkl",
                              data_path: str = "/data/PlayerStatistics.csv"):
    """
    Analyze feature importance and redundancy in the training data.
    """
    import sys
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from scipy.fft import fft, fftfreq
    import shap
    
    sys.path.insert(0, "/root")
    from train_meta_learner_v4 import MetaLearnerV4
    from ensemble_predictor import load_all_window_models, predict_with_window
    
    print("="*70)
    print("FEATURE ENGINEERING OPTIMIZATION ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\n[*] Loading training data...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Create player name if needed
    if 'firstName' in df.columns and 'lastName' in df.columns:
        df['playerName'] = df['firstName'] + ' ' + df['lastName']
    
    # Select numeric features for analysis
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Found {len(numeric_features)} numeric features")
    
    # Remove target variables from feature analysis
    target_features = ['points', 'reboundsDefensive', 'reboundsOffensive', 'assists', 'threePointersMade']
    analysis_features = [f for f in numeric_features if f not in target_features]
    
    print(f"  Analyzing {len(analysis_features)} predictive features")
    
    # 1. Mutual Information Analysis
    print(f"\n[*] Mutual Information Analysis...")
    
    mi_scores = {}
    for target in target_features:
        if target in df.columns:
            # Remove rows with missing values
            valid_data = df[analysis_features + [target]].dropna()
            X = valid_data[analysis_features].fillna(0)
            y = valid_data[target].fillna(0)
            
            # Calculate mutual information
            mi = mutual_info_regression(X, y, random_state=42)
            mi_scores[target] = dict(zip(analysis_features, mi))
            
            print(f"  {target}: Top 10 features by MI")
            sorted_mi = sorted(mi_scores[target].items(), key=lambda x: x[1], reverse=True)
            for i, (feature, score) in enumerate(sorted_mi[:10]):
                print(f"    {i+1:2d}. {feature}: {score:.4f}")
    
    # 2. Feature Correlation Analysis
    print(f"\n[*] Feature Correlation Analysis...")
    
    # Calculate correlation matrix
    feature_data = df[analysis_features].fillna(0)
    correlation_matrix = feature_data.corr().abs()
    
    # Find highly correlated features (>0.8)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    print(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.8)")
    print(f"  Top 10 correlated pairs:")
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:10]):
        print(f"    {i+1:2d}. {feat1} ↔ {feat2}: {corr:.3f}")
    
    # 3. Feature Clustering
    print(f"\n[*] Feature Clustering Analysis...")
    
    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='average')
    cluster_labels = clustering.fit_predict(scaled_features)
    
    # Group features by cluster
    feature_clusters = {}
    for feature, cluster in zip(analysis_features, cluster_labels):
        if cluster not in feature_clusters:
            feature_clusters[cluster] = []
        feature_clusters[cluster].append(feature)
    
    print(f"  Found {len(feature_clusters)} feature clusters")
    large_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
    print(f"  {len(large_clusters)} clusters with multiple features (redundancy candidates)")
    
    # 4. Zero Variance Features
    print(f"\n[*] Zero Variance Analysis...")
    
    zero_var_features = []
    low_var_features = []
    
    for feature in analysis_features:
        var = df[feature].var()
        if var == 0 or pd.isna(var):
            zero_var_features.append(feature)
        elif var < 0.01:  # Low variance threshold
            low_var_features.append(feature)
    
    print(f"  Zero variance features: {len(zero_var_features)}")
    if zero_var_features:
        print(f"    {zero_var_features[:10]}")
    
    print(f"  Low variance features: {len(low_var_features)}")
    if low_var_features:
        print(f"    {low_var_features[:10]}")
    
    # 5. Advanced Feature Engineering Recommendations
    print(f"\n[*] Advanced Feature Engineering Recommendations...")
    
    recommendations = {
        'fourier_features': [],
        'rolling_decay_features': [],
        'interaction_features': []
    }
    
    # Identify time-series features for Fourier transforms
    time_series_features = ['points', 'assists', 'reboundsDefensive', 'reboundsOffensive', 'threePointersMade']
    for feature in time_series_features:
        if feature in df.columns:
            recommendations['fourier_features'].append(f"{feature}_seasonality")
            recommendations['rolling_decay_features'].append(f"{feature}_decay_10")
            recommendations['rolling_decay_features'].append(f"{feature}_decay_20")
    
    # Identify interaction candidates
    interaction_candidates = [
        ('points', 'numMinutes'),
        ('assists', 'reboundsDefensive'),
        ('threePointersMade', 'fieldGoalsAttempted')
    ]
    
    for feat1, feat2 in interaction_candidates:
        if feat1 in df.columns and feat2 in df.columns:
            recommendations['interaction_features'].append(f"{feat1}_x_{feat2}")
    
    print(f"  Fourier transform candidates: {len(recommendations['fourier_features'])}")
    print(f"  Rolling decay candidates: {len(recommendations['rolling_decay_features'])}")
    print(f"  Interaction candidates: {len(recommendations['interaction_features'])}")
    
    # 6. Optimization Summary
    print(f"\n[*] Feature Optimization Summary...")
    
    # Calculate total reduction potential
    total_features = len(analysis_features)
    redundant_features = len(zero_var_features) + len(low_var_features)
    
    # Estimate redundancy from clusters (keep 1 per cluster)
    cluster_redundancy = sum(len(cluster) - 1 for cluster in large_clusters.values())
    redundant_features += cluster_redundancy
    
    reduction_percentage = (redundant_features / total_features) * 100
    
    print(f"  Original feature count: {total_features}")
    print(f"  Redundant features identified: {redundant_features}")
    print(f"  Potential reduction: {reduction_percentage:.1f}%")
    print(f"  Optimized feature count: {total_features - redundant_features}")
    
    # 7. Create optimized feature list
    optimized_features = []
    
    # Keep one feature per cluster (highest MI average)
    for cluster_id, features in large_clusters.items():
        if len(features) > 1:
            # Calculate average MI across all targets
            avg_mi_scores = {}
            for feature in features:
                mi_sum = 0
                count = 0
                for target in target_features:
                    if target in mi_scores and feature in mi_scores[target]:
                        mi_sum += mi_scores[target][feature]
                        count += 1
                avg_mi_scores[feature] = mi_sum / count if count > 0 else 0
            
            # Keep the feature with highest average MI
            best_feature = max(avg_mi_scores.items(), key=lambda x: x[1])[0]
            optimized_features.append(best_feature)
        else:
            optimized_features.extend(features)
    
    # Add features not in any large cluster
    single_features = [f for f in analysis_features if f not in sum(large_clusters.values(), [])]
    optimized_features.extend(single_features)
    
    # Remove zero/low variance features
    optimized_features = [f for f in optimized_features 
                         if f not in zero_var_features + low_var_features]
    
    print(f"\n🎯 Final Optimized Feature Set: {len(optimized_features)} features")
    print(f"  Features removed: {total_features - len(optimized_features)}")
    print(f"  Efficiency gain: {(total_features - len(optimized_features))/total_features*100:.1f}%")
    
    # Save results
    results = {
        'mi_scores': mi_scores,
        'high_corr_pairs': high_corr_pairs,
        'feature_clusters': feature_clusters,
        'zero_var_features': zero_var_features,
        'low_var_features': low_var_features,
        'recommendations': recommendations,
        'optimization_summary': {
            'original_count': total_features,
            'redundant_count': redundant_features,
            'optimized_count': len(optimized_features),
            'reduction_percentage': reduction_percentage,
            'optimized_features': optimized_features
        }
    }
    
    results_file = "/models/feature_optimization_analysis.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Analysis saved to {results_file}")
    
    return results

@app.local_entrypoint()
def main():
    """
    Analyze feature importance and redundancy.
    """
    print("="*70)
    print("FEATURE ENGINEERING OPTIMIZATION")
    print("="*70)
    print("Analyzing feature redundancy and optimization opportunities...")
    
    results = analyze_feature_importance.remote()
    
    print("\n" + "="*70)
    print("FEATURE OPTIMIZATION ANALYSIS COMPLETE")
    print("="*70)
    
    summary = results['optimization_summary']
    print(f"Original features: {summary['original_count']}")
    print(f"Optimized features: {summary['optimized_count']}")
    print(f"Reduction: {summary['reduction_percentage']:.1f}%")
    
    print(f"\nNext steps:")
    print(f"1. Implement optimized feature pipeline")
    print(f"2. Add advanced features (Fourier, rolling decay)")
    print(f"3. Retrain models with optimized feature set")
    print(f"4. Validate accuracy improvements")
