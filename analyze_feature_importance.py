"""
Feature Importance Analysis for CE50 Structural Features

Analyzes which of the 40 structural features contribute most to CE50 prediction
and identifies potential redundancies with Morgan fingerprints.

Author: Enhanced CE50 Prediction System
Date: 2026-01-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from structural_features import StructuralFeatureExtractor
from ce50_ensemble_predictor import DualFingerprintGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(filepath='ce50_300compounds_training.csv'):
    """Load CE50 data"""
    df = pd.read_csv(filepath)
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles', 'CE50': 'ce50'})
    df['pce50'] = -np.log10(df['ce50'])
    df = df.dropna(subset=['smiles', 'ce50'])
    return df


def extract_all_features(df):
    """Extract fingerprints and structural features"""
    print("Extracting features...")

    fp_gen = DualFingerprintGenerator()
    extractor = StructuralFeatureExtractor()

    fps = fp_gen.generate_all_features(
        df['smiles'].values,
        use_structural=True,
        structural_extractor=extractor
    )

    X_fingerprints = fps['binary']  # Use binary fingerprints
    X_structural = fps['structural']
    feature_names = extractor.feature_names

    print(f"  ✓ Fingerprints: {X_fingerprints.shape}")
    print(f"  ✓ Structural:   {X_structural.shape}")

    return X_fingerprints, X_structural, feature_names


def train_models_and_get_importance(X_fp, X_struct, y, feature_names):
    """Train models and extract feature importance"""
    print("\nTraining models for feature importance analysis...")

    # Split data
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_fp_train = X_fp[idx_train]
    X_fp_test = X_fp[idx_test]
    X_struct_train = X_struct[idx_train]
    X_struct_test = X_struct[idx_test]

    # Model 1: Fingerprints only
    print("\n  Model 1: Fingerprints only (baseline)")
    scaler_fp = StandardScaler()
    X_fp_train_scaled = scaler_fp.fit_transform(X_fp_train)
    X_fp_test_scaled = scaler_fp.transform(X_fp_test)

    model_fp = RandomForestRegressor(
        n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_fp.fit(X_fp_train_scaled, y_train)
    score_fp = model_fp.score(X_fp_test_scaled, y_test)
    print(f"    Test R² = {score_fp:.4f}")

    # Model 2: Structural features only
    print("\n  Model 2: Structural features only")
    scaler_struct = StandardScaler()
    X_struct_train_scaled = scaler_struct.fit_transform(X_struct_train)
    X_struct_test_scaled = scaler_struct.transform(X_struct_test)

    model_struct = RandomForestRegressor(
        n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_struct.fit(X_struct_train_scaled, y_train)
    score_struct = model_struct.score(X_struct_test_scaled, y_test)
    print(f"    Test R² = {score_struct:.4f}")

    # Get structural feature importances
    struct_importances = model_struct.feature_importances_

    # Model 3: Combined (fingerprints + structural)
    print("\n  Model 3: Combined (fingerprints + structural)")
    X_combined_train = np.hstack([X_fp_train, X_struct_train])
    X_combined_test = np.hstack([X_fp_test, X_struct_test])

    scaler_combined = StandardScaler()
    X_combined_train_scaled = scaler_combined.fit_transform(X_combined_train)
    X_combined_test_scaled = scaler_combined.transform(X_combined_test)

    model_combined = RandomForestRegressor(
        n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_combined.fit(X_combined_train_scaled, y_train)
    score_combined = model_combined.score(X_combined_test_scaled, y_test)
    print(f"    Test R² = {score_combined:.4f}")

    # Extract importances from combined model
    # Last 40 features are structural
    combined_importances = model_combined.feature_importances_
    struct_importances_in_combined = combined_importances[-len(feature_names):]

    # Permutation importance for structural features
    print("\n  Computing permutation importance (this may take a minute)...")
    perm_importance = permutation_importance(
        model_combined, X_combined_test_scaled, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    struct_perm_importance = perm_importance.importances_mean[-len(feature_names):]

    return {
        'score_fp': score_fp,
        'score_struct': score_struct,
        'score_combined': score_combined,
        'struct_importances': struct_importances,
        'struct_importances_in_combined': struct_importances_in_combined,
        'struct_perm_importance': struct_perm_importance,
        'feature_names': feature_names
    }


def create_importance_visualizations(results, output_file='ce50_feature_importance.png'):
    """Create comprehensive feature importance visualizations"""
    print("\nGenerating feature importance visualizations...")

    feature_names = results['feature_names']
    n_features = len(feature_names)

    fig = plt.figure(figsize=(20, 14))

    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    models = ['Fingerprints\nOnly', 'Structural\nOnly', 'Combined']
    scores = [results['score_fp'], results['score_struct'], results['score_combined']]
    colors = ['#4A90E2', '#FFA500', '#50C878']

    bars = ax1.bar(models, scores, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Test R² Score', fontsize=13, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(scores) * 1.2])
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Top 15 Structural Features (standalone model)
    ax2 = plt.subplot(3, 3, 2)
    top_indices = np.argsort(results['struct_importances'])[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = results['struct_importances'][top_indices]

    y_pos = np.arange(len(top_features))
    bars = ax2.barh(y_pos, top_importances, color='#FFA500', edgecolor='black', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features, fontsize=10)
    ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Top 15 Structural Features\n(Standalone Model)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Top 15 Structural Features (combined model)
    ax3 = plt.subplot(3, 3, 3)
    top_indices_comb = np.argsort(results['struct_importances_in_combined'])[-15:]
    top_features_comb = [feature_names[i] for i in top_indices_comb]
    top_importances_comb = results['struct_importances_in_combined'][top_indices_comb]

    y_pos = np.arange(len(top_features_comb))
    bars = ax3.barh(y_pos, top_importances_comb, color='#50C878', edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_features_comb, fontsize=10)
    ax3.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax3.set_title('Top 15 Structural Features\n(Combined Model)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Permutation Importance (top 15)
    ax4 = plt.subplot(3, 3, 4)
    top_indices_perm = np.argsort(results['struct_perm_importance'])[-15:]
    top_features_perm = [feature_names[i] for i in top_indices_perm]
    top_perm_imp = results['struct_perm_importance'][top_indices_perm]

    y_pos = np.arange(len(top_features_perm))
    bars = ax4.barh(y_pos, top_perm_imp, color='#9B59B6', edgecolor='black', linewidth=1)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features_perm, fontsize=10)
    ax4.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax4.set_title('Top 15 by Permutation Importance\n(Impact on Predictions)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Feature Category Breakdown
    ax5 = plt.subplot(3, 3, 5)

    # Categorize features
    categories = {
        'Bond Features': [],
        'UFF Features': [],
        'Topology Features': [],
        'Atom Features': []
    }

    for i, name in enumerate(feature_names):
        importance = results['struct_importances'][i]
        if 'bond' in name or 'c_c' in name or 'c_n' in name or 'c_o' in name:
            categories['Bond Features'].append(importance)
        elif 'uff' in name or 'vdw' in name or 'charge' in name or 'coordination' in name or 'strain' in name:
            categories['UFF Features'].append(importance)
        elif 'angle' in name or 'dihedral' in name or 'topology' in name or 'branching' in name:
            categories['Topology Features'].append(importance)
        else:
            categories['Atom Features'].append(importance)

    category_means = {k: np.mean(v) if v else 0 for k, v in categories.items()}

    bars = ax5.bar(category_means.keys(), category_means.values(),
                   color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3'],
                   edgecolor='black', linewidth=2)
    ax5.set_ylabel('Average Importance', fontsize=12, fontweight='bold')
    ax5.set_title('Feature Category Importance', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 6. Importance Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(results['struct_importances'], bins=20, color='#FFA500',
             edgecolor='black', alpha=0.7)
    ax6.axvline(np.mean(results['struct_importances']), color='red',
                linestyle='--', linewidth=2, label=f'Mean = {np.mean(results["struct_importances"]):.4f}')
    ax6.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax6.set_title('Structural Feature Importance Distribution', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)

    # 7. Correlation: Standalone vs Combined Importance
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(results['struct_importances'],
                results['struct_importances_in_combined'],
                alpha=0.6, s=100, edgecolors='k', linewidths=0.5)

    # Add diagonal line
    max_val = max(results['struct_importances'].max(),
                  results['struct_importances_in_combined'].max())
    ax7.plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.7)

    ax7.set_xlabel('Importance (Standalone)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Importance (Combined)', fontsize=12, fontweight='bold')
    ax7.set_title('Feature Importance:\nStandalone vs Combined Model', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Calculate correlation
    corr = np.corrcoef(results['struct_importances'],
                       results['struct_importances_in_combined'])[0, 1]
    ax7.text(0.05, 0.95, f'Correlation = {corr:.3f}',
            transform=ax7.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 8. Top 10 Most Important Features (All Methods)
    ax8 = plt.subplot(3, 3, 8)

    # Combine rankings
    rank_standalone = np.argsort(-results['struct_importances'])
    rank_combined = np.argsort(-results['struct_importances_in_combined'])
    rank_perm = np.argsort(-results['struct_perm_importance'])

    # Get top features from permutation importance
    top_10_features = [feature_names[i] for i in rank_perm[:10]]

    # Create comparison data
    data_matrix = []
    for feat in top_10_features:
        idx = feature_names.index(feat)
        data_matrix.append([
            results['struct_importances'][idx],
            results['struct_importances_in_combined'][idx],
            results['struct_perm_importance'][idx]
        ])

    data_matrix = np.array(data_matrix)

    # Normalize for visualization
    data_matrix_norm = data_matrix / data_matrix.max(axis=0, keepdims=True)

    im = ax8.imshow(data_matrix_norm.T, cmap='YlOrRd', aspect='auto')
    ax8.set_xticks(np.arange(len(top_10_features)))
    ax8.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9)
    ax8.set_yticks([0, 1, 2])
    ax8.set_yticklabels(['Standalone', 'Combined', 'Permutation'], fontsize=11)
    ax8.set_title('Top 10 Features Across Methods\n(Normalized)', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax8)
    cbar.set_label('Normalized Importance', fontsize=11)

    # 9. Feature Redundancy Analysis
    ax9 = plt.subplot(3, 3, 9)

    # Calculate importance drop when moving from standalone to combined
    importance_drop = results['struct_importances'] - results['struct_importances_in_combined']

    # Features with biggest drop are most redundant with fingerprints
    redundant_indices = np.argsort(-importance_drop)[:10]
    redundant_features = [feature_names[i] for i in redundant_indices]
    redundant_drops = importance_drop[redundant_indices]

    y_pos = np.arange(len(redundant_features))
    bars = ax9.barh(y_pos, redundant_drops, color='#E74C3C', edgecolor='black', linewidth=1)
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(redundant_features, fontsize=10)
    ax9.set_xlabel('Importance Drop', fontsize=12, fontweight='bold')
    ax9.set_title('Top 10 Most Redundant Features\n(Overlap with Fingerprints)', fontsize=13, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')

    plt.suptitle('CE50 Structural Feature Importance Analysis (298 Compounds)\n' +
                 'Identifying Key Features for Prediction Enhancement',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")

    return fig


def print_detailed_analysis(results):
    """Print detailed feature importance analysis"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    feature_names = results['feature_names']

    print("\n📊 MODEL PERFORMANCE:")
    print(f"  Fingerprints only:  R² = {results['score_fp']:.4f}")
    print(f"  Structural only:    R² = {results['score_struct']:.4f}")
    print(f"  Combined:           R² = {results['score_combined']:.4f}")

    improvement = ((results['score_combined'] - results['score_fp']) / results['score_fp'] * 100)
    print(f"\n  Combined vs Fingerprints: {improvement:+.2f}%")

    print("\n" + "-"*80)
    print("\n🏆 TOP 10 MOST IMPORTANT STRUCTURAL FEATURES (Permutation):")
    top_perm_indices = np.argsort(results['struct_perm_importance'])[-10:][::-1]
    for rank, idx in enumerate(top_perm_indices, 1):
        name = feature_names[idx]
        perm_imp = results['struct_perm_importance'][idx]
        standalone_imp = results['struct_importances'][idx]
        combined_imp = results['struct_importances_in_combined'][idx]
        print(f"  {rank:2d}. {name:30s}")
        print(f"      Permutation: {perm_imp:.5f} | Standalone: {standalone_imp:.5f} | Combined: {combined_imp:.5f}")

    print("\n" + "-"*80)
    print("\n🔄 TOP 10 MOST REDUNDANT FEATURES (Overlap with Fingerprints):")
    importance_drop = results['struct_importances'] - results['struct_importances_in_combined']
    redundant_indices = np.argsort(-importance_drop)[:10]
    for rank, idx in enumerate(redundant_indices, 1):
        name = feature_names[idx]
        drop = importance_drop[idx]
        standalone = results['struct_importances'][idx]
        combined = results['struct_importances_in_combined'][idx]
        drop_pct = (drop / standalone * 100) if standalone > 0 else 0
        print(f"  {rank:2d}. {name:30s}")
        print(f"      Drop: {drop:.5f} ({drop_pct:.1f}%) | {standalone:.5f} → {combined:.5f}")

    print("\n" + "-"*80)
    print("\n💡 RECOMMENDATIONS:")

    # High value features (high permutation importance)
    high_value = np.sum(results['struct_perm_importance'] > 0.001)
    print(f"\n  ✓ Keep {high_value} features with permutation importance > 0.001")

    # Low redundancy features (similar importance in standalone vs combined)
    importance_ratio = results['struct_importances_in_combined'] / (results['struct_importances'] + 1e-10)
    low_redundancy = np.sum(importance_ratio > 0.7)
    print(f"  ✓ Focus on {low_redundancy} features with low fingerprint redundancy")

    # Feature categories
    bond_features = [name for name in feature_names if 'bond' in name or 'c_c' in name or 'c_n' in name]
    uff_features = [name for name in feature_names if 'uff' in name or 'vdw' in name or 'charge' in name]
    print(f"\n  📌 Feature categories:")
    print(f"     Bond features: {len(bond_features)}")
    print(f"     UFF features:  {len(uff_features)}")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CE50 STRUCTURAL FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Load data
    df = load_data()
    print(f"\n✓ Loaded {len(df)} compounds")

    # Extract features
    X_fp, X_struct, feature_names = extract_all_features(df)
    y = df['pce50'].values

    # Train models and get importance
    results = train_models_and_get_importance(X_fp, X_struct, y, feature_names)

    # Create visualizations
    fig = create_importance_visualizations(results)

    # Print analysis
    print_detailed_analysis(results)

    print("\n✓ Feature importance analysis complete!")
    print("📊 Visualization: ce50_feature_importance.png")

    plt.show()


if __name__ == '__main__':
    main()
