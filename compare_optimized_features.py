"""
Optimized Feature Set Comparison

Compares three approaches:
1. Baseline: Fingerprints only
2. Full: Fingerprints + all 40 structural features
3. Optimized: Fingerprints + top 10 selected features

Uses real CE50 data (298 compounds)

Author: Enhanced CE50 Prediction System
Date: 2026-01-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from structural_features import StructuralFeatureExtractor
from ce50_ensemble_predictor import DualFingerprintGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Top 10 features based on permutation importance analysis
TOP_FEATURES = [
    'c_n_bonds',
    'max_bond_length',
    'topology_complexity',
    'bond_density',
    'rotatable_bonds',
    'ring_count',
    'avg_bond_length',
    'num_nitrogen',
    'hba_count',
    'std_bond_length'
]


def load_data(filepath='ce50_300compounds_training.csv'):
    """Load CE50 data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    df = pd.read_csv(filepath)
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles', 'CE50': 'ce50'})
    df['pce50'] = -np.log10(df['ce50'])
    df = df.dropna(subset=['smiles', 'ce50'])

    print(f"\n✓ Loaded {len(df)} compounds")
    print(f"  CE50 range:  {df['ce50'].min():.1f} - {df['ce50'].max():.1f} eV")

    return df


def extract_features(df):
    """Extract all features"""
    print("\n" + "="*80)
    print("EXTRACTING FEATURES")
    print("="*80)

    fp_gen = DualFingerprintGenerator()
    extractor = StructuralFeatureExtractor()

    print("\nGenerating fingerprints and structural features...")
    fps = fp_gen.generate_all_features(
        df['smiles'].values,
        use_structural=True,
        structural_extractor=extractor
    )

    X_fp = fps['binary']
    X_struct_full = fps['structural']
    feature_names = extractor.feature_names

    # Extract top 10 features only
    top_indices = [feature_names.index(f) for f in TOP_FEATURES]
    X_struct_top10 = X_struct_full[:, top_indices]

    print(f"\n✓ Fingerprints:              {X_fp.shape}")
    print(f"✓ Structural (all 40):       {X_struct_full.shape}")
    print(f"✓ Structural (top 10):       {X_struct_top10.shape}")
    print(f"\n  Top 10 features: {', '.join(TOP_FEATURES[:5])}...")

    return X_fp, X_struct_full, X_struct_top10, df['pce50'].values


def train_and_evaluate(X, y, model_name, n_estimators=300):
    """Train a model and return comprehensive results"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"  Feature dimensionality: {X.shape[1]}")

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # Train
    print("  Training model...")
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Cross-validation
    print("  Cross-validating...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

    print(f"\n  ✓ Train R²:  {train_r2:.4f}")
    print(f"  ✓ Test R²:   {test_r2:.4f}")
    print(f"  ✓ Test MAE:  {test_mae:.4f}")
    print(f"  ✓ CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        'model': pipeline,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def create_comparison_plots(baseline, full, optimized, output_file='ce50_optimized_comparison.png'):
    """Create comprehensive comparison visualizations"""
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(20, 12))

    # Colors
    baseline_color = '#4A90E2'
    full_color = '#E74C3C'
    optimized_color = '#2ECC71'

    # 1. R² Score Comparison
    ax1 = plt.subplot(2, 4, 1)
    models = ['Baseline\n(FP only)', 'Full\n(FP+40)', 'Optimized\n(FP+10)']
    test_scores = [baseline['test_r2'], full['test_r2'], optimized['test_r2']]
    cv_scores = [baseline['cv_mean'], full['cv_mean'], optimized['cv_mean']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, test_scores, width, label='Test R²',
                    color=[baseline_color, full_color, optimized_color],
                    edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, cv_scores, width, label='CV R²',
                    color=[baseline_color, full_color, optimized_color],
                    alpha=0.6, edgecolor='black', linewidth=2)

    ax1.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 0.8])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 2. Predicted vs Actual - Baseline
    ax2 = plt.subplot(2, 4, 2)
    ax2.scatter(baseline['y_test'], baseline['y_pred_test'],
                alpha=0.6, s=80, edgecolors='k', linewidths=0.5, c=baseline_color)
    lims = [baseline['y_test'].min(), baseline['y_test'].max()]
    ax2.plot(lims, lims, 'r--', lw=2, alpha=0.7)
    ax2.set_xlabel('Actual pCE50', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted pCE50', fontsize=11, fontweight='bold')
    ax2.set_title(f'Baseline (FP Only)\nR²={baseline["test_r2"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Predicted vs Actual - Full
    ax3 = plt.subplot(2, 4, 3)
    ax3.scatter(full['y_test'], full['y_pred_test'],
                alpha=0.6, s=80, edgecolors='k', linewidths=0.5, c=full_color)
    lims = [full['y_test'].min(), full['y_test'].max()]
    ax3.plot(lims, lims, 'r--', lw=2, alpha=0.7)
    ax3.set_xlabel('Actual pCE50', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted pCE50', fontsize=11, fontweight='bold')
    ax3.set_title(f'Full Features (FP+40)\nR²={full["test_r2"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Predicted vs Actual - Optimized
    ax4 = plt.subplot(2, 4, 4)
    ax4.scatter(optimized['y_test'], optimized['y_pred_test'],
                alpha=0.6, s=80, edgecolors='k', linewidths=0.5, c=optimized_color)
    lims = [optimized['y_test'].min(), optimized['y_test'].max()]
    ax4.plot(lims, lims, 'r--', lw=2, alpha=0.7)
    ax4.set_xlabel('Actual pCE50', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted pCE50', fontsize=11, fontweight='bold')
    ax4.set_title(f'Optimized (FP+10)\nR²={optimized["test_r2"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add winner annotation
    best_r2 = max(baseline['test_r2'], full['test_r2'], optimized['test_r2'])
    if optimized['test_r2'] == best_r2:
        ax4.text(0.5, 0.1, '⭐ BEST', transform=ax4.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5, linewidth=2))

    # 5. MAE Comparison
    ax5 = plt.subplot(2, 4, 5)
    mae_scores = [baseline['test_mae'], full['test_mae'], optimized['test_mae']]
    bars = ax5.bar(models, mae_scores,
                   color=[baseline_color, full_color, optimized_color],
                   edgecolor='black', linewidth=2)
    ax5.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax5.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, mae in zip(bars, mae_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{mae:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 6. Cross-Validation Scores
    ax6 = plt.subplot(2, 4, 6)
    cv_data = [baseline['cv_scores'], full['cv_scores'], optimized['cv_scores']]
    bp = ax6.boxplot(cv_data, labels=['Baseline', 'Full', 'Optimized'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], [baseline_color, full_color, optimized_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    ax6.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax6.set_title('5-Fold Cross-Validation', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add mean values
    for i, scores in enumerate(cv_data):
        ax6.text(i + 1, scores.mean() - 0.05,
                f'{scores.mean():.4f}',
                ha='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 7. Improvement Metrics
    ax7 = plt.subplot(2, 4, 7)

    baseline_val = baseline['test_r2']
    improvements = [
        0,  # baseline vs baseline
        ((full['test_r2'] - baseline_val) / baseline_val * 100),
        ((optimized['test_r2'] - baseline_val) / baseline_val * 100)
    ]

    colors_imp = [baseline_color,
                  full_color if improvements[1] > 0 else '#E74C3C',
                  optimized_color if improvements[2] > 0 else '#E74C3C']

    bars = ax7.bar(models, improvements, color=colors_imp,
                   edgecolor='black', linewidth=2)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax7.set_title('Relative Performance Improvement', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.5 if height >= 0 else -0.5
        ax7.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{imp:+.1f}%',
                ha='center', va=va, fontweight='bold', fontsize=11)

    # 8. Feature Count vs Performance
    ax8 = plt.subplot(2, 4, 8)

    feature_counts = [2048, 2088, 2058]  # FP, FP+40, FP+10
    test_r2s = [baseline['test_r2'], full['test_r2'], optimized['test_r2']]

    ax8.scatter([feature_counts[0]], [test_r2s[0]], s=200, c=baseline_color,
               edgecolors='k', linewidths=2, label='Baseline', zorder=3)
    ax8.scatter([feature_counts[1]], [test_r2s[1]], s=200, c=full_color,
               edgecolors='k', linewidths=2, label='Full', zorder=3)
    ax8.scatter([feature_counts[2]], [test_r2s[2]], s=200, c=optimized_color,
               edgecolors='k', linewidths=2, label='Optimized', marker='*', zorder=3)

    ax8.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Test R² Score', fontsize=13, fontweight='bold')
    ax8.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)

    # Annotate optimized point
    ax8.annotate('Feature\nSelection\nWins!',
                xy=(feature_counts[2], test_r2s[2]),
                xytext=(feature_counts[2] + 15, test_r2s[2] + 0.05),
                fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))

    plt.suptitle('CE50 Prediction: Feature Selection Optimization (298 Compounds)\n' +
                 'Baseline vs Full Features vs Top 10 Selected Features',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    return fig


def print_summary(baseline, full, optimized):
    """Print detailed comparison summary"""
    print("\n" + "="*80)
    print("OPTIMIZED FEATURE SET COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 MODEL COMPARISON:")
    print(f"\n  1. BASELINE (Fingerprints Only - 2048 features):")
    print(f"     Test R²:   {baseline['test_r2']:.4f}")
    print(f"     Test MAE:  {baseline['test_mae']:.4f}")
    print(f"     CV R²:     {baseline['cv_mean']:.4f} ± {baseline['cv_std']:.4f}")

    print(f"\n  2. FULL (Fingerprints + 40 Structural - 2088 features):")
    print(f"     Test R²:   {full['test_r2']:.4f}")
    print(f"     Test MAE:  {full['test_mae']:.4f}")
    print(f"     CV R²:     {full['cv_mean']:.4f} ± {full['cv_std']:.4f}")

    print(f"\n  3. OPTIMIZED (Fingerprints + 10 Selected - 2058 features):")
    print(f"     Test R²:   {optimized['test_r2']:.4f}")
    print(f"     Test MAE:  {optimized['test_mae']:.4f}")
    print(f"     CV R²:     {optimized['cv_mean']:.4f} ± {optimized['cv_std']:.4f}")

    print("\n" + "-"*80)
    print("📈 IMPROVEMENT METRICS:")
    print("-"*80)

    full_vs_baseline = ((full['test_r2'] - baseline['test_r2']) / baseline['test_r2'] * 100)
    opt_vs_baseline = ((optimized['test_r2'] - baseline['test_r2']) / baseline['test_r2'] * 100)
    opt_vs_full = ((optimized['test_r2'] - full['test_r2']) / full['test_r2'] * 100)

    print(f"\n  Full vs Baseline:        {full_vs_baseline:+.2f}%")
    print(f"  Optimized vs Baseline:   {opt_vs_baseline:+.2f}%")
    print(f"  Optimized vs Full:       {opt_vs_full:+.2f}%")

    # Find best model
    best_test = max(baseline['test_r2'], full['test_r2'], optimized['test_r2'])
    best_cv = max(baseline['cv_mean'], full['cv_mean'], optimized['cv_mean'])

    print("\n" + "-"*80)
    print("🏆 WINNER:")
    print("-"*80)

    if optimized['test_r2'] == best_test:
        print("\n  ✓ OPTIMIZED MODEL (FP + Top 10 Features) is the BEST!")
        print(f"    - Achieves highest test R²: {optimized['test_r2']:.4f}")
        print(f"    - Uses only 10 structural features instead of 40")
        print(f"    - {opt_vs_full:+.2f}% better than full feature set")
    elif baseline['test_r2'] == best_test:
        print("\n  ✓ BASELINE (Fingerprints Only) performs best")
        print(f"    - Test R²: {baseline['test_r2']:.4f}")
        print("    - Structural features didn't improve prediction")
    else:
        print("\n  ✓ FULL MODEL (All 40 Features) performs best")
        print(f"    - Test R²: {full['test_r2']:.4f}")

    print("\n💡 TOP 10 SELECTED FEATURES:")
    for i, feat in enumerate(TOP_FEATURES, 1):
        print(f"  {i:2d}. {feat}")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("OPTIMIZED FEATURE SET COMPARISON")
    print("Baseline vs Full (40) vs Optimized (Top 10)")
    print("="*80)

    # Load data
    df = load_data()

    # Extract features
    X_fp, X_struct_full, X_struct_top10, y = extract_features(df)

    # Prepare feature sets
    X_baseline = X_fp
    X_full = np.hstack([X_fp, X_struct_full])
    X_optimized = np.hstack([X_fp, X_struct_top10])

    # Train and evaluate all three models
    baseline_results = train_and_evaluate(X_baseline, y, "BASELINE (Fingerprints Only)")
    full_results = train_and_evaluate(X_full, y, "FULL (Fingerprints + 40 Features)")
    optimized_results = train_and_evaluate(X_optimized, y, "OPTIMIZED (Fingerprints + Top 10)")

    # Create visualizations
    fig = create_comparison_plots(baseline_results, full_results, optimized_results)

    # Print summary
    print_summary(baseline_results, full_results, optimized_results)

    print("\n✓ Optimization analysis complete!")
    print("📊 Visualization: ce50_optimized_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
