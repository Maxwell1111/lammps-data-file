"""
Full 4-Model Ensemble Comparison: Baseline vs Enhanced

Compares the complete ensemble approach:
- Baseline: 4 models (RF/XGB × Binary/Count fingerprints)
- Enhanced: 4 models with structural features added

Uses real experimental CE50 data (300 compounds)

Author: Enhanced CE50 Prediction System
Date: 2026-01-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from structural_features import StructuralFeatureExtractor
from ce50_ensemble_predictor import EnsembleModel, DualFingerprintGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(filepath='ce50_300compounds_training.csv'):
    """Load real CE50 data"""
    print("="*80)
    print("LOADING REAL CE50 EXPERIMENTAL DATA")
    print("="*80)

    df = pd.read_csv(filepath)

    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles', 'CE50': 'ce50'})

    df['pce50'] = -np.log10(df['ce50'])
    df = df.dropna(subset=['smiles', 'ce50'])

    print(f"\n✓ Loaded {len(df)} compounds")
    print(f"  CE50 range:  {df['ce50'].min():.1f} - {df['ce50'].max():.1f} eV")
    print(f"  pCE50 range: {df['pce50'].min():.4f} - {df['pce50'].max():.4f}")
    print(f"  Sample compounds: {', '.join(df['Compound_Name'].head(3).values)}")

    return df


def train_baseline_ensemble(df):
    """Train baseline 4-model ensemble (fingerprints only)"""
    print("\n" + "="*80)
    print("TRAINING BASELINE ENSEMBLE (Fingerprints Only)")
    print("="*80)

    # Generate fingerprints
    fp_gen = DualFingerprintGenerator()
    fps = fp_gen.generate_both(df['smiles'].values)

    X_binary = fps['binary']
    X_count = fps['count']
    y = df['pce50'].values

    print(f"\nFeature shapes:")
    print(f"  Binary fingerprints: {X_binary.shape}")
    print(f"  Count fingerprints:  {X_count.shape}")

    # Create and train ensemble
    ensemble = EnsembleModel(use_structural=False)
    models, scores = ensemble.train_ensemble(X_binary, X_count, y)

    # Evaluate
    results = ensemble.evaluate_ensemble()

    # Add y_test to results for plotting
    for model_name in results.keys():
        results[model_name]['y_test'] = ensemble.y_test

    return ensemble, results, scores


def train_enhanced_ensemble(df):
    """Train enhanced 4-model ensemble (fingerprints + structural)"""
    print("\n" + "="*80)
    print("TRAINING ENHANCED ENSEMBLE (Fingerprints + Structural Features)")
    print("="*80)

    # Generate fingerprints and structural features
    fp_gen = DualFingerprintGenerator()
    extractor = StructuralFeatureExtractor()

    print("\nExtracting features (this may take a few minutes)...")
    fps = fp_gen.generate_all_features(
        df['smiles'].values,
        use_structural=True,
        structural_extractor=extractor
    )

    X_binary = fps['binary']
    X_count = fps['count']
    X_structural = fps['structural']
    y = df['pce50'].values

    print(f"\nFeature shapes:")
    print(f"  Binary fingerprints:    {X_binary.shape}")
    print(f"  Count fingerprints:     {X_count.shape}")
    print(f"  Structural features:    {X_structural.shape}")
    print(f"  Total per model:        {X_binary.shape[1] + X_structural.shape[1]} features")

    # Create and train ensemble
    ensemble = EnsembleModel(use_structural=True)
    models, scores = ensemble.train_ensemble(X_binary, X_count, y, X_structural=X_structural)

    # Evaluate
    results = ensemble.evaluate_ensemble()

    # Add y_test to results for plotting
    for model_name in results.keys():
        results[model_name]['y_test'] = ensemble.y_test

    return ensemble, results, scores


def create_ensemble_comparison_plots(baseline_results, baseline_scores,
                                     enhanced_results, enhanced_scores,
                                     output_file='ce50_full_ensemble_comparison.png'):
    """Create comprehensive ensemble comparison visualizations"""
    print("\n" + "="*80)
    print("GENERATING ENSEMBLE COMPARISON VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(20, 14))

    # Colors
    baseline_color = '#4A90E2'
    enhanced_color = '#50C878'
    model_colors = {
        'rf_binary': '#FF6B6B',
        'rf_count': '#4ECDC4',
        'xgb_binary': '#FFD93D',
        'xgb_count': '#95E1D3'
    }

    # 1. Individual Model R² Scores - Baseline
    ax1 = plt.subplot(3, 3, 1)
    models = list(baseline_scores.keys())
    scores = list(baseline_scores.values())
    bars = ax1.bar(range(len(models)), scores,
                   color=[model_colors[m] for m in models],
                   edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('_', '\n').upper() for m in models], fontsize=10)
    ax1.set_ylabel('Cross-Val R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline: Individual Model CV Scores', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, max(scores) * 1.2])
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 2. Individual Model R² Scores - Enhanced
    ax2 = plt.subplot(3, 3, 2)
    models = list(enhanced_scores.keys())
    scores = list(enhanced_scores.values())
    bars = ax2.bar(range(len(models)), scores,
                   color=[model_colors[m] for m in models],
                   edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', '\n').upper() for m in models], fontsize=10)
    ax2.set_ylabel('Cross-Val R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Enhanced: Individual Model CV Scores', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(scores) * 1.2])
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Average Ensemble Performance Comparison
    ax3 = plt.subplot(3, 3, 3)
    baseline_avg_cv = np.mean(list(baseline_scores.values()))
    enhanced_avg_cv = np.mean(list(enhanced_scores.values()))
    baseline_avg_test = np.mean([r['r2'] for r in baseline_results.values()])
    enhanced_avg_test = np.mean([r['r2'] for r in enhanced_results.values()])

    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [baseline_avg_test, enhanced_avg_test], width,
                    label='Test R²', color=[baseline_color, enhanced_color],
                    edgecolor='black', linewidth=2)
    bars2 = ax3.bar(x + width/2, [baseline_avg_cv, enhanced_avg_cv], width,
                    label='CV R²', color=[baseline_color, enhanced_color],
                    alpha=0.6, edgecolor='black', linewidth=2)

    ax3.set_ylabel('Average R² Score', fontsize=12, fontweight='bold')
    ax3.set_title('Ensemble Average Performance', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Baseline', 'Enhanced'], fontsize=11)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add improvement annotation
    improvement = ((enhanced_avg_test - baseline_avg_test) / baseline_avg_test * 100)
    if improvement > 0:
        ax3.text(0.5, 0.4, f'+{improvement:.1f}%\nimprovement',
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4, linewidth=2))

    # 4-7. Predicted vs Actual for each model type (Baseline)
    for idx, (model_name, result) in enumerate(baseline_results.items()):
        ax = plt.subplot(3, 3, 4 + idx)
        y_test = baseline_results['rf_binary']['y_test']  # Same for all
        y_pred = result['predictions']

        ax.scatter(y_test, y_pred, alpha=0.6, s=60,
                  edgecolors='k', linewidths=0.5, c=model_colors[model_name])
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, 'r--', lw=2, alpha=0.7)
        ax.set_xlabel('Actual pCE50', fontsize=11)
        ax.set_ylabel('Predicted pCE50', fontsize=11)
        ax.set_title(f'Baseline: {model_name.upper()}\nR²={result["r2"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 8. Best Model Per Ensemble - Test Set R²
    ax8 = plt.subplot(3, 3, 8)
    baseline_best = max(baseline_results.items(), key=lambda x: x[1]['r2'])
    enhanced_best = max(enhanced_results.items(), key=lambda x: x[1]['r2'])

    comparison_data = {
        'Baseline\nBest Model': baseline_best[1]['r2'],
        'Enhanced\nBest Model': enhanced_best[1]['r2']
    }

    bars = ax8.bar(comparison_data.keys(), comparison_data.values(),
                   color=[baseline_color, enhanced_color],
                   edgecolor='black', linewidth=2)
    ax8.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
    ax8.set_title('Best Individual Model Comparison', fontsize=13, fontweight='bold')
    ax8.set_ylim([0, 1.0])
    ax8.grid(True, alpha=0.3, axis='y')

    # Add model names and values
    ax8.text(0, baseline_best[1]['r2'] + 0.05,
            f"{baseline_best[0].upper()}\n{baseline_best[1]['r2']:.4f}",
            ha='center', fontweight='bold', fontsize=10)
    ax8.text(1, enhanced_best[1]['r2'] + 0.05,
            f"{enhanced_best[0].upper()}\n{enhanced_best[1]['r2']:.4f}",
            ha='center', fontweight='bold', fontsize=10)

    # 9. Model Consistency (Std Dev of predictions)
    ax9 = plt.subplot(3, 3, 9)
    baseline_r2s = [r['r2'] for r in baseline_results.values()]
    enhanced_r2s = [r['r2'] for r in enhanced_results.values()]

    bp = ax9.boxplot([baseline_r2s, enhanced_r2s],
                     labels=['Baseline', 'Enhanced'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], [baseline_color, enhanced_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    ax9.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
    ax9.set_title('Ensemble Model Consistency', fontsize=13, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

    # Add statistics
    ax9.text(1, np.mean(baseline_r2s) - 0.05,
            f'μ={np.mean(baseline_r2s):.3f}\nσ={np.std(baseline_r2s):.3f}',
            ha='center', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax9.text(2, np.mean(enhanced_r2s) - 0.05,
            f'μ={np.mean(enhanced_r2s):.3f}\nσ={np.std(enhanced_r2s):.3f}',
            ha='center', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Full 4-Model Ensemble Comparison: Real CE50 Data (298 Compounds)\n' +
                 'Baseline (Fingerprints) vs Enhanced (Fingerprints + Structural Features)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    return fig


def print_detailed_summary(baseline_results, baseline_scores,
                          enhanced_results, enhanced_scores):
    """Print comprehensive comparison summary"""
    print("\n" + "="*80)
    print("FULL 4-MODEL ENSEMBLE COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 BASELINE ENSEMBLE (Fingerprints Only):")
    print("\n  Individual Model Performance (Cross-Validation):")
    for model_name, score in baseline_scores.items():
        print(f"    {model_name:15s}: CV R² = {score:.4f}")

    print("\n  Individual Model Performance (Test Set):")
    for model_name, result in baseline_results.items():
        print(f"    {model_name:15s}: Test R² = {result['r2']:.4f}, MAE = {result['mae']:.4f}")

    baseline_avg_cv = np.mean(list(baseline_scores.values()))
    baseline_avg_test = np.mean([r['r2'] for r in baseline_results.values()])
    baseline_best_test = max([r['r2'] for r in baseline_results.values()])

    print(f"\n  Ensemble Statistics:")
    print(f"    Average CV R²:    {baseline_avg_cv:.4f}")
    print(f"    Average Test R²:  {baseline_avg_test:.4f}")
    print(f"    Best Model R²:    {baseline_best_test:.4f}")

    print("\n" + "-"*80)

    print("\n🚀 ENHANCED ENSEMBLE (Fingerprints + Structural Features):")
    print("\n  Individual Model Performance (Cross-Validation):")
    for model_name, score in enhanced_scores.items():
        print(f"    {model_name:15s}: CV R² = {score:.4f}")

    print("\n  Individual Model Performance (Test Set):")
    for model_name, result in enhanced_results.items():
        print(f"    {model_name:15s}: Test R² = {result['r2']:.4f}, MAE = {result['mae']:.4f}")

    enhanced_avg_cv = np.mean(list(enhanced_scores.values()))
    enhanced_avg_test = np.mean([r['r2'] for r in enhanced_results.values()])
    enhanced_best_test = max([r['r2'] for r in enhanced_results.values()])

    print(f"\n  Ensemble Statistics:")
    print(f"    Average CV R²:    {enhanced_avg_cv:.4f}")
    print(f"    Average Test R²:  {enhanced_avg_test:.4f}")
    print(f"    Best Model R²:    {enhanced_best_test:.4f}")

    print("\n" + "="*80)
    print("📈 IMPROVEMENT METRICS:")
    print("="*80)

    cv_improvement = ((enhanced_avg_cv - baseline_avg_cv) / baseline_avg_cv * 100)
    test_improvement = ((enhanced_avg_test - baseline_avg_test) / baseline_avg_test * 100)
    best_improvement = ((enhanced_best_test - baseline_best_test) / baseline_best_test * 100)

    print(f"\n  Average CV R² improvement:         {cv_improvement:+.2f}%")
    print(f"  Average Test R² improvement:       {test_improvement:+.2f}%")
    print(f"  Best Model R² improvement:         {best_improvement:+.2f}%")
    print(f"\n  Absolute CV R² gain:               {enhanced_avg_cv - baseline_avg_cv:+.4f}")
    print(f"  Absolute Test R² gain:             {enhanced_avg_test - baseline_avg_test:+.4f}")
    print(f"  Absolute Best Model R² gain:       {enhanced_best_test - baseline_best_test:+.4f}")

    # Model-by-model comparison
    print(f"\n  Model-by-Model CV Improvements:")
    for model_name in baseline_scores.keys():
        baseline_cv = baseline_scores[model_name]
        enhanced_cv = enhanced_scores[model_name]
        improvement = ((enhanced_cv - baseline_cv) / baseline_cv * 100)
        print(f"    {model_name:15s}: {improvement:+.2f}% ({baseline_cv:.4f} → {enhanced_cv:.4f})")

    print("\n" + "="*80)
    print("💡 CONCLUSION:")
    print("="*80)

    if enhanced_avg_test > baseline_avg_test:
        print("\n  ✓ Enhanced ensemble OUTPERFORMS baseline!")
        print("  ✓ Structural features successfully improve CE50 prediction")
        print(f"  ✓ Average improvement: {test_improvement:.2f}%")
    else:
        print("\n  ⚠ Results are mixed - some models improved, others similar")
        print("  ℹ Consider feature selection and hyperparameter tuning")

    if enhanced_best_test > baseline_best_test:
        print(f"  ✓ Best enhanced model outperforms best baseline model by {best_improvement:.2f}%")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("FULL 4-MODEL ENSEMBLE COMPARISON")
    print("Baseline (Fingerprints) vs Enhanced (Fingerprints + Structural)")
    print("="*80)

    # Load data
    df = load_data()

    # Train baseline ensemble
    baseline_ensemble, baseline_results, baseline_scores = train_baseline_ensemble(df)

    # Train enhanced ensemble
    enhanced_ensemble, enhanced_results, enhanced_scores = train_enhanced_ensemble(df)

    # Create visualizations
    fig = create_ensemble_comparison_plots(
        baseline_results, baseline_scores,
        enhanced_results, enhanced_scores
    )

    # Print summary
    print_detailed_summary(
        baseline_results, baseline_scores,
        enhanced_results, enhanced_scores
    )

    print("\n✓ Full ensemble comparison complete!")
    print("📊 Visualization: ce50_full_ensemble_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
