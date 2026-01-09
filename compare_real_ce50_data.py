"""
CE50 Real Data Comparison: Baseline vs Enhanced Models

Uses actual experimental CE50 data (300 compounds) to compare:
- Baseline: Morgan fingerprints only
- Enhanced: Morgan fingerprints + structural features

Author: Enhanced CE50 Prediction System
Date: 2026-01-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from structural_features import StructuralFeatureExtractor
from ce50_ensemble_predictor import DualFingerprintGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_real_ce50_data(filepath='ce50_300compounds_training.csv'):
    """Load real experimental CE50 data"""
    print(f"Loading real CE50 data from {filepath}...")

    df = pd.read_csv(filepath)

    # Rename columns if needed
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles', 'CE50': 'ce50'})

    # Convert to pCE50
    df['pce50'] = -np.log10(df['ce50'])

    # Remove any invalid entries
    df = df.dropna(subset=['smiles', 'ce50'])

    print(f"  Loaded {len(df)} compounds")
    print(f"  CE50 range: {df['ce50'].min():.1f} - {df['ce50'].max():.1f}")
    print(f"  pCE50 range: {df['pce50'].min():.4f} - {df['pce50'].max():.4f}")

    return df


def extract_features(df, use_structural=False):
    """Extract fingerprints and optionally structural features"""
    print(f"\nExtracting features (use_structural={use_structural})...")

    fp_gen = DualFingerprintGenerator()

    if use_structural:
        print("  Extracting structural features (this may take a few minutes)...")
        extractor = StructuralFeatureExtractor()
        fps = fp_gen.generate_all_features(
            df['smiles'].values,
            use_structural=True,
            structural_extractor=extractor
        )

        # Concatenate binary fingerprints with structural features
        X = np.hstack([fps['binary'], fps['structural']])
        print(f"  ✓ Feature shape: {X.shape} (2048 fingerprints + {fps['structural'].shape[1]} structural)")
    else:
        fps = fp_gen.generate_both(df['smiles'].values)
        X = fps['binary']
        print(f"  ✓ Feature shape: {X.shape} (fingerprints only)")

    return X


def train_and_evaluate_model(X, y, model_name, n_estimators=300):
    """Train Random Forest and return comprehensive results"""
    print(f"\nTraining {model_name}...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

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
            n_jobs=-1,
            verbose=0
        ))
    ])

    # Train
    print("  Training...")
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Cross-validation
    print("  Cross-validating...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

    print(f"  ✓ Train R²:  {train_r2:.4f}")
    print(f"  ✓ Test R²:   {test_r2:.4f}")
    print(f"  ✓ Test MAE:  {test_mae:.4f}")
    print(f"  ✓ Test RMSE: {test_rmse:.4f}")
    print(f"  ✓ CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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


def create_comparison_plots(baseline_results, enhanced_results, output_file='ce50_real_comparison.png'):
    """Create comprehensive comparison visualizations"""
    print("\nGenerating comparison plots...")

    fig = plt.figure(figsize=(18, 12))

    # Color scheme
    baseline_color = '#4A90E2'
    enhanced_color = '#50C878'

    # 1. Predicted vs Actual - Baseline
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(baseline_results['y_test'], baseline_results['y_pred_test'],
                alpha=0.6, s=100, edgecolors='k', linewidths=0.5, c=baseline_color)
    lims = [min(baseline_results['y_test'].min(), baseline_results['y_pred_test'].min()),
            max(baseline_results['y_test'].max(), baseline_results['y_pred_test'].max())]
    ax1.plot(lims, lims, 'r--', lw=2.5, label='Perfect Prediction', alpha=0.7)
    ax1.set_xlabel('Actual pCE50', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted pCE50', fontsize=13, fontweight='bold')
    ax1.set_title('Baseline Model\n(Morgan Fingerprints Only)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    textstr = f"R² = {baseline_results['test_r2']:.4f}\nMAE = {baseline_results['test_mae']:.4f}\nRMSE = {baseline_results['test_rmse']:.4f}"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=baseline_color, linewidth=2, alpha=0.9))

    # 2. Predicted vs Actual - Enhanced
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(enhanced_results['y_test'], enhanced_results['y_pred_test'],
                alpha=0.6, s=100, edgecolors='k', linewidths=0.5, c=enhanced_color)
    lims = [min(enhanced_results['y_test'].min(), enhanced_results['y_pred_test'].min()),
            max(enhanced_results['y_test'].max(), enhanced_results['y_pred_test'].max())]
    ax2.plot(lims, lims, 'r--', lw=2.5, label='Perfect Prediction', alpha=0.7)
    ax2.set_xlabel('Actual pCE50', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Predicted pCE50', fontsize=13, fontweight='bold')
    ax2.set_title('Enhanced Model\n(Fingerprints + Structural Features)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    textstr = f"R² = {enhanced_results['test_r2']:.4f}\nMAE = {enhanced_results['test_mae']:.4f}\nRMSE = {enhanced_results['test_rmse']:.4f}"
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=enhanced_color, linewidth=2, alpha=0.9))

    # 3. R² Comparison
    ax3 = plt.subplot(2, 3, 3)
    models = ['Baseline\n(Fingerprints)', 'Enhanced\n(+ Structural)']
    r2_test = [baseline_results['test_r2'], enhanced_results['test_r2']]
    r2_cv = [baseline_results['cv_mean'], enhanced_results['cv_mean']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax3.bar(x - width/2, r2_test, width, label='Test R²',
                    color=[baseline_color, enhanced_color], edgecolor='black', linewidth=2)
    bars2 = ax3.bar(x + width/2, r2_cv, width, label='CV R² (mean)',
                    color=[baseline_color, enhanced_color], alpha=0.6, edgecolor='black', linewidth=2)

    ax3.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax3.set_title('Model Comparison: R² Scores', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add improvement annotation
    if enhanced_results['test_r2'] > baseline_results['test_r2']:
        improvement = ((enhanced_results['test_r2'] - baseline_results['test_r2']) /
                       baseline_results['test_r2'] * 100)
        ax3.text(0.5, 0.35, f'+{improvement:.1f}%\nimprovement',
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=16, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4, linewidth=2))

    # 4. MAE Comparison
    ax4 = plt.subplot(2, 3, 4)
    mae_scores = [baseline_results['test_mae'], enhanced_results['test_mae']]
    bars = ax4.bar(models, mae_scores, color=[baseline_color, enhanced_color],
                   edgecolor='black', linewidth=2)
    ax4.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax4.set_title('Model Comparison: MAE\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    if enhanced_results['test_mae'] < baseline_results['test_mae']:
        reduction = ((baseline_results['test_mae'] - enhanced_results['test_mae']) /
                     baseline_results['test_mae'] * 100)
        ax4.text(0.5, 0.6, f'{reduction:.1f}%\nreduction',
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=16, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4, linewidth=2))

    # 5. Residuals Distribution
    ax5 = plt.subplot(2, 3, 5)
    baseline_residuals = baseline_results['y_test'] - baseline_results['y_pred_test']
    enhanced_residuals = enhanced_results['y_test'] - enhanced_results['y_pred_test']

    bp = ax5.boxplot([baseline_residuals, enhanced_residuals],
                     labels=['Baseline', 'Enhanced'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], [baseline_color, enhanced_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2.5, alpha=0.7, label='Zero Error')
    ax5.set_ylabel('Residuals (Actual - Predicted)', fontsize=13, fontweight='bold')
    ax5.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Cross-Validation Scores
    ax6 = plt.subplot(2, 3, 6)
    cv_data = [baseline_results['cv_scores'], enhanced_results['cv_scores']]
    bp = ax6.boxplot(cv_data, labels=['Baseline', 'Enhanced'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], [baseline_color, enhanced_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    ax6.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax6.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add mean values
    for i, scores in enumerate(cv_data):
        ax6.text(i + 1, scores.mean() - 0.05, f'μ = {scores.mean():.4f}\nσ = {scores.std():.4f}',
                ha='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('CE50 Prediction: Real Experimental Data (300 Compounds)\nBaseline vs Enhanced Model Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")

    return fig


def print_summary(baseline_results, enhanced_results):
    """Print detailed comparison summary"""
    print("\n" + "="*80)
    print("REAL CE50 DATA: MODEL COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 BASELINE MODEL (Morgan Fingerprints Only):")
    print(f"  Test R²:   {baseline_results['test_r2']:.4f}")
    print(f"  Test MAE:  {baseline_results['test_mae']:.4f}")
    print(f"  Test RMSE: {baseline_results['test_rmse']:.4f}")
    print(f"  CV R²:     {baseline_results['cv_mean']:.4f} ± {baseline_results['cv_std']:.4f}")

    print("\n🚀 ENHANCED MODEL (Fingerprints + Structural Features):")
    print(f"  Test R²:   {enhanced_results['test_r2']:.4f}")
    print(f"  Test MAE:  {enhanced_results['test_mae']:.4f}")
    print(f"  Test RMSE: {enhanced_results['test_rmse']:.4f}")
    print(f"  CV R²:     {enhanced_results['cv_mean']:.4f} ± {enhanced_results['cv_std']:.4f}")

    print("\n📈 IMPROVEMENT METRICS:")
    r2_improvement = ((enhanced_results['test_r2'] - baseline_results['test_r2']) /
                      baseline_results['test_r2'] * 100)
    mae_change = ((enhanced_results['test_mae'] - baseline_results['test_mae']) /
                   baseline_results['test_mae'] * 100)
    cv_improvement = ((enhanced_results['cv_mean'] - baseline_results['cv_mean']) /
                      baseline_results['cv_mean'] * 100)

    print(f"  R² improvement:        {r2_improvement:+.2f}%")
    print(f"  MAE change:            {mae_change:+.2f}%")
    print(f"  CV R² improvement:     {cv_improvement:+.2f}%")
    print(f"  Absolute R² gain:      {enhanced_results['test_r2'] - baseline_results['test_r2']:+.4f}")
    print(f"  Absolute MAE change:   {enhanced_results['test_mae'] - baseline_results['test_mae']:+.4f}")

    print("\n💡 CONCLUSION:")
    if enhanced_results['test_r2'] > baseline_results['test_r2']:
        print("  ✓ Enhanced model OUTPERFORMS baseline on test set!")
        print("  ✓ Structural features successfully improve CE50 prediction")
    else:
        print("  ⚠ Models show similar performance on this dataset")
        print("  ℹ Structural features may need further tuning or feature selection")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CE50 PREDICTION: REAL EXPERIMENTAL DATA COMPARISON")
    print("="*80)
    print("\n Using actual CE50 experimental data (300 compounds)")

    # Load real data
    df = load_real_ce50_data()

    # Extract features for baseline model
    X_baseline = extract_features(df, use_structural=False)

    # Extract features for enhanced model
    X_enhanced = extract_features(df, use_structural=True)

    # Target variable
    y = df['pce50'].values

    # Train and evaluate baseline model
    baseline_results = train_and_evaluate_model(
        X_baseline, y, "Baseline (Fingerprints Only)", n_estimators=300
    )

    # Train and evaluate enhanced model
    enhanced_results = train_and_evaluate_model(
        X_enhanced, y, "Enhanced (Fingerprints + Structural)", n_estimators=300
    )

    # Create comparison plots
    fig = create_comparison_plots(baseline_results, enhanced_results)

    # Print summary
    print_summary(baseline_results, enhanced_results)

    print("\n✓ Analysis complete!")
    print("📊 Visualization: ce50_real_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
