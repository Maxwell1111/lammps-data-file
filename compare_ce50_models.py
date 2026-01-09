"""
CE50 Model Comparison: Baseline vs Enhanced

This script demonstrates the performance improvement from adding structural features
to the CE50 prediction ensemble. It creates comparison visualizations.

NOTE: This demonstration uses synthetic data for illustration.
For real validation, replace with actual experimental CE50 data.

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

from structural_features import StructuralFeatureExtractor
from ce50_ensemble_predictor import DualFingerprintGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_ce50_data(n_samples=150):
    """
    Generate synthetic CE50 data with realistic patterns.

    In real usage, replace this with actual experimental data.
    """
    print("Generating synthetic CE50 data (for demonstration only)...")

    # Diverse set of molecules
    smiles_list = []

    # Alkanes (low fragmentation energy)
    for n in range(1, 8):
        smiles_list.append('C' * n)

    # Alcohols
    for n in range(1, 7):
        smiles_list.append('C' * n + 'O')

    # Amines
    for n in range(1, 6):
        smiles_list.append('C' * n + 'N')

    # Aromatics (higher fragmentation energy)
    aromatics = [
        'c1ccccc1',  # benzene
        'Cc1ccccc1',  # toluene
        'CCc1ccccc1',  # ethylbenzene
        'c1ccc(C)cc1C',  # xylene
        'c1ccc(O)cc1',  # phenol
        'c1ccc(N)cc1',  # aniline
        'c1ccc(Cl)cc1',  # chlorobenzene
    ]
    smiles_list.extend(aromatics * 2)

    # Branched molecules
    branched = [
        'CC(C)C',  # isobutane
        'CC(C)CC',  # isopentane
        'CC(C)(C)C',  # neopentane
        'CC(C)CO',  # isobutanol
    ]
    smiles_list.extend(branched * 3)

    # Heterocycles
    heterocycles = [
        'c1cccnc1',  # pyridine
        'c1ccoc1',  # furan
        'c1ccsc1',  # thiophene
        'C1CCCCC1',  # cyclohexane
    ]
    smiles_list.extend(heterocycles * 2)

    # Complex molecules
    complex_mols = [
        'CC(=O)OC',  # methyl acetate
        'CC(=O)C',  # acetone
        'CCN(CC)CC',  # triethylamine
        'c1ccc2ccccc2c1',  # naphthalene
    ]
    smiles_list.extend(complex_mols * 3)

    # Ensure we have enough samples
    while len(smiles_list) < n_samples:
        smiles_list.extend(smiles_list[:n_samples - len(smiles_list)])

    smiles_list = smiles_list[:n_samples]

    # Generate synthetic CE50 values based on molecular properties
    # Real CE50 depends on bond strengths, size, aromaticity, etc.
    ce50_values = []

    for smiles in smiles_list:
        # Base CE50
        base_ce50 = 30.0

        # Size factor (larger molecules need more energy)
        size_factor = len(smiles) * 0.5

        # Aromatic rings increase stability
        aromatic_factor = smiles.count('c') * 1.5

        # Branching increases fragmentation sites
        branch_factor = smiles.count('(') * -2.0

        # Heteroatoms affect fragmentation
        hetero_factor = (smiles.count('O') + smiles.count('N')) * 1.0

        # Add some realistic noise
        noise = np.random.normal(0, 3.0)

        ce50 = base_ce50 + size_factor + aromatic_factor + branch_factor + hetero_factor + noise
        ce50 = max(15.0, min(60.0, ce50))  # Realistic range
        ce50_values.append(ce50)

    df = pd.DataFrame({
        'smiles': smiles_list,
        'ce50': ce50_values
    })
    df['pce50'] = -np.log10(df['ce50'])

    return df


def extract_features(df, use_structural=False):
    """Extract fingerprints and optionally structural features"""
    print(f"\nExtracting features (use_structural={use_structural})...")

    fp_gen = DualFingerprintGenerator()

    if use_structural:
        extractor = StructuralFeatureExtractor()
        fps = fp_gen.generate_all_features(
            df['smiles'].values,
            use_structural=True,
            structural_extractor=extractor
        )

        # Concatenate binary fingerprints with structural features
        X = np.hstack([fps['binary'], fps['structural']])
        print(f"  Feature shape: {X.shape} (fingerprints + structural)")
    else:
        fps = fp_gen.generate_both(df['smiles'].values)
        X = fps['binary']
        print(f"  Feature shape: {X.shape} (fingerprints only)")

    return X


def train_and_evaluate_model(X, y, model_name):
    """Train a model and return predictions"""
    print(f"\nTraining {model_name}...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # Train
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
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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
        'cv_scores': cv_scores
    }


def create_comparison_plots(baseline_results, enhanced_results):
    """Create comprehensive comparison visualizations"""
    print("\nGenerating comparison plots...")

    fig = plt.figure(figsize=(16, 12))

    # 1. Predicted vs Actual - Baseline
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(baseline_results['y_test'], baseline_results['y_pred_test'],
                alpha=0.6, s=80, edgecolors='k', linewidths=0.5)
    ax1.plot([baseline_results['y_test'].min(), baseline_results['y_test'].max()],
             [baseline_results['y_test'].min(), baseline_results['y_test'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual pCE50', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted pCE50', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Model (Fingerprints Only)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    textstr = f"R² = {baseline_results['test_r2']:.4f}\nMAE = {baseline_results['test_mae']:.4f}"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 2. Predicted vs Actual - Enhanced
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(enhanced_results['y_test'], enhanced_results['y_pred_test'],
                alpha=0.6, s=80, edgecolors='k', linewidths=0.5, color='green')
    ax2.plot([enhanced_results['y_test'].min(), enhanced_results['y_test'].max()],
             [enhanced_results['y_test'].min(), enhanced_results['y_test'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual pCE50', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted pCE50', fontsize=12, fontweight='bold')
    ax2.set_title('Enhanced Model (Fingerprints + Structural)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    textstr = f"R² = {enhanced_results['test_r2']:.4f}\nMAE = {enhanced_results['test_mae']:.4f}"
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. R² Comparison
    ax3 = plt.subplot(2, 3, 3)
    models = ['Baseline\n(Fingerprints)', 'Enhanced\n(+ Structural)']
    r2_scores = [baseline_results['test_r2'], enhanced_results['test_r2']]
    colors = ['lightblue', 'lightgreen']
    bars = ax3.bar(models, r2_scores, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax3.set_title('Model Comparison: R² Score', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement annotation
    improvement = ((enhanced_results['test_r2'] - baseline_results['test_r2']) /
                   baseline_results['test_r2'] * 100)
    ax3.text(0.5, 0.5, f'+{improvement:.1f}%\nimprovement',
            transform=ax3.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 4. MAE Comparison
    ax4 = plt.subplot(2, 3, 4)
    mae_scores = [baseline_results['test_mae'], enhanced_results['test_mae']]
    bars = ax4.bar(models, mae_scores, color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_title('Model Comparison: MAE (Lower is Better)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    reduction = ((baseline_results['test_mae'] - enhanced_results['test_mae']) /
                 baseline_results['test_mae'] * 100)
    ax4.text(0.5, 0.5, f'{reduction:.1f}%\nreduction',
            transform=ax4.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 5. Residuals Comparison
    ax5 = plt.subplot(2, 3, 5)
    baseline_residuals = baseline_results['y_test'] - baseline_results['y_pred_test']
    enhanced_residuals = enhanced_results['y_test'] - enhanced_results['y_pred_test']

    violin_parts = ax5.violinplot([baseline_residuals, enhanced_residuals],
                                   positions=[1, 2],
                                   showmeans=True, showmedians=True)
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels(['Baseline', 'Enhanced'])
    ax5.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax5.set_title('Residuals Distribution', fontsize=13, fontweight='bold')
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Cross-Validation Scores
    ax6 = plt.subplot(2, 3, 6)
    cv_data = [baseline_results['cv_scores'], enhanced_results['cv_scores']]
    bp = ax6.boxplot(cv_data, labels=['Baseline', 'Enhanced'],
                     patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    ax6.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax6.set_title('5-Fold Cross-Validation Scores', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add mean values
    for i, scores in enumerate(cv_data):
        ax6.text(i + 1, scores.mean() + 0.02, f'{scores.mean():.4f}',
                ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('ce50_model_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: ce50_model_comparison.png")

    return fig


def print_summary(baseline_results, enhanced_results):
    """Print detailed comparison summary"""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    print("\nBASELINE MODEL (Fingerprints Only):")
    print(f"  Test R²:  {baseline_results['test_r2']:.4f}")
    print(f"  Test MAE: {baseline_results['test_mae']:.4f}")
    print(f"  Test RMSE: {baseline_results['test_rmse']:.4f}")
    print(f"  CV R² (mean ± std): {baseline_results['cv_scores'].mean():.4f} ± {baseline_results['cv_scores'].std():.4f}")

    print("\nENHANCED MODEL (Fingerprints + Structural Features):")
    print(f"  Test R²:  {enhanced_results['test_r2']:.4f}")
    print(f"  Test MAE: {enhanced_results['test_mae']:.4f}")
    print(f"  Test RMSE: {enhanced_results['test_rmse']:.4f}")
    print(f"  CV R² (mean ± std): {enhanced_results['cv_scores'].mean():.4f} ± {enhanced_results['cv_scores'].std():.4f}")

    print("\nIMPROVEMENT:")
    r2_improvement = ((enhanced_results['test_r2'] - baseline_results['test_r2']) /
                      baseline_results['test_r2'] * 100)
    mae_reduction = ((baseline_results['test_mae'] - enhanced_results['test_mae']) /
                     baseline_results['test_mae'] * 100)

    print(f"  R² improvement:    +{r2_improvement:.1f}%")
    print(f"  MAE reduction:     -{mae_reduction:.1f}%")
    print(f"  Absolute R² gain:  +{enhanced_results['test_r2'] - baseline_results['test_r2']:.4f}")

    if enhanced_results['test_r2'] > baseline_results['test_r2']:
        print(f"\n✓ Enhanced model outperforms baseline!")

    print("\n" + "="*70)


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("CE50 PREDICTION: BASELINE vs ENHANCED MODEL COMPARISON")
    print("="*70)
    print("\nNOTE: This demonstration uses synthetic data.")
    print("For real validation, use actual experimental CE50 data.\n")

    # Generate data
    df = generate_synthetic_ce50_data(n_samples=150)
    print(f"  Generated {len(df)} molecules")
    print(f"  CE50 range: {df['ce50'].min():.1f} - {df['ce50'].max():.1f}")

    # Extract features for baseline model
    X_baseline = extract_features(df, use_structural=False)

    # Extract features for enhanced model
    X_enhanced = extract_features(df, use_structural=True)

    # Target variable
    y = df['pce50'].values

    # Train and evaluate baseline model
    baseline_results = train_and_evaluate_model(
        X_baseline, y, "Baseline (Fingerprints Only)"
    )

    # Train and evaluate enhanced model
    enhanced_results = train_and_evaluate_model(
        X_enhanced, y, "Enhanced (Fingerprints + Structural)"
    )

    # Create comparison plots
    fig = create_comparison_plots(baseline_results, enhanced_results)

    # Print summary
    print_summary(baseline_results, enhanced_results)

    plt.show()

    print("\nVisualization saved: ce50_model_comparison.png")
    print("Review the charts to see the performance improvement!")


if __name__ == '__main__':
    main()
