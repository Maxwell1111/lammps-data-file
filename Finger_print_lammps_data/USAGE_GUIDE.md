# CE50 Prediction: Usage Guide

Complete guide for using the optimized CE50 prediction model.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Your Own Model](#training-your-own-model)
4. [Making Predictions](#making-predictions)
5. [Batch Processing](#batch-processing)
6. [Feature Extraction](#feature-extraction)
7. [Model Evaluation](#model-evaluation)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv ce50_env

# Activate on macOS/Linux
source ce50_env/bin/activate

# Activate on Windows
ce50_env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
# Test imports
python -c "from ce50_predictor import CE50Predictor; print('Installation successful!')"
```

---

## Quick Start

### Minimal Example

```python
from ce50_predictor import CE50Predictor
import pandas as pd

# Load training data
df = pd.read_csv('ce50_300compounds_training.csv')

# Initialize predictor with optimized features
predictor = CE50Predictor(use_optimized_features=True)

# Train the model
results = predictor.train(
    smiles_list=df['smiles'].values,
    ce50_values=df['ce50'].values,
    test_size=0.2,
    random_state=42
)

# Print training results
print(f"Training R²: {results['train_r2']:.3f}")
print(f"Test R²: {results['test_r2']:.3f}")

# Make a prediction
result = predictor.predict("CCO")  # Ethanol
print(f"\nPredicted CE50: {result['ce50']:.2f} eV")
print(f"Confidence: {result['confidence']}")
```

**Expected Output:**
```
Generating features...
Extracting structural features for 298 molecules...
Using 10 optimized features

Training ensemble models...
Training Random Forest (binary)...
Training Random Forest (count)...
Training XGBoost (binary)...
Training XGBoost (count)...

Training R²: 0.746
Test R²: 0.453

Predicted CE50: 12.34 eV
Confidence: high
```

---

## Training Your Own Model

### Using Provided Training Data

```python
from ce50_predictor import CE50Predictor
import pandas as pd
import numpy as np

# Load the provided dataset (298 compounds)
df = pd.read_csv('ce50_300compounds_training.csv')

print(f"Dataset: {len(df)} compounds")
print(f"CE50 range: {df['ce50'].min():.1f} - {df['ce50'].max():.1f} eV")

# Initialize with optimized features (top 10)
predictor = CE50Predictor(use_optimized_features=True)

# Train with custom test split
results = predictor.train(
    smiles_list=df['smiles'].values,
    ce50_values=df['ce50'].values,
    test_size=0.2,
    random_state=42
)

# View detailed results
print("\n" + "="*50)
print("TRAINING RESULTS")
print("="*50)
print(f"Training set: {results['train_size']} compounds")
print(f"Test set:     {results['test_size']} compounds")
print(f"Training R²:  {results['train_r2']:.3f}")
print(f"Test R²:      {results['test_r2']:.3f}")

if 'cv_scores' in results and results['cv_scores']:
    cv_mean = np.mean(results['cv_scores'])
    cv_std = np.std(results['cv_scores'])
    print(f"CV R² (5-fold): {cv_mean:.3f} ± {cv_std:.3f}")

# Save trained models
predictor.save_models('my_trained_models')
```

### Using Your Own Data

Your CSV file must have these columns:
- `smiles`: SMILES strings
- `ce50`: CE50 values (in eV)

```python
from ce50_predictor import CE50Predictor
import pandas as pd

# Load your custom dataset
df = pd.read_csv('my_ce50_data.csv')

# Basic data validation
print(f"Loaded {len(df)} compounds")
print(f"Missing values: {df.isnull().sum().sum()}")

# Remove any invalid entries
df = df.dropna(subset=['smiles', 'ce50'])

# Check SMILES validity
from rdkit import Chem
valid_smiles = []
valid_ce50 = []

for smiles, ce50 in zip(df['smiles'], df['ce50']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        valid_smiles.append(smiles)
        valid_ce50.append(ce50)
    else:
        print(f"Invalid SMILES: {smiles}")

print(f"Valid compounds: {len(valid_smiles)}/{len(df)}")

# Train predictor
predictor = CE50Predictor(use_optimized_features=True)
results = predictor.train(valid_smiles, valid_ce50)

# Save for later use
predictor.save_models('custom_models')
```

---

## Making Predictions

### Single Molecule Prediction

```python
from ce50_predictor import CE50Predictor

# Load trained model
predictor = CE50Predictor(use_optimized_features=True)
predictor.load_models('my_trained_models')

# Predict for a molecule
smiles = "c1ccccc1"  # Benzene
result = predictor.predict(smiles)

print(f"SMILES: {smiles}")
print(f"Predicted CE50: {result['ce50']:.2f} eV")
print(f"Predicted pCE50: {result['pce50']:.4f}")
print(f"Confidence: {result['confidence']}")
print(f"Uncertainty: {result['uncertainty']:.4f}")
```

### Understanding Results

```python
result = {
    'ce50': 15.23,        # Predicted CE50 in eV
    'pce50': -1.1827,     # pCE50 = -log10(CE50)
    'confidence': 'high', # 'high', 'medium', or 'low'
    'uncertainty': 0.032  # Std dev across ensemble models
}

# Confidence levels:
# - high:   uncertainty < 0.05 (ensemble models agree)
# - medium: uncertainty 0.05-0.10 (moderate disagreement)
# - low:    uncertainty > 0.10 (high disagreement - be cautious)
```

---

## Batch Processing

### Predict for Multiple Molecules

```python
from ce50_predictor import CE50Predictor
import pandas as pd

# Load model
predictor = CE50Predictor(use_optimized_features=True)
predictor.load_models('my_trained_models')

# Prepare SMILES list
smiles_list = [
    'CCO',           # Ethanol
    'c1ccccc1',      # Benzene
    'CC(=O)O',       # Acetic acid
    'CCN',           # Ethylamine
    'c1ccc(cc1)O',   # Phenol
]

# Batch prediction
results = predictor.predict_batch(smiles_list)

# Display results as table
data = []
for smiles, result in zip(smiles_list, results):
    data.append({
        'SMILES': smiles,
        'CE50 (eV)': f"{result['ce50']:.2f}",
        'pCE50': f"{result['pce50']:.4f}",
        'Confidence': result['confidence'],
        'Uncertainty': f"{result['uncertainty']:.4f}"
    })

df_results = pd.DataFrame(data)
print(df_results.to_string(index=False))
```

**Output:**
```
       SMILES  CE50 (eV)    pCE50 Confidence  Uncertainty
          CCO      12.34  -1.0913       high       0.0245
    c1ccccc1      15.67  -1.1952       high       0.0312
     CC(=O)O      11.23  -1.0503     medium       0.0678
          CCN      13.45  -1.1287       high       0.0289
  c1ccc(cc1)O      14.89  -1.1728       high       0.0334
```

### Processing Large Datasets

```python
from ce50_predictor import CE50Predictor
import pandas as pd
from tqdm import tqdm

# Load model
predictor = CE50Predictor(use_optimized_features=True)
predictor.load_models('my_trained_models')

# Load large dataset
df = pd.read_csv('large_compound_library.csv')
print(f"Processing {len(df)} compounds...")

# Process in chunks to manage memory
chunk_size = 100
results_list = []

for i in tqdm(range(0, len(df), chunk_size)):
    chunk_smiles = df['smiles'].iloc[i:i+chunk_size].tolist()
    chunk_results = predictor.predict_batch(chunk_smiles)
    results_list.extend(chunk_results)

# Add predictions to dataframe
df['predicted_ce50'] = [r['ce50'] for r in results_list]
df['confidence'] = [r['confidence'] for r in results_list]
df['uncertainty'] = [r['uncertainty'] for r in results_list]

# Filter high-confidence predictions
high_conf = df[df['confidence'] == 'high']
print(f"\nHigh confidence predictions: {len(high_conf)}/{len(df)}")

# Save results
df.to_csv('predictions_output.csv', index=False)
```

---

## Feature Extraction

### Extract Structural Features Only

```python
from structural_features import StructuralFeatureExtractor

# Initialize extractor
extractor = StructuralFeatureExtractor()

# Extract all 40 features
smiles = "CCO"
features = extractor.extract_features(smiles)

# Display feature names and values
feature_dict = dict(zip(extractor.feature_names, features))

print("Bond Features:")
print(f"  C-N bonds: {feature_dict['c_n_bonds']:.0f}")
print(f"  C-O bonds: {feature_dict['c_o_bonds']:.0f}")
print(f"  Max bond length: {feature_dict['max_bond_length']:.3f} Å")
print(f"  Avg bond length: {feature_dict['avg_bond_length']:.3f} Å")

print("\nTopology Features:")
print(f"  Ring count: {feature_dict['ring_count']:.0f}")
print(f"  Rotatable bonds: {feature_dict['rotatable_bonds']:.0f}")
print(f"  Topology complexity: {feature_dict['topology_complexity']:.3f}")

print("\nUFF Features:")
print(f"  Avg UFF energy: {feature_dict['avg_uff_energy']:.3f}")
print(f"  Avg angle strain: {feature_dict['avg_angle_strain']:.3f}")
```

### Export Features for Analysis

```python
from structural_features import StructuralFeatureExtractor
import pandas as pd

extractor = StructuralFeatureExtractor()

# Load compound list
df = pd.read_csv('ce50_300compounds_training.csv')

# Extract features for all compounds
features_list = []
for smiles in df['smiles']:
    features = extractor.extract_features(smiles)
    if features is not None:
        features_list.append(features)
    else:
        features_list.append([None] * extractor.n_features)

# Create dataframe
df_features = pd.DataFrame(features_list, columns=extractor.feature_names)
df_features['smiles'] = df['smiles'].values
df_features['ce50'] = df['ce50'].values

# Save for external analysis
df_features.to_csv('extracted_features.csv', index=False)

# Correlation analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation with CE50
correlations = df_features.corr()['ce50'].sort_values(ascending=False)
print("Top 10 features correlated with CE50:")
print(correlations[1:11])  # Skip ce50 itself

# Visualize
plt.figure(figsize=(10, 6))
correlations[1:21].plot(kind='barh')
plt.xlabel('Correlation with CE50')
plt.title('Feature Correlation Analysis')
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300)
```

---

## Model Evaluation

### Cross-Validation

```python
from ce50_ensemble_predictor import EnsembleModel, DualFingerprintGenerator
from structural_features import StructuralFeatureExtractor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('ce50_300compounds_training.csv')

# Generate features
fp_gen = DualFingerprintGenerator()
extractor = StructuralFeatureExtractor()

fps_binary = fp_gen.generate(df['smiles'].values, use_counts=False)

# Extract structural features
structural_features = []
for smiles in df['smiles']:
    features = extractor.extract_features(smiles)
    if features is None:
        features = np.zeros(extractor.n_features)
    structural_features.append(features)
X_structural = np.array(structural_features)

# Select top 10 features
TOP_FEATURES = [
    'c_n_bonds', 'max_bond_length', 'topology_complexity',
    'bond_density', 'rotatable_bonds', 'ring_count',
    'avg_bond_length', 'num_nitrogen', 'hba_count', 'std_bond_length'
]
top_indices = [extractor.feature_names.index(f) for f in TOP_FEATURES]
X_structural_optimized = X_structural[:, top_indices]

# Convert to pCE50
y = -np.log10(df['ce50'].values)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(fps_binary), 1):
    print(f"\nFold {fold}/5...")

    # Split data
    X_train = fps_binary[train_idx]
    X_test = fps_binary[test_idx]
    X_struct_train = X_structural_optimized[train_idx]
    X_struct_test = X_structural_optimized[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Train ensemble
    ensemble = EnsembleModel(use_structural=True)
    models, scores = ensemble.train_ensemble(
        X_train, X_train, y_train,
        X_structural=X_struct_train
    )

    # Evaluate on test fold
    X_combined_test = np.hstack([X_test, X_struct_test])
    predictions = []
    for model_name, model_info in ensemble.models.items():
        y_pred = model_info['model'].predict(X_combined_test)
        predictions.append(y_pred)

    y_pred_mean = np.mean(predictions, axis=0)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred_mean)
    cv_scores.append(r2)
    print(f"  Fold {fold} R²: {r2:.3f}")

# Final statistics
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
print(f"Mean R²: {np.mean(cv_scores):.3f}")
print(f"Std Dev: {np.std(cv_scores):.3f}")
print(f"Min R²:  {np.min(cv_scores):.3f}")
print(f"Max R²:  {np.max(cv_scores):.3f}")
```

### Performance Comparison

```python
from ce50_predictor import CE50Predictor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('ce50_300compounds_training.csv')

# Compare different configurations
configs = [
    {'name': 'Baseline (FP only)', 'use_structural': False, 'use_optimized': False},
    {'name': 'Full (FP + 40)', 'use_structural': True, 'use_optimized': False},
    {'name': 'Optimized (FP + 10)', 'use_structural': True, 'use_optimized': True},
]

results = []

for config in configs:
    print(f"\nTraining: {config['name']}")
    predictor = CE50Predictor(
        use_optimized_features=config['use_optimized'],
        use_structural=config['use_structural']
    )

    train_results = predictor.train(
        df['smiles'].values,
        df['ce50'].values,
        test_size=0.2,
        random_state=42
    )

    results.append({
        'Model': config['name'],
        'Test R²': train_results['test_r2'],
        'Train R²': train_results['train_r2'],
    })

# Display comparison
df_comparison = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(df_comparison.to_string(index=False))
```

---

## Advanced Usage

### Custom Feature Selection

```python
from ce50_predictor import CE50Predictor
from structural_features import StructuralFeatureExtractor
import pandas as pd
import numpy as np

# See feature_importance_ranking.csv for all features ranked by importance

# Define your custom feature set
CUSTOM_FEATURES = [
    'c_n_bonds',         # Top importance
    'max_bond_length',
    'bond_density',
    # Add more features as needed
]

# Modify predictor to use custom features
predictor = CE50Predictor(use_optimized_features=False)

# Manually select features
extractor = StructuralFeatureExtractor()
custom_indices = [extractor.feature_names.index(f) for f in CUSTOM_FEATURES]

# You would need to modify the training pipeline to use custom_indices
# This is an advanced use case - see ce50_ensemble_predictor.py for details
```

### Ensemble Model Inspection

```python
from ce50_predictor import CE50Predictor

predictor = CE50Predictor(use_optimized_features=True)
# ... train or load models ...

# Inspect individual models
for model_name, model_info in predictor.ensemble.models.items():
    print(f"\n{model_name}:")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Fingerprint: {model_info['fingerprint_type']}")

    # For Random Forest, inspect feature importances
    if model_info['model_type'] == 'RandomForest':
        importances = model_info['model'].feature_importances_
        print(f"  Top 5 feature importances: {importances[:5]}")
```

---

## Troubleshooting

### Issue: Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'rdkit'

# Solution: Install via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit
```

### Issue: 3D Generation Failures

```python
# Error: 3D generation failed for SMILES

from rdkit import Chem

smiles = "your_smiles_here"
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES - check syntax")
else:
    print("Valid SMILES")
    # Check if molecule is too large or unusual
    print(f"Atoms: {mol.GetNumAtoms()}")
    print(f"Bonds: {mol.GetNumBonds()}")
```

### Issue: Low Confidence Predictions

```python
# Low confidence indicates the molecule may be outside the training domain

from ce50_predictor import CE50Predictor

predictor = CE50Predictor(use_optimized_features=True)
# ... load model ...

result = predictor.predict("unusual_molecule_smiles")

if result['confidence'] == 'low':
    print(f"Warning: Low confidence (uncertainty = {result['uncertainty']:.4f})")
    print("Prediction may be unreliable.")
    print("Consider:")
    print("  1. Adding similar molecules to training set")
    print("  2. Using experimental validation")
    print("  3. Interpreting prediction with caution")
```

### Issue: Memory Errors with Large Datasets

```python
# Process in smaller chunks

def predict_large_dataset(predictor, smiles_list, chunk_size=50):
    all_results = []

    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i+chunk_size]
        results = predictor.predict_batch(chunk)
        all_results.extend(results)

        # Optional: Clear memory
        if i % 500 == 0:
            import gc
            gc.collect()

    return all_results
```

### Issue: UFF Lookup Warnings

```
# Warning: UFF lookup failed for element X

# This is normal - not all atom types are in the UFF table
# The model gracefully handles this and continues
# No action needed unless you see many failures (>10%)
```

---

## Getting Help

If you encounter issues:

1. Check this guide and README.md
2. Review MODEL_PERFORMANCE.md for expected behavior
3. Examine feature_importance_ranking.csv for feature details
4. Check the GitHub repository for updates
5. Verify your RDKit installation is working: `python -c "from rdkit import Chem"`

---

## Citation

If you use this model in your research, please cite:

```
Enhanced CE50 Prediction using Morgan Fingerprints and LAMMPS Structural Features
GitHub: https://github.com/Maxwell1111/lammps-data-file/tree/master/Finger_print_lammps_data
Year: 2026
```

---

**Ready to start? Try the Quick Start example above!**
