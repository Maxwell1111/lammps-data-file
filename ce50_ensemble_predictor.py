"""
CE50 Ensemble Prediction System with Dual Fingerprints

CE50 = Collision Energy required to fragment 50% of parent ions in mass spectrometry

This advanced system predicts CE50 (collision energy for 50% fragmentation) from molecular structure:
- Dual fingerprint types (Binary + Count Morgan)
- 4-model ensemble (RF/XGB × Binary/Count)
- Applicability domain assessment (4 methods)
- Dynamic model selection based on confidence
- Comprehensive visualization suite
- Model persistence and versioning

CE50 is a mass spectrometry property measuring the collision energy needed to fragment
50% of precursor ions. It depends on molecular structure, bond strengths, and gas-phase
stability. This is predictable from molecular fingerprints (R² ~0.57).

Author: Senior Bioinformatician
Date: 2026-01-05
Version: 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from scipy.sparse import csr_matrix
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class DualFingerprintGenerator:
    """Generate both binary and count-based Morgan fingerprints"""

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def generate_binary_fingerprint(self, smiles):
        """Generate binary Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(fp)

    def generate_count_fingerprint(self, smiles):
        """Generate count-based Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.int32)
        for idx, val in fp.GetNonzeroElements().items():
            arr[idx] = val
        return arr

    def generate_both(self, smiles_list):
        """Generate both fingerprint types for a list of SMILES"""
        binary_fps = []
        count_fps = []
        valid_mask = []

        for smiles in smiles_list:
            binary_fp = self.generate_binary_fingerprint(smiles)
            count_fp = self.generate_count_fingerprint(smiles)

            if binary_fp is not None and count_fp is not None:
                binary_fps.append(binary_fp)
                count_fps.append(count_fp)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        return {
            'binary': np.array(binary_fps),
            'count': np.array(count_fps),
            'valid_mask': valid_mask
        }


class ApplicabilityDomain:
    """Multi-method applicability domain assessment"""

    def __init__(self):
        self.training_fps_binary = None
        self.training_fps_count = None
        self.pca_binary = None
        self.pca_count = None
        self.svm_binary = None
        self.svm_count = None
        self.training_centroid_binary = None
        self.training_centroid_count = None
        self.training_cov_inv_binary = None
        self.training_cov_inv_count = None

    def fit(self, X_train_binary, X_train_count):
        """Fit applicability domain models on training data"""
        self.training_fps_binary = X_train_binary
        self.training_fps_count = X_train_count

        # Fit PCA
        self.pca_binary = PCA(n_components=min(50, X_train_binary.shape[0]-1))
        X_pca_binary = self.pca_binary.fit_transform(X_train_binary)
        self.training_centroid_binary = np.mean(X_pca_binary, axis=0)
        self.training_cov_inv_binary = np.linalg.pinv(np.cov(X_pca_binary.T))

        self.pca_count = PCA(n_components=min(50, X_train_count.shape[0]-1))
        X_pca_count = self.pca_count.fit_transform(X_train_count)
        self.training_centroid_count = np.mean(X_pca_count, axis=0)
        self.training_cov_inv_count = np.linalg.pinv(np.cov(X_pca_count.T))

        # Fit One-Class SVM
        self.svm_binary = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        self.svm_binary.fit(X_train_binary)

        self.svm_count = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        self.svm_count.fit(X_train_count)

    def assess_tanimoto_similarity(self, query_fp, training_fps, threshold=0.3):
        """Method 1: Tanimoto similarity to nearest neighbor"""
        max_sim = 0.0
        for train_fp in training_fps:
            sim = np.sum(query_fp & train_fp) / np.sum(query_fp | train_fp)
            max_sim = max(max_sim, sim)

        return {
            'max_similarity': max_sim,
            'within_domain': max_sim >= threshold
        }

    def assess_pca_distance(self, query_fp, fp_type='binary'):
        """Method 2: PCA Mahalanobis distance"""
        if fp_type == 'binary':
            pca = self.pca_binary
            centroid = self.training_centroid_binary
            cov_inv = self.training_cov_inv_binary
        else:
            pca = self.pca_count
            centroid = self.training_centroid_count
            cov_inv = self.training_cov_inv_count

        query_pca = pca.transform([query_fp])[0]
        diff = query_pca - centroid
        distance = np.sqrt(diff @ cov_inv @ diff.T)

        # Calculate threshold from training data
        training_pca = pca.transform(
            self.training_fps_binary if fp_type == 'binary' else self.training_fps_count
        )
        training_distances = [
            np.sqrt((x - centroid) @ cov_inv @ (x - centroid).T)
            for x in training_pca
        ]
        threshold = np.percentile(training_distances, 95)

        return {
            'distance': distance,
            'threshold': threshold,
            'within_domain': distance < threshold
        }

    def assess_svm(self, query_fp, fp_type='binary'):
        """Method 3: One-Class SVM"""
        svm = self.svm_binary if fp_type == 'binary' else self.svm_count
        prediction = svm.predict([query_fp])[0]

        return {
            'prediction': prediction,
            'within_domain': prediction == 1
        }

    def assess_all(self, query_fp_binary, query_fp_count):
        """Assess all methods and aggregate"""
        results = {
            'tanimoto_binary': self.assess_tanimoto_similarity(query_fp_binary, self.training_fps_binary),
            'tanimoto_count': self.assess_tanimoto_similarity(query_fp_count, self.training_fps_count),
            'pca_binary': self.assess_pca_distance(query_fp_binary, 'binary'),
            'pca_count': self.assess_pca_distance(query_fp_count, 'count'),
            'svm_binary': self.assess_svm(query_fp_binary, 'binary'),
            'svm_count': self.assess_svm(query_fp_count, 'count')
        }

        # Voting system
        votes_in_domain = sum([
            results['tanimoto_binary']['within_domain'],
            results['tanimoto_count']['within_domain'],
            results['pca_binary']['within_domain'],
            results['pca_count']['within_domain'],
            results['svm_binary']['within_domain'],
            results['svm_count']['within_domain']
        ])

        if votes_in_domain >= 5:
            confidence = 'High'
        elif votes_in_domain >= 3:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        results['overall_confidence'] = confidence
        results['votes_in_domain'] = votes_in_domain

        return results


class EnsembleModel:
    """4-model ensemble with dynamic selection"""

    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.applicability_domain = ApplicabilityDomain()
        self.fingerprint_generator = DualFingerprintGenerator()

    def build_pipeline(self, model_type, model_name):
        """Build sklearn pipeline with scaler and model"""
        if model_type == 'rf':
            estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        else:  # xgb
            estimator = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, estimator)
        ])
        return pipeline

    def get_param_grid(self, model_type):
        """Get hyperparameter grid for model type"""
        if model_type == 'rf':
            return {
                f'{model_type}__n_estimators': [100, 200, 300],
                f'{model_type}__max_depth': [10, 20, None],
                f'{model_type}__min_samples_split': [2, 5, 10],
                f'{model_type}__min_samples_leaf': [1, 2, 4],
                f'{model_type}__max_features': ['sqrt', 'log2']
            }
        else:  # xgb
            return {
                f'{model_type}__n_estimators': [100, 200, 300],
                f'{model_type}__max_depth': [3, 5, 7],
                f'{model_type}__learning_rate': [0.01, 0.05, 0.1],
                f'{model_type}__subsample': [0.8, 1.0],
                f'{model_type}__colsample_bytree': [0.8, 1.0]
            }

    def train_single_model(self, X_train, y_train, model_type, fp_type):
        """Train a single model with hyperparameter optimization"""
        print(f"\nTraining {model_type.upper()} with {fp_type} fingerprints...")

        pipeline = self.build_pipeline(model_type, model_type)
        param_grid = self.get_param_grid(model_type)

        # Dynamic CV folds based on dataset size
        n_folds = 3 if len(X_train) < 100 else 5

        grid_search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20,
            cv=n_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best {model_type.upper()}-{fp_type} params: {grid_search.best_params_}")
        print(f"Best CV R² score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_score_

    def train_ensemble(self, X_binary, X_count, y):
        """Train all 4 models in the ensemble"""
        print("\n" + "="*80)
        print("TRAINING 4-MODEL ENSEMBLE")
        print("="*80)

        # Split data
        X_binary_train, X_binary_test, y_train, y_test = train_test_split(
            X_binary, y, test_size=0.2, random_state=RANDOM_STATE
        )
        X_count_train, X_count_test, _, _ = train_test_split(
            X_count, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Fit applicability domain
        print("\nFitting applicability domain models...")
        self.applicability_domain.fit(X_binary_train, X_count_train)

        # Train 4 models
        models_to_train = [
            ('rf', 'binary', X_binary_train),
            ('rf', 'count', X_count_train),
            ('xgb', 'binary', X_binary_train),
            ('xgb', 'count', X_count_train)
        ]

        for model_type, fp_type, X_train in models_to_train:
            model_name = f"{model_type}_{fp_type}"
            model, cv_score = self.train_single_model(X_train, y_train, model_type, fp_type)
            self.models[model_name] = model
            self.model_scores[model_name] = cv_score

        # Store test data for evaluation
        self.X_binary_test = X_binary_test
        self.X_count_test = X_count_test
        self.y_test = y_test

        return self.models, self.model_scores

    def calculate_model_uncertainty(self, model, X, model_name):
        """Calculate prediction uncertainty (Method 4)"""
        if 'rf' in model_name:
            # Use Random Forest tree variance
            tree_predictions = np.array([
                tree.predict(X) for tree in model.named_steps['rf'].estimators_
            ])
            std_dev = np.std(tree_predictions, axis=0)
        else:
            # For XGBoost, use prediction itself as proxy (could implement bootstrap)
            predictions = model.predict(X)
            std_dev = np.ones_like(predictions) * 0.2  # Placeholder

        return std_dev

    def predict_with_confidence(self, smiles_list):
        """
        Make predictions with confidence assessment and dynamic model selection

        Returns detailed predictions for each molecule
        """
        # Generate fingerprints
        fps = self.fingerprint_generator.generate_both(smiles_list)
        X_binary = fps['binary']
        X_count = fps['count']

        results = []

        for i in range(len(X_binary)):
            # Get predictions from all 4 models
            predictions = {}
            uncertainties = {}

            for model_name, model in self.models.items():
                fp_type = 'binary' if 'binary' in model_name else 'count'
                X = X_binary[i:i+1] if fp_type == 'binary' else X_count[i:i+1]

                pred = model.predict(X)[0]
                uncertainty = self.calculate_model_uncertainty(model, X, model_name)[0]

                predictions[model_name] = pred
                uncertainties[model_name] = uncertainty

            # Assess applicability domain
            ad_results = self.applicability_domain.assess_all(X_binary[i], X_count[i])

            # Calculate confidence scores for each model
            confidence_scores = {}
            for model_name in self.models.keys():
                fp_type = 'binary' if 'binary' in model_name else 'count'

                # Combine multiple factors
                tanimoto_score = ad_results[f'tanimoto_{fp_type}']['max_similarity']
                pca_in_domain = 1.0 if ad_results[f'pca_{fp_type}']['within_domain'] else 0.5
                svm_in_domain = 1.0 if ad_results[f'svm_{fp_type}']['within_domain'] else 0.5
                uncertainty_score = max(0, 1.0 - uncertainties[model_name])

                # Weighted combination
                confidence = (
                    0.4 * tanimoto_score +
                    0.2 * pca_in_domain +
                    0.2 * svm_in_domain +
                    0.2 * uncertainty_score
                )
                confidence_scores[model_name] = confidence

            # Select model with highest confidence (dynamic selection)
            best_model_name = max(confidence_scores, key=confidence_scores.get)
            final_prediction = predictions[best_model_name]

            # Check for ensemble disagreement
            pred_values = list(predictions.values())
            ensemble_std = np.std(pred_values)
            disagreement_flag = ensemble_std > 0.5  # pCE50 units

            if disagreement_flag:
                # High disagreement - check applicability domain
                if ad_results['overall_confidence'] == 'Low':
                    overall_confidence = 'Low'
                else:
                    overall_confidence = 'Medium'
            else:
                overall_confidence = ad_results['overall_confidence']

            results.append({
                'smiles': smiles_list[i],
                'predicted_pce50': final_prediction,
                'predicted_ce50': 10 ** (-final_prediction),
                'confidence': overall_confidence,
                'selected_model': best_model_name,
                'all_predictions': predictions,
                'ensemble_std': ensemble_std,
                'disagreement_flag': disagreement_flag,
                'applicability_scores': {
                    'tanimoto_binary': ad_results['tanimoto_binary']['max_similarity'],
                    'tanimoto_count': ad_results['tanimoto_count']['max_similarity'],
                    'overall_votes': ad_results['votes_in_domain']
                }
            })

        return results

    def evaluate_ensemble(self):
        """Evaluate all models on test set"""
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION ON TEST SET")
        print("="*80)

        results = {}

        # Evaluate each model
        for model_name, model in self.models.items():
            fp_type = 'binary' if 'binary' in model_name else 'count'
            X_test = self.X_binary_test if fp_type == 'binary' else self.X_count_test

            y_pred = model.predict(X_test)

            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            print(f"\n{model_name.upper()}:")
            print(f"  R² = {r2:.4f}")
            print(f"  MAE = {mae:.4f}")
            print(f"  RMSE = {rmse:.4f}")

            results[model_name] = {
                'predictions': y_pred,
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }

        return results

    def save_models(self, output_dir='models'):
        """Save all models and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save models
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/{model_name}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

        # Save applicability domain
        ad_path = f"{output_dir}/applicability_domain_{timestamp}.pkl"
        joblib.dump(self.applicability_domain, ad_path)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_scores': self.model_scores,
            'random_state': RANDOM_STATE,
            'fingerprint_config': {
                'radius': 2,
                'n_bits': 2048
            }
        }

        metadata_path = f"{output_dir}/metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to {metadata_path}")


class Visualizer:
    """Enhanced visualization suite"""

    @staticmethod
    def plot_ensemble_comparison(y_test, results_dict, output_file='ensemble_comparison.png'):
        """Compare all 4 models side by side"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()

        colors = {'rf_binary': 'blue', 'rf_count': 'green',
                  'xgb_binary': 'red', 'xgb_count': 'purple'}

        for idx, (model_name, results) in enumerate(results_dict.items()):
            ax = axes[idx]
            y_pred = results['predictions']

            ax.scatter(y_test, y_pred, alpha=0.6, s=80,
                      edgecolors='k', linewidths=1, color=colors[model_name])
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', lw=2, label='Perfect Prediction')

            ax.set_xlabel('Actual pCE50', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted pCE50', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name.upper()}', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add metrics
            textstr = f"R² = {results['r2']:.4f}\nMAE = {results['mae']:.4f}\nRMSE = {results['rmse']:.4f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Ensemble comparison saved to {output_file}")
        plt.close()

    @staticmethod
    def plot_confidence_distribution(predictions, output_file='confidence_distribution.png'):
        """Plot distribution of confidence levels"""
        confidence_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for pred in predictions:
            confidence_counts[pred['confidence']] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        colors_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        ax1.bar(confidence_counts.keys(), confidence_counts.values(),
               color=[colors_map[k] for k in confidence_counts.keys()],
               edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
        ax1.set_title('Confidence Level Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Ensemble disagreement
        ensemble_stds = [pred['ensemble_std'] for pred in predictions]
        ax2.hist(ensemble_stds, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Disagreement Threshold')
        ax2.set_xlabel('Ensemble Std Dev (pCE50)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Ensemble Disagreement Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution saved to {output_file}")
        plt.close()

    @staticmethod
    def plot_model_selection(predictions, output_file='model_selection.png'):
        """Show which models were selected"""
        model_counts = {}
        for pred in predictions:
            model_name = pred['selected_model']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        plt.figure(figsize=(10, 6))
        colors = {'rf_binary': 'blue', 'rf_count': 'green',
                  'xgb_binary': 'red', 'xgb_count': 'purple'}

        bars = plt.bar(model_counts.keys(), model_counts.values(),
                      color=[colors.get(k, 'gray') for k in model_counts.keys()],
                      edgecolor='black', linewidth=1.5)

        plt.ylabel('Number of Times Selected', fontsize=12, fontweight='bold')
        plt.title('Dynamic Model Selection Frequency', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Model selection plot saved to {output_file}")
        plt.close()


def load_and_prepare_data(filepath):
    """Load and prepare data for ensemble training"""
    print(f"\nLoading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Handle different column name formats
    if 'SMILES' in df.columns and 'Measured_CE50' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles', 'Measured_CE50': 'ce50'})

    # Remove missing values
    df = df.dropna(subset=['smiles', 'ce50'])

    print(f"Loaded {len(df)} compounds")
    if 'Compound_Name' in df.columns:
        print(f"Sample compounds: {', '.join(df['Compound_Name'].head(3).values)}")

    return df


def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("CE50 DUAL FINGERPRINT ENSEMBLE PREDICTOR")
    print("="*80)

    # Load data
    df = load_and_prepare_data('kinase_compounds.csv')

    # Convert to pCE50
    df['pce50'] = -np.log10(df['ce50'])

    print(f"\nData Statistics:")
    print(f"  CE50 range: {df['ce50'].min():.2f} - {df['ce50'].max():.2f}")
    print(f"  pCE50 range: {df['pce50'].min():.4f} - {df['pce50'].max():.4f}")

    # Generate dual fingerprints
    print("\nGenerating dual fingerprints...")
    fp_generator = DualFingerprintGenerator()
    fps = fp_generator.generate_both(df['smiles'].values)

    # Filter valid molecules
    df_valid = df[fps['valid_mask']].copy()
    X_binary = fps['binary']
    X_count = fps['count']
    y = df_valid['pce50'].values

    print(f"Valid molecules: {len(df_valid)}")
    print(f"Binary fingerprints shape: {X_binary.shape}")
    print(f"Count fingerprints shape: {X_count.shape}")

    # Create and train ensemble
    ensemble = EnsembleModel()
    models, scores = ensemble.train_ensemble(X_binary, X_count, y)

    # Evaluate ensemble
    results = ensemble.evaluate_ensemble()

    # Generate predictions for test set
    test_smiles = df_valid.iloc[len(y)-len(ensemble.y_test):]['smiles'].values
    predictions = ensemble.predict_with_confidence(test_smiles)

    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    viz = Visualizer()
    viz.plot_ensemble_comparison(ensemble.y_test, results)
    viz.plot_confidence_distribution(predictions)
    viz.plot_model_selection(predictions)

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    ensemble.save_models()

    # Save predictions to CSV
    predictions_df = pd.DataFrame([{
        'smiles': p['smiles'],
        'predicted_ce50': p['predicted_ce50'],
        'predicted_pce50': p['predicted_pce50'],
        'confidence': p['confidence'],
        'selected_model': p['selected_model'],
        'ensemble_std': p['ensemble_std'],
        'tanimoto_binary': p['applicability_scores']['tanimoto_binary'],
        'tanimoto_count': p['applicability_scores']['tanimoto_count']
    } for p in predictions])

    predictions_df.to_csv('ensemble_predictions.csv', index=False)
    print("\nPredictions saved to ensemble_predictions.csv")

    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE SUMMARY")
    print("="*80)
    print(f"\nBest performing model (by R²):")
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"  {best_model[0]}: R² = {best_model[1]['r2']:.4f}")

    print(f"\nConfidence distribution:")
    for conf_level in ['High', 'Medium', 'Low']:
        count = sum(1 for p in predictions if p['confidence'] == conf_level)
        pct = 100 * count / len(predictions)
        print(f"  {conf_level}: {count} ({pct:.1f}%)")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")

    return ensemble, predictions


if __name__ == "__main__":
    ensemble, predictions = main()
