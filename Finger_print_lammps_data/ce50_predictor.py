"""
CE50 Predictor - Simple Interface for CE50 Predictions

This module provides an easy-to-use wrapper around the ensemble predictor
for making CE50 predictions from SMILES strings.

Usage:
    from ce50_predictor import CE50Predictor

    predictor = CE50Predictor(use_optimized_features=True)
    result = predictor.predict("CCO")
    print(f"Predicted CE50: {result['ce50']:.2f} eV")
"""

import numpy as np
import warnings
from ce50_ensemble_predictor import EnsembleModel, DualFingerprintGenerator

# Top 10 optimized features based on importance analysis
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


class CE50Predictor:
    """
    Simple interface for CE50 predictions.

    This class provides a user-friendly wrapper around the ensemble model
    with support for optimized feature selection.

    Parameters
    ----------
    use_optimized_features : bool, default=True
        If True, use only the top 10 most important structural features.
        If False, use all 40 structural features.
    use_structural : bool, default=True
        If True, include structural features from LAMMPS analysis.
        If False, use only Morgan fingerprints (baseline model).

    Attributes
    ----------
    ensemble : EnsembleModel
        The underlying ensemble model
    fp_generator : DualFingerprintGenerator
        Fingerprint generator with structural feature support
    """

    def __init__(self, use_optimized_features=True, use_structural=True):
        self.use_optimized_features = use_optimized_features
        self.use_structural = use_structural

        # Initialize ensemble model
        self.ensemble = EnsembleModel(use_structural=use_structural)

        # Initialize fingerprint generator
        self.fp_generator = DualFingerprintGenerator()

        # Load structural extractor if needed
        if use_structural:
            try:
                from structural_features import StructuralFeatureExtractor
                self.structural_extractor = StructuralFeatureExtractor()

                # Get indices of top features if using optimized mode
                if use_optimized_features:
                    feature_names = self.structural_extractor.feature_names
                    self.top_feature_indices = [
                        feature_names.index(f) for f in TOP_FEATURES
                    ]
                else:
                    self.top_feature_indices = None
            except ImportError:
                warnings.warn(
                    "Structural features module not available. "
                    "Falling back to fingerprints only."
                )
                self.use_structural = False
                self.structural_extractor = None
        else:
            self.structural_extractor = None

        self.is_trained = False

    def train(self, smiles_list, ce50_values, test_size=0.2, random_state=42):
        """
        Train the ensemble model on your data.

        Parameters
        ----------
        smiles_list : array-like of str
            List of SMILES strings
        ce50_values : array-like of float
            Corresponding CE50 values (in eV)
        test_size : float, default=0.2
            Fraction of data to use for testing
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        results : dict
            Dictionary containing training results with keys:
            - 'models': List of trained models
            - 'test_r2': Test set R²
            - 'train_r2': Training set R²
            - 'cv_scores': Cross-validation scores
        """
        from sklearn.model_selection import train_test_split

        print("Generating features...")

        # Generate fingerprints
        fps_binary = self.fp_generator.generate(smiles_list, use_counts=False)
        fps_count = self.fp_generator.generate(smiles_list, use_counts=True)

        # Generate structural features if needed
        X_structural = None
        if self.use_structural and self.structural_extractor is not None:
            structural_features = []
            print(f"Extracting structural features for {len(smiles_list)} molecules...")

            for i, smiles in enumerate(smiles_list):
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(smiles_list)}")

                features = self.structural_extractor.extract_features(smiles)
                if features is None:
                    features = np.zeros(self.structural_extractor.n_features)
                structural_features.append(features)

            X_structural = np.array(structural_features)

            # Select top features if in optimized mode
            if self.use_optimized_features and self.top_feature_indices is not None:
                X_structural = X_structural[:, self.top_feature_indices]
                print(f"Using {len(self.top_feature_indices)} optimized features")
            else:
                print(f"Using all {X_structural.shape[1]} structural features")

        # Convert CE50 to pCE50 (negative log scale)
        y = -np.log10(np.array(ce50_values))

        # Train/test split
        indices = np.arange(len(smiles_list))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        X_binary_train = fps_binary[train_idx]
        X_binary_test = fps_binary[test_idx]
        X_count_train = fps_count[train_idx]
        X_count_test = fps_count[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if X_structural is not None:
            X_structural_train = X_structural[train_idx]
            X_structural_test = X_structural[test_idx]
        else:
            X_structural_train = None
            X_structural_test = None

        print("\nTraining ensemble models...")
        models, scores = self.ensemble.train_ensemble(
            X_binary_train, X_binary_train,  # Use binary for combined
            y_train,
            X_structural=X_structural_train
        )

        # Store test data for predictions
        self.ensemble.X_binary_test = X_binary_test
        self.ensemble.X_count_test = X_count_test
        self.ensemble.X_structural_test = X_structural_test
        self.ensemble.y_test = y_test

        self.is_trained = True

        return {
            'models': models,
            'test_r2': scores.get('test_r2'),
            'train_r2': scores.get('train_r2'),
            'cv_scores': scores.get('cv_scores'),
            'test_size': len(test_idx),
            'train_size': len(train_idx)
        }

    def predict(self, smiles):
        """
        Predict CE50 for a single molecule.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'ce50': Predicted CE50 value (in eV)
            - 'pce50': Predicted pCE50 value (-log10(CE50))
            - 'confidence': Confidence level ('high', 'medium', 'low')
            - 'uncertainty': Standard deviation across ensemble models
        """
        if not self.is_trained:
            raise ValueError(
                "Model not trained. Call train() first or load a trained model."
            )

        results = self.predict_batch([smiles])
        return results[0]

    def predict_batch(self, smiles_list):
        """
        Predict CE50 for multiple molecules.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings

        Returns
        -------
        results : list of dict
            List of prediction dictionaries (see predict() for format)
        """
        if not self.is_trained:
            raise ValueError(
                "Model not trained. Call train() first or load a trained model."
            )

        # Generate fingerprints
        fps_binary = self.fp_generator.generate(smiles_list, use_counts=False)
        fps_count = self.fp_generator.generate(smiles_list, use_counts=True)

        # Generate structural features if needed
        X_structural = None
        if self.use_structural and self.structural_extractor is not None:
            structural_features = []

            for smiles in smiles_list:
                features = self.structural_extractor.extract_features(smiles)
                if features is None:
                    features = np.zeros(self.structural_extractor.n_features)
                structural_features.append(features)

            X_structural = np.array(structural_features)

            # Select top features if in optimized mode
            if self.use_optimized_features and self.top_feature_indices is not None:
                X_structural = X_structural[:, self.top_feature_indices]

        # Combine features based on model configuration
        predictions = []

        for i in range(len(smiles_list)):
            # Get predictions from all ensemble models
            model_predictions = []

            for model_name, model_info in self.ensemble.models.items():
                model = model_info['model']
                fp_type = model_info['fingerprint_type']

                # Select appropriate fingerprint
                if fp_type == 'binary':
                    X_fp = fps_binary[i:i+1]
                else:  # count
                    X_fp = fps_count[i:i+1]

                # Combine with structural features if available
                if X_structural is not None and self.use_structural:
                    X_combined = np.hstack([X_fp, X_structural[i:i+1]])
                else:
                    X_combined = X_fp

                # Make prediction (in pCE50 scale)
                pce50_pred = model.predict(X_combined)[0]
                model_predictions.append(pce50_pred)

            # Ensemble average
            pce50_mean = np.mean(model_predictions)
            pce50_std = np.std(model_predictions)

            # Convert back to CE50
            ce50_pred = 10 ** (-pce50_mean)

            # Determine confidence based on ensemble disagreement
            if pce50_std < 0.05:
                confidence = 'high'
            elif pce50_std < 0.10:
                confidence = 'medium'
            else:
                confidence = 'low'

            predictions.append({
                'ce50': ce50_pred,
                'pce50': pce50_mean,
                'confidence': confidence,
                'uncertainty': pce50_std
            })

        return predictions

    def save_models(self, output_dir='trained_models'):
        """
        Save trained models to disk.

        Parameters
        ----------
        output_dir : str, default='trained_models'
            Directory to save models
        """
        if not self.is_trained:
            raise ValueError("No trained models to save.")

        self.ensemble.save_models(output_dir)
        print(f"Models saved to {output_dir}/")

    def load_models(self, model_dir='trained_models'):
        """
        Load trained models from disk.

        Parameters
        ----------
        model_dir : str, default='trained_models'
            Directory containing saved models
        """
        self.ensemble.load_models(model_dir)
        self.is_trained = True
        print(f"Models loaded from {model_dir}/")


def main():
    """Example usage of CE50Predictor"""

    # Example 1: Simple prediction (would need trained models)
    print("Example 1: Simple Prediction")
    print("-" * 50)

    predictor = CE50Predictor(use_optimized_features=True)

    # Note: This requires a trained model
    # predictor.load_models('trained_models')
    # result = predictor.predict("CCO")
    # print(f"SMILES: CCO (Ethanol)")
    # print(f"Predicted CE50: {result['ce50']:.2f} eV")
    # print(f"Confidence: {result['confidence']}")

    print("(Would need trained models loaded)")

    print("\n" + "=" * 50)
    print("Example 2: Training on Custom Data")
    print("-" * 50)

    # This example shows how to train
    print("See USAGE_GUIDE.md for complete training examples")


if __name__ == "__main__":
    main()
