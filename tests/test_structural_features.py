"""
Unit tests for structural_features module

Tests the 3D structure generation and feature extraction capabilities
for CE50 prediction enhancement.
"""

import pytest
import numpy as np
from structural_features import (
    smiles_to_3d_coords,
    StructuralFeatureExtractor,
    atomic_num_to_symbol
)


class TestAtomicNumToSymbol:
    """Test atomic number to symbol conversion"""

    def test_common_elements(self):
        assert atomic_num_to_symbol(1) == 'H'
        assert atomic_num_to_symbol(6) == 'C'
        assert atomic_num_to_symbol(7) == 'N'
        assert atomic_num_to_symbol(8) == 'O'

    def test_unknown_element(self):
        assert atomic_num_to_symbol(999) == 'X'


class TestSmilesTo 3D:
    """Test SMILES to 3D coordinate conversion"""

    def test_simple_molecules(self):
        """Test 3D generation for simple molecules"""
        # Methane
        result = smiles_to_3d_coords('C')
        assert result is not None
        atoms, mol = result
        assert len(atoms) == 5  # 1C + 4H

        # Ethanol
        result = smiles_to_3d_coords('CCO')
        assert result is not None
        atoms, mol = result
        assert len(atoms) == 9  # 2C + 1O + 6H

    def test_benzene(self):
        """Test aromatic structure"""
        result = smiles_to_3d_coords('c1ccccc1')
        assert result is not None
        atoms, mol = result
        assert len(atoms) == 12  # 6C + 6H

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        result = smiles_to_3d_coords('INVALID')
        assert result is None

    def test_atom_format(self):
        """Test that atoms are in correct format (x, y, z, atomic_num)"""
        result = smiles_to_3d_coords('C')
        assert result is not None
        atoms, mol = result

        for atom in atoms:
            assert len(atom) == 4
            assert isinstance(atom[0], float)  # x
            assert isinstance(atom[1], float)  # y
            assert isinstance(atom[2], float)  # z
            assert isinstance(atom[3], (int, np.integer))  # atomic_num


class TestStructuralFeatureExtractor:
    """Test the StructuralFeatureExtractor class"""

    @pytest.fixture
    def extractor(self):
        return StructuralFeatureExtractor()

    def test_initialization(self, extractor):
        """Test extractor initializes correctly"""
        assert extractor.uff_table is not None
        assert len(extractor.feature_names) > 0
        assert extractor.n_features == len(extractor.feature_names)

    def test_feature_extraction_methane(self, extractor):
        """Test feature extraction on methane"""
        features = extractor.extract_features('C')

        assert features is not None
        assert len(features) == extractor.n_features
        assert features.shape == (extractor.n_features,)

        # Check some expected values for methane
        feature_dict = dict(zip(extractor.feature_names, features))
        assert feature_dict['heavy_atom_count'] == 5  # 1C + 4H
        assert feature_dict['bond_count'] == 4  # 4 C-H bonds
        assert feature_dict['num_carbon'] == 1
        assert feature_dict['ring_count'] == 0

    def test_feature_extraction_ethanol(self, extractor):
        """Test feature extraction on ethanol (CCO)"""
        features = extractor.extract_features('CCO')

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Ethanol: 2C + 1O + 6H = 9 atoms, 8 bonds
        assert feature_dict['heavy_atom_count'] == 9
        assert feature_dict['bond_count'] == 8
        assert feature_dict['num_carbon'] == 2
        assert feature_dict['num_oxygen'] == 1
        assert feature_dict['c_c_bonds'] == 1
        assert feature_dict['c_o_bonds'] == 1
        assert feature_dict['ring_count'] == 0

    def test_feature_extraction_benzene(self, extractor):
        """Test feature extraction on benzene"""
        features = extractor.extract_features('c1ccccc1')

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Benzene should have aromatic ring
        assert feature_dict['ring_count'] >= 1
        assert feature_dict['aromatic_ring_count'] >= 1
        assert feature_dict['num_carbon'] == 6

    def test_bond_features_present(self, extractor):
        """Test that bond features are extracted"""
        features = extractor.extract_features('CC')

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Check bond features exist and are reasonable
        assert 'avg_bond_length' in feature_dict
        assert 'c_c_bonds' in feature_dict
        assert feature_dict['avg_bond_length'] > 0
        assert feature_dict['c_c_bonds'] == 1

    def test_uff_features_present(self, extractor):
        """Test that UFF features are extracted"""
        features = extractor.extract_features('C')

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Check UFF features exist
        assert 'avg_uff_energy' in feature_dict
        assert 'avg_vdw_distance' in feature_dict
        assert 'avg_coordination' in feature_dict

        # Values should be reasonable
        assert feature_dict['avg_uff_energy'] >= 0
        assert feature_dict['avg_vdw_distance'] > 0

    def test_topology_features_present(self, extractor):
        """Test that topology features are extracted"""
        features = extractor.extract_features('C(C)(C)C')  # isobutane - branched

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Check topology features
        assert 'angle_count' in feature_dict
        assert 'branching_index' in feature_dict

        # Branched molecule should have angles
        assert feature_dict['angle_count'] > 0

    def test_atom_features_present(self, extractor):
        """Test that atom features are extracted"""
        features = extractor.extract_features('CCO')

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Check atom features
        assert 'heteroatom_ratio' in feature_dict
        assert 'molecular_weight' in feature_dict
        assert 'sp3_ratio' in feature_dict

        # Ethanol has 1 heteroatom (O) out of 9 total
        assert 0 < feature_dict['heteroatom_ratio'] < 1
        assert feature_dict['molecular_weight'] > 0

    def test_invalid_smiles_returns_none(self, extractor):
        """Test that invalid SMILES returns None"""
        features = extractor.extract_features('INVALID')
        assert features is None

    def test_feature_consistency(self, extractor):
        """Test that same SMILES gives consistent features"""
        features1 = extractor.extract_features('CCO')
        features2 = extractor.extract_features('CCO')

        assert features1 is not None
        assert features2 is not None
        np.testing.assert_array_almost_equal(features1, features2, decimal=5)

    def test_all_features_are_numbers(self, extractor):
        """Test that all extracted features are numeric"""
        features = extractor.extract_features('CCO')

        assert features is not None
        for feat in features:
            assert isinstance(feat, (int, float, np.number))
            assert not np.isnan(feat)
            assert not np.isinf(feat)

    def test_complex_molecule(self, extractor):
        """Test feature extraction on a more complex molecule"""
        # Caffeine
        caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
        features = extractor.extract_features(caffeine_smiles)

        assert features is not None
        feature_dict = dict(zip(extractor.feature_names, features))

        # Caffeine has multiple rings
        assert feature_dict['ring_count'] >= 2
        assert feature_dict['aromatic_ring_count'] >= 1
        assert feature_dict['num_nitrogen'] > 0

    def test_heteroatom_ratios(self, extractor):
        """Test heteroatom ratio calculations"""
        # Pure hydrocarbon - no heteroatoms
        features_hc = extractor.extract_features('CCCC')
        feat_dict_hc = dict(zip(extractor.feature_names, features_hc))
        # Should have low heteroatom ratio (only counts heavy atoms, but still low)

        # Molecule with heteroatoms
        features_hetero = extractor.extract_features('CCON')
        feat_dict_hetero = dict(zip(extractor.feature_names, features_hetero))

        # Heteroatom molecule should have higher ratio
        assert feat_dict_hetero['heteroatom_ratio'] > feat_dict_hc['heteroatom_ratio']


class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_batch_processing(self):
        """Test processing multiple molecules"""
        extractor = StructuralFeatureExtractor()
        smiles_list = ['C', 'CC', 'CCO', 'c1ccccc1']

        features_list = [extractor.extract_features(s) for s in smiles_list]

        # All should succeed
        for features in features_list:
            assert features is not None

        # All should have same number of features
        n_features = len(features_list[0])
        for features in features_list:
            assert len(features) == n_features

    def test_feature_matrix_creation(self):
        """Test creating a feature matrix from SMILES list"""
        extractor = StructuralFeatureExtractor()
        smiles_list = ['C', 'CC', 'CCO']

        features_list = [extractor.extract_features(s) for s in smiles_list]
        feature_matrix = np.array(features_list)

        assert feature_matrix.shape == (3, extractor.n_features)
        assert not np.any(np.isnan(feature_matrix))
        assert not np.any(np.isinf(feature_matrix))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
