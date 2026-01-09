"""
Structural Feature Extraction for CE50 Prediction Enhancement

This module extracts 3D structural features from SMILES strings to enhance
CE50 (collision energy for 50% fragmentation) predictions. It bridges the gap
between 1D SMILES and 3D molecular structure using RDKit conformer generation,
then extracts features using lammps-data-file topology analysis and UFF parameters.

Features extracted:
- Bond characteristics (lengths, counts, types)
- UFF force field parameters (energies, charges, coordination)
- Topology metrics (angles, dihedrals, complexity)
- Atom-level properties (hybridization, heteroatoms, rings)

Author: Enhanced CE50 Prediction System
Date: 2026-01-08
"""

import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem

from lammps_data.bonds import extrapolate_bonds, calculate_bond_length
from lammps_data.angles import get_angles, calculate_angle
from lammps_data.dihedrals import get_dihedrals
from lammps_data.uff_table import UffTable, \
    NonExistentElementError, NonExistentCoordinationError, NonExistentAngleError


# Atomic number to element symbol mapping (partial - extend as needed)
ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
    5: 'B', 14: 'Si', 33: 'As', 34: 'Se',
}


def atomic_num_to_symbol(atomic_num):
    """Convert atomic number to element symbol"""
    return ATOMIC_NUM_TO_SYMBOL.get(atomic_num, 'X')


def smiles_to_3d_coords(smiles, random_seed=42):
    """
    Convert SMILES string to 3D coordinates using RDKit ETKDG algorithm.

    This function generates a 3D conformer from a SMILES string using the
    Extended Torsion Knowledge Distance Geometry (ETKDG) method, which uses
    experimental torsion angle preferences for more realistic structures.

    Parameters:
    -----------
    smiles : str
        SMILES string representation of molecule
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    tuple or None
        (atoms_list, rdkit_mol) where atoms_list is [(x, y, z, atomic_num), ...]
        Returns None if SMILES is invalid or 3D generation fails

    Examples:
    ---------
    >>> atoms, mol = smiles_to_3d_coords('CCO')  # ethanol
    >>> len(atoms)
    9  # 2C + 1O + 6H
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens (important for complete structure)
    mol = Chem.AddHs(mol)

    # Generate 3D conformer using ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed

    # Try ETKDG first
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        # Fallback: basic embedding with random coordinates
        success = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if success != 0:
            return None

    # Optimize geometry with UFF force field
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # UFF optimization failed, but we can still use the embedded structure
        pass

    # Extract coordinates in lammps-data-file format: (x, y, z, atomic_num)
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append((pos.x, pos.y, pos.z, atom.GetAtomicNum()))

    return atoms, mol


class StructuralFeatureExtractor:
    """
    Extract 3D structural features from SMILES strings for CE50 prediction.

    This class converts SMILES to 3D coordinates, then extracts ~40 structural
    features related to molecular fragmentation including bond properties,
    UFF force field parameters, topology, and atom characteristics.

    Attributes:
    -----------
    uff_table : UffTable
        Universal Force Field parameters table
    feature_names : list
        Names of all extracted features (for interpretability)
    n_features : int
        Total number of features extracted

    Examples:
    ---------
    >>> extractor = StructuralFeatureExtractor()
    >>> features = extractor.extract_features('CCO')
    >>> features.shape
    (40,)
    """

    def __init__(self):
        self.uff_table = UffTable()
        self.feature_names = self._initialize_feature_names()
        self.n_features = len(self.feature_names)

    def _initialize_feature_names(self):
        """Initialize list of feature names for interpretability"""
        names = [
            # Bond features (10)
            'bond_count', 'avg_bond_length', 'min_bond_length', 'max_bond_length',
            'std_bond_length', 'c_c_bonds', 'c_n_bonds', 'c_o_bonds',
            'c_h_bonds', 'bond_density',

            # UFF features (8)
            'avg_uff_energy', 'min_uff_energy', 'max_uff_energy',
            'avg_vdw_distance', 'avg_charge', 'avg_coordination',
            'avg_angle_strain', 'max_angle_strain',

            # Topology features (6)
            'angle_count', 'dihedral_count', 'angles_per_atom_avg',
            'angles_per_atom_max', 'topology_complexity', 'branching_index',

            # Atom features (16)
            'heavy_atom_count', 'heteroatom_ratio', 'n_o_ratio',
            'sp_ratio', 'sp2_ratio', 'sp3_ratio',
            'ring_count', 'aromatic_ring_count',
            'rotatable_bonds', 'hbd_count', 'hba_count',
            'molecular_weight', 'num_carbon', 'num_nitrogen',
            'num_oxygen', 'num_hetero'
        ]
        return names

    def extract_features(self, smiles):
        """
        Main extraction pipeline: SMILES → 3D → features.

        Parameters:
        -----------
        smiles : str
            SMILES string to extract features from

        Returns:
        --------
        np.array or None
            Array of ~40 structural features, or None if extraction fails
        """
        # 1. Convert SMILES to 3D coordinates
        result = smiles_to_3d_coords(smiles)
        if result is None:
            return None
        atoms, rdkit_mol = result

        # 2. Extract topology using lammps-data-file
        bonds = extrapolate_bonds(atoms)
        angles = get_angles(bonds)
        dihedrals = get_dihedrals(bonds)

        # 3. Compute features
        features = {}

        # Extract each feature category
        features.update(self._bond_features(bonds, atoms))
        features.update(self._uff_features(bonds, atoms, angles))
        features.update(self._topology_features(bonds, angles, dihedrals, atoms))
        features.update(self._atom_features(atoms, rdkit_mol))

        # Convert to numpy array in consistent order
        feature_array = np.array([features[name] for name in self.feature_names])

        return feature_array

    def _bond_features(self, bonds, atoms):
        """
        Extract bond-related features.

        Calculates bond lengths, counts by element pair, and bond density.
        Longer/weaker bonds are more likely to fragment.
        """
        if len(bonds) == 0:
            return {
                'bond_count': 0,
                'avg_bond_length': 0.0,
                'min_bond_length': 0.0,
                'max_bond_length': 0.0,
                'std_bond_length': 0.0,
                'c_c_bonds': 0,
                'c_n_bonds': 0,
                'c_o_bonds': 0,
                'c_h_bonds': 0,
                'bond_density': 0.0,
            }

        bond_lengths = []
        bond_types = defaultdict(int)

        for bond in bonds:
            try:
                length = calculate_bond_length(bond, atoms)
                bond_lengths.append(length)
            except Exception:
                # Skip bonds with calculation errors
                continue

            # Classify bond by element pair
            elem1 = atoms[bond[0]][3]  # atomic number
            elem2 = atoms[bond[1]][3]
            bond_key = tuple(sorted([elem1, elem2]))
            bond_types[bond_key] += 1

        # Handle case where no valid bond lengths were calculated
        if len(bond_lengths) == 0:
            bond_lengths = [0.0]

        n_atoms = len(atoms)

        return {
            'bond_count': len(bonds),
            'avg_bond_length': np.mean(bond_lengths),
            'min_bond_length': np.min(bond_lengths),
            'max_bond_length': np.max(bond_lengths),
            'std_bond_length': np.std(bond_lengths) if len(bond_lengths) > 1 else 0.0,
            'c_c_bonds': bond_types.get((6, 6), 0),
            'c_n_bonds': bond_types.get((6, 7), 0),
            'c_o_bonds': bond_types.get((6, 8), 0),
            'c_h_bonds': bond_types.get((1, 6), 0),
            'bond_density': len(bonds) / n_atoms if n_atoms > 0 else 0.0,
        }

    def _uff_features(self, bonds, atoms, angles):
        """
        Extract UFF force field parameters.

        UFF parameters encode fundamental chemical information:
        - Energy (Di): bond dissociation energy
        - Distance (xi): van der Waals radius
        - Charge (Zi): partial charge
        - Angle strain: deviation from optimal geometry

        These directly relate to fragmentation energetics.
        """
        bond_energies = []
        vdw_distances = []
        charges = []
        coordination_nums = []
        angle_strains = []

        # Per-atom UFF lookup
        for atom_idx, atom in enumerate(atoms):
            element = atomic_num_to_symbol(atom[3])

            # Count bonds for this atom (coordination number)
            atom_bonds = [b for b in bonds if atom_idx in b]
            n_bonds = len(atom_bonds)

            if n_bonds == 0:
                continue

            # Calculate average angle for this atom
            atom_angles = [a for a in angles if atom_idx == a[1]]  # middle atom
            if atom_angles:
                angle_values = []
                for angle_triple in atom_angles:
                    try:
                        angle_val = calculate_angle(
                            atoms[angle_triple[0]][:3],
                            atoms[angle_triple[1]][:3],
                            atoms[angle_triple[2]][:3]
                        )
                        angle_values.append(angle_val)
                    except Exception:
                        continue

                avg_angle = np.mean(angle_values) if angle_values else 109.47
            else:
                # Default angles based on coordination
                if n_bonds == 1:
                    avg_angle = 180.0
                elif n_bonds == 2:
                    avg_angle = 180.0
                elif n_bonds == 3:
                    avg_angle = 120.0
                else:
                    avg_angle = 109.47

            # Lookup UFF parameters
            try:
                # Try with tight tolerance first
                uff_row = self.uff_table.lookup(element, n_bonds, avg_angle, tolerance=10)

                bond_energies.append(uff_row['energy'])
                vdw_distances.append(uff_row['distance'])
                charges.append(abs(uff_row['charge']))
                coordination_nums.append(uff_row['coordination'])

                # Angle strain: deviation from UFF optimal angle
                strain = abs(avg_angle - uff_row['angle'])
                angle_strains.append(strain)

            except (NonExistentElementError, NonExistentCoordinationError, NonExistentAngleError):
                # Try looser tolerance if tight one fails
                try:
                    uff_row = self.uff_table.lookup(element, n_bonds, avg_angle, tolerance=5)
                    bond_energies.append(uff_row['energy'])
                    vdw_distances.append(uff_row['distance'])
                    charges.append(abs(uff_row['charge']))
                    coordination_nums.append(uff_row['coordination'])
                    strain = abs(avg_angle - uff_row['angle'])
                    angle_strains.append(strain)
                except:
                    # Atom not in UFF table or unusual coordination - skip
                    pass
            except Exception:
                # Catch any other lookup errors (like multiplicity)
                pass

        # Return aggregated UFF features
        return {
            'avg_uff_energy': np.mean(bond_energies) if bond_energies else 0.0,
            'min_uff_energy': np.min(bond_energies) if bond_energies else 0.0,
            'max_uff_energy': np.max(bond_energies) if bond_energies else 0.0,
            'avg_vdw_distance': np.mean(vdw_distances) if vdw_distances else 0.0,
            'avg_charge': np.mean(charges) if charges else 0.0,
            'avg_coordination': np.mean(coordination_nums) if coordination_nums else 0.0,
            'avg_angle_strain': np.mean(angle_strains) if angle_strains else 0.0,
            'max_angle_strain': np.max(angle_strains) if angle_strains else 0.0,
        }

    def _topology_features(self, bonds, angles, dihedrals, atoms):
        """
        Extract topological complexity features.

        Higher branching and complexity create multiple potential fragmentation sites.
        """
        n_atoms = len(atoms)

        # Angles per atom (measure of branching)
        angles_per_atom = defaultdict(int)
        for angle in angles:
            angles_per_atom[angle[1]] += 1  # middle atom

        # Branching index: ratio of branched atoms to total atoms
        branched_atoms = sum(1 for count in angles_per_atom.values() if count > 2)
        branching_index = branched_atoms / n_atoms if n_atoms > 0 else 0.0

        return {
            'angle_count': len(angles),
            'dihedral_count': len(dihedrals),
            'angles_per_atom_avg': np.mean(list(angles_per_atom.values())) if angles_per_atom else 0.0,
            'angles_per_atom_max': np.max(list(angles_per_atom.values())) if angles_per_atom else 0.0,
            'topology_complexity': (len(angles) + len(dihedrals)) / n_atoms if n_atoms > 0 else 0.0,
            'branching_index': branching_index,
        }

    def _atom_features(self, atoms, rdkit_mol):
        """
        Extract atom-level features using RDKit.

        Includes element counts, hybridization, rings, and molecular descriptors.
        """
        n_atoms = len(atoms)
        atomic_nums = [a[3] for a in atoms]

        # Element counts
        c_count = sum(1 for an in atomic_nums if an == 6)
        n_count = sum(1 for an in atomic_nums if an == 7)
        o_count = sum(1 for an in atomic_nums if an == 8)
        s_count = sum(1 for an in atomic_nums if an == 16)
        p_count = sum(1 for an in atomic_nums if an == 15)
        hetero_count = n_count + o_count + s_count + p_count

        # Hybridization from RDKit
        sp_count = sp2_count = sp3_count = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Only count carbon hybridization
                hyb = atom.GetHybridization()
                if hyb == Chem.HybridizationType.SP:
                    sp_count += 1
                elif hyb == Chem.HybridizationType.SP2:
                    sp2_count += 1
                elif hyb == Chem.HybridizationType.SP3:
                    sp3_count += 1

        # Ring analysis
        ring_info = rdkit_mol.GetRingInfo()
        aromatic_count = sum(
            1 for ring in ring_info.AtomRings()
            if all(rdkit_mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        )

        # Molecular descriptors
        from rdkit.Chem import Descriptors, Lipinski

        rotatable_bonds = Descriptors.NumRotatableBonds(rdkit_mol)
        hbd = Lipinski.NumHDonors(rdkit_mol)
        hba = Lipinski.NumHAcceptors(rdkit_mol)
        mol_wt = Descriptors.MolWt(rdkit_mol)

        return {
            'heavy_atom_count': n_atoms,
            'heteroatom_ratio': hetero_count / n_atoms if n_atoms > 0 else 0.0,
            'n_o_ratio': (n_count + o_count) / n_atoms if n_atoms > 0 else 0.0,
            'sp_ratio': sp_count / c_count if c_count > 0 else 0.0,
            'sp2_ratio': sp2_count / c_count if c_count > 0 else 0.0,
            'sp3_ratio': sp3_count / c_count if c_count > 0 else 0.0,
            'ring_count': ring_info.NumRings(),
            'aromatic_ring_count': aromatic_count,
            'rotatable_bonds': rotatable_bonds,
            'hbd_count': hbd,
            'hba_count': hba,
            'molecular_weight': mol_wt,
            'num_carbon': c_count,
            'num_nitrogen': n_count,
            'num_oxygen': o_count,
            'num_hetero': hetero_count,
        }
