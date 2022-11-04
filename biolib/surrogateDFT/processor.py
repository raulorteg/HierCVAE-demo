from collections import OrderedDict, defaultdict

import numpy as np
import torch
from rdkit import Chem

from .scalers import MinMaxScaler


class DatasetProcessor:
    def __init__(self, device: str):

        self.device = device
        self._load_dicts()
        self.scaler = MinMaxScaler()

    def _load_dicts(self):

        atom_dict = OrderedDict(
            {
                ("C", "aromatic"): 0,
                ("O", "aromatic"): 1,
                ("S", "aromatic"): 2,
                "C": 3,
                "H": 4,
                ("N", "aromatic"): 5,
                "Si": 6,
                ("Se", "aromatic"): 7,
            }
        )
        bond_dict = OrderedDict({"AROMATIC": 0, "DOUBLE": 1, "SINGLE": 2})
        fingerprint_dict = OrderedDict(
            {
                (0, ((0, 0), (2, 0), (4, 2))): 0,
                (0, ((0, 0), (0, 0), (4, 2))): 1,
                (0, ((0, 0), (0, 0), (1, 0))): 2,
                (1, ((0, 0), (0, 0))): 3,
                (0, ((0, 0), (0, 0), (2, 0))): 4,
                (2, ((0, 0), (0, 0))): 5,
                (0, ((0, 0), (0, 0), (3, 1))): 6,
                (3, ((0, 1), (3, 2), (4, 2))): 7,
                (3, ((3, 2), (3, 2), (4, 2), (4, 2))): 8,
                (0, ((0, 0), (0, 0), (0, 0))): 9,
                (4, ((0, 2),)): 10,
                (4, ((3, 2),)): 11,
                (5, ((0, 0), (0, 0), (4, 2))): 12,
                (0, ((0, 0), (4, 2), (5, 0))): 13,
                (0, ((0, 0), (0, 0), (5, 0))): 14,
                (0, ((0, 0), (1, 0), (4, 2))): 15,
                (3, ((0, 1), (4, 2), (6, 2))): 16,
                (6, ((3, 2), (3, 2), (4, 2), (4, 2))): 17,
                (4, ((5, 2),)): 18,
                (4, ((6, 2),)): 19,
                (5, ((0, 0), (0, 0))): 20,
                (3, ((3, 1), (4, 2), (6, 2))): 21,
                (6, ((0, 2), (3, 2), (4, 2), (4, 2))): 22,
                (0, ((0, 0), (0, 0), (6, 2))): 23,
                (0, ((0, 0), (0, 0), (3, 2))): 24,
                (3, ((0, 2), (3, 1), (4, 2))): 25,
                (0, ((0, 0), (1, 0), (3, 2))): 26,
                (5, ((0, 0), (2, 0))): 27,
                (2, ((5, 0), (5, 0))): 28,
                (0, ((0, 0), (0, 2), (5, 0))): 29,
                (3, ((0, 2), (0, 2), (4, 2), (4, 2))): 30,
                (0, ((0, 0), (0, 0), (0, 2))): 31,
                (0, ((0, 0), (4, 2), (7, 0))): 32,
                (0, ((0, 0), (0, 2), (2, 0))): 33,
                (0, ((0, 0), (0, 0), (7, 0))): 34,
                (7, ((0, 0), (0, 0))): 35,
                (3, ((3, 1), (3, 2), (6, 2))): 36,
                (3, ((0, 2), (3, 1), (6, 2))): 37,
                (3, ((3, 1), (3, 2), (3, 2))): 38,
                (3, ((0, 2), (3, 2), (4, 2), (4, 2))): 39,
                (3, ((3, 1), (3, 2), (4, 2))): 40,
                (0, ((0, 0), (2, 0), (3, 2))): 41,
                (0, ((0, 0), (0, 2), (7, 0))): 42,
                (6, ((0, 2), (0, 2), (4, 2), (4, 2))): 43,
                (0, ((0, 0), (0, 2), (1, 0))): 44,
                (0, ((0, 0), (3, 2), (5, 0))): 45,
                (3, ((0, 2), (3, 1), (3, 2))): 46,
                (0, ((0, 2), (2, 0), (5, 0))): 47,
                (0, ((2, 0), (4, 2), (5, 0))): 48,
                (3, ((4, 2), (4, 2), (4, 2), (5, 2))): 49,
                (5, ((0, 0), (0, 0), (3, 2))): 50,
                (0, ((2, 0), (3, 2), (5, 0))): 51,
                (0, ((0, 2), (5, 0), (5, 0))): 52,
                (0, ((4, 2), (5, 0), (5, 0))): 53,
                (0, ((0, 0), (3, 2), (7, 0))): 54,
                (0, ((3, 2), (5, 0), (5, 0))): 55,
            }
        )
        edge_dict = OrderedDict(
            {
                ((0, 0), 0): 0,
                ((0, 2), 0): 1,
                ((0, 4), 2): 2,
                ((0, 1), 0): 3,
                ((0, 3), 1): 4,
                ((3, 3), 2): 5,
                ((3, 4), 2): 6,
                ((0, 5), 0): 7,
                ((4, 5), 2): 8,
                ((3, 6), 2): 9,
                ((4, 6), 2): 10,
                ((3, 3), 1): 11,
                ((0, 6), 2): 12,
                ((0, 3), 2): 13,
                ((2, 5), 0): 14,
                ((0, 0), 2): 15,
                ((0, 7), 0): 16,
                ((3, 5), 2): 17,
            }
        )

        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.bond_dict))
        self.edge_dict = defaultdict(lambda: len(self.bond_dict))

        for key, value in atom_dict.items():
            self.atom_dict[key] = value

        for key, value in bond_dict.items():
            self.bond_dict[key] = value

        for key, value in fingerprint_dict.items():
            self.fingerprint_dict[key] = value

        for key, value in edge_dict.items():
            self.edge_dict[key] = value

        """
        self.atom_dict = {('C', 'aromatic'): 0, ('O', 'aromatic'): 1, ('S', 'aromatic'): 2, 'C': 3, 'H': 4, ('N', 'aromatic'): 5, 'Si': 6, ('Se', 'aromatic'): 7}
        self.bond_dict = {'AROMATIC': 0, 'DOUBLE': 1, 'SINGLE': 2}
        self.fingerprint_dict = {(0, ((0, 0), (2, 0), (4, 2))): 0, (0, ((0, 0), (0, 0), (4, 2))): 1, (0, ((0, 0), (0, 0), (1, 0))): 2, (1, ((0, 0), (0, 0))): 3, (0, ((0, 0), (0, 0), (2, 0))): 4, (2, ((0, 0), (0, 0))): 5, (0, ((0, 0), (0, 0), (3, 1))): 6, (3, ((0, 1), (3, 2), (4, 2))): 7, (3, ((3, 2), (3, 2), (4, 2), (4, 2))): 8, (0, ((0, 0), (0, 0), (0, 0))): 9, (4, ((0, 2),)): 10, (4, ((3, 2),)): 11, (5, ((0, 0), (0, 0), (4, 2))): 12, (0, ((0, 0), (4, 2), (5, 0))): 13, (0, ((0, 0), (0, 0), (5, 0))): 14, (0, ((0, 0), (1, 0), (4, 2))): 15, (3, ((0, 1), (4, 2), (6, 2))): 16, (6, ((3, 2), (3, 2), (4, 2), (4, 2))): 17, (4, ((5, 2),)): 18, (4, ((6, 2),)): 19, (5, ((0, 0), (0, 0))): 20, (3, ((3, 1), (4, 2), (6, 2))): 21, (6, ((0, 2), (3, 2), (4, 2), (4, 2))): 22, (0, ((0, 0), (0, 0), (6, 2))): 23, (0, ((0, 0), (0, 0), (3, 2))): 24, (3, ((0, 2), (3, 1), (4, 2))): 25, (0, ((0, 0), (1, 0), (3, 2))): 26, (5, ((0, 0), (2, 0))): 27, (2, ((5, 0), (5, 0))): 28, (0, ((0, 0), (0, 2), (5, 0))): 29, (3, ((0, 2), (0, 2), (4, 2), (4, 2))): 30, (0, ((0, 0), (0, 0), (0, 2))): 31, (0, ((0, 0), (4, 2), (7, 0))): 32, (0, ((0, 0), (0, 2), (2, 0))): 33, (0, ((0, 0), (0, 0), (7, 0))): 34, (7, ((0, 0), (0, 0))): 35, (3, ((3, 1), (3, 2), (6, 2))): 36, (3, ((0, 2), (3, 1), (6, 2))): 37, (3, ((3, 1), (3, 2), (3, 2))): 38, (3, ((0, 2), (3, 2), (4, 2), (4, 2))): 39, (3, ((3, 1), (3, 2), (4, 2))): 40, (0, ((0, 0), (2, 0), (3, 2))): 41, (0, ((0, 0), (0, 2), (7, 0))): 42, (6, ((0, 2), (0, 2), (4, 2), (4, 2))): 43, (0, ((0, 0), (0, 2), (1, 0))): 44, (0, ((0, 0), (3, 2), (5, 0))): 45, (3, ((0, 2), (3, 1), (3, 2))): 46, (0, ((0, 2), (2, 0), (5, 0))): 47, (0, ((2, 0), (4, 2), (5, 0))): 48, (3, ((4, 2), (4, 2), (4, 2), (5, 2))): 49, (5, ((0, 0), (0, 0), (3, 2))): 50, (0, ((2, 0), (3, 2), (5, 0))): 51, (0, ((0, 2), (5, 0), (5, 0))): 52, (0, ((4, 2), (5, 0), (5, 0))): 53, (0, ((0, 0), (3, 2), (7, 0))): 54, (0, ((3, 2), (5, 0), (5, 0))): 55}
        self.edge_dict = {((0, 0), 0): 0, ((0, 2), 0): 1, ((0, 4), 2): 2, ((0, 1), 0): 3, ((0, 3), 1): 4, ((3, 3), 2): 5, ((3, 4), 2): 6, ((0, 5), 0): 7, ((4, 5), 2): 8, ((3, 6), 2): 9, ((4, 6), 2): 10, ((3, 3), 1): 11, ((0, 6), 2): 12, ((0, 3), 2): 13, ((2, 5), 0): 14, ((0, 0), 2): 15, ((0, 7), 0): 16, ((3, 5), 2): 17}
        """

    def _atoms_symbol2index(self, mol):
        """Transform the atom types in a molecule (e.g., H, C, and O)
        into the indices (e.g., H=0, C=1, and O=2).
        Note that each atom index considers the aromaticity.
        """

        # get the symbol for every atom
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]

        # if the atom is aromatic then also add this information
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], "aromatic")

        # transform the symbols into indexes
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def _create_ijbonddict(self, mol):
        """Create a dictionary, in which each key is a node ID
        and each value is the tuples of its neighboring node
        and chemical bond (e.g., single and double) IDs.
        """
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    def _extract_fingerprints(self, radius, atoms, i_jbond_dict):
        """Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        """

        if (len(atoms) == 1) or (radius == 0):
            nodes = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_.append(self.fingerprint_dict[fingerprint])

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = nodes_
                i_jedge_dict = i_jedge_dict_

        return np.array(nodes)

    def process_file(self, filename: str):

        processed_dataset = []

        # load the file into memmory
        with open(filename, "r") as f:

            # process line by line
            for line in f.readlines():

                # split smiles string and property
                smiles, value = line.split()

                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms_idxs = self._atoms_symbol2index(mol)
                molecular_size = len(atoms_idxs)
                i_jbond_dict = self._create_ijbonddict(mol)
                fingerprints = self._extract_fingerprints(1, atoms_idxs, i_jbond_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)

                # send the data to the appropriate device (cuda or cpu)
                fingerprints = torch.LongTensor(fingerprints).to(self.device)
                adjacency = torch.FloatTensor(adjacency).to(self.device)
                value = torch.FloatTensor([[float(value)]]).to(self.device)

                # save in buffer
                processed_dataset.append(
                    (fingerprints, adjacency, molecular_size, value)
                )

        return processed_dataset

    def process_list(self, smiles: str):

        processed_dataset = []
        for smile in smiles:

            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            atoms_idxs = self._atoms_symbol2index(mol)
            molecular_size = len(atoms_idxs)
            i_jbond_dict = self._create_ijbonddict(mol)
            fingerprints = self._extract_fingerprints(1, atoms_idxs, i_jbond_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            # send the data to the appropriate device (cuda or cpu)
            fingerprints = torch.LongTensor(fingerprints).to(self.device)
            adjacency = torch.FloatTensor(adjacency).to(self.device)
            value = torch.FloatTensor([[float(0.0)]]).to(self.device)

            # save in buffer
            processed_dataset.append((fingerprints, adjacency, molecular_size, value))

        return processed_dataset
