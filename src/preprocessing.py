import numpy as np
import pandas as pd
from rdkit import Chem
from collections import defaultdict
import torch
from torch.utils.data.dataset import Subset
from sklearn.model_selection import StratifiedKFold


class Featurizer():

    def __init__(self):
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))


    def  get_atom_ids(self, mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)
    
    
    def get_ijbond_dict(self, mol):
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict
    
    
    def get_atom_fingerprints(self, mol, radius):
        """Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        """
        atoms = self.get_atom_ids(mol)
        i_jbond_dict  = self.get_ijbond_dict(mol)

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


def create_datasets(train_path, test_path, radius, task, device):
    
    feat = Featurizer()
    
    def preprocessing(SMILES, y):
        DATASET = []
        for smiles, y in zip(SMILES, y):
            # generate mol object
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            # get molucular_size
            molecular_size = len(feat.get_atom_ids(mol))
            # get finger_prints
            fingerprints = feat.get_atom_fingerprints(mol, radius)
            fingerprints = torch.LongTensor(fingerprints).to(device)
            # get adjacency matrix
            adjacency = Chem.GetAdjacencyMatrix(mol)
            adjacency = torch.FloatTensor(adjacency).to(device)
            # objective variable tensor
            if task == 'classification':
                property = torch.LongTensor([y]).to(device)
            if task == 'regression':
                property = torch.FloatTensor([y]).to(device)
            # append
            DATASET.append((fingerprints, adjacency, molecular_size, property))
        return DATASET
    
    ### (i) load SMILES and objective ariable from training dataset ###
    df_train = pd.read_csv(f'{train_path}')
    y = np.array(df_train.y_true)
    # train valid split
    skf = StratifiedKFold(n_splits=5, random_state=9, shuffle=True)
    datasets_train = []
    datasets_valid = []
    valid_indexes = []
    for tr_idx, va_idx in skf.split(df_train , y):
        smiles_tr = df_train.iloc[tr_idx].SMILES.tolist()
        smiles_va = df_train.iloc[va_idx].SMILES.tolist()
        y_tr = np.array(df_train.iloc[tr_idx].y_true)
        y_va = np.array(df_train.iloc[va_idx].y_true)
        data_tr = preprocessing(smiles_tr, y_tr)
        data_va = preprocessing(smiles_va, y_va)
        datasets_train.append(data_tr)
        datasets_valid.append(data_va)
        valid_indexes.append(va_idx)
    ### (ii) load SMILES and objective variable from testset, respectively ###
    df_test = pd.read_csv(f'{test_path}')
    SMILES_TE = df_test.SMILES.tolist()
    N = len(SMILES_TE)
    K = int(N/2) # K zeros, N-K ones
    y_dummy = np.array([0] * K + [1] * (N-K))
    np.random.shuffle(y_dummy)
    dataset_test = preprocessing(SMILES_TE, y_dummy)
    
    N_fingerprints = len(feat.fingerprint_dict)
    
    return datasets_train, datasets_valid, dataset_test, N_fingerprints, valid_indexes