# MolecularGNN
This repository is the version with minor changes to [molecularGNN_smiles](https://github.com/masashitsubaki/molecularGNN_smiles).The original algorithm for obtaining atom IDs based on r-radius subgraphs has been retained, the preprocessing convenience has been improved. Specifically, changes have been made in the following three areas. Now, this implementation is compatible with only classification task.  
* The SMILES input format has been modified to support csv.  
* Creating Featurizer class, we have made it easier to manipulate the molecular graph fingerprint acquisition for GNN from SMILES.  
* The model validation process has been modified. In the preprocessing phase, the training dataset is partitioned based on *k*-fold cross-validation to generate subsets of *k* different training and validation sets. During training of the GNN, the test set predictions of each model built from each subset are saved.
