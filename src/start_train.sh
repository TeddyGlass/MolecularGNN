#!/bin/bash

date=201218
train_path=../datasets/Ames_SMILES_train.csv
test_path=../datasets/Ames_SMILES_test.csv
task=classification

dim=100
radius=1
layer_hidden=6
layer_output=6
batch_train=32
batch_test=32
lr=1e-4
lr_decay=0.9
decay_interval=3
iteration=200

python MolecularGNN.py $date $train_path $test_path $task $radius $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $iteration
