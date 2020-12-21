import preprocessing as pp

import sys
import os
import argparse
import pickle
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

import requests
import json


class MolecularGNN(nn.Module):
    
    def __init__(self, task, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGNN, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
        
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if task == 'regression':
            self.W_property = nn.Linear(dim, 1)
    
    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices
    
    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors)) # message passing func.
        return hidden_vectors + torch.matmul(matrix, hidden_vectors) # update func.

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    
    def gnn(self, inputs):
        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    
    def forward(self, data_batch, train):
        inputs = data_batch[:-1]
        y_true = torch.cat(data_batch[-1])
        if train:
            molecular_vectors = self.gnn(inputs)
            pred = self.mlp(molecular_vectors)
            if task == 'classification':
                loss = F.cross_entropy(pred, y_true)
            elif task == 'regression':
                loss = F.mse_loss(pred, y_true)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors_ = self.gnn(inputs)
                pred_ = self.mlp(molecular_vectors_)
                if task == 'classification':
                    loss_ = F.cross_entropy(pred_, y_true)
                elif task == 'regression':
                    loss_ =  F.mse_loss(pred_, y_true)
            return pred_, y_true, loss_


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        loss_history = []
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
            loss_history.append(loss.item())
            
        return loss_total, loss_total/len(loss_history)
    

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        if task == 'classification':
            P, Y, L = [], [], []
            for i in range(0, N, batch_test):
                data_batch = list(zip(*dataset[i:i+batch_test]))
                p, y, l = self.model.forward(data_batch, train=False)
                p = p.to('cpu').data.numpy()
                y = y.to('cpu').data.numpy()
                l = l.to('cpu').data.numpy()
                P.append(p)
                Y.append(y)
                L.append(l)
            auc = roc_auc_score(np.concatenate(Y), np.concatenate(P)[:,1])
            return  np.concatenate(P)[:,1], np.mean(L), auc
        elif task == 'regression':
            SAE = 0  # sum absolute error.
            for i in range(0, N, batch_test):
                data_batch = list(zip(*dataset[i:i+batch_test]))
                pred, y_true = self.model.forward(data_batch, train=False)
                SAE += sum(np.abs(pred-y_true))
            MAE = SAE / N  # mean absolute error.
            return MAE


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('date')
    parser.add_argument('train_path')
    parser.add_argument('test_path')
    parser.add_argument('task')
    parser.add_argument('radius', type=int)
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_hidden', type=int)
    parser.add_argument('layer_output', type=int)
    parser.add_argument('batch_train', type=int)
    parser.add_argument('batch_test', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('decay_interval', type=int)
    parser.add_argument('iteration', type=int)
    args = parser.parse_args()
    
    
    # to device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')
    print('-'*100)
    
    ### Parameters ###
    # processing params
    radius = args.radius
    task = args.task
    train_path = args.train_path
    test_path = args.test_path   
    #GNN params
    dim=args.dim
    layer_hidden=args.layer_hidden
    layer_output=args.layer_output
    batch_train=args.batch_train
    batch_test=args.batch_test
    lr=args.lr
    lr_decay=args.lr_decay
    decay_interval=args.decay_interval
    iteration=args.iteration
    settings = f'dim{dim}--layer_hidden{layer_hidden}--layer_output{layer_output}--lr{lr}--lr_decay{lr_decay}--decay_interval{decay_interval}--batch{batch_train}'
    print(settings)
    
    # Slack Message
    url = 'http://xxx.xxx.xxx'
    message = {
        "text": f"GNN train start"
    }
    message_ = {
        "text": f"GNN train end.)"
    }
    requests.post(url, data = json.dumps(message))
    
    # Preprocessing  Datasets
    print('Creating datasets from molecular graph.')
    print('Trainingset is splitted and converted into subsets based on K-foldCV')
    print('Just a moment......')
    datasets_train, datasets_valid, dataset_test, N_fingerprints, valid_indexes = pp.create_datasets(
        train_path,
        test_path,
        radius,
        task,
        device
    )
    
    # Make directiry for Saving Results
    os.mkdir(f'{args.date}')
    ### Trainig and Prediction ##
    for a in range(5):
        dataset_train =  datasets_train[a]
        dataset_valid = datasets_valid[a]
        dataset_test = dataset_test
        N_fingerprints = N_fingerprints
        print('-'*100)
        print('The preprocess has finished!')
        print('# of training data allocations:', a)
        print('# of training data samples:', len(dataset_train))
        print('# of development data samples:', len(dataset_valid))
        print('# of test data samples:', len(dataset_test))
        print('-'*100)

        # Create Model
        print('Creating a model.')
        torch.manual_seed(1234)
        model = MolecularGNN(
            task,
            N_fingerprints,
            dim,
            layer_hidden,
            layer_output,
        ).to(device)
        trainer = Trainer(model)
        tester = Tester(model)
        print('# of model parameters:',
              sum([np.prod(p.size()) for p in model.parameters()]))
        print('-'*100)

        # training
        if task == 'classification':
            result = 'Epoch\tTime(sec)\tLoss_train\tLogloss_valid\tAUC_valid'
        # if task == 'regression':
        #     result = 'Epoch\tTime(sec)\tLoss_train\tMAE_valid'

        HISTORY = []
        VA_PREDS = []
        TE_PREDS = []

        start = timeit.default_timer()
        for epoch in range(iteration):
            epoch += 1
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            loss_train, loss_train_ave= trainer.train(dataset_train)
            if task == 'classification':
                pred_valid, loss_valid, auc_valid = tester.test(dataset_valid)
                pred_test, loss_test, auc_test = tester.test(dataset_test)
            # if task == 'regression':
            #     pred_valid = tester.test(dataset_valid)
            #     pred_test = tester.test(dataset_test)
            time = timeit.default_timer() - start

            if epoch == 1:
                minutes = time * iteration / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The training will finish in about',
                      hours, 'hours', minutes, 'minutes.')
                print('-'*100)
                print(result)

            result = '\t'.join(
                map(
                    str, 
                    [
                        epoch,
                        round(time, 5),
                        round(loss_train_ave, 7),
                        round(loss_valid, 7),
                        round(auc_valid, 7),
                    ]
                )
            )
            HISTORY.append([epoch, time, loss_train_ave, loss_valid, auc_valid])
            VA_PREDS.append(pred_valid)
            TE_PREDS.append(pred_test)
            print(result)

        items = [HISTORY, VA_PREDS, TE_PREDS, valid_indexes]
        with open(f'./{args.date}/results{a}--{settings}.pkl', 'wb') as f:
            pickle.dump(items, f)
        message_ = {
            "text": f"GNN train end. (AUC={round(max([h[4] for h in HISTORY]), 4)}) {settings}"
        }
        requests.post(url, data = json.dumps(message_))
    
