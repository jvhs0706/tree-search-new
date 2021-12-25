from utils import *
from model import *
from encoding import *

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help = 'MILP instance type to process.',
        choices = ['setcover', 'cauctions', 'indset'],
    )
    parser.add_argument(
        'problem_params', 
        help = 'Problem parameters to identify the instances.',
        nargs = '*'
    )
    parser.add_argument(
        '-v', '--variable_features',
        help='Variable features.',
        nargs = '+',
        type = lambda z: 'v_' + z,
        required = True
    )
    parser.add_argument(
        '-c', '--constraint_features',
        help='Constraint features.',
        nargs = '+',
        type = lambda z: 'c_' + z,
        required = True
    )
    parser.add_argument(
        '-e', '--edge_features',
        help='Edge features.',
        nargs = '+',
        type = lambda z: 'e_' + z,
        required = True
    )
    parser.add_argument(
        '-ns', '--num_step', 
        help = 'Number of training steps.',
        type = int,
        required = True
    )
    parser.add_argument(
        '-K', '--num_prob_map',
        help = 'Number of probability maps.',
        type = int, 
        required= True
    )
    args = parser.parse_args()
    
    data_dir = f'data/instances/{args.problem}'
    train_dir = '_'.join([f'{data_dir}/train'] + [str(s) for s in args.problem_params])
    valid_dir = '_'.join([f'{data_dir}/valid'] + [str(s) for s in args.problem_params])

    num_training_instances = len(os.listdir(train_dir))

    model = Model(v_feats=len(args.variable_features), c_feats=len(args.constraint_features), e_feats=len(args.edge_features), g_hidden_neurons=[64, 64], f_hidden_neurons=[64, 64], out_neurons=[32, args.num_prob_map])
    encoder = BinaryIPEncoder(*args.variable_features, *args.constraint_features, *args.edge_features)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    Loss = torch.nn.BCELoss(reduction = 'none')

    tree_size_stat, best_map_stat = [], []

    for i in range(args.num_step):
        idx = np.random.randint(num_training_instances)
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        ip_instance = load_model(f'{train_dir}/instance_{idx+1}.lp')
        training_set = tree_search_train(ip_instance, model.predictor, encoder = encoder, p0 = 0.02, p1 = 0.98)
        for ip, sol in training_set:
            V, C, E = encoder(ip)
            prob_maps = model(V, C, E)

            _loss, _index = Loss(input = prob_maps, target = torch.tensor(sol).unsqueeze(-1).expand(*prob_maps.shape).float()).mean(axis = 0).min(dim = 0)
            loss = loss + _loss / len(training_set)
            best_map_stat.append(_index.item())

        loss.backward()
        optimizer.step()
        tree_size_stat.append(len(training_set))
        print(f'Step {i}, loss {loss.item()}, tree_size {len(training_set)}.')
    
    tree_size_stat, best_map_stat = np.array(tree_size_stat), np.array(best_map_stat)
    plt.pie([(best_map_stat == k).sum() for k in range(best_map_stat.max() + 1)])
    plt.legend()
    plt.show()
    
    plt.plot(tree_size_stat)
    plt.show()