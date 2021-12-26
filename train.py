from utils import *
from model import *
from encoding import *

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import numpy as np
import json

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
        default = BinaryIPEncoder.variable_features,
        choices= BinaryIPEncoder.variable_features
    )
    parser.add_argument(
        '-c', '--constraint_features',
        help='Constraint features.',
        nargs = '+',
        default = BinaryIPEncoder.constraint_features,
        choices= BinaryIPEncoder.constraint_features
    )
    parser.add_argument(
        '-e', '--edge_features',
        help='Edge features.',
        nargs = '+',
        default = BinaryIPEncoder.edge_features,
        choices= BinaryIPEncoder.edge_features
    )
    parser.add_argument(
        '-nt', '--num_training', 
        help = 'Number of training steps.',
        type = int,
        required = True
    )
    parser.add_argument(
        '-nv', '--num_validation',
        help = 'Number of validation steps.',
        type = int,
        default = 0
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
    num_validation_instances = len(os.listdir(valid_dir))

    model = Model(v_dim=len(args.variable_features), c_dim=len(args.constraint_features), e_dim=len(args.edge_features), K = args.num_prob_map)
    encoder = BinaryIPEncoder(*['v_'+feat for feat in args.variable_features], *['c_'+feat for feat in args.constraint_features], *['e_'+feat for feat in args.edge_features])

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    Loss = torch.nn.BCELoss(reduction = 'none')

    shuffled_indices = np.arange(args.num_training + args.num_validation)
    np.random.shuffle(shuffled_indices)
    training_mask = shuffled_indices < args.num_training 
    
    loss_hist, tree_size_hist = [], []
    train_best_map_hist, valid_best_map_hist = np.array([0] * args.num_prob_map), np.array([0] * args.num_prob_map)

    for i in range(args.num_training + args.num_validation):
        if training_mask[i]:
            idx = np.random.randint(num_training_instances)
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            ip_instance = load_instance(f'{train_dir}/instance_{idx+1}.lp') 
        else: 
            idx = np.random.randint(num_validation_instances)
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            ip_instance = load_instance(f'{valid_dir}/instance_{idx+1}.lp') 
            model.eval()
        
        problems = tree_search_train(ip_instance, model.predictor, encoder = encoder, p0 = 0.05, p1 = 0.95)
        V, C, E, sols = [], [], [], []
        
        v_slices, c_slices, V, C, E, sols = [], [], [], [], [], []
        for ip, sol in problems:
            v, c, e = encoder(ip)
            V.append(v), C.append(c), E.append(e.coalesce())
            sols.append(sol)
            num_v, num_c = v.shape[0], c.shape[0]
            v_start, c_start = 0 if len(v_slices) == 0 else v_slices[-1].stop, 0 if len(c_slices) == 0 else c_slices[-1].stop
            v_slices.append(slice(v_start, v_start + num_v))
            c_slices.append(slice(c_start, c_start + num_c))

        V = torch.cat(V, dim = 0)
        C = torch.cat(C, dim = 0)

        E_indices = torch.cat([e.indices() + torch.tensor([[cs.start], [vs.start]]) for e, cs, vs in zip(E, c_slices, v_slices)], axis = 1)
        E_values = torch.cat([e.values() for e in E])
        E_shape = (C.shape[0], V.shape[0], E_values.shape[-1])
        E = torch.sparse_coo_tensor(E_indices, E_values, E_shape)

        prob_maps = model(V, C, E)

        for vs, sol in zip(v_slices, sols):
            _loss, _index = Loss(input = prob_maps[vs], target = torch.tensor(sol).unsqueeze(-1).expand(*prob_maps[vs].shape).float()).mean(axis = 0).min(dim = 0)
            loss = loss + _loss / len(problems)
            if training_mask[i]:
                train_best_map_hist[_index.item()] += 1
            else:
                valid_best_map_hist[_index.item()] += 1
        loss_hist.append(loss.item()), tree_size_hist.append(len(problems))
        
        if training_mask[i]:
            loss.backward()
            optimizer.step()
            print(f'Step {i} (train), loss {loss.item()}, tree_size {len(problems)}.')
        else:
            model.train()
            print(f'Step {i} (valid), loss {loss.item()}, tree_size {len(problems)}.')
            
    # summarize the history of loss and the tree size
    loss_hist, tree_size_hist = np.array(loss_hist), np.array(tree_size_hist)
    f, axes = plt.subplots(1, 2)
    axes[0].set_title('Loss history')
    axes[0].plot(np.arange(args.num_training + args.num_validation)[training_mask], loss_hist[training_mask], label = 'train')
    axes[0].plot(np.arange(args.num_training + args.num_validation)[~training_mask], loss_hist[~training_mask], label = 'valid')
    axes[0].legend()

    axes[1].set_title('Tree size history')
    axes[1].plot(np.arange(args.num_training + args.num_validation)[training_mask], tree_size_hist[training_mask], label = 'train')
    axes[1].plot(np.arange(args.num_training + args.num_validation)[~training_mask], tree_size_hist[~training_mask], label = 'valid')
    axes[1].legend()

    plt.show()
    
    # summarize the history of best maps
    f, axes = plt.subplots(1, 2)
    axes[0].set_title('train')
    axes[0].pie(train_best_map_hist, labels = np.arange(len(train_best_map_hist)))
    axes[0].legend()

    axes[1].set_title('valid')
    axes[1].pie(valid_best_map_hist, labels = np.arange(len(valid_best_map_hist)))
    axes[1].legend()

    plt.show()

    # save the model
    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    os.makedirs(model_dir, exist_ok = True)
    config = {
        'variable_features': args.variable_features,
        'constraint_features': args.constraint_features,
        'edge_features': args.edge_features,
        'num_prob_map': args.num_prob_map
    }
    with open(model_dir +'/config.json', 'w') as f:
        json.dump(config, f)
    torch.save(model.state_dict(), model_dir + '/model_state_dict')
