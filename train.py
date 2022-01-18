from utils import *
from model import *
from encoding import *

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
    parser.add_argument(
        '-p0', '--threshold_prob_0',
        help = 'Threshold probability for predicting 0.',
        type = float, 
        required= True
    )
    parser.add_argument(
        '-p1', '--threshold_prob_1',
        help = 'Threshold probability for predicting 1.',
        type = float,
        required = True
    )
    parser.add_argument(
        '-M', '--max_num_node',
        help = 'Maximum number of nodes in any tree search problem.',
        type = int, 
        required = True
    )
    parser.add_argument(
        '-bn', '--batch_norm', 
        help = 'Do batch normalization in the model', 
        action = 'store_true'
    )
    parser.add_argument(
        '-a', '--accelerate',
        help = 'Accelerate by merging different problems of the same instance into one graph, but it requires large memory.',
        action = 'store_true'
    )
    args = parser.parse_args()
    
    data_dir = f'data/instances/{args.problem}'
    train_dir = '_'.join([f'{data_dir}/train'] + args.problem_params)
    valid_dir = '_'.join([f'{data_dir}/valid'] + args.problem_params)

    num_training_instances = len(os.listdir(train_dir))
    num_validation_instances = len(os.listdir(valid_dir))

    model = Model(v_dim=len(args.variable_features), c_dim=len(args.constraint_features), e_dim=len(args.edge_features), K = args.num_prob_map, bn = args.batch_norm)
    encoder = BinaryIPEncoder(*['v_'+feat for feat in args.variable_features], *['c_'+feat for feat in args.constraint_features], *['e_'+feat for feat in args.edge_features])

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
    Loss = torch.nn.BCELoss(reduction = 'none')

    shuffled_indices = np.arange(args.num_training + args.num_validation)
    np.random.shuffle(shuffled_indices)
    training_mask = shuffled_indices < args.num_training 
    
    loss_hist, tree_size_hist = [], []
    train_best_map_hist, valid_best_map_hist = np.array([0] * args.num_prob_map), np.array([0] * args.num_prob_map)
    
    
    for i in range(args.num_training + args.num_validation):
        if training_mask[i]:
            idx = np.random.randint(num_training_instances)
            ip_instance = load_instance(f'{train_dir}/instance_{idx+1}.lp') 
            optimizer.zero_grad()
        else: 
            idx = np.random.randint(num_validation_instances)
            ip_instance = load_instance(f'{valid_dir}/instance_{idx+1}.lp') 
            model.eval()
        loss = torch.tensor(0.0, requires_grad = bool(training_mask[i]))

        problems = tree_search_train(ip_instance, model.predictor, args.max_num_node, encoder = encoder, p0 = args.threshold_prob_0, p1 = args.threshold_prob_1)
        
        if args.accelerate:
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

        else:
            for ip, sol in problems:
                v, c, e = encoder(ip)
                prob_maps = model(v, c, e)
                _loss, _index = Loss(input = prob_maps, target = torch.tensor(sol).unsqueeze(-1).expand(*prob_maps.shape).float()).mean(axis = 0).min(dim = 0)
                loss = loss + _loss / len(problems)
                if training_mask[i]:
                    train_best_map_hist[_index.item()] += 1
                else:
                    valid_best_map_hist[_index.item()] += 1
        
        loss_hist.append(loss.item()), tree_size_hist.append(len(problems))
        
        if training_mask[i]:
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'Step {i} (train), loss {loss.item():.3f}, tree_size {len(problems)}.')
        else:
            model.train()
            print(f'Step {i} (valid), loss {loss.item():.3f}, tree_size {len(problems)}.')
            
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
    model_config = {
        'variable_features': args.variable_features,
        'constraint_features': args.constraint_features,
        'edge_features': args.edge_features,
        'num_prob_map': args.num_prob_map, 
        'batch_norm': args.batch_norm,
        'threshold_prob_0': args.threshold_prob_0,
        'threshold_prob_1': args.threshold_prob_1
    }
    with open(model_dir +'/model_config.json', 'w') as f:
        json.dump(model_config, f)
    torch.save(model.state_dict(), model_dir + '/model_state_dict')
