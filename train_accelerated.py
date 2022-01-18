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
    # arguments
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
        '-M', '--max_num_node',
        help = 'Maximum number of nodes.',
        type = lambda z: int(z) if z is not None else np.inf,
        default = np.inf
    )
    parser.add_argument(
        '-H', '--max_tree_height',
        help = 'Maximum tree height.',
        type = lambda z: int(z) if z is not None else np.inf,
        default = np.inf
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
        '-bn', '--batch_norm', 
        help = 'Do batch normalization in the model', 
        action = 'store_true'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        help = 'The (initial) learning rate of the Adam optimizer.',
        type = float,
        default = 1e-3
    )
    parser.add_argument(
        '-ss', '--step_size',
        help = 'The step size of the StepLR lr_scheduler.',
        type = int, 
        default = 128
    )
    parser.add_argument(
        '-g', '--gamma', 
        help = 'The gamma value of the StepLR lr_scheduler.',
        type = float,
        default = 0.5
    )
    args = parser.parse_args()
    
    # data directories
    data_dir = f'data/instances/{args.problem}'
    train_dir = '_'.join([f'{data_dir}/train'] + args.problem_params)
    valid_dir = '_'.join([f'{data_dir}/valid'] + args.problem_params)

    # number of training instances
    num_training_instances = len(os.listdir(train_dir))
    num_validation_instances = len(os.listdir(valid_dir))

    # initialize the model, the encoder, the optimizer, the scheduler, and the loss function
    model = Model(v_dim=len(args.variable_features), c_dim=len(args.constraint_features), e_dim=len(args.edge_features), K = args.num_prob_map, bn = args.batch_norm)
    encoder = BinaryIPEncoder(*['v_'+feat for feat in args.variable_features], *['c_'+feat for feat in args.constraint_features], *['e_'+feat for feat in args.edge_features])
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    Loss = torch.nn.BCELoss(reduction = 'none')

    # shuffle the appearences of the training instances and validation instances
    shuffled_indices = np.arange(args.num_training + args.num_validation)
    np.random.shuffle(shuffled_indices)
    training_mask = shuffled_indices < args.num_training 
    
    # the history of loss, the tree height, and the tree size will be kept
    loss_hist, tree_height_hist, num_node_hist = [], [], []
    train_best_map_hist, valid_best_map_hist = np.array([0] * args.num_prob_map), np.array([0] * args.num_prob_map)
    
    # training loop
    for i in range(args.num_training + args.num_validation):
        if training_mask[i]:
            idx = np.random.randint(num_training_instances)
            ip_instance = load_instance(f'{train_dir}/instance_{idx+1}.lp') 
        else: 
            idx = np.random.randint(num_validation_instances)
            ip_instance = load_instance(f'{valid_dir}/instance_{idx+1}.lp') 
        
        tree_height, num_node = 0, 0
        root_node = assign_values(ip_instance, [], [])

        # build the tree for one instance by increasing depth
        if type(root_node) == gp.Model:
            nodes = [root_node]
            mode = 'train' if training_mask[i] else 'valid'

            loss, var_count = torch.tensor(0.0, requires_grad = bool(training_mask[i]), dtype = torch.float), 0
            while nodes:
                new_nodes = []
                try:
                    out, proposals_batch = model.predictor_batch(nodes, encoder, p0 = args.threshold_prob_0, p1 = args.threshold_prob_1, mode = mode)
                    for proposals, ip, _out in zip(proposals_batch, nodes, out):
                        
                        # compute the loss for one depth level
                        opt_sol, _ = solve_instance(ip)
                        _loss, _index = Loss(input = _out, target = torch.tensor(opt_sol, dtype = torch.float).unsqueeze(-1).expand(*_out.shape)).sum(axis = 0).min(dim = 0)
                        if mode == 'train':
                            train_best_map_hist[_index.item()] += 1
                        else:
                            valid_best_map_hist[_index.item()] += 1
                        loss, var_count = loss + _loss, var_count + ip.NumVars
                        
                        # generate the next depth level
                        for proposal in proposals:
                            if proposal is not None:
                                new_ip = assign_values(ip, *proposal)
                                if type(new_ip) == gp.Model:
                                    new_nodes.append(new_ip)
                                
                except AttributeError as e:
                    pass
                
                # update the bookkeeping variables and the list of nodes
                tree_height, num_node = tree_height + 1, num_node + len(nodes)
                if tree_height >= args.max_tree_height or num_node >= args.max_num_node:
                    break
                else:
                    nodes = new_nodes

            # compute the overall loss
            loss = loss / var_count
            
            # do backprop if it is a training instance
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # update the history
            loss_hist.append(loss.item()), tree_height_hist.append(tree_height), num_node_hist.append(num_node)
            print(f'Step {i} ({mode}), loss {loss:.3f}, tree_height {tree_height}, num_node {num_node}.')

    # save the model and the configurations
    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    os.makedirs(model_dir, exist_ok = True)
    model_config = {
        'variable_features': args.variable_features,
        'constraint_features': args.constraint_features,
        'edge_features': args.edge_features,
        'num_prob_map': args.num_prob_map, 
        'batch_norm': args.batch_norm,
        'threshold_prob_0': args.threshold_prob_0,
        'threshold_prob_1': args.threshold_prob_1,
        'accelerated': True
    }
    with open(model_dir +'/model_config.json', 'w') as f:
        json.dump(model_config, f)
    
    training_config = {
        'num_step': args.num_training,
        'learning_rate': args.learning_rate,
        'step_size': args.step_size,
        'gamma': args.gamma
    }
    with open(model_dir +'/training_config.json', 'w') as f:
        json.dump(training_config, f)

    torch.save(model.state_dict(), model_dir + '/model_state_dict')
    
    # summarize the history of loss and the tree size
    loss_hist, tree_height_hist, num_node_hist = np.array(loss_hist), np.array(tree_height_hist), np.array(num_node_hist)
    f, axes = plt.subplots(1, 3, constrained_layout=True)
    axes[0].set_title('Loss history')
    axes[0].plot(np.arange(args.num_training + args.num_validation)[training_mask], loss_hist[training_mask], label = 'train')
    axes[0].plot(np.arange(args.num_training + args.num_validation)[~training_mask], loss_hist[~training_mask], label = 'valid')
    axes[0].legend()

    axes[1].set_title('Tree height history')
    axes[1].plot(np.arange(args.num_training + args.num_validation)[training_mask], tree_height_hist[training_mask], label = 'train')
    axes[1].plot(np.arange(args.num_training + args.num_validation)[~training_mask], tree_height_hist[~training_mask], label = 'valid')
    axes[1].legend()

    axes[2].set_title('Number of nodes history')
    axes[2].plot(np.arange(args.num_training + args.num_validation)[training_mask], num_node_hist[training_mask], label = 'train')
    axes[2].plot(np.arange(args.num_training + args.num_validation)[~training_mask], num_node_hist[~training_mask], label = 'valid')
    axes[2].legend()
    
    f.savefig(model_dir + '/training_hist.png')
    
    # summarize the history of best maps
    f, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title('train')
    axes[0].pie(train_best_map_hist, labels = np.arange(len(train_best_map_hist)))
    axes[0].legend()

    axes[1].set_title('valid')
    axes[1].pie(valid_best_map_hist, labels = np.arange(len(valid_best_map_hist)))
    axes[1].legend()

    f.savefig(model_dir + '/best_map_hist.png')

    