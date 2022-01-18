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
import time

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
        '-p0', '--threshold_prob_0',
        help = 'Threshold value for predicting 0.',
        type = float, 
        required = True
    )
    parser.add_argument(
        '-p1', '--threshold_prob_1',
        help = 'Threshold value for predicting 1.',
        type = float,
        required = True
    )
    parser.add_argument(
        '-M', '--max_tree_size',
        help = 'Maximum tree size.',
        type = int
    )
    parser.add_argument(
        '-H', '--max_tree_height',
        help = 'Maximum tree height.',
        type = int
    )
    parser.add_argument(
        '-N', '--first_num_instances',
        help = 'Only considering the first few instances.',
        type = int,
    )
    parser.add_argument(
        '-a', '--accelerate', 
        help = 'Accerelate by computing multiple problems at the same depth.', 
        action = 'store_true'
    )

    args = parser.parse_args()

    data_dir = f'data/instances/{args.problem}'
    test_dir = '_'.join([f'{data_dir}/test'] + args.problem_params)
        
    num_testing_instances = min(len(os.listdir(test_dir)), args.first_num_instances) if args.first_num_instances is not None else len(os.listdir(test_dir))

    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    with open(model_dir +'/model_config.json', 'r') as f:
        model_config = json.load(f)
    model = Model(v_dim=len(model_config['variable_features']), c_dim=len(model_config['constraint_features']), 
        e_dim=len(model_config['edge_features']), K = model_config['num_prob_map'], bn = model_config['batch_norm'])
    model.load_state_dict(torch.load(model_dir + '/model_state_dict'))
    encoder = BinaryIPEncoder(*['v_'+feat for feat in model_config['variable_features']], *['c_'+feat for feat in model_config['constraint_features']], *['e_'+feat for feat in model_config['edge_features']])    
    
    solver_time, gurobi_time = 0.0, 0.0
    solved = 0
    gap, gap_ratio = 0.0, 0.0
    
    with torch.no_grad():
        for i in range(num_testing_instances):
            ip_instance = load_instance(f'{test_dir}/instance_{i+1}.lp')
            
            if args.accelerate:
                tic = time.time()
                obj_val = tree_search_accelerated(ip_instance, model.predictor_batch, max_depth=args.max_tree_height, encoder = encoder, p0 = args.threshold_prob_0, p1 = args.threshold_prob_1)
                toc = time.time()
            else:
                tic = time.time()
                obj_val = tree_search(ip_instance, model.predictor, max_num_node = args.max_tree_size, encoder = encoder, p0 = args.threshold_prob_0, p1 = args.threshold_prob_1)
                toc = time.time()
            solver_time += (toc - tic)

            tic = time.time()
            _, opt_obj_val = solve_instance(ip_instance)
            toc = time.time()
            gurobi_time += (toc - tic)

            if np.abs(obj_val) < np.inf:
                solved += 1
                gap += abs(obj_val - opt_obj_val)
                gap_ratio += abs((obj_val - opt_obj_val)/opt_obj_val)
        
    print(f'Average running time: solver {solver_time / num_testing_instances:.3f} s, gurobi {gurobi_time / num_testing_instances:.3f} s.')
    print(f'Solved: {solved} out of {num_testing_instances}.')
    print(f'Average gap {gap / solved}, average normalized gap {gap_ratio / solved}.')
    