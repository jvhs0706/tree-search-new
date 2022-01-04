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
        'p0',
        help = 'Threshold value for predicting 0.',
        type = float
    )
    parser.add_argument(
        'p1',
        help = 'Threshold value for predicting 1.',
        type = float
    )
    args = parser.parse_args()
    
    data_dir = f'data/instances/{args.problem}'
    test_dir = '_'.join([f'{data_dir}/test'] + args.problem_params)

    num_testing_instances = len(os.listdir(test_dir))

    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    with open(model_dir +'/model_config.json', 'r') as f:
        model_config = json.load(f)
    model = Model(v_dim=len(model_config['variable_features']), c_dim=len(model_config['constraint_features']), 
        e_dim=len(model_config['edge_features']), K = model_config['num_prob_map'])
    model.load_state_dict(torch.load(model_dir + '/model_state_dict'))
    encoder = BinaryIPEncoder(*['v_'+feat for feat in model_config['variable_features']], *['c_'+feat for feat in model_config['constraint_features']], *['e_'+feat for feat in model_config['edge_features']])    
    
    with torch.no_grad():
        for i in range(num_testing_instances):
            ip_instance = load_instance(f'{test_dir}/instance_{i+1}.lp')
            obj_val = tree_search(ip_instance, model.predictor, encoder = encoder, p0 = args.p0, p1 = args.p1)
            _, opt_obj_val = solve_instance(ip_instance)
            print(f'Instance {i}, solution objective {obj_val}, optimal objective {opt_obj_val}.')
        
    