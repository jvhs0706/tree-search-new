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
        '-s', '--source_model_params',
        help = 'Parameters of the source model',
        nargs = '*'
    )
    parser.add_argument(
        '-n', '--num_transfer', 
        help = 'Number of transfer learning steps.',
        type = int,
        required = True
    )
    parser.add_argument(
        '-c', '--components',
        help = 'Components of the model that are NOT frozen.',
        nargs = '*',
        default = ['tail']
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
        help = 'Maximum number of nodes.',
        type = lambda z: int(z) if z is not None else np.inf,
        default = np.inf
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
        default = 32
    )
    parser.add_argument(
        '-g', '--gamma', 
        help = 'The gamma value of the StepLR lr_scheduler.',
        type = float,
        default = 0.5
    )
    parser.add_argument(
        '-d', '--device',
        help = 'The device for pytorch.',
        default = 'cuda' if torch.cuda.is_available() else 'cpu',
        type = torch.device
    )
    args = parser.parse_args()
    
    # data directories
    data_dir = f'data/instances/{args.problem}'
    transfer_dir = '_'.join([f'{data_dir}/transfer'] + args.problem_params)

    # number of training instances
    num_transfer_instances = len(os.listdir(transfer_dir))

    # initialize the model, the encoder, the optimizer, the scheduler, and the loss function
    source_model_dir = f'models/'+ '_'.join([args.problem] + args.source_model_params)
    with open(f'{source_model_dir}/model_config.json') as f:
        sm_config = json.load(f)
    model = Model(v_dim=len(sm_config['variable_features']), c_dim=len(sm_config['constraint_features']), e_dim=len(sm_config['edge_features']), K = sm_config['num_prob_map'], bn = sm_config['batch_norm']).to(args.device)
    model.load_state_dict(torch.load(source_model_dir + '/model_state_dict'))

    for cname, component in model.named_children():
        if cname not in args.components:
            component.eval()
            for p in component.parameters():
                p.requires_grad = False
    
    encoder = BinaryIPEncoder(*['v_'+feat for feat in sm_config['variable_features']], *['c_'+feat for feat in sm_config['constraint_features']], *['e_'+feat for feat in sm_config['edge_features']], device = args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    Loss = torch.nn.BCELoss(reduction = 'none')

    # the history of loss, the tree height, and the tree size will be kept
    loss_hist, num_node_hist = [], []
    transfer_best_map_hist = np.array([0] * sm_config['num_prob_map'])

    print(f'Using device {args.device}...')
    
    for i in range(args.num_transfer):
        idx = np.random.randint(num_transfer_instances)
        ip_instance = load_instance(f'{transfer_dir}/instance_{idx+1}.lp') 

        num_node, num_var = 0, 0
        root_node = assign_values(ip_instance, [], [])

        if type(root_node) == gp.Model:
            nodes = deque([root_node])
            loss = torch.tensor(0.0)

            while nodes:
                qip = nodes.popleft()
                try:
                    # compute the loss for one node
                    out, proposals = model.predictor(qip, encoder, p0 = args.threshold_prob_0, p1 = args.threshold_prob_1, mode = 'train')
                    opt_sol, _ = solve_instance(qip, OutputFlag = 0)
                    _loss, _index = Loss(input = out, target = torch.tensor(opt_sol, dtype = torch.float, device = args.device).unsqueeze(-1).expand(*out.shape)).sum(axis = 0).min(dim = 0)
                    transfer_best_map_hist[_index.item()] += 1
                    loss, num_node, num_var = loss + _loss, num_node + 1, num_var + qip.NumVars
                    
                    # generate the child nodes
                    for proposal in proposals:
                        new_ip = assign_values(qip, *proposal)
                        if type(new_ip) == gp.Model:
                            nodes.append(new_ip)
                
                except AttributeError as e:
                    pass

                # break if necessary
                if num_node >= args.max_num_node:
                    break

            # compute the overall loss
            loss = loss / num_var
            
            # do backprop 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # update the history
            loss_hist.append(loss.item()), num_node_hist.append(num_node)
            print(f'Step {i} (transfer), loss {loss:.3f}, num_node {num_node}.')
    
    # save the model and the configurations
    model_dir = f'models/'+ '_'.join([args.problem] + args.problem_params)
    os.makedirs(model_dir, exist_ok = True)
    model_config = {
        'variable_features': sm_config['variable_features'],
        'constraint_features': sm_config['constraint_features'],
        'edge_features': sm_config['edge_features'],
        'num_prob_map': sm_config['num_prob_map'], 
        'batch_norm': sm_config['batch_norm']
    }

    with open(model_dir +'/model_config.json', 'w') as f:
        json.dump(model_config, f)

    training_config = {
        'num_step': args.num_transfer,
        'learning_rate': args.learning_rate,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'threshold_prob_0': args.threshold_prob_0,
        'threshold_prob_1': args.threshold_prob_1,
        'accelerated': False
    }
    with open(model_dir +'/training_config.json', 'w') as f:
        json.dump(training_config, f)

    torch.save(model.state_dict(), model_dir + '/model_state_dict')

    # summarize the history of loss and the tree size
    loss_hist, num_node_hist = np.array(loss_hist), np.array(num_node_hist)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title('Loss history')
    axes[0].plot(loss_hist, label = 'transfer')
    axes[0].legend()

    axes[1].set_title('Number of nodes history')
    axes[1].plot(num_node_hist, label = 'transfer')
    axes[1].legend()
    
    fig.savefig(model_dir + '/training_hist.png')
    
    # summarize the history of best maps
    fig, ax = plt.subplots(constrained_layout=True)

    plt.title('transfer')

    ax.set_title('transfer')
    ax.pie(transfer_best_map_hist, labels = np.arange(len(transfer_best_map_hist)))
    ax.legend()

    fig.savefig(model_dir + '/best_map_hist.png')