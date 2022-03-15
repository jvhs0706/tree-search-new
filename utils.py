import gurobipy as gp 
from gurobipy import GRB

import numpy as np

from collections import deque

def load_instance(filename):
    ip = gp.read(filename)
    return ip

def assign_values(ip, indicies, values):
    '''
    ip: gp.Model, the model
    indicies: iterable, the indicies that will be assigned values
    values: bool or iterable, values that the corresponding indicies will be assigned
    '''
    ip_copy = ip.copy()
    variables = ip_copy.getVars()
    
    if type(values) == bool:
        for idx in indicies:
            ip_copy.addConstr(variables[idx] == values)
    elif len(indicies) == len(values):
        for idx, val in zip(indicies, values):
            ip_copy.addConstr(variables[idx] == bool(val))
    else:
        raise TypeError
    ip_copy.setParam('OutputFlag', 0)
    
    try:
        presolved = ip_copy.presolve()
        if presolved.NumVars == 0:
            ip_copy.optimize()
            return ip_copy.ObjVal
        elif presolved.NumVars == presolved.NumBinVars:
            return presolved
        else:
            raise TypeError
    except gp.GurobiError as e:
        assert str(e) == 'Unable to create presolved model', str(e)

def tree_search(ip, predictor, max_num_node, random_pop = False, *predictor_args, **predictor_kwargs):
    '''
    predictor(ip, *predictor_args, **predictor_kwargs) -> [(indicies_1, values_1), (indicies_2, values_2), ..., (indicies_M, values_M)]
    max_num_node: maximum number of nodes explored
    '''
    
    root_node = assign_values(ip, [], [])
    if type(root_node) == gp.Model:
        nodes = [root_node] if random_pop else deque([root_node])
        best_val, num_node = np.inf, 0 # minimal ip.ModelSense * objval
        while nodes:
            qip = nodes.pop(np.random.randint(len(nodes))) if random_pop else nodes.popleft()
            proposals = predictor(qip, *predictor_args, **predictor_kwargs)
            num_node += 1
            for (indicies, values) in proposals:
                new_ip = assign_values(qip, indicies, values)
                if new_ip is None:# Not feasible
                    pass
                elif type(new_ip) == gp.Model:
                    nodes.append(new_ip)
                else:
                    best_val = min(best_val, ip.ModelSense * new_ip)
            if num_node >= max_num_node:
                break
        return best_val * ip.ModelSense
    
    elif root_node is None:
        return ip.ModelSense * np.inf
    
    else:
        return root_node 

def tree_search_accelerated(ip, batch_predictor, max_tree_height, max_num_node, *predictor_args, **predictor_kwargs):
    root_node = assign_values(ip, [], [])
    tree_height, num_node = 0, 0
    if type(root_node) == gp.Model:
        
        nodes = [root_node]
        best_val = np.inf # minimal ip.ModelSense * objval
        while nodes:
            new_nodes = []
            proposals_batch = batch_predictor(nodes, *predictor_args, **predictor_kwargs)
            for proposals, ip in zip(proposals_batch, nodes):
                for (indicies, values) in proposals:
                    new_ip = assign_values(ip, indicies, values)
                    if new_ip is None: # Not feasible
                        pass
                    elif type(new_ip) == gp.Model:
                        new_nodes.append(new_ip)
                    else:
                        best_val = min(best_val, ip.ModelSense * new_ip)
            tree_height, num_node = tree_height + 1, num_node + len(nodes)
            nodes = new_nodes
            if tree_height >= max_tree_height or num_node >= max_num_node:
                break
            
        return best_val * ip.ModelSense
    
    elif root_node is None:
        return ip.ModelSense * np.inf
    
    else:
        return root_node 

def solve_instance(ip, **kwargs):
    for p, val in kwargs.items():
        ip.setParam(p, val)
    ip.optimize()
    variables = ip.getVars()
    return np.array([var.X for var in variables]).round().astype(bool), ip.ObjVal

class DatasetSplitter:
    '''
    Split the training steps and testing steps
    It does not remember anything
    '''
    def __init__(self, num_train: int, num_valid: int):
        assert num_train >= 0 and num_valid >= 0
        self.num_train, self.num_valid = num_train, num_valid

    def __call__(self):
        if np.random.rand() < self.num_train / (self.num_train + self.num_valid):
            self.num_train -= 1
            return 'train'
        else:
            self.num_valid -= 1
            return 'valid' 

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.num_train + self.num_valid > 0:
            return self()
        else:
            raise StopIteration

    def __len__(self):
        '''
        How many steps are ``left``
        '''
        return self.num_train + self.num_valid
