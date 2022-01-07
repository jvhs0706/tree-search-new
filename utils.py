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

def tree_search(ip, predictor, max_num_node: int, *predictor_args, **predictor_kwargs):
    '''
    predictor(ip, *predictor_args, **predictor_kwargs) -> [(indicies_1, values_1), (indicies_2, values_2), ..., (indicies_M, values_M)]
    max_num_node: maximum number of nodes explored
    '''
    ip.setParam('OutputFlag', 0)
    
    root_node = assign_values(ip, [], [])
    if type(root_node) == gp.Model:
        nodes = deque([root_node])
        best_val = np.inf # minimal ip.ModelSense * objval
        for i in range(max_num_node):
            if nodes:
                qip = nodes.popleft()
                proposals = predictor(qip, *predictor_args, **predictor_kwargs)
                for (indicies, values) in proposals:
                    new_ip = assign_values(qip, indicies, values)
                    if new_ip is None:# Not feasible
                        pass
                    elif type(new_ip) == gp.Model:
                        nodes.append(new_ip)
                    else:
                        best_val = min(best_val, ip.ModelSense * new_ip)
            else:
                break
        return best_val * ip.ModelSense
    
    elif root_node is None:
        return ip.ModelSense * np.inf
    
    else:
        return root_node 

def solve_instance(ip):
    ip.setParam('OutputFlag', 0)
    ip.optimize()
    variables = ip.getVars()
    return np.array([var.X for var in variables]).round().astype(bool), ip.ObjVal

def tree_search_train(ip, predictor, max_num_node: int, *predictor_args, **predictor_kwargs):
    '''
    predictor(ip, *predictor_args, **predictor_kwargs) -> [(indicies_1, values_1), (indicies_2, values_2), ..., (indicies_M, values_M)]
    max_num_node: maximum number of nodes explored
    '''

    ip.setParam('OutputFlag', 0)

    best_val = np.inf # minimal ip.ModelSense * objval
    training_set = []
    
    root_node = assign_values(ip, [], [])
    if type(root_node) == gp.Model:
        nodes = deque([root_node])
        for i in range(max_num_node):
            if nodes:
                qip = nodes.popleft()
                qip.setParam('OutputFlag', 0)
                try:
                    opt_sol, _ = solve_instance(qip)
                    training_set.append((qip, opt_sol))
                    proposals = predictor(qip, *predictor_args, **predictor_kwargs)
                    for (indicies, values) in proposals:
                        new_ip = assign_values(qip, indicies, values)
                        if type(new_ip) == gp.Model:
                            nodes.append(new_ip)
                except AttributeError as e:
                    pass
            else:
                break

    return training_set