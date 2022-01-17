import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import scipy.sparse
import scipy.sparse.linalg

import gurobipy as gp 
from gurobipy import GRB

class BinaryIPEncoder:
    '''
    All avaiable encoding features
        variable encoding ('v_'): 
            coef: Objective coefficient, normalized to [-1, 1].
            lp_sol: LP solution value fractionality.
            lp_sol_is_at_lb: LP solution value equals lower bound.
            lp_sol_is_at_ub: LP solution value equals upper bound.

        constraint encoding ('c_'):
            obj_cos_sim: Cosine similarity with objective.
            type: {'>': -1, '=': 0, '<': 1}.
            bias: Bias value, normalized with constraint coefficients.
            lp_is_tight: Tightness indicator in LP solution.
            lp_dualsol: Dual solution value, normalized.
            lp_slack: Dual slack in LP solution.

        edge encoding ('e_'):
            coef: Constraint coefficient.
    '''

    variable_features = ['coef', 'lp_sol', 'lp_sol_is_at_lb', 'lp_sol_is_at_ub']
    constraint_features = ['obj_cos_sim', 'type', 'bias', 'lp_is_tight', 'lp_dualsol', 'lp_slack']
    edge_features = ['coef'] 

    constr_sense_encoding = {'>': -1, '=': 0, '<': 1}

    @staticmethod
    def _lp_relax(ip_model):
        lp_model = ip_model.copy()
        lp_model.setParam('OutputFlag', 0)
        max_obj_coef = max([var.Obj for var in lp_model.getVars()])
        for var in lp_model.getVars():
            var.VType, var.LB, var.UB = 'C', 0.0, 1.0
            var.Obj = var.Obj / max_obj_coef
        lp_model.optimize()
        return lp_model

    @staticmethod
    def _get_topology(ip_model):
        coo_mat = ip_model.getA().tocoo()
        return np.stack([coo_mat.row, coo_mat.col]), coo_mat.shape        
    
    @staticmethod
    def encode_v_coef(ip):
        arr = np.array([var.Obj for var in ip.getVars()])
        return arr / arr.max()

    @staticmethod
    def encode_v_lp_sol(lp):
        arr = np.array([var.X for var in lp.getVars()])
        return arr

    @staticmethod
    def encode_v_lp_sol_is_at_lb(lp, epsilon = 1e-8):
        arr = np.array([var.X < epsilon for var in lp.getVars()])
        return arr

    @staticmethod
    def encode_v_lp_sol_is_at_ub(lp, epsilon = 1e-8):
        arr = np.array([var.X > 1-epsilon for var in lp.getVars()])
        return arr

    @staticmethod
    def encode_c_obj_cos_sim(ip):
        A = ip.getA()
        c = np.array([var.Obj for var in ip.getVars()])
        arr = (A @ c) / (scipy.sparse.linalg.norm(A, ord = 2, axis = 1) * np.linalg.norm(c, ord = 2))
        return arr

    @staticmethod
    def encode_c_type(ip):
        arr = np.array([BinaryIPEncoder.constr_sense_encoding[constr.Sense] for constr in ip.getConstrs()])
        return arr

    @staticmethod
    def encode_c_bias(ip):
        arr = np.array([constr.RHS for constr in ip.getConstrs()])
        return arr

    @staticmethod
    def encode_c_lp_is_tight(lp, epsilon = 1e-8):
        arr = np.array([np.abs(constr.Slack) < epsilon for constr in lp.getConstrs()])
        return arr

    @staticmethod
    def encode_c_lp_dualsol(lp):
        arr = np.array([constr.Pi for constr in lp.getConstrs()])
        return arr

    @staticmethod
    def encode_c_lp_slack(lp):
        arr = np.array([constr.Slack if constr.Sense != '=' else 0.0 for constr in lp.getConstrs()])
        return arr

    @staticmethod
    def encode_e_coef(ip):
        arr = ip.getA().tocoo().data
        return arr

    def __init__(self, *args, epsilon = 1e-8):
        '''
        Usage: BinaryIPEncoder('v_feat1', 'v_feat2', ...)
        '''
    
        self.variable_feats = [feat for feat in BinaryIPEncoder.variable_features if f'v_{feat}' in args]
        self.constraint_feats = [feat for feat in BinaryIPEncoder.constraint_features if f'c_{feat}' in args]
        self.edge_feats = [feat for feat in BinaryIPEncoder.edge_features if f'e_{feat}' in args]        

    def __call__(self, ip):
        lp = BinaryIPEncoder._lp_relax(ip)
        edge_index, prob_size = BinaryIPEncoder._get_topology(ip)

        V = torch.tensor(np.stack([getattr(BinaryIPEncoder, f'encode_v_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.variable_feats], axis = -1), dtype = torch.float)
        C = torch.tensor(np.stack([getattr(BinaryIPEncoder, f'encode_c_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.constraint_feats], axis = -1), dtype = torch.float)
        E = torch.sparse_coo_tensor(edge_index, np.stack([getattr(BinaryIPEncoder, f'encode_e_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.edge_feats], -1), (*prob_size, len(self.edge_features)), dtype = torch.float)
        return V, C, E

    def encode_batch(self, ips):
        V, C, E_indices, E_values = [], [], [], []
        num_constr, num_var = 0, 0
        for ip in ips:
            lp = BinaryIPEncoder._lp_relax(ip)
            edge_index, prob_size = BinaryIPEncoder._get_topology(ip)
            m, n = prob_size

            V.append(np.stack([getattr(BinaryIPEncoder, f'encode_v_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.variable_feats], axis = -1)) 
            C.append(np.stack([getattr(BinaryIPEncoder, f'encode_c_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.constraint_feats], axis = -1))
            E_indices.append(edge_index + np.array([[num_constr], [num_var]]))
            E_values.append(np.stack([getattr(BinaryIPEncoder, f'encode_e_{feat}')(lp if feat[:2] == 'lp' else ip) for feat in self.edge_feats], -1))
            
            num_constr, num_var = num_constr + m, num_var + n
    
        V = torch.tensor(np.concatenate(V, axis = 0), dtype = torch.float)
        C = torch.tensor(np.concatenate(C, axis = 0), dtype = torch.float)
        E = torch.sparse_coo_tensor(np.concatenate(E_indices, axis = 1), np.concatenate(E_values, axis = 0), (num_constr, num_var, len(self.edge_features)), dtype = torch.float)
        return V, C, E