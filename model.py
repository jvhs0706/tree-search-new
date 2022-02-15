import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def dense_stack(*args, output_relu = True):
    seq = nn.Sequential()
    for i in range(1, len(args)):
        seq.add_module(f'dense {i-1}', nn.Linear(args[i-1], args[i]))
        if i < len(args) - 1 or output_relu:
            seq.add_module(f'relu {i-1}', nn.ReLU())
    return seq

class HalfConv(nn.Module):
    '''
    Aggregating information from V side to U side.
    
    input: a bipartite graph
    u: features of the nodes on the same side (U, F) 
    v: features of the nodes on the opposite side (V, G)
    e: featuers of of the edges (V, U, H)

    g_dims (F+G+H, ..., D)
    f_dims (F+D, ...)
    '''
    def __init__(self, u_dim: int, v_dim: int, e_dim: int, f_dims, g_dims, bn: bool):
        super().__init__()    
        self.g = dense_stack(u_dim + v_dim + e_dim, *g_dims)
        self.f = dense_stack(u_dim + g_dims[-1], *f_dims)
        if bn:
            self.bn = nn.BatchNorm1d(f_dims[-1])
        
    def forward(self, u, v, e):
        V, U, H = e.shape
        U, F = u.shape
        V, G = v.shape
        assert e.shape == (V, U, H)
        
        e = e.coalesce()
        e_indices, e_values = e.indices(), e.values()
        g_out = self.g(torch.cat([u[e_indices[1]], v[e_indices[0]], e_values], axis = -1)) # (nnz, D)
        g_out = torch.sparse.sum(torch.sparse_coo_tensor(e_indices, g_out, (V, U, g_out.shape[-1])), dim = -3) # (V, U, D) to (U, D)
        out = self.f(torch.cat([u, g_out.to_dense()], axis = -1))
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out

class Model(nn.Module):
    def __init__(self, v_dim: int, c_dim: int, e_dim: int, K: int, bn: bool = False):
        '''
        v_dim: dimension of variable feature vectors
        c_dim: dimension of constraint feature vectors
        e_dim: dimension of edge feature vectors
        K: number of probablity maps to predict
        '''
        super().__init__()
        self.initial_embedding_v = nn.Sequential(dense_stack(v_dim, 64, 64), nn.BatchNorm1d(64)) if bn else dense_stack(v_dim, 64, 64)
        self.initial_embedding_c = nn.Sequential(dense_stack(c_dim, 64, 64), nn.BatchNorm1d(64)) if bn else dense_stack(c_dim, 64, 64)

        self.c_side_convolution = HalfConv(64, 64, 1, (64, 64), (64, 64), bn)
        self.v_side_convolution = HalfConv(64, 64, 1, (64, 64), (64, 64), bn)

        self.tail = dense_stack(64, 64, K, output_relu=False)
    
    def forward(self, v, c, e):
        v1 = self.initial_embedding_v(v)
        c1 = self.initial_embedding_c(c)

        c2 = self.c_side_convolution(c1, v1, e.transpose(0, 1))
        v2 = self.v_side_convolution(v1, c2, e)
        return torch.sigmoid(self.tail(v2))
    
    def predictor(self, ip, encoder, p0: float, p1: float, mode = 'test'):
        proposals = []
        assert 0.0 <= p0 < p1 <= 1.0
        
        if mode == 'test':
            with torch.no_grad():
                # self.eval()
                V, C, E = encoder(ip)
                probs = self(V, C, E).cpu().numpy() # (n, K)
                n, K = probs.shape
                mask = np.logical_or(probs < p0, probs > p1)
                proposals = [(np.arange(n)[mask[:, k]], probs[mask[:, k], k] > (p1 + p0)/2) for k in range(K) if mask[:, k].sum() > 0]
                return proposals
        
        elif mode == 'valid':
            with torch.no_grad():
                # self.eval()
                V, C, E = encoder(ip)
                probs = self(V, C, E) # (n, K)
                n, K = probs.shape
                mask = torch.logical_or(probs < p0, probs > p1)
                proposals = [(torch.arange(n)[mask[:, k]], probs[mask[:, k], k] > (p1 + p0)/2) for k in range(K) if mask[:, k].sum() > 0]
                return probs, proposals
        
        elif mode == 'train':
            # self.train()
            V, C, E = encoder(ip)
            probs = self(V, C, E) # (n, K)
            with torch.no_grad():
                n, K = probs.shape
                mask = torch.logical_or(probs < p0, probs > p1)
                proposals = [(torch.arange(n)[mask[:, k]], probs[mask[:, k], k] > (p1 + p0)/2) for k in range(K) if mask[:, k].sum() > 0]
            return probs, proposals
        
        else:
            raise TypeError

    def predictor_batch(self, ips, encoder, p0: float, p1: float, mode = 'test'):
        proposals_batch = []
        assert 0.0 <= p0 < p1 <= 1.0
        if mode == 'test':
            with torch.no_grad():
                # self.eval()
                V, C, E = encoder.encode_batch(ips)
                out = self(V, C, E).cpu().numpy()
                _, K = out.shape
                indices = np.cumsum([0] + [ip.NumVars for ip in ips])
                mask = np.logical_or(out < p0, out > p1)
                for i, ip in enumerate(ips):
                    _out, _mask = out[indices[i]: indices[i+1]], mask[indices[i]: indices[i+1]]
                    proposals_batch.append([(np.arange(ips[i].numVars)[_mask[:, k]], (_out[_mask[:, k], k] > (p1 + p0)/2)) for k in range(K) if _mask[:, k].sum() > 0])
                return proposals_batch
        
        elif mode == 'valid':
            with torch.no_grad():
                # self.eval()
                V, C, E = encoder.encode_batch(ips)
                out = self(V, C, E)
                _, K = out.shape
                indices = np.cumsum([0] + [ip.NumVars for ip in ips])
                mask = torch.logical_or(out < p0, out > p1)
                probs = []
                for i, ip in enumerate(ips):
                    _out, _mask = out[indices[i]: indices[i+1]], mask[indices[i]: indices[i+1]]
                    proposals_batch.append([(torch.arange(ips[i].numVars)[_mask[:, k]], _out[_mask[:, k], k] > (p1 + p0)/2) for k in range(K) if _mask[:, k].sum() > 0])
                    probs.append(_out)
                return out, probs, proposals_batch

        elif mode == 'train':
            # self.train()
            V, C, E = encoder.encode_batch(ips)
            out = self(V, C, E)
            with torch.no_grad():
                _, K = out.shape
                indices = np.cumsum([0] + [ip.NumVars for ip in ips])
                mask = torch.logical_or(out < p0, out > p1)
                probs = []
                for i, ip in enumerate(ips):
                    _out, _mask = out[indices[i]: indices[i+1]], mask[indices[i]: indices[i+1]]
                    proposals_batch.append([(torch.arange(ips[i].numVars)[_mask[:, k]], _out[_mask[:, k], k] > (p1 + p0)/2) for k in range(K) if _mask[:, k].sum() > 0])
                    probs.append(_out)
            return out, probs, proposals_batch
        