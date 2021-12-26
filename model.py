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

    g_args (F+G+H, ..., D)
    f_args (F+D, ...)
    '''
    def __init__(self, u_dim: int, v_dim: int, e_dim: int, f_dims, g_dims):
        super().__init__()    
        self.g = dense_stack(u_dim + v_dim + e_dim, *g_dims)
        self.f = dense_stack(u_dim + g_dims[-1], *f_dims)
        
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
        return out

class Model(nn.Module):
    def __init__(self, v_dim: int, c_dim: int, e_dim: int, K: int):
        '''
        v_dim: dimension of variable feature vectors
        c_dim: dimension of constraint feature vectors
        e_dim: dimension of edge feature vectors
        K: number of probablity maps to predict
        '''
        super().__init__()
        self.initial_embedding_v = dense_stack(v_dim, 64, 64)
        self.initial_embedding_v_bn = nn.BatchNorm1d(64)
        self.initial_embedding_c = dense_stack(c_dim, 64, 64)
        self.initial_embedding_c_bn = nn.BatchNorm1d(64)

        self.c_side_convolution = HalfConv(64, 64, 1, (64, 64), (64, 64))
        self.c_side_convolution_bn = nn.BatchNorm1d(64)
        self.v_side_convolution = HalfConv(64, 64, 1, (64, 64), (64, 64))
        self.v_side_convolution_bn = nn.BatchNorm1d(64)

        self.tail = dense_stack(64, 64, K, output_relu=False)
    
    def forward(self, v, c, e):
        v1 = self.initial_embedding_v(v)
        v1 = self.initial_embedding_v_bn(v1)
        c1 = self.initial_embedding_c(c)
        c1 = self.initial_embedding_c_bn(c1)

        c2 = self.c_side_convolution(c1, v1, e.transpose(0, 1))
        c2 = self.c_side_convolution_bn(c2)
        v2 = self.v_side_convolution(v1, c2, e)
        v2 = self.v_side_convolution_bn(v2)
        return torch.sigmoid(self.tail(v2))
    
    def predictor(self, ip, encoder, p0: float, p1: float):
        with torch.no_grad():
            self.eval()
            assert 0.0 <= p0 < p1 <= 1.0
            V, C, E = encoder(ip)
            out = self(V, C, E) # (n, K)
            proposals = []
            n, K = out.shape
            for k in range(K):
                probs = out[:, k].detach().numpy()
                mask = np.logical_or(probs < p0, probs > p1)
                if mask.sum() > 0:
                    proposals.append((np.arange(n)[mask], probs[mask] > (p1+p0)/2))
            self.train()
            return proposals