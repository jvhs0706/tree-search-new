import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def _dense_stack(*args, output_relu = True):
    seq = nn.Sequential()
    for i in range(1, len(args)):
        seq.add_module(f'dense {i-1}', nn.Linear(args[i-1], args[i]))
        if i < len(args) - 1 or output_relu:
            seq.add_module(f'relu {i-1}', nn.ReLU())
    return seq

class _HalfConvolution(nn.Module):
    '''
    input: a bipartite graph
    u: features of the nodes on the same side (U, F) 
    v: features of the nodes on the opposite side (V, G)
    e: featuers of of the edges (V, U, H)

    g_args (F+G+H, ..., D)
    f_args (F+D, ...)
    '''
    def __init__(self, *, f_args, g_args):
        super().__init__()
        self.g = _dense_stack(*g_args)
        self.f = _dense_stack(*f_args)
        
    def forward(self, u, v, e):
        U, F = u.shape
        V, G = v.shape
        _, _, H = e.shape
        assert e.shape == (V, U, H)
        
        e = e.coalesce()
        e_indices, e_values = e.indices(), e.values()
        g_out = self.g(torch.cat([u[e_indices[1]], v[e_indices[0]], e_values], axis = -1)) # (nnz, D)
        _, D = g_out.shape 
        g_out = torch.sparse.sum(torch.sparse_coo_tensor(e_indices, g_out, (V, U, D)), dim = -3) # (V, U, D) to (U, D)
        out = self.f(torch.cat([u, g_out.to_dense()], axis = -1))
        return out

class Model(nn.Module):
    def __init__(self, *, v_feats: int, c_feats: int, e_feats: list, g_hidden_neurons: list, f_hidden_neurons: list, out_neurons: list):
        super().__init__()
        self.half_conv = _HalfConvolution(
            g_args = [v_feats+c_feats+e_feats] + g_hidden_neurons,
            f_args = [v_feats + g_hidden_neurons[-1]] + f_hidden_neurons
        )
        self.tail = _dense_stack(f_hidden_neurons[-1], *out_neurons, output_relu = False)
    
    def forward(self, u, v, e):
        out = self.half_conv(u, v, e)  
        return F.sigmoid(self.tail(out))
    
    def predictor(self, ip, encoder, p0: float, p1: float):
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
        
        return proposals