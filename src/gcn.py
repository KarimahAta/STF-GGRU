import torch
import torch.nn as nn
import torch.nn.functional as F

class G0(nn.Module):
    def __init__(self, i0, o0, a0=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(i0, o0))
        self.b = nn.Parameter(torch.zeros(o0))
        self.a = a0

    def forward(self, x, A):
        # x: [B, N, F]
        # A: [N, N]
        h = torch.einsum("bnf,fo->bno", x, self.w) + self.b
        y = torch.einsum("nm,bmf->bnf", A, h)
        if self.a:
            y = F.relu(y)
        return y

class GCN(nn.Module):
    def __init__(self, f0, f1, f2):
        super().__init__()
        self.g1 = G0(f0, f1)
        self.g2 = G0(f1, f2, a0=False)

    def forward(self, x, A):
        x = self.g1(x, A)
        x = self.g2(x, A)
        return x

def nA(A, e=1e-6):
    # normalized adjacency
    d = torch.sum(A, dim=1)
    d = torch.pow(d + e, -0.5)
    D = torch.diag(d)
    return D @ A @ D
