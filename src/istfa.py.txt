import torch
import torch.nn as nn
import torch.nn.functional as F

class ISTFA(nn.Module):
    def __init__(self, k=8, w=0.5, e=1e-8):
        super().__init__()
        self.k = int(k)
        self.w = float(w)
        self.e = e

    def _knn(self, z):
        # z: [N,F]
        d = torch.cdist(z, z)                     # [N,N]
        _, ix = torch.topk(-d, self.k, dim=-1)    # [N,k]
        A = torch.zeros_like(d)
        A.scatter_(1, ix, 1.0)
        A.fill_diagonal_(1.0)
        A = A / (A.sum(dim=-1, keepdim=True) + self.e)
        return A

    def _cka(self, z):
        # z: [N,F]
        z = z - z.mean(dim=0, keepdim=True)
        K = z @ z.t()                              # [N,N]
        K = K / (torch.norm(K) + self.e)
        # center (approx)
        K = K - K.mean(dim=0, keepdim=True)
        K = K - K.mean(dim=1, keepdim=True)
        # scale to row-stochastic
        K = K / (K.abs().sum(dim=-1, keepdim=True) + self.e)
        K = K + torch.eye(K.size(0), device=K.device) * 0.01
        K = K / (K.sum(dim=-1, keepdim=True) + self.e)
        return K

    def forward(self, x, f_k=1, f_c=1):
        # x: [B,N,F]  (we use mean over batch to get stable A)
        z = x.mean(dim=0)                          # [N,F]

        A1 = self._knn(z) if f_k else None
        A2 = self._cka(z) if f_c else None

        if (A1 is None) and (A2 is None):
            N = z.size(0)
            A = torch.eye(N, device=z.device)
        elif A1 is None:
            A = A2
        elif A2 is None:
            A = A1
        else:
            A = self.w * A1 + (1.0 - self.w) * A2

        A = F.normalize(A, p=1, dim=-1)
        return A
