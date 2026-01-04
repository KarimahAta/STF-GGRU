import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn import GCN, nA
from istfa import ISTFA

class STF_GGRU(nn.Module):
    def __init__(self, f=3, h=64, k=8, w=0.5, dp=0.3,
                 f_t=1, f_s=1, f_i=1, f_k=1, f_c=1):
        super().__init__()

        self.f_t = int(f_t)   # temporal
        self.f_s = int(f_s)   # spatial
        self.f_i = int(f_i)   # ISTFA
        self.f_k = int(f_k)   # D-KNN
        self.f_c = int(f_c)   # CKA

        self.k = int(k)
        self.w = float(w)
        self.h = int(h)

        self.i = ISTFA(k=self.k, w=self.w)
        self.g = GCN(f, h, h)

        self.p = nn.Linear(f, h)          # proj for temporal path
        self.r = nn.GRU(h, h, batch_first=True)
        self.o = nn.Linear(h, 1)

        self.a = nn.Parameter(torch.tensor(0.5))  # gate
        self.dp = float(dp)

    def _A(self, x):
        # x: [B,N,F]
        if self.f_i:
            A = self.i(x, f_k=self.f_k, f_c=self.f_c)     # [N,N]
        else:
            N = x.size(1)
            A = torch.eye(N, device=x.device)
        return nA(A)

    def _T(self, x):
        # temporal encoding
        # x: [B,T,N,F] -> ht: [B,N,H]
        B, T, N, F0 = x.shape
        u = x.permute(0, 2, 1, 3).contiguous()            # [B,N,T,F]
        u = u.view(B * N, T, F0)                          # [B*N,T,F]
        u = self.p(u)                                     # [B*N,T,H]
        y, _ = self.r(u)                                  # [B*N,T,H]
        ht = y[:, -1, :].view(B, N, self.h)               # [B,N,H]
        return ht

    def _S(self, x, A):
        # spatial encoding
        # x: [B,N,F], A:[N,N] -> hs: [B,N,H]
        hs = self.g(x, A)                                 # [B,N,H]
        return hs

    def forward(self, x):
        # x: [B,T,N,F]
        B, T, N, F0 = x.shape

        x0 = x[:, -1, :, :]                               # [B,N,F]

        if self.f_t:
            ht = self._T(x)                               # [B,N,H]
        else:
            ht = self.p(x0)                               # [B,N,H]

        A = self._A(x0)                                   # [N,N]

        if self.f_s:
            hs = self._S(x0, A)                           # [B,N,H]
        else:
            hs = self.p(x0)                               # [B,N,H]

        g = torch.sigmoid(self.a)
        hq = g * ht + (1.0 - g) * hs                      # [B,N,H]

        if self.dp > 0:
            hq = F.dropout(hq, p=self.dp, training=self.training)

        y = self.o(hq).squeeze(-1)                        # [B,N]
        return y
