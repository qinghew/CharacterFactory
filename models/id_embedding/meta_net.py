import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from einops import rearrange
import torch.nn.init as init

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class Residual(nn.Module):
    def __init__(self,
                 fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True, pre_norm=False, activate = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.activate = activate
        if self.activate == True:
            self.non_linear = leaky_relu()

    def forward(self, input):
        if hasattr(self, 'pre_norm') and self.pre_norm:
            out = self.norm(input)
            out = F.linear(out, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        
        if self.activate == True:
            out = self.non_linear(out)
        return out


class StyleVectorizer(nn.Module):
    def __init__(self, dim_in, dim_out, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            if i == 0: 
                layers.extend([EqualLinear(dim_in, dim_out, lr_mul, pre_norm=False, activate = True)]) 
            elif i == depth - 1:
                layers.extend([EqualLinear(dim_out, dim_out, lr_mul, pre_norm=True, activate = False)]) 
            else:
                layers.extend([Residual(EqualLinear(dim_out, dim_out, lr_mul, pre_norm=True, activate = True))])

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim_out, eps=1e-5)
        
    def forward(self, x):
        return self.norm(self.net(x))

