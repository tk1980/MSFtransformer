import torch
from torch import nn
from einops import rearrange

# Grouped Projection
class GroupLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, groups=1, mode='interleave'):
        super(GroupLinear, self).__init__(in_features//groups, out_features, bias)
        self.groups = groups
        self.einsum = 'bngd,ged->bnge' if mode == 'block' else 'bngd,ged->bneg'

    def forward(self, input):
        b, n, _ = input.shape # 3-dims
        g, d, e = self.groups, self.in_features, self.out_features
        X = torch.einsum(self.einsum, input.reshape(b,n,g,d), self.weight.reshape(g,e//g,d))
        if self.bias is not None:
            return X.reshape(b,n,-1) + self.bias
        else:
            return X.reshape(b,n,-1)


#Transformer attention block
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, groups = 1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        if groups == 1:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        else:
            self.to_qkv = GroupLinear(dim, inner_dim * 3, bias = False, groups=groups)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AttentionMSF(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, groups=1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        if groups == 1:
            self.to_qkvp = nn.Linear(dim, inner_dim * 4, bias = False)
        else:
            self.to_qkvp = GroupLinear(dim, inner_dim * 4, bias = False, groups=groups)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkvp = self.to_qkvp(x).chunk(4, dim = -1)
        q, k, v, p = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkvp)

        # Gaussian kernel
        dist = (torch.matmul(q, k.transpose(-1, -2)) - 0.5 * k.pow(2).sum(-1).unsqueeze(-2)) * self.scale
        
        attn = self.attend(dist)

        out = torch.matmul(attn, v) - p
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



def get_attention_module(att_name, nc, heads, dim_head):
    att = att_name.split('-')
    groups = int(att[1][1:]) if len(att) > 1 and att[1].startswith('g') else 1
    if att[0] == 'orig':
        return Attention(nc, heads, dim_head, groups = groups)
    elif att[0] == 'msf':
        return AttentionMSF(nc, heads, dim_head, groups = groups)