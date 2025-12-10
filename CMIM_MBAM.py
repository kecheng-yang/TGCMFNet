import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch import nn, einsum
import math


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, l, **kwargs):
        x_, l_ = self.fn(x, l, **kwargs)
        return x + x_, l + l_


class Layernorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, l, **kwargs):
        return self.fn(self.norm1(x), self.norm2(l), **kwargs)


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, l):
        return self.net1(x), self.net2(l)


class Atten(nn.Module):
    def __init__(self, num_atte, heads, cross=True) -> None:
        super().__init__()

        if cross == True:
            in_channels = 2
        else:
            in_channels = 1

        self.heads = heads
        self.layers = nn.ModuleList([])

        if num_atte != 1:
            for i in range(heads):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, num_atte, 1),
                        nn.BatchNorm2d(num_atte),
                        nn.LeakyReLU(),
                        nn.Conv2d(num_atte, 1, 1)
                    ))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x[:, i])
            else:
                out = torch.cat([out, layer(x[:, i])], 1)
        return out


def cc(img1, img2):
    N, C, _, _ = img1.shape

    KLloss = torch.nn.KLDivLoss(reduction="batchmean")
    img1 = img1.reshape(N, -1)
    img2 = img2.reshape(N, -1)
    img1 = F.log_softmax(img1, dim=1)
    img2 = F.softmax(img2, dim=1)
    return KLloss(img1, img2)


class Encoder(nn.Module):
    def __init__(self, dim, head_dim, heads, num_atte, dropout=0.1, cross=True):
        super().__init__()

        self.dim = dim
        self.cross = cross

        self.scale = (head_dim / heads) ** -0.5  # 1/sqrt(dim)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, 3 * head_dim)
        self.to_qkv1 = nn.Linear(dim, 3 * head_dim)

        self.to_cls_token = nn.Identity()

        self.mlp = nn.Linear(dim, dim)
        self.mlp1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        if self.cross == True:
            self.atte = Atten(num_atte, heads)
        else:
            self.atte_h = Atten(num_atte, heads, cross=cross)
            self.atte_l = Atten(num_atte, heads, cross=cross)

    def forward(self, x, l, mask):
        b, n, _, h = *x.shape, self.heads
        p_size = int(math.sqrt(n - 1))

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [q, k, v])  # split into multi head attentions

        q1, k1, v1 = self.to_qkv1(l).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                         [q1, k1, v1])  # split into multi head attentions

        if self.cross == True:
            # A融合
            dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)
            dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)

            sup = torch.stack([dots, dots1], 2)
            sup = self.atte(sup)
            dots = (dots + sup)
            dots1 = (dots1 + sup)

        else:
            dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)
            dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)

            sup = dots.unsqueeze(2)
            # sup = torch.stack([dots, dots], 2)
            sup1 = dots1.unsqueeze(2)
            # sup1 = torch.stack([dots1, dots1], 2)
            sup = self.atte_h(sup)
            sup1 = self.atte_l(sup1)
            dots = (dots + sup)
            dots1 = (dots1 + sup1)

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        attn1 = dots1.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)  # product of v times whatever inside softmax

        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out1 = rearrange(out1, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.mlp(out)
        out1 = self.mlp1(out1)

        out = self.dropout(out)
        out1 = self.dropout1(out1)

        return out, out1


class CMIM_MBAM(nn.Module):
    def __init__(self, dim=32, hidden_dim=8, head_dim=64, heads=8, num_atte=8, depth=4, dropout=0.1):
        super().__init__()

        self.depth = depth
        self.CMIM = nn.ModuleList([])

        for i in range(int(depth)):
            self.CMIM.append(nn.ModuleList([
                Residual(Layernorm(dim, Encoder(dim, head_dim, heads, num_atte, dropout, cross=True))),
                # Residual(Layernorm(dim,Feedforward(dim,hidden_dim,dropout))),

                Residual(Layernorm(dim, Encoder(dim, head_dim, heads, num_atte, dropout, cross=False))),
                Residual(Layernorm(dim, Feedforward(dim, hidden_dim, dropout))),
            ]))

        self.MBAM = Modality_Bridging_Alignment_Module()

    def forward(self, x, l, mask=None):
        mbam_loss = 0
        for i, (attention, attention1, mlp1) in enumerate(self.CMIM):
            x, l = attention(x, l, mask=mask)  # go to attention
            # x,l = mlp(x,l)  # go to MLP_Block
            x, l = attention1(x, l, mask=mask)  # go to attention
            x, l = mlp1(x, l)  # go to MLP_Block
            mbam_loss += self.MBAM(x[:, 0], l[:, 0])


        mbam_loss = mbam_loss / len(self.CMIM)
        return x, l, mbam_loss


class Modality_Bridging_Alignment_Module(torch.nn.Module):
    def __init__(self, use_momentum=True):
        super(Modality_Bridging_Alignment_Module, self).__init__()

        self.AFFM_online = Attention_Feature_Fusion_Module()
        self.projector_online = torch.nn.Sequential(nn.Linear(32, 512),
                                                    nn.BatchNorm1d(512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, 256))

        self.predictor_online = torch.nn.Sequential(nn.Linear(256, 512),
                                                    nn.BatchNorm1d(512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, 256))
        self.use_momentum = use_momentum

    def forward(self, x, l):
        x = self.AFFM_online(x, x)
        l = self.AFFM_online(l, l)
        x_l = self.AFFM_online(x, l)

        p1 = self.predictor_online(self.projector_online(x))
        p2 = self.predictor_online(self.projector_online(l))
        p3 = self.predictor_online(self.projector_online(x_l))


        loss1 = loss_fn(p1, p2.detach())
        loss2 = loss_fn(p1, p3.detach())
        loss3 = loss_fn(p2, p3.detach())
        loss = (loss1 + loss2 + loss3).mean()
        return loss


class Attention_Feature_Fusion_Module(torch.nn.Module):
    def __init__(self):
        super(Attention_Feature_Fusion_Module, self).__init__()
        self.MLP_1 = torch.nn.Sequential(nn.Linear(32, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 32),
                                         )
        self.MLP_2 = torch.nn.Sequential(nn.Linear(32, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 32),
                                         )
        self.MLP_3 = torch.nn.Sequential(nn.Linear(32 * 3, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1_ = torch.unsqueeze(self.MLP_1(torch.squeeze(x1)), dim=1)  # N * 1 * C
        x2_ = torch.unsqueeze(self.MLP_2(torch.squeeze(x2)), dim=-1)  # N * C * 1
        alpha = torch.squeeze(self.sigmoid(torch.matmul(x1_, x2_)), dim=-1)
        add_x = torch.squeeze(x1) * alpha + torch.squeeze(x2) * (1 - alpha)
        out = self.MLP_3(torch.cat([x1, add_x, x2], dim=-1))
        return out


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)