# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
import torch.utils.data as Data

from CMIM_MBAM import CMIM_MBAM
from einops import rearrange, repeat
import clip
from collections import OrderedDict
from prettytable import PrettyTable
import time

class Cascade(nn.Module):
    def __init__(self, ch_in, ch_out, batch):
        super(Cascade, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1))
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, stride=1, padding=1)
        if batch:
            self.bn = nn.BatchNorm2d(ch_in * 2)
        else:
            self.bn = nn.Identity()
        self.conv3 = nn.Conv2d(ch_in * 2, ch_out, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU()
        self.skip = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.act1(x1) + x
        x3 = self.bn(self.conv2(x2))
        x3 = self.conv3(x3) + self.skip(x1)
        out = self.act2(x3)
        return out

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, batch: bool):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_mid, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        if batch:
            self.bn1 = nn.BatchNorm2d(ch_mid)
            self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(in_channels=ch_mid, out_channels=ch_out, kernel_size=3, stride=1, padding=1)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1))
            # nn.BatchNorm2d(ch_out))

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = out + self.extra(x)
        return out

class CNN(nn.Module):
    def __init__(self, bands: int, FM: int, NCLidar, batch: bool):
        super(CNN, self).__init__()
        self.conv_HSI_1 = ResBlk(bands, FM * 2, FM, batch=batch)
        self.conv_LIDAR_1 = nn.Sequential(nn.Conv2d(NCLidar, FM, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(FM),
                                          nn.LeakyReLU())
        self.conv_HSI_2 = ResBlk(FM, FM * 4, FM * 2, batch=batch)
        self.conv_LIDAR_2 = Cascade(FM, FM * 2, batch)
        self.pool_LIDAR = nn.AvgPool2d(2)
        self.pool_HSI = nn.AvgPool2d(2)
        # self.conv_HSI_3 = ResBlk(FM * 2, FM * 4, FM * 4, batch=batch)
        # self.conv_LIDAR_3 = Cascade(FM * 2, FM * 4, batch)

    def forward(self, data_HSI, data_LiDAR):
        data_HSI = self.conv_HSI_1(data_HSI)
        data_HSI = self.conv_HSI_2(data_HSI)
        data_LiDAR = self.conv_LIDAR_1(data_LiDAR)
        data_LiDAR = self.conv_LIDAR_2(data_LiDAR)
        data_HSI = self.pool_HSI(data_HSI)
        # data_HSI= F.leaky_relu(data_HSI)
        data_LiDAR = F.leaky_relu(self.pool_LIDAR(data_LiDAR))
        # data_HSI = self.conv_HSI_3(data_HSI)
        # data_LiDAR = self.conv_LIDAR_3(data_LiDAR)

        return data_HSI, data_LiDAR

class CMIM_AND_MBAM(nn.Module):
    def __init__(self, patchsize, depth=2):
        super().__init__()
        self.patchsize = patchsize
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.CMIM_MBAM = CMIM_MBAM(dim=32, hidden_dim=8, head_dim=32, heads=2, num_atte=8, depth=depth,
                                                 dropout=0.1)
        self.pos_embedding1 = nn.Parameter(torch.empty(1, 1+(patchsize//2)**2, 32))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)
        self.pos_embedding2 = nn.Parameter(torch.empty(1, 1+(patchsize//2)**2, 32))
        torch.nn.init.normal_(self.pos_embedding2, std=.02)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, data_HSI, data_LiDAR):
        data_HSI = rearrange(data_HSI, 'b c h w -> b (h w) c')
        data_LiDAR = rearrange(data_LiDAR, 'b c h w -> b (h w) c')

        cls_tokens1 = self.cls_token.expand(data_HSI.shape[0], -1, -1)
        data_HSI = torch.cat([cls_tokens1, data_HSI], 1)
        cls_tokens2 = self.cls_token.expand(data_LiDAR.shape[0], -1, -1)
        data_LiDAR = torch.cat([cls_tokens2, data_LiDAR], 1)

        data_HSI = data_HSI + self.pos_embedding1
        data_LiDAR = data_LiDAR + self.pos_embedding2

        data_HSI = self.dropout1(data_HSI)
        data_LiDAR = self.dropout2(data_LiDAR)

        data_HSI, data_LiDAR, cffl_loss = self.CMIM_MBAM(data_HSI, data_LiDAR)
        data_HSI = rearrange(data_HSI[:, 1:], 'b (h w) c -> b c h w', h=self.patchsize // 2,
                                 w=self.patchsize // 2)
        data_LiDAR = rearrange(data_LiDAR[:, 1:], 'b (h w) c -> b c h w', h=self.patchsize // 2,
                                   w=self.patchsize // 2)
        return data_HSI, data_LiDAR, cffl_loss


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        # self.skipcat = nn.ModuleList([])
        # for _ in range(depth - 2):
        #     self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):

        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer_Text(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TGCMFNet(nn.Module):
    def __init__(self, FM=16, NC=144, NCLidar=1, Classes=15, patchsize=16,
                 context_length=None, transformer_width=None, transformer_layers=None, transformer_heads=None,
                 vocab_size=None, embed_dim=None, block=2):
        super(TGCMFNet, self).__init__()

        self.context_length = context_length
        self.transformer_text = Transformer_Text(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_projection = nn.Parameter(torch.randn(FM * 4, embed_dim))

        self.cnn = CNN(NC, FM, NCLidar, True)
        self.CMIM_AND_MBAM = CMIM_AND_MBAM(patchsize, depth=block)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, (patchsize // 2) ** 2 + 1, FM * 4))
        self.cls_token = nn.Parameter(torch.randn(1, 1, FM * 4))
        self.ViT = Transformer(dim=FM * 4, depth=5, heads=4, dim_head=16, mlp_head=8, dropout=0.3)
        self.cls = nn.Sequential(
            nn.LayerNorm(FM * 4),
            nn.Linear(FM * 4, Classes)
        )

        # self.mapper = nn.Linear(64, 32)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        #.type(self.dtype)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_text(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


    def VTAM(self, feature, text, text_queue_1, text_queue_2, label):
        image_features = feature @ self.image_projection
        text_features = self.encode_text(text)
        text_features_q1 = self.encode_text(text_queue_1)
        text_features_q2 = self.encode_text(text_queue_2)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        loss_img = F.cross_entropy(logits_per_image, label)
        loss_text = F.cross_entropy(logits_per_text, label.long())
        loss_clip = (loss_img + loss_text) / 2



        # q1
        # normalized features
        text_features_q1 = text_features_q1 / text_features_q1.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features_q1.t()
        logits_per_text = logit_scale * text_features_q1 @ image_features.t()


        loss_img = F.cross_entropy(logits_per_image, label.long())
        loss_text = F.cross_entropy(logits_per_text, label.long())
        loss_clip1 = (loss_img + loss_text) / 2


        # q2
        # normalized features
        text_features_q2 = text_features_q2 / text_features_q2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features_q2.t()
        logits_per_text = logit_scale * text_features_q2 @ image_features.t()

        loss_img = F.cross_entropy(logits_per_image, label.long())
        loss_text = F.cross_entropy(logits_per_text, label.long())
        loss_clip2 = (loss_img + loss_text) / 2
        return loss_clip, loss_clip1, loss_clip2


    def forward(self, h_spa, l_spa, b_y=None, text=None, text_queue_1=None, text_queue_2=None):
        batch = h_spa.shape[0]
        h_spa, l_spa  =self.cnn(h_spa, l_spa)
        h_spa, l_spa, mbam_loss = self.CMIM_AND_MBAM(h_spa, l_spa)
        # mbam_loss  = torch.tensor(0)



        data_fusion = torch.cat([h_spa, l_spa], dim=1)


        feature_local = self.pool(data_fusion)
        feature_local = feature_local.flatten(1)

        # [b,c,h,w]->[b,c,h*w]
        data_fusion = data_fusion.flatten(2)
        # [b,c,l]->[n,l,d]得到vit的输入
        data_fusion = torch.einsum('ndl->nld', data_fusion)

        # 加位置编码和clstoken
        data_fusion = data_fusion + self.encoder_pos_embed[:, 1:, :]
        cls_token = repeat(self.cls_token, '() l d->b l d', b=batch)
        data_fusion = torch.cat([cls_token, data_fusion], dim=1)
        data_fusion += self.encoder_pos_embed[:, 0]
        data_fusion = self.ViT(data_fusion)


        # 分类
        feature_global = data_fusion[:, 0]
        feature_global = feature_global.view(batch, -1)
        feature = feature_local + feature_global

        if text != None:
            loss_clip, loss_clip1, loss_clip2 = self.VTAM(feature, text, text_queue_1, text_queue_2, b_y)
            vtam_loss = (1 - 0.3) * loss_clip + 0.3 * (loss_clip1 + loss_clip2)/2
        else:
            vtam_loss = torch.Tensor([0]).to(h_spa.device)



        # mapper = self.mapper(feature)
        pro = self.cls(feature)
        pro = torch.softmax(pro, dim=-1)

        return pro, vtam_loss, mbam_loss


def test(cnn, TestLabel, TestPatch1, TestPatch2):
    with torch.no_grad():
        pred_y = np.empty((len(TestLabel)), dtype='float32')
        pro_y = np.empty((len(TestLabel)), dtype='float32')
        patch = 1000
        number = len(TestLabel) // patch
        for i in range(number):
            temp_TestPatch1 = TestPatch1[i * patch:(i + 1) * patch, :, :, :]
            temp_TestPatch2 = TestPatch2[i * patch:(i + 1) * patch, :, :, :]
            temp_TestPatch1 = temp_TestPatch1.cuda()
            temp_TestPatch2 = temp_TestPatch2.cuda()
            pred = cnn(temp_TestPatch1, temp_TestPatch2, b_y=None)[0]

            temp_pred_y = torch.max(pred, 1)[1].squeeze()
            temp_pro_y = torch.max(pred, 1)[0].squeeze()
            pred_y[i * patch:(i + 1) * patch] = temp_pred_y.cpu()
            pro_y[i * patch:(i + 1) * patch] = temp_pro_y.detach().cpu()
            del temp_TestPatch1, temp_TestPatch2, pred, temp_pred_y, temp_pro_y

        if (i + 1) * patch < len(TestLabel):
            temp_TestPatch1 = TestPatch1[(i + 1) * patch:len(TestLabel), :, :, :]
            temp_TestPatch2 = TestPatch2[(i + 1) * patch:len(TestLabel), :, :, :]
            temp_TestPatch1 = temp_TestPatch1.cuda()
            temp_TestPatch2 = temp_TestPatch2.cuda()
            pred = cnn(temp_TestPatch1, temp_TestPatch2, b_y=None)[0]
            temp_pred_y = torch.max(pred, 1)[1].squeeze()
            temp_pro_y = torch.max(pred, 1)[0].squeeze()
            pred_y[(i + 1) * patch:len(TestLabel)] = temp_pred_y.cpu()
            pro_y[(i + 1) * patch:len(TestLabel)] = temp_pro_y.detach().cpu()
            del temp_TestPatch1, temp_TestPatch2, pred, temp_pred_y, temp_pro_y
        return pred_y, pro_y

def train_network(train_loader, TestPatch1, TestPatch2, TestLabel, LR, EPOCH, SEED, NC, NCLidar, Classes, batchsize,
                  patchsize, num_labelled, factor_lambda, dataset_name, label_values, label_queue, depth):
    pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3


    net = TGCMFNet(FM=16, NC=NC, NCLidar=NCLidar, Classes=Classes, patchsize=patchsize,
                  context_length=context_length, transformer_width=transformer_width, transformer_layers=transformer_layers, transformer_heads=transformer_heads,
                  vocab_size=vocab_size, embed_dim=embed_dim, block=depth)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')


    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # Optimize all net parameters
    loss_fun1 = nn.CrossEntropyLoss()  # Cross entropy loss
    val_acc = []
    BestAcc = 0
    """Pre-training"""
    net.train()
    for epoch in range(EPOCH):
        time1 = time.time()
        loss_list = []
        class_loss_list = []
        vtam_loss_list = []
        mbam_loss_list = []

        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):  # Supervised train_loader
            b_x1, b_x2, b_y = b_x1.cuda(), b_x2.cuda(), b_y.cuda()  # Move data to GPU
            '''
                添加文本
            '''
            text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_values[k]}').to(k.device) for k in b_y.long()])
            text_queue_1 = [label_queue[label_values[k]][0] for k in b_y.long()]

            text_queue_2 = [label_queue[label_values[k]][1] for k in b_y.long()]
            text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
            text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])

            pro, vtam_loss, mbam_loss = net(b_x1, b_x2, b_y=b_y, text=text, text_queue_1=text_queue_1, text_queue_2=text_queue_2)  # Model output according to labelled train set
            class_loss = loss_fun1(pro, b_y.long())
            ce_loss = class_loss + factor_lambda * vtam_loss + factor_lambda * mbam_loss  # Multi-scale classification loss
            loss_list.append(ce_loss.item())
            class_loss_list.append(class_loss.item())
            vtam_loss_list.append(vtam_loss.item())
            mbam_loss_list.append(mbam_loss.item())



            net.zero_grad()  # Reset gradient
            ce_loss.backward()  # Backward
            optimizer.step()  # Update parameters of net

        net.eval()



        pred_y, pro_y = test(net, TestLabel, TestPatch1, TestPatch2)

        pred_y = torch.from_numpy(pred_y).long()
        accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)



        val_acc.append(accuracy.data.cpu().numpy())
        # Save the parameters in network
        if accuracy > BestAcc:
            # torch.save(net.state_dict(),
            #            './log/' + dataset_name + '_UACL_baseline_pretrain.pkl')
            torch.save(net.state_dict(),
                        './log/' + dataset_name + '.pkl')
            BestAcc = accuracy
            best_y = pred_y

            out = PrettyTable()
            # print('epoch:{:0>3d}'.format(epoch))
            out.add_column("loss", ['value'])
            out.add_column('accuracy', ['{:.4f}'.format(accuracy)])
            out.add_column('loss', ['{:.4f}'.format(np.mean(loss_list) if len(loss_list) > 0 else 0)])
            out.add_column('class loss', ['{:.4f}'.format(np.mean(class_loss_list)) if len(class_loss_list) > 0 else 0])
            out.add_column('vtam loss', ['{:.4f}'.format(np.mean(vtam_loss_list)) if len(vtam_loss_list) > 0 else 0])
            out.add_column('mbam loss', ['{:.4f}'.format(np.mean(mbam_loss_list)) if len(mbam_loss_list) > 0 else 0])
            print(out)

        net.train()  # Open Batch Normalization and Dropout
        time2  =time.time()
        print('seed:{} epoch:{:0>3d} time:{:.4f}'.format(SEED, epoch, time2-time1))
        # print("time:%.4f"%(time2-time1))
    # print("pretrain BestACC {}".format(BestAcc))

    net.load_state_dict(torch.load(
        './log/' + dataset_name+'.pkl'))
    net.eval()

    pred_y, pro_y = test(net, TestLabel, TestPatch1, TestPatch2)

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

    Classes = np.unique(TestLabel)
    EachAcc = np.empty(len(Classes))

    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum_all = 0

        for j in range(len(TestLabel)):
            if TestLabel[j] == cla:
                sum_all = sum_all + 1
                # sum_all += 1

            if TestLabel[j] == cla and pred_y[j] == cla:
                right = right + 1
                # right += 1

        # EachAcc[i] = right.__float__() / sum_all.__float__()
        # AA = np.mean(EachAcc)

    # print(OA)
    # print(EachAcc)
    # print(AA)
    return pred_y, val_acc

