import torch
from torch import nn
from functools import partial
from torch.nn.parameter import Parameter

from .utils import pytorch_utils as pt_utils

NORM_DICT = {
    "batch_norm": nn.BatchNorm1d,
    "id": nn.Identity,
    "layer_norm": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}

class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.self_attn = nn.MultiheadAttention(embed_dim=cfg.feat_dim, num_heads=cfg.num_heads, dropout=cfg.attn_dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=cfg.feat_dim, num_heads=cfg.num_heads, dropout=cfg.attn_dropout, batch_first=True)

        self.fg_indicator = Parameter(torch.ones(1, 1, cfg.feat_dim))
        self.bg_indicator = Parameter(torch.zeros(1, 1, cfg.feat_dim))

        self.ffn = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.ffn_cfg.hidden_dim, bias=cfg.ffn_cfg.use_bias),
            ACTIVATION_DICT[cfg.ffn_cfg.activation](),
            nn.Dropout(cfg.ffn_cfg.dropout, inplace=False),
            nn.Linear(cfg.ffn_cfg.hidden_dim, cfg.feat_dim, bias=cfg.ffn_cfg.use_bias),
        )

        self.self_pos_emb = pt_utils.Seq(3).conv1d(cfg.feat_dim, bn=True).conv1d(cfg.feat_dim, activation=None)
        self.cross_pos_emb = pt_utils.Seq(3).conv1d(cfg.feat_dim, bn=True).conv1d(cfg.feat_dim, activation=None)
        self.bg_point_memory_pos_emb = pt_utils.Seq(3).conv1d(cfg.feat_dim, bn=True).conv1d(cfg.feat_dim, activation=None)

        self.self_dropout = nn.Dropout(cfg.dropout)
        self.cross_dropout = nn.Dropout(cfg.dropout)
        self.ffn_dropout = nn.Dropout(cfg.ffn_cfg.dropout)

        self.self_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.cross_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.fg_token_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.bg_point_memory_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.ffn_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

    def forward(self, input):
        feat = input.pop("feat")  # b,c,n
        xyz = input.pop("xyz")  # b,n,3

        x = feat.permute(0, 2, 1).contiguous()  # b,n,c
        N = x.shape[1]  # number of points

        # cross-attn
        fg_tokens = input.pop("fg_tokens")  # b,num_fg_tokens,c
        B, num_fg_tokens, _ = fg_tokens.shape
        m_xyz = input.pop("memory_xyz")  # b,t*n,3
        m_feat = input.pop("memory_feat")  # b,c,t*n
        m_mask = input.pop("memory_mask")  # b,t*n

        fg_tokens = self.fg_token_norm(fg_tokens)
        point_memory = m_feat.permute(0, 2, 1).contiguous()  # b,t*n,c
        point_memory = self.bg_point_memory_norm(point_memory)
        point_memory_pe = m_xyz.permute(0, 2, 1).contiguous()  # b,3,t*n
        point_memory_pe = self.bg_point_memory_pos_emb(point_memory_pe).permute(0, 2, 1).contiguous()  # b,t*n,c

        cross_pe = xyz.permute(0, 2, 1).contiguous()  # b,3,n
        cross_pe = self.cross_pos_emb(cross_pe).permute(0, 2, 1).contiguous()  # b,n,c
        q = self.cross_norm(x) + cross_pe
        k = torch.cat([fg_tokens, point_memory + point_memory_pe], dim=1)  # b,num_fg_tokens + t*n,c
        v = torch.cat([fg_tokens + self.fg_indicator, point_memory + self.bg_indicator], dim=1)

        # get attention mask based on memory mask
        # allow attending only background points (m_mask < 0.5) and all foreground tokens
        fg_token_attn_mask = torch.zeros(B * self.cfg.num_heads, N, num_fg_tokens, device=fg_tokens.device)  # b*num_heads,n,num_fg_tokens
        bg_point_attn_mask = m_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.cfg.num_heads, N, 1) # b, 1, 1, t*n -> b, num_heads, n, t*n
        bg_point_attn_mask = bg_point_attn_mask.view(B * self.cfg.num_heads, N, -1)  # b*num_heads,n,t*n
        attn_mask = torch.cat([fg_token_attn_mask, bg_point_attn_mask], dim=-1)
        attn_mask = attn_mask > 0.5  # b*num_heads,n,num_fg_tokens + t*n

        x2, _ = self.cross_attn(q, k, v, need_weights=False, attn_mask=attn_mask)
        x = x + self.cross_dropout(x2)  # b,n,c

        # self-attn
        self_pe = xyz.permute(0, 2, 1).contiguous()  # b,3,n
        self_pe = self.self_pos_emb(self_pe).permute(0, 2, 1).contiguous()  # b,n,c
        x2 = self.self_norm(x)
        x2, _ = self.self_attn(x2 + self_pe, x2 + self_pe, x2, need_weights=False)
        x = x + self.self_dropout(x2)  # b,n,c

        # ffn
        x2 = self.ffn_norm(x)
        x2 = self.ffn(x2)
        x = x + self.ffn_dropout(x2)  # b,n,c

        x = x.permute(0, 2, 1).contiguous()  # b,c,n

        output_dict = dict(feat=x, xyz=xyz)
        return output_dict


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for layer_cfg in cfg.layers_cfg:
            self.layers.append(TransformerLayer(layer_cfg))

    def forward(self, input_dict):
        memory = input_dict.pop("memory")
        output_dict = dict()

        for i, layer in enumerate(self.layers):
            input_dict.update(
                memory_feat=memory["feat"].flatten(2),  # b,c,t,n -> b,c,t*n
                memory_xyz=memory["xyz"].flatten(1,2),  # b,t,n,3 -> b,t*n,3
                memory_mask=memory["mask"].flatten(1),  # b,t,n -> b,t*n
                fg_tokens=memory["fg_tokens"],  # b,num_fg_tokens,c
            )
            input_dict = layer(input_dict)

        output_dict.update(input_dict)
        return output_dict
