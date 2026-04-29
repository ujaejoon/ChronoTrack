import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .attention import MaskedMultiHeadAttention


class MemoryEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoderLayer, self).__init__()
        self.cfg = cfg

        self.token_indicator = nn.Parameter(torch.zeros(1, 1, cfg.token_dim))

        self.pos_embed = nn.Linear(3, cfg.token_dim)
        self.cross_attn = MaskedMultiHeadAttention(cfg.token_dim, cfg.num_heads, cfg.attn_dropout, bias=True)
        self.self_attn = nn.MultiheadAttention(cfg.token_dim, cfg.num_heads, dropout=cfg.attn_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_dim * cfg.mlp_expansion),
            nn.ReLU(),
            nn.Linear(cfg.token_dim * cfg.mlp_expansion, cfg.token_dim),
        )

        self.cross_norm_token = nn.LayerNorm(cfg.token_dim)
        self.cross_norm_prev_token = nn.LayerNorm(cfg.token_dim)
        self.cross_norm_feat = nn.LayerNorm(cfg.token_dim)
        self.self_norm = nn.LayerNorm(cfg.token_dim)
        self.mlp_norm = nn.LayerNorm(cfg.token_dim)

        self.cross_dropout = nn.Dropout(cfg.dropout)
        self.self_dropout = nn.Dropout(cfg.dropout)
        self.mlp_dropout = nn.Dropout(cfg.dropout)

    def forward(self, xyz, feat, mask, fg_tokens, prev_fg_tokens):
        """
        Args:
            xyz : b, num_points, 3
            feat : b, c, num_points
            mask : b, num_points
            fg_tokens : b, num_fg_tokens, c
            prev_fg_tokens : b, num_fg_tokens, c or None
        Returns:
            fg_tokens : b, num_fg_tokens, c
        """
        # cross attention
        pos = self.pos_embed(xyz)  # (B, num_points, C)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, num_points, C)
        norm_feat = self.cross_norm_feat(feat)
        norm_feat_pos = norm_feat + pos  # (B, num_points, C)

        q = self.cross_norm_token(fg_tokens)
        if prev_fg_tokens is not None:
            prev_fg_tokens = self.cross_norm_prev_token(prev_fg_tokens)  # (B, num_prev_tokens, C)
            k = torch.cat([norm_feat_pos, prev_fg_tokens + self.token_indicator], dim=1)
            v = torch.cat([norm_feat, prev_fg_tokens], dim=1)
            token_mask = torch.ones(mask.size(0), prev_fg_tokens.size(1), dtype=mask.dtype, device=mask.device)  # (B, num_prev_tokens)
            mask = torch.cat([mask, token_mask], dim=1)  # (B, num_points + num_prev_tokens)
        else:
            k = norm_feat_pos
            v = norm_feat

        cross_out, _, _ = self.cross_attn(q, k, v, mask, self.cfg.mask_threshold)  # (B, num_fg_tokens, C)
        fg_tokens = fg_tokens + self.cross_dropout(cross_out)

        # self attention
        q = k = v = self.self_norm(fg_tokens)
        self_out, _ = self.self_attn(q, k, v, need_weights=False)
        fg_tokens = fg_tokens + self.self_dropout(self_out)

        # mlp
        mlp_out = self.mlp(self.mlp_norm(fg_tokens))
        fg_tokens = fg_tokens + self.mlp_dropout(mlp_out)

        return fg_tokens


class MemoryEncoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg
        self.fg_tokens = nn.Parameter(torch.empty(1, cfg.num_fg_tokens, cfg.token_dim))
        self.layers = nn.ModuleList([MemoryEncoderLayer(cfg) for _ in range(cfg.num_layers)])

        nn.init.trunc_normal_(self.fg_tokens)

    def forward(self, xyz, feat, mask, prev_fg_tokens):
        """
        Args:
            xyz : b, num_points, 3
            feat : b, c, num_points
            mask : b, num_points
            prev_fg_tokens : b, num_fg_tokens, c or None
        Returns:
            new_fg_tokens: b, num_fg_tokens, c
        """
        if prev_fg_tokens is None:
            new_fg_tokens = self.fg_tokens.expand(xyz.size(0), -1, -1)
        else:
            new_fg_tokens = prev_fg_tokens

        for layer in self.layers:
            new_fg_tokens = layer(xyz, feat, mask, new_fg_tokens, prev_fg_tokens)

        return new_fg_tokens
