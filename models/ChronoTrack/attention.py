import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MaskedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.merge_heads = Rearrange("b h l d -> b l (h d)", h=num_heads)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def value_proj(self, x: Tensor) -> Tensor:
        return self.out_proj(self.v_proj(x))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, threshold: Tensor) -> Tensor:
        """
        Args:
            q: query tensor with shape (B, L, C)
            k: key tensor with shape (B, S, C)
            v: value tensor with shape (B, S, C)
            mask: mask tensor with shape (B, S) or (B, L, S)
            threshold: threshold value for hard mask
        Returns:
            x: output tensor with shape (B, L, C)
            out_attn_weight: raw attention weights with shape (B, L, S) averaged over heads
            out_masked_attn_weight: masked attention weights with shape (B, L, S) averaged over heads
        """
        # projection
        q = self.q_proj(q) * (self.head_dim**-0.5)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # split heads
        q = self.split_heads(q)  # (B, H, L, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)
        q = rearrange(q, "b h l d -> (b h) l d")  # (B * H, L, D)
        k = rearrange(k, "b h s d -> (b h) s d")  # (B * H, S, D)
        v = rearrange(v, "b h s d -> (b h) s d")  # (B * H, S, D)

        # masked attention
        attn_weight = q @ k.transpose(-2, -1)  # (B * H, L, S)
        attn_weight = rearrange(attn_weight, "(b h) l s -> b h l s", h=self.num_heads)  # (B, H, L, S)
        out_attn_weight = attn_weight.softmax(dim=-1).mean(dim=1)  # (B, L, S)

        if mask is not None and threshold is not None:
            # expand mask
            if mask.dim() == 2:
                expanded_mask = repeat(mask, "b s -> b h l s", h=self.num_heads, l=q.size(1))
                # valid = not all key are masked
                valid_batch = ~((mask < threshold).all(dim=1))  # (B,)
                valid_batch = rearrange(valid_batch, "b -> b 1 1 1")
            elif mask.dim() == 3:
                expanded_mask = repeat(mask, "b l s -> b h l s", h=self.num_heads)
                # valid = not all key are masked
                valid_batch = ~((mask < threshold).all(dim=2))
                valid_batch = rearrange(valid_batch, "b l -> b 1 l 1")
            else:
                raise ValueError("Mask must be 2D or 3D tensor")

            masked_attn_weight = torch.where(expanded_mask < threshold, float("-inf"), attn_weight)

            new_attn_weight = torch.where(valid_batch, masked_attn_weight, attn_weight)
        else:
            new_attn_weight = attn_weight

        out_masked_attn_weight = new_attn_weight.softmax(dim=-1)  # (B, H, L, S)

        attn_weight = rearrange(new_attn_weight, "b h l s -> (b h) l s")
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        x = attn_weight @ v  # (B * H, L, D)
        x = rearrange(x, "(b h) l d -> b h l d", h=self.num_heads)  # (B, H, L, D)
        x = self.merge_heads(x)  # (B, L, C)
        x = self.out_proj(x)

        return x, out_attn_weight, out_masked_attn_weight
