import torch
import torch.nn as nn
import numpy as np

""" Applied in Transformers models, Llama

These comes under the Class.

These are used in Llama Rotary Embedding
"""


def rotate_half(x):
    """Rotates half the hidden dims of the inputs"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_roatary_pos_emb(q, k, cos, sin, position_ids):
    """The first two dim of cos and sin are always 1, so, squeeze them"""
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
