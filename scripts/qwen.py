#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw

import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG_QWEN = os.environ.get("DEBUG_QWEN", "0") == "1"

# Get the first available attribute from a list of attribute names.
# Returns the attribute value if found, otherwise returns default.
def _get_first_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = int(v.item())
                    else:
                        continue
                return v
            except Exception:
                continue
    return default


# Rotate half of the tensor for RoPE (Rotary Position Embedding).
# Splits the last dimension in half and swaps with negation.
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# Apply RoPE (Rotary Position Embedding) to input tensor.
# Uses LLaMA-style rotation with cos and sin tensors.
def _apply_rope_llama(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


# Fallback RoPE implementation for models without built-in rotary embedding.
# Generates cos/sin from position IDs using LLaMA-style computation.
class RotaryEmbeddingFallback(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.rope_scaling = rope_scaling

        if self.head_dim % 2 != 0:
            raise RuntimeError(f"RoPE requires even head_dim, got {self.head_dim}")

        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Generate cos and sin tensors from position IDs.
    def forward(self, position_ids_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids_1d.to(torch.float32)

        # support simple linear rope scaling if provided
        if isinstance(self.rope_scaling, dict):
            t = str(self.rope_scaling.get("type", "")).lower()
            if t == "linear":
                factor = float(self.rope_scaling.get("factor", 1.0))
                if factor != 1.0:
                    pos = pos / factor

        freqs = torch.einsum("s,d->sd", pos, self.inv_freq)  # [S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)              # [S, D]
        return torch.cos(emb), torch.sin(emb)


# Build key mask for attention computation.
# Combines cache mask and chunk mask for the full key sequence.
def _make_key_mask(attention_mask: torch.Tensor, past_len: torch.Tensor, past_len_cache: torch.Tensor, seq_len: int) -> torch.Tensor:
    chunk_mask = attention_mask[:, -seq_len:]
    chunk_mask = (chunk_mask > 0).to(chunk_mask.dtype)

    cache_valid = (past_len > 0).to(chunk_mask.dtype)
    B = chunk_mask.shape[0]
    cache_mask = torch.ones((B, past_len_cache), device=chunk_mask.device, dtype=chunk_mask.dtype) * cache_valid

    return torch.cat([cache_mask, chunk_mask], dim=1)


# Unified KV delta model wrapper for Qwen3.
# Wraps HuggingFace model to output KV deltas instead of full KV cache.
# Supports both prefill and decode phases with a single model.
class Qwen3UnifiedKvDelta(nn.Module):
    def __init__(self, hf_model: nn.Module, max_total_len: int):
        super().__init__()
        self.m = hf_model
        self.max_total_len = int(max_total_len)

        self.core = getattr(hf_model, "model")
        self.layers = getattr(self.core, "layers")
        self.norm = getattr(self.core, "norm")
        self.lm_head = getattr(hf_model, "lm_head")

        cfg = getattr(hf_model, "config", None)
        if cfg is None:
            raise RuntimeError("hf_model.config is required")

        self.hidden_size = int(_get_first_attr(cfg, ["hidden_size", "n_embd", "dim"], 0))
        if self.hidden_size <= 0:
            raise RuntimeError("Cannot infer hidden_size from config")

        self.num_heads = int(_get_first_attr(cfg, ["num_attention_heads", "num_heads", "n_head"], 0))
        if self.num_heads <= 0:
            attn0_tmp = self.layers[0].self_attn
            self.num_heads = int(_get_first_attr(attn0_tmp, ["num_heads", "n_heads"], 0))
        if self.num_heads <= 0:
            raise RuntimeError("Cannot infer num_heads (config/attn both invalid)")

        attn0 = self.layers[0].self_attn
        if not (hasattr(attn0, "q_proj") and hasattr(attn0.q_proj, "weight")):
            raise RuntimeError("Cannot access layer0.self_attn.q_proj.weight")
        if not (hasattr(attn0, "k_proj") and hasattr(attn0.k_proj, "weight")):
            raise RuntimeError("Cannot access layer0.self_attn.k_proj.weight")

        q_out = int(attn0.q_proj.weight.shape[0])
        if q_out % self.num_heads != 0:
            raise RuntimeError(f"q_proj.out_features not divisible by num_heads: {q_out} % {self.num_heads} != 0")
        self.head_dim = int(q_out // self.num_heads)

        k_out = int(attn0.k_proj.weight.shape[0])
        if k_out % self.head_dim != 0:
            raise RuntimeError(f"k_proj.out_features not divisible by head_dim: {k_out} % {self.head_dim} != 0")
        self.num_kv_heads = int(k_out // self.head_dim)

        if self.head_dim % 2 != 0:
            raise RuntimeError(f"RoPE requires even head_dim, got {self.head_dim}")

        self.group_size = int(self.num_heads // self.num_kv_heads)
        if self.group_size * self.num_kv_heads != self.num_heads:
            raise RuntimeError(f"Invalid GQA: num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")

        self.qkv_dim = int(self.num_heads * self.head_dim)

        rope_theta = float(_get_first_attr(cfg, ["rope_theta"], 10000.0))
        rope_scaling = _get_first_attr(cfg, ["rope_scaling"], None)
        self.rope_fallback = RotaryEmbeddingFallback(self.head_dim, base=rope_theta, rope_scaling=rope_scaling)

        if DEBUG_QWEN:
            kv_cfg = int(_get_first_attr(cfg, ["num_key_value_heads", "num_kv_heads", "n_head_kv"], 0))
            print(
                "[DEBUG_QWEN] inferred:",
                "hidden_size=", self.hidden_size,
                "num_heads=", self.num_heads,
                "num_kv_heads=", self.num_kv_heads,
                "head_dim=", self.head_dim,
                "qkv_dim=", self.qkv_dim,
                "group_size=", self.group_size,
                "kv_cfg=", kv_cfg,
                "has_qk_norm=", (hasattr(attn0, "q_norm") and hasattr(attn0, "k_norm")),
                "has_rotary_emb=", hasattr(attn0, "rotary_emb"),
                "has_core_rotary_emb=", hasattr(self.core, "rotary_emb"),
            )

    # Get cos/sin for RoPE from attention module or core module.
    # Falls back to RotaryEmbeddingFallback if no built-in rotary embedding found.
    def _get_cos_sin(self, attn_mod: nn.Module, cache_position_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) attention-level rotary_emb (some variants)
        rotary = getattr(attn_mod, "rotary_emb", None)
        # 2) core-level rotary_emb (shared)
        if rotary is None:
            rotary = getattr(self.core, "rotary_emb", None)

        if rotary is not None:
            # Try common signatures:
            # - cos, sin = rotary(value_states, position_ids)
            # - cos, sin = rotary(position_ids)
            pos_ids = cache_position_1d.view(1, -1)  # [1,S]
            try:
                # pass a dummy tensor with correct dtype/device if needed
                dummy = torch.zeros((1, 1, pos_ids.shape[1], self.head_dim), device=cache_position_1d.device, dtype=torch.float32)
                cos, sin = rotary(dummy, pos_ids)
                return cos, sin
            except Exception:
                pass
            try:
                cos, sin = rotary(pos_ids)
                return cos, sin
            except Exception:
                pass
            # If rotary exists but signature mismatch, fall through to fallback

        # Fallback: LLaMA-style cos/sin [S, head_dim]
        return self.rope_fallback(cache_position_1d)

    # Apply Q/K normalization if available in attention module.
    def _apply_qk_norm_if_any(self, attn_mod: nn.Module, q: torch.Tensor, k_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = getattr(attn_mod, "q_norm", None)
        k_norm = getattr(attn_mod, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k_kv = k_norm(k_kv)
        return q, k_kv

    # Process attention chunk with KV cache.
    # Computes attention scores using cached KV and new KV, then outputs deltas.
    def _attn_chunk(
        self,
        attn_mod: nn.Module,
        x: torch.Tensor,                 # [B,S,hidden]
        attention_mask: torch.Tensor,    # [B,total_seq]
        cache_position: torch.Tensor,    # [S]
        cache_k_full: torch.Tensor,      # [B,max_total_len,kv,hd]
        cache_v_full: torch.Tensor,      # [B,max_total_len,kv,hd]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape

        past_len = cache_position[0]
        past_len_cache = torch.clamp(past_len, min=1)

        cache_k = cache_k_full[:, :past_len_cache]     # [B,Pc,kv,hd]
        cache_v = cache_v_full[:, :past_len_cache]

        # projections
        q = attn_mod.q_proj(x)  # [B,S,qkv_dim]
        k = attn_mod.k_proj(x)  # [B,S,kv*hd]
        v = attn_mod.v_proj(x)  # [B,S,kv*hd]

        # reshape
        q = q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # [B,H,S,hd]
        k_kv = k.reshape(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3) # [B,kv,S,hd]
        v_kv = v.reshape(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3) # [B,kv,S,hd]

        # QK-Norm (IMPORTANT) — do it before RoPE
        q, k_kv = self._apply_qk_norm_if_any(attn_mod, q, k_kv)

        # RoPE
        cos, sin = self._get_cos_sin(attn_mod, cache_position)  # broadcastable
        q = _apply_rope_llama(q, cos, sin)
        k_kv = _apply_rope_llama(k_kv, cos, sin)

        # expand kv -> heads (GQA)
        k_h = k_kv.repeat_interleave(self.group_size, dim=1)  # [B,H,S,hd]
        v_h = v_kv.repeat_interleave(self.group_size, dim=1)

        # cache expand
        cache_k_h = cache_k.permute(0, 2, 1, 3).repeat_interleave(self.group_size, dim=1)  # [B,H,Pc,hd]
        cache_v_h = cache_v.permute(0, 2, 1, 3).repeat_interleave(self.group_size, dim=1)

        # scale
        scaling = getattr(attn_mod, "scaling", None)
        if scaling is None:
            scale = float(self.head_dim) ** -0.5
        else:
            try:
                scale = float(scaling)
            except Exception:
                scale = float(self.head_dim) ** -0.5

        # scores
        scores_cache = torch.matmul(q.float(), cache_k_h.float().transpose(-1, -2)) * scale  # [B,H,S,Pc]
        scores_new = torch.matmul(q.float(), k_h.float().transpose(-1, -2)) * scale          # [B,H,S,S]
        scores = torch.cat([scores_cache, scores_new], dim=-1)                                # [B,H,S,K]
        K = scores.shape[-1]

        # padding mask
        key_mask = _make_key_mask(attention_mask, past_len, past_len_cache, S)  # [B,K]
        key_mask_f = (key_mask > 0).to(scores.dtype)
        pad = (1.0 - key_mask_f).unsqueeze(1).unsqueeze(1) * 1e4

        # causal by absolute positions
        cache_pos = torch.arange(past_len_cache, device=cache_position.device, dtype=cache_position.dtype)
        key_pos_total = torch.cat([cache_pos, cache_position], dim=0)  # [K]
        q_pos = cache_position.view(S, 1)
        k_pos = key_pos_total.view(1, K)
        causal_ok = (k_pos <= q_pos)

        causal = torch.where(
            causal_ok,
            torch.zeros((), device=scores.device, dtype=scores.dtype),
            torch.full((), -1e4, device=scores.device, dtype=scores.dtype),
        ).view(1, 1, S, K)

        scores = scores + causal - pad
        attn = torch.softmax(scores, dim=-1).to(q.dtype)

        attn_cache = attn[..., :past_len_cache]
        attn_new = attn[..., past_len_cache:]

        out_cache = torch.matmul(attn_cache, cache_v_h)  # [B,H,S,hd]
        out_new = torch.matmul(attn_new, v_h)            # [B,H,S,hd]
        out = out_cache + out_new

        # back to hidden
        out = out.permute(0, 2, 1, 3).reshape(B, S, self.qkv_dim)
        out = attn_mod.o_proj(out)

        # deltas: store RoPE-applied K (IMPORTANT) and raw V
        k_delta = k_kv.permute(0, 2, 1, 3).contiguous()  # [B,S,kv,hd]
        v_delta = v_kv.permute(0, 2, 1, 3).contiguous()  # [B,S,kv,hd]

        return out, k_delta, v_delta

    # Forward pass through MLP (gate, up, down projections with SiLU).
    def _mlp(self, mlp_mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate = mlp_mod.gate_proj(x)
        up = mlp_mod.up_proj(x)
        h = F.silu(gate) * up
        return mlp_mod.down_proj(h)

    # Forward pass through the unified KV delta model.
    # Returns logits and KV deltas for all layers.
    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, cache_position: torch.Tensor, *cache_flat: torch.Tensor):
        L = len(self.layers)
        if len(cache_flat) != 2 * L:
            raise RuntimeError(f"expect {2*L} cache tensors, got {len(cache_flat)}")

        model_dtype = next(self.m.parameters()).dtype
        x = inputs_embeds.to(model_dtype)

        deltas: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = layer.input_layernorm(x)

            cache_k = cache_flat[2 * i].to(model_dtype)
            cache_v = cache_flat[2 * i + 1].to(model_dtype)

            attn_out, k_delta, v_delta = self._attn_chunk(
                layer.self_attn,
                x_norm,
                attention_mask,
                cache_position,
                cache_k,
                cache_v,
            )

            x = residual + attn_out
            residual = x

            x_norm2 = layer.post_attention_layernorm(x)
            mlp_out = self._mlp(layer.mlp, x_norm2)
            x = residual + mlp_out

            deltas.append(k_delta.to(model_dtype))
            deltas.append(v_delta.to(model_dtype))

        x = self.norm(x)
        logits = self.lm_head(x).float()
        return (logits, *deltas)
