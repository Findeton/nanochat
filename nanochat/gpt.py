"""
GPT model with unified associative attention over live context and persistent
latent memory tokens.

Key ideas:
- ordinary context tokens and persistent memory tokens are both read through
  attention
- persistent memory differs mainly in how it is written / updated
- runtime memory state is latent torch tensors, not explicit symbolic slots
- writes and reads are intentionally sparse so memory stays selective instead of
  diffusing across every slot
"""

import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.episodic_memory import PersistentMemoryTokens
from nanochat.flash_attention import flash_attn
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    ve_gate_channels: int = 32
    window_pattern: str = "SSSL"
    episodic_dim: int = 256
    episodic_slots: int = 48
    episodic_decay: float = 0.995
    episodic_gate_init: float = -0.5
    episodic_kind_bias_init: float = 0.0
    episodic_beta: float = 24.0
    episodic_top_k: int = 8
    episodic_max_write_tokens: int = 64
    episodic_summary_budget: int = 4
    episodic_anchor_budget: int = 8
    episodic_low_rank: int = 128
    episodic_mode: str = "symmetric_associative_attention"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_proj = Linear(in_features, rank, bias=False)
        self.out_proj = Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.out_proj(self.in_proj(x))


class SharedEpisodicController(nn.Module):
    """
    Shared read/write controller for persistent latent memory tokens.

    The controller learns:
    - write salience and surprise-like priority
    - symmetric write attention from memory slots back to context tokens
    - recall queries used to retrieve the relevant persistent memory subset
    """

    def __init__(self, config):
        super().__init__()
        rank = config.episodic_low_rank
        total_slots = max(1, config.episodic_slots)
        summary_slots = max(1, total_slots // 4)
        anchor_slots = max(0, total_slots - summary_slots)
        if anchor_slots == 0:
            summary_slots = total_slots
        self.summary_slots = summary_slots
        self.anchor_slots = anchor_slots
        self.total_slots = summary_slots + anchor_slots
        self.write_key_proj = Linear(config.n_embd, config.episodic_dim, bias=False)
        self.read_query_proj = Linear(config.n_embd, config.episodic_dim, bias=False)
        self.recall_query_proj = Linear(config.n_embd, config.episodic_dim, bias=False)
        self.write_value_proj = LowRankLinear(config.n_embd, config.n_embd, rank)
        self.write_gate = Linear(config.n_embd, 1, bias=False)
        self.write_surprise_gate = Linear(config.n_embd, 1, bias=False)
        self.summary_slot_queries = nn.Parameter(torch.zeros(summary_slots, config.episodic_dim))
        self.anchor_slot_queries = nn.Parameter(torch.zeros(anchor_slots, config.episodic_dim))
        self.kind_bias = nn.Parameter(torch.zeros(2))
        self.no_write_bias = nn.Parameter(torch.zeros(()))
        self.write_recency_bias = nn.Parameter(torch.zeros(()))
        self.summary_budget = min(summary_slots, max(1, config.episodic_summary_budget))
        self.anchor_budget = min(anchor_slots, max(1, config.episodic_anchor_budget))
        self.score_scale = config.episodic_dim ** -0.5
        self.score_cap = None if config.episodic_beta <= 0 else float(config.episodic_beta)

    def slot_queries(self):
        return torch.cat([self.summary_slot_queries, self.anchor_slot_queries], dim=0)

    def scale_memory_scores(self, scores):
        scores = scores * self.score_scale
        if self.score_cap is not None:
            scores = scores.clamp(min=-self.score_cap, max=self.score_cap)
        return scores

    def slot_kind_ids(self, device):
        return torch.cat(
            [
                torch.zeros(self.summary_slots, device=device, dtype=torch.long),
                torch.ones(self.anchor_slots, device=device, dtype=torch.long),
            ],
            dim=0,
        )


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = config.ve_gate_channels
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def _expand_kv_heads(self, x):
        if self.n_kv_head == self.n_head:
            return x
        repeat = self.n_head // self.n_kv_head
        return x.repeat_interleave(repeat, dim=2)

    def _manual_attention(self, q, k, v, window_size, memory_kv=None):
        B, T, _, D = q.shape
        qh = q.permute(0, 2, 1, 3)
        kh = self._expand_kv_heads(k).permute(0, 2, 1, 3)
        vh = self._expand_kv_heads(v).permute(0, 2, 1, 3)

        scores_ctx = torch.einsum("bhtd,bhsd->bhts", qh, kh) / (D ** 0.5)
        positions = torch.arange(T, device=q.device)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        if window_size[0] > 0:
            causal = causal & (positions.unsqueeze(1) - positions.unsqueeze(0) < window_size[0])
        scores_ctx = scores_ctx.masked_fill(~causal.view(1, 1, T, T), float("-inf"))

        if memory_kv is not None and memory_kv["k"].size(1) > 0:
            km = self._expand_kv_heads(memory_kv["k"]).permute(0, 2, 1, 3)
            vm = self._expand_kv_heads(memory_kv["v"]).permute(0, 2, 1, 3)
            scores_mem = torch.einsum("bhtd,bhmd->bhtm", qh, km) / (D ** 0.5)
            scores = torch.cat([scores_mem + memory_kv["scores"].to(dtype=scores_mem.dtype).unsqueeze(1), scores_ctx], dim=-1)
            att = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            mem_len = km.size(-2)
            att_mem = att[..., :mem_len]
            att_ctx = att[..., mem_len:]
            out = torch.einsum("bhtm,bhmd->bhtd", att_mem, vm)
            out = out + torch.einsum("bhts,bhsd->bhtd", att_ctx, vh)
        else:
            att = F.softmax(scores_ctx.float(), dim=-1).to(dtype=q.dtype)
            out = torch.einsum("bhts,bhsd->bhtd", att, vh)

        return out.permute(0, 2, 1, 3).contiguous()

    @torch.no_grad()
    def memory_attention_summary(self, x, cos_sin, window_size, memory_kv=None):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        qh = q.permute(0, 2, 1, 3)
        kh = self._expand_kv_heads(k).permute(0, 2, 1, 3)
        scores_ctx = torch.einsum("bhtd,bhsd->bhts", qh, kh) / (self.head_dim ** 0.5)
        positions = torch.arange(T, device=x.device)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        if window_size[0] > 0:
            causal = causal & (positions.unsqueeze(1) - positions.unsqueeze(0) < window_size[0])
        scores_ctx = scores_ctx.masked_fill(~causal.view(1, 1, T, T), float("-inf"))

        if memory_kv is None or memory_kv["k"].size(1) == 0:
            return {
                "actual_mem_mass_last": 0.0,
                "actual_mem_mass_mean": 0.0,
                "actual_mem_score_max_last": 0.0,
                "actual_ctx_score_max_last": float(scores_ctx[..., -1, :].max().item()),
                "retrieved_score_max": 0.0,
            }

        km = self._expand_kv_heads(memory_kv["k"]).permute(0, 2, 1, 3)
        scores_mem = torch.einsum("bhtd,bhmd->bhtm", qh, km) / (self.head_dim ** 0.5)
        scores_mem = scores_mem + memory_kv["scores"].to(dtype=scores_mem.dtype).unsqueeze(1)
        scores = torch.cat([scores_mem, scores_ctx], dim=-1)
        att = F.softmax(scores.float(), dim=-1)
        mem_len = scores_mem.size(-1)
        return {
            "actual_mem_mass_last": float(att[..., -1, :mem_len].sum(dim=-1).mean().item()),
            "actual_mem_mass_mean": float(att[..., :mem_len].sum(dim=-1).mean().item()),
            "actual_mem_score_max_last": float(scores_mem[..., -1, :].max().item()),
            "actual_ctx_score_max_last": float(scores_ctx[..., -1, :].max().item()),
            "retrieved_score_max": float(memory_kv["scores"].max().item()) if memory_kv["scores"].numel() else 0.0,
        }

    def forward(self, x, ve, cos_sin, window_size, kv_cache, memory_kv=None):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        if memory_kv is not None and kv_cache is not None:
            raise NotImplementedError("KV-cache decoding is not implemented for persistent memory tokens")

        if memory_kv is not None:
            y = self._manual_attention(q, k, v, window_size, memory_kv=memory_kv)
        elif kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.episodic_max_write_tokens = config.episodic_max_write_tokens
        self.attn = CausalSelfAttention(config, layer_idx)
        self.episodic_memory_score_bias = nn.Parameter(torch.tensor(float(config.episodic_gate_init)))
        self.episodic_memory = PersistentMemoryTokens(
            token_dim=config.n_embd,
            key_dim=config.episodic_dim,
            max_slots=config.episodic_slots,
            top_k=config.episodic_top_k,
            decay=config.episodic_decay,
            score_scale=config.episodic_dim ** -0.5,
            score_cap=config.episodic_beta,
        )
        self.mlp = MLP(config)

    def compute_write_salience(self, key_source, controller):
        return controller.write_gate(norm(key_source)).squeeze(-1)

    def _select_write_positions(self, key_source, value_source, controller):
        max_write_tokens = min(max(1, self.episodic_max_write_tokens), key_source.size(1))
        if key_source.dim() != 3 or key_source.size(1) <= max_write_tokens:
            positions = torch.arange(key_source.size(1), device=key_source.device, dtype=torch.long).unsqueeze(0).expand(key_source.size(0), -1)
            return key_source, value_source, positions
        top_idx = self.compute_write_salience(key_source, controller).topk(max_write_tokens, dim=1).indices
        top_idx = top_idx.sort(dim=1).values
        key_gather = top_idx.unsqueeze(-1).expand(-1, -1, key_source.size(-1))
        value_gather = top_idx.unsqueeze(-1).expand(-1, -1, value_source.size(-1))
        return key_source.gather(1, key_gather), value_source.gather(1, value_gather), top_idx

    def _budget_mask(self, strengths, budget):
        budget = min(max(int(budget), 0), strengths.size(1))
        if budget <= 0:
            return torch.zeros_like(strengths)
        if budget >= strengths.size(1):
            return torch.ones_like(strengths)
        top_idx = strengths.topk(budget, dim=1).indices
        mask = torch.zeros_like(strengths)
        return mask.scatter(1, top_idx, 1.0)

    def build_episodic_state(self, key_source, value_source, controller, reward=0.0):
        key_source, value_source, source_positions = self._select_write_positions(key_source, value_source, controller)
        write_keys = controller.write_key_proj(norm(key_source))
        write_values = value_source + controller.write_value_proj(norm(value_source))

        # Symmetric write: memory slots query context-token keys using the same
        # associative geometry later used for recall. A no-write option keeps
        # memory scarce instead of forcing every token into a slot.
        salience_logits = self.compute_write_salience(key_source, controller)
        surprise_logits = controller.write_surprise_gate(norm(value_source)).squeeze(-1)
        if source_positions.numel() > 0:
            recency = source_positions.to(dtype=write_keys.dtype)
            recency = recency / recency.amax(dim=1, keepdim=True).clamp_min(1.0)
        else:
            recency = write_keys.new_zeros(write_keys.size(0), write_keys.size(1))
        write_priority = salience_logits + surprise_logits + controller.write_recency_bias.to(dtype=write_keys.dtype) * recency

        slot_queries = controller.slot_queries().to(device=write_keys.device, dtype=write_keys.dtype)
        slot_queries = slot_queries.unsqueeze(0).expand(write_keys.size(0), -1, -1)
        write_scores = controller.scale_memory_scores(torch.einsum("bse,bte->bst", norm(slot_queries), norm(write_keys)))
        write_scores = write_scores + write_priority.unsqueeze(1)

        token_count = max(int(write_scores.size(-1)), 1)
        no_write_score = write_scores.new_full((*write_scores.shape[:-1], 1), math.log(token_count))
        no_write_score = no_write_score + controller.no_write_bias.to(dtype=write_scores.dtype)
        slot_to_token_with_null = torch.softmax(torch.cat([write_scores, no_write_score], dim=-1).float(), dim=-1).to(dtype=write_values.dtype)
        slot_to_token = slot_to_token_with_null[..., :-1]
        token_to_slot = torch.softmax(write_scores.float(), dim=1).to(dtype=write_values.dtype)

        att = slot_to_token * token_to_slot
        raw_strengths = att.sum(dim=-1).clamp(min=0.0, max=1.0)
        att = att / raw_strengths.unsqueeze(-1).clamp_min(1e-6)

        priority_weight = torch.sigmoid(write_priority).unsqueeze(1)
        memory_tokens = torch.einsum("bst,btd->bsd", att, write_values)
        memory_keys = torch.einsum("bst,bte->bse", att, write_keys)
        memory_strengths = (slot_to_token * token_to_slot * priority_weight).sum(dim=-1).clamp(min=0.0, max=1.0)
        memory_strengths = memory_strengths * (0.3 + 0.7 * float(reward))

        summary_slots = controller.summary_slots
        summary_mask = self._budget_mask(memory_strengths[:, :summary_slots], controller.summary_budget)
        anchor_mask = self._budget_mask(memory_strengths[:, summary_slots:], controller.anchor_budget)
        slot_mask = torch.cat([summary_mask, anchor_mask], dim=1).to(dtype=memory_strengths.dtype)
        memory_tokens = memory_tokens * slot_mask.unsqueeze(-1)
        memory_keys = memory_keys * slot_mask.unsqueeze(-1)
        memory_strengths = memory_strengths * slot_mask
        source_pos = torch.einsum("bst,bt->bs", att, source_positions.to(dtype=att.dtype)).round().to(dtype=torch.long)
        source_pos = torch.where(slot_mask > 0, source_pos, source_pos.new_full(source_pos.shape, -1))
        kind_ids = controller.slot_kind_ids(write_keys.device).unsqueeze(0).expand(write_keys.size(0), -1)
        return self.episodic_memory.build_state(
            memory_tokens,
            memory_keys,
            memory_strengths,
            kind_ids=kind_ids,
            source_positions=source_pos,
        )

    def _memory_state(self, x, state_override=None):
        if state_override is not None:
            return state_override
        return self.episodic_memory.live_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)

    def _masked_recall_source(self, x, recall_mask=None):
        if recall_mask is None:
            return x.mean(dim=1, keepdim=True)
        mask = recall_mask.to(device=x.device, dtype=x.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1)
        if mask.size(1) != x.size(1):
            raise ValueError(f"Recall mask length {mask.size(1)} does not match sequence length {x.size(1)}")
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask).sum(dim=1, keepdim=True) / denom

    def _retrieve_memory_subset(self, x, controller, state_override=None, recall_mask=None):
        state = self._memory_state(x, state_override=state_override)
        recall_source = self._masked_recall_source(x, recall_mask=recall_mask)
        recall_query = controller.recall_query_proj(norm(recall_source))
        slots = self.episodic_memory.retrieve_slots(recall_query, state_override=state)
        memory_tokens = slots["values"].squeeze(1)
        memory_keys = slots["keys"].squeeze(1)
        memory_strengths = slots["strengths"].squeeze(1)
        kind_ids = slots["kind_ids"].squeeze(1).to(dtype=torch.long).clamp(min=0, max=controller.kind_bias.numel() - 1)
        source_positions = slots["source_positions"].squeeze(1)
        if memory_tokens.size(1) == 0:
            empty_scores = x.new_zeros(x.size(0), 0)
            return {
                "summary_query": recall_query,
                "tokens": memory_tokens,
                "keys": memory_keys,
                "strengths": memory_strengths,
                "kind_ids": kind_ids,
                "source_positions": source_positions,
                "scores": empty_scores,
            }
        kind_bias = controller.kind_bias.to(dtype=x.dtype).gather(0, kind_ids.reshape(-1)).view_as(kind_ids)
        scores = slots["scores"].squeeze(1) + kind_bias + self.episodic_memory_score_bias.to(dtype=x.dtype)
        return {
            "summary_query": recall_query,
            "tokens": memory_tokens,
            "keys": memory_keys,
            "strengths": memory_strengths,
            "kind_ids": kind_ids,
            "source_positions": source_positions,
            "scores": scores,
        }

    def read_episodic_memory_details(self, x, controller, state_override=None, recall_mask=None):
        retrieved = self._retrieve_memory_subset(x, controller, state_override=state_override, recall_mask=recall_mask)
        memory_tokens = retrieved["tokens"]
        memory_keys = retrieved["keys"]
        if memory_tokens.size(1) == 0:
            empty_att = x.new_zeros(x.size(0), x.size(1), 0)
            empty_values = x.new_zeros(x.size(0), x.size(1), 0, x.size(-1))
            return {
                "query": controller.read_query_proj(norm(x)),
                "summary_query": retrieved["summary_query"],
                "memory_values": empty_values,
                "attention": empty_att,
                "retrieved": torch.zeros_like(x),
                "scores": retrieved["scores"],
                "tokens": memory_tokens,
            }

        token_query = controller.read_query_proj(norm(x))
        memory_scores = controller.scale_memory_scores(torch.einsum("bte,bke->btk", norm(token_query), norm(memory_keys)))
        memory_scores = memory_scores + retrieved["scores"].unsqueeze(1)
        att = torch.softmax(memory_scores.float(), dim=-1).to(dtype=x.dtype)
        memory_values = memory_tokens.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        retrieved_values = torch.einsum("btk,bkd->btd", att, memory_tokens)
        return {
            "query": token_query,
            "summary_query": retrieved["summary_query"],
            "memory_values": memory_values,
            "attention": att,
            "retrieved": retrieved_values,
            "scores": retrieved["scores"],
            "tokens": memory_tokens,
        }

    def build_episodic_attention_kv(self, x, controller, state_override=None, recall_mask=None):
        retrieved = self._retrieve_memory_subset(x, controller, state_override=state_override, recall_mask=recall_mask)
        memory_tokens = retrieved["tokens"]
        if memory_tokens.size(1) == 0:
            return {
                "tokens": memory_tokens,
                "k": x.new_zeros(x.size(0), 0, self.n_kv_head, self.head_dim),
                "v": x.new_zeros(x.size(0), 0, self.n_kv_head, self.head_dim),
                "scores": x.new_zeros(x.size(0), x.size(1), 0),
                "keys": retrieved["keys"],
            }
        memory_k = self.attn.c_k(norm(memory_tokens)).view(x.size(0), memory_tokens.size(1), self.n_kv_head, self.head_dim)
        memory_v = self.attn.c_v(memory_tokens).view(x.size(0), memory_tokens.size(1), self.n_kv_head, self.head_dim)
        scores = retrieved["scores"].unsqueeze(1).expand(-1, x.size(1), -1)
        return {
            "tokens": memory_tokens,
            "k": memory_k,
            "v": memory_v,
            "scores": scores,
            "keys": retrieved["keys"],
        }

    def forward(self, x, ve, cos_sin, window_size, kv_cache, controller, episodic_state=None, recall_mask=None):
        memory_kv = self.build_episodic_attention_kv(x, controller, state_override=episodic_state, recall_mask=recall_mask)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, memory_kv=memory_kv)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.episodic_controller = SharedEpisodicController(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            }
        )
        self._special_token_ids = {}
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def set_tokenizer_special_ids(self, tokenizer):
        self._special_token_ids = {
            "user_start": int(tokenizer.encode_special("<|user_start|>")),
            "user_end": int(tokenizer.encode_special("<|user_end|>")),
            "assistant_start": int(tokenizer.encode_special("<|assistant_start|>")),
            "assistant_end": int(tokenizer.encode_special("<|assistant_end|>")),
        }

    def _derive_memory_metadata(self, token_ids):
        token_ids = self._coerce_idx(token_ids)
        B, T = token_ids.size()
        device = token_ids.device
        source_positions = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        role_ids = torch.ones((B, T), device=device, dtype=torch.long)
        group_ids = torch.zeros((B, T), device=device, dtype=torch.long)
        next_token_ids = torch.full((B, T), -1, device=device, dtype=torch.long)

        special = self._special_token_ids
        if not special:
            if T > 1:
                next_token_ids[:, :-1] = token_ids[:, 1:]
            return {
                "source_positions": source_positions,
                "role_ids": role_ids,
                "group_ids": group_ids,
                "next_token_ids": next_token_ids,
            }

        user_start = special["user_start"]
        user_end = special["user_end"]
        assistant_start = special["assistant_start"]
        assistant_end = special["assistant_end"]

        role_ids.zero_()
        group_ids.fill_(-1)
        for b in range(B):
            current_role = 0
            current_group = -1
            for t in range(T):
                token = int(token_ids[b, t].item())
                if token == user_start:
                    current_role = 1
                    current_group += 1
                    continue
                if token == assistant_start:
                    current_role = 2
                    current_group += 1
                    continue
                if token == user_end or token == assistant_end:
                    current_role = 0
                    continue
                role_ids[b, t] = current_role if current_role != 0 else 1
                group_ids[b, t] = current_group if current_group >= 0 else 0
            valid_groups = group_ids[b].unique(sorted=True)
            for group_id in valid_groups.tolist():
                if group_id < 0:
                    continue
                positions = torch.nonzero(group_ids[b] == group_id, as_tuple=False).flatten()
                if positions.numel() <= 1:
                    continue
                next_token_ids[b, positions[:-1]] = token_ids[b, positions[1:]]

        return {
            "source_positions": source_positions,
            "role_ids": role_ids,
            "group_ids": group_ids,
            "next_token_ids": next_token_ids,
        }

    def _coerce_idx(self, idx):
        if isinstance(idx, list):
            idx = torch.tensor([idx], dtype=torch.long, device=self.get_device())
        elif idx.ndim == 1:
            idx = idx.unsqueeze(0).to(device=self.get_device(), dtype=torch.long)
        else:
            idx = idx.to(device=self.get_device(), dtype=torch.long)
        return idx

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for proj in [
            self.episodic_controller.write_key_proj,
            self.episodic_controller.read_query_proj,
            self.episodic_controller.recall_query_proj,
        ]:
            torch.nn.init.uniform_(proj.weight, -s, s)
        torch.nn.init.zeros_(self.episodic_controller.write_gate.weight)
        torch.nn.init.zeros_(self.episodic_controller.write_surprise_gate.weight)
        torch.nn.init.uniform_(self.episodic_controller.summary_slot_queries, -s, s)
        torch.nn.init.uniform_(self.episodic_controller.anchor_slot_queries, -s, s)
        torch.nn.init.constant_(self.episodic_controller.kind_bias, self.config.episodic_kind_bias_init)
        torch.nn.init.zeros_(self.episodic_controller.no_write_bias)
        torch.nn.init.zeros_(self.episodic_controller.write_recency_bias)
        torch.nn.init.uniform_(self.episodic_controller.write_value_proj.in_proj.weight, -s, s)
        torch.nn.init.zeros_(self.episodic_controller.write_value_proj.out_proj.weight)

        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.constant_(block.episodic_memory_score_bias, self.config.episodic_gate_init)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _build_recall_mask(self, idx):
        special = self._special_token_ids
        if not special:
            return torch.ones_like(idx, dtype=torch.bool)
        assistant_start = special.get("assistant_start")
        if assistant_start is None:
            return torch.ones_like(idx, dtype=torch.bool)
        mask = torch.zeros_like(idx, dtype=torch.bool)
        for b in range(idx.size(0)):
            assistant_positions = torch.nonzero(idx[b] == assistant_start, as_tuple=False).flatten()
            if assistant_positions.numel() == 0:
                mask[b].fill_(True)
            else:
                mask[b, : int(assistant_positions[-1].item()) + 1] = True
        return mask

    def _prepare_inputs(self, idx, kv_cache=None):
        _, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        if kv_cache is None:
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        return x, cos_sin

    def _run_trunk(self, idx, x, cos_sin, kv_cache=None, collect_block_ios=False, detach_block_ios=True, memory_override=None, recall_mask=None):
        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        block_ios = [] if collect_block_ios else None

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            block_input = x.detach() if collect_block_ios and detach_block_ios else x
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            block_memory = None if memory_override is None else memory_override[i]
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[i],
                kv_cache,
                controller=self.episodic_controller,
                episodic_state=block_memory,
                recall_mask=recall_mask,
            )
            if collect_block_ios:
                block_output = x.detach() if detach_block_ios else x
                block_ios.append((block_input, block_output))
            if i == backout_layer:
                x_backout = x

        return x, x_backout, block_ios

    @torch.no_grad()
    def forward_and_collect(self, idx):
        return self.collect_block_ios(idx, detach=True)

    def collect_block_ios(self, idx, detach=False):
        idx = self._coerce_idx(idx)
        x, cos_sin = self._prepare_inputs(idx, kv_cache=None)
        _, _, block_ios = self._run_trunk(
            idx,
            x,
            cos_sin,
            kv_cache=None,
            collect_block_ios=True,
            detach_block_ios=detach,
        )
        return block_ios

    def empty_memory_override(self, batch_size, device=None, dtype=None):
        device = self.get_device() if device is None else device
        dtype = self.transformer.wte.weight.dtype if dtype is None else dtype
        return [
            block.episodic_memory.empty_state(batch_size=batch_size, device=device, dtype=dtype)
            for block in self.transformer.h
        ]

    def build_memory_state_from_block_ios(self, block_ios, reward=0.0):
        return [
            block.build_episodic_state(key_source, value_source, self.episodic_controller, reward=reward)
            for block, (key_source, value_source) in zip(self.transformer.h, block_ios)
        ]

    def _build_recall_trace_state(self, idx, key_positions, value_positions, reward=0.0):
        idx = self._coerce_idx(idx)
        if idx.size(1) == 0:
            return self.empty_memory_override(idx.size(0), device=idx.device, dtype=self.transformer.wte.weight.dtype)
        block_ios = self.collect_block_ios(idx, detach=False)
        states = []
        for block, (key_source_full, value_source_full) in zip(self.transformer.h, block_ios):
            key_pos = self._normalize_positions(key_positions, key_source_full.size(1), key_source_full.device)
            value_pos = self._normalize_positions(value_positions, value_source_full.size(1), value_source_full.device)
            if key_pos is None or value_pos is None or key_pos.numel() == 0 or value_pos.numel() == 0:
                states.append(block.episodic_memory.empty_state(batch_size=idx.size(0), device=idx.device, dtype=key_source_full.dtype))
                continue
            key_source = self._gather_hidden_positions(key_source_full, key_pos)
            value_source = self._gather_hidden_positions(value_source_full, value_pos)
            states.append(block.build_episodic_state(key_source, value_source, self.episodic_controller, reward=reward))
        return states

    def _update_memory_override(self, current_override, write_states):
        if current_override is None:
            current_override = self.empty_memory_override(
                batch_size=write_states[0]["tokens"].size(0),
                device=write_states[0]["tokens"].device,
                dtype=write_states[0]["tokens"].dtype,
            )
        return [
            block.episodic_memory.updated_state(current_state, write_state)
            for block, current_state, write_state in zip(self.transformer.h, current_override, write_states)
        ]

    def _stack_memory_overrides(self, per_example_states):
        if not per_example_states:
            return self.empty_memory_override(0)
        stacked = []
        for layer_idx in range(self.config.n_layer):
            keys = per_example_states[0][layer_idx].keys()
            layer_state = {}
            for key in keys:
                layer_state[key] = torch.cat([example[layer_idx][key] for example in per_example_states], dim=0)
            stacked.append(layer_state)
        return stacked

    def _split_memory_events(self, token_row, write_mode="turn"):
        token_row = [int(t) for t in token_row]
        if not token_row:
            return []
        if write_mode == "full_context":
            return [{"kind": "memorize", "tokens": token_row}]

        special = self._special_token_ids
        if not special:
            return [{"kind": "memorize", "tokens": token_row}]

        bos = token_row[0]
        user_start = special["user_start"]
        user_end = special["user_end"]
        assistant_start = special["assistant_start"]
        assistant_end = special["assistant_end"]

        events = []
        i = 0
        n = len(token_row)
        while i < n:
            if token_row[i] != user_start:
                i += 1
                continue
            user_end_idx = next((j for j in range(i + 1, n) if token_row[j] == user_end), None)
            if user_end_idx is None:
                break
            if write_mode == "user":
                segment = [bos] + token_row[i:user_end_idx + 1]
                events.append({"kind": "memorize", "tokens": segment})
                i = user_end_idx + 1
                continue

            assistant_start_idx = next((j for j in range(user_end_idx + 1, n) if token_row[j] == assistant_start), None)
            assistant_end_idx = None if assistant_start_idx is None else next((j for j in range(assistant_start_idx + 1, n) if token_row[j] == assistant_end), None)
            if assistant_start_idx is None or assistant_end_idx is None:
                break

            segment = [bos] + token_row[i:assistant_end_idx + 1]
            events.append({"kind": "memorize", "tokens": segment})
            if write_mode == "turn_recall":
                query_positions = list(range(2, 2 + max(user_end_idx - i - 1, 0)))
                value_start = 2 + max(user_end_idx - i - 1, 0) + 2
                value_positions = list(range(value_start, value_start + max(assistant_end_idx - assistant_start_idx - 1, 0)))
                if query_positions and value_positions:
                    events.append(
                        {
                            "kind": "recall_trace",
                            "tokens": segment,
                            "query_positions": query_positions,
                            "value_positions": value_positions,
                        }
                    )
            i = assistant_end_idx + 1
        return events

    def _build_memory_state_single(self, idx, reward=0.0, write_mode="turn"):
        idx = self._coerce_idx(idx)
        if idx.size(1) == 0:
            return self.empty_memory_override(1, device=idx.device, dtype=self.transformer.wte.weight.dtype)
        if write_mode == "full_context":
            block_ios = self.collect_block_ios(idx, detach=False)
            return self.build_memory_state_from_block_ios(block_ios, reward=reward)

        state = self.empty_memory_override(1, device=idx.device, dtype=self.transformer.wte.weight.dtype)
        events = self._split_memory_events(idx[0].tolist(), write_mode=write_mode)
        if not events:
            return state
        for event in events:
            tokens = self._coerce_idx(event["tokens"])
            if event["kind"] == "memorize":
                write_state = self.build_memory_state(tokens, reward=reward, write_mode="full_context")
            elif event["kind"] == "recall_trace":
                write_state = self._build_recall_trace_state(
                    tokens,
                    event["query_positions"],
                    event["value_positions"],
                    reward=reward,
                )
            else:
                raise ValueError(f"Unknown memory event kind: {event['kind']}")
            state = self._update_memory_override(state, write_state)
        return state

    def build_memory_state(self, idx, reward=0.0, write_mode="turn"):
        idx = self._coerce_idx(idx)
        if idx.size(1) == 0:
            return self.empty_memory_override(idx.size(0), device=idx.device, dtype=self.transformer.wte.weight.dtype)
        per_example_states = [
            self._build_memory_state_single(idx[b:b + 1], reward=reward, write_mode=write_mode)
            for b in range(idx.size(0))
        ]
        return self._stack_memory_overrides(per_example_states)

    def collect_block_outputs(self, idx, positions=None, memory_override=None):
        idx = self._coerce_idx(idx)
        if idx.size(1) == 0:
            return [self.transformer.wte.weight.new_zeros((idx.size(0), 0, self.config.n_embd)) for _ in self.transformer.h]
        x, cos_sin = self._prepare_inputs(idx, kv_cache=None)
        recall_mask = self._build_recall_mask(idx)
        _, _, block_ios = self._run_trunk(
            idx,
            x,
            cos_sin,
            kv_cache=None,
            collect_block_ios=True,
            detach_block_ios=False,
            memory_override=memory_override,
            recall_mask=recall_mask,
        )
        outputs = [block_output for _, block_output in block_ios]
        if positions is None:
            return outputs
        if isinstance(positions, int):
            return [out[:, positions:positions + 1, :] for out in outputs]

        pos = torch.as_tensor(positions, device=idx.device, dtype=torch.long)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(idx.size(0), -1)
        gathered = []
        for out in outputs:
            gather_idx = pos.unsqueeze(-1).expand(-1, -1, out.size(-1))
            gathered.append(out.gather(1, gather_idx))
        return gathered

    @torch.no_grad()
    def collect_memory_diagnostics(self, idx, memory_override=None):
        idx = self._coerce_idx(idx)
        x, cos_sin = self._prepare_inputs(idx, kv_cache=None)
        recall_mask = self._build_recall_mask(idx)
        x0 = x
        diagnostics = []

        def _entropy(att):
            if att.numel() == 0:
                return att.new_zeros(())
            p = att.clamp_min(1e-9)
            return (-(p * p.log()).sum(dim=-1)).mean()

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            block_memory = None if memory_override is None else memory_override[i]
            memory_kv = block.build_episodic_attention_kv(
                x,
                self.episodic_controller,
                state_override=block_memory,
                recall_mask=recall_mask,
            )
            attn_memory = block.attn.memory_attention_summary(
                norm(x),
                cos_sin,
                self.window_sizes[i],
                memory_kv=memory_kv,
            )
            attn_out = block.attn(
                norm(x),
                ve,
                cos_sin,
                self.window_sizes[i],
                kv_cache=None,
                memory_kv=memory_kv,
            )
            x_after_attn = x + attn_out
            mem_details = block.read_episodic_memory_details(
                x_after_attn,
                self.episodic_controller,
                state_override=block_memory,
                recall_mask=recall_mask,
            )
            mlp_out = block.mlp(norm(x_after_attn))
            x = x_after_attn + mlp_out

            slot_count = int((block.episodic_memory.live_state(batch_size=1)["strengths"][0] > 1e-6).sum().item()) if block_memory is None else int((block_memory["strengths"][0] > 1e-6).sum().item())
            diagnostics.append(
                {
                    "layer": i,
                    "slot_count": slot_count,
                    "memory_gate": float(torch.sigmoid(block.episodic_memory_score_bias).item()),
                    "kv_token_count": int(mem_details["memory_values"].size(-2)),
                    "attn_out": attn_out.detach(),
                    "mem_read": mem_details["retrieved"].detach(),
                    "mlp_out": mlp_out.detach(),
                    "mem_attention": mem_details["attention"].detach(),
                    "slot_scores": mem_details["scores"].detach(),
                    "attn_norm": float(attn_out.norm(dim=-1).mean().item()),
                    "mem_norm": float(mem_details["retrieved"].norm(dim=-1).mean().item()),
                    "mlp_norm": float(mlp_out.norm(dim=-1).mean().item()),
                    "mem_entropy": float(_entropy(mem_details["attention"]).item()),
                    "actual_mem_mass_last": attn_memory["actual_mem_mass_last"],
                    "actual_mem_mass_mean": attn_memory["actual_mem_mass_mean"],
                    "actual_mem_score_max_last": attn_memory["actual_mem_score_max_last"],
                    "actual_ctx_score_max_last": attn_memory["actual_ctx_score_max_last"],
                    "retrieved_score_max": attn_memory["retrieved_score_max"],
                    "slot_strength_mean": float((block_memory["strengths"][0] if block_memory is not None else block.episodic_memory.live_state(batch_size=1)["strengths"][0]).mean().item()),
                    "slot_strength_max": float((block_memory["strengths"][0] if block_memory is not None else block.episodic_memory.live_state(batch_size=1)["strengths"][0]).max().item()),
                }
            )
        return diagnostics

    def _normalize_positions(self, positions, length, device):
        if positions is None:
            return None
        if length <= 0:
            return torch.zeros((0,), device=device, dtype=torch.long)
        pos = torch.as_tensor(positions, device=device, dtype=torch.long).flatten()
        pos = pos.clamp(min=0, max=length - 1)
        if pos.numel() == 0:
            return pos
        return pos.unique(sorted=True)

    def _gather_hidden_positions(self, hidden, positions):
        if positions is None or positions.numel() == 0:
            return hidden[:, :0, :]
        gather_idx = positions.view(1, -1, 1).expand(hidden.size(0), -1, hidden.size(-1))
        return hidden.gather(1, gather_idx)

    @torch.no_grad()
    def write_to_memory_banks(self, block_ios, token_ids=None, reward=0.0):
        for block, (key_source, value_source) in zip(self.transformer.h, block_ios):
            state = block.build_episodic_state(key_source, value_source, self.episodic_controller, reward=reward)
            block.episodic_memory.write(state)

    @torch.no_grad()
    def write_recall_trace(self, idx, key_positions, value_positions, reward=0.0):
        states = self._build_recall_trace_state(idx, key_positions, value_positions, reward=reward)
        for block, state in zip(self.transformer.h, states):
            block.episodic_memory.write(state)

    @torch.no_grad()
    def replay_memory_sequence(self, idx, reward=0.0, write_mode="turn"):
        idx = self._coerce_idx(idx)
        state = self.build_memory_state(idx, reward=reward, write_mode=write_mode)
        self.clear_memory_banks()
        for block, bank_state in zip(self.transformer.h, state):
            block.episodic_memory.import_state(bank_state)

    @torch.no_grad()
    def clear_memory_banks(self):
        for block in self.transformer.h:
            block.episodic_memory.clear()

    @torch.no_grad()
    def has_live_memory(self):
        return any((block.episodic_memory.strengths > 1e-6).any().item() for block in self.transformer.h)

    @torch.no_grad()
    def memorize(self, idx, reward=0.0):
        idx = self._coerce_idx(idx)
        if idx.size(1) == 0:
            return
        block_ios = self.forward_and_collect(idx)
        self.write_to_memory_banks(block_ios, token_ids=idx, reward=reward)

    @torch.no_grad()
    def memory_state_dict(self):
        return {
            "version": 3,
            "n_layer": self.config.n_layer,
            "episodic_dim": self.config.episodic_dim,
            "episodic_value_dim": self.config.n_embd,
            "banks": [block.episodic_memory.export_state() for block in self.transformer.h],
        }

    @torch.no_grad()
    def load_memory_state_dict(self, state_dict, strict=True):
        if strict:
            assert state_dict["n_layer"] == self.config.n_layer, "Memory state layer count mismatch"
            assert len(state_dict.get("banks", [])) == self.config.n_layer, "Memory state bank count mismatch"
        self.clear_memory_banks()
        for block, bank_state in zip(self.transformer.h, state_dict.get("banks", [])):
            block.episodic_memory.import_state(bank_state)

    @torch.no_grad()
    def save_memory_state(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.memory_state_dict(), path)

    @torch.no_grad()
    def load_memory_state(self, path, strict=True):
        state_dict = torch.load(path, map_location=self.get_device(), weights_only=False)
        self.load_memory_state_dict(state_dict, strict=strict)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        episodic_scalars = sum(block.episodic_memory_score_bias.numel() for block in self.transformer.h) + self.episodic_controller.kind_bias.numel()
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
            + episodic_scalars
        )
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        episodic_controller = sum(p.numel() for p in self.episodic_controller.parameters())
        episodic_scalars = sum(block.episodic_memory_score_bias.numel() for block in self.transformer.h)
        transformer_params = sum(p.numel() for p in self.transformer.h.parameters()) - episodic_scalars
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
            + episodic_scalars
        )
        total = wte + value_embeds + lm_head + episodic_controller + transformer_params + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "episodic_controller": episodic_controller,
            "transformer_params": transformer_params,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, _, _, _ = get_dist_info()

        block_matrix_params = []
        episodic_scalar_params = []
        episodic_matrix_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("episodic_memory_score_bias") or name == "episodic_controller.kind_bias":
                episodic_scalar_params.append(param)
            elif name.startswith("episodic_controller.") or ".episodic_" in name:
                episodic_matrix_params.append(param)
            elif name.startswith("transformer.h."):
                block_matrix_params.append(param)

        embedding_params = list(self.transformer.wte.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        matrix_params = [p for p in block_matrix_params if p.ndim >= 2]
        scalar_block_params = [p for p in block_matrix_params if p.ndim < 2]
        scalar_block_params.extend(episodic_scalar_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=episodic_matrix_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=scalar_block_params, lr=0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )

        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _compute_logits(self, idx, kv_cache=None, memory_override=None):
        x, cos_sin = self._prepare_inputs(idx, kv_cache=kv_cache)
        recall_mask = self._build_recall_mask(idx)
        x, x_backout, _ = self._run_trunk(
            idx,
            x,
            cos_sin,
            kv_cache=kv_cache,
            collect_block_ios=False,
            memory_override=memory_override,
            recall_mask=recall_mask,
        )
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        top_block = self.transformer.h[-1]
        top_override = None if memory_override is None else memory_override[-1]
        top_details = top_block.read_episodic_memory_details(
            x,
            self.episodic_controller,
            state_override=top_override,
            recall_mask=recall_mask,
        )
        top_memory = top_details["retrieved"]
        memory_probe_logits = self.lm_head(norm(top_memory))
        logits = logits[..., :self.config.vocab_size].float()
        memory_probe_logits = memory_probe_logits[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        memory_probe_logits = softcap * torch.tanh(memory_probe_logits / softcap)
        return logits, memory_probe_logits, top_details["query"], top_memory

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", memory_override=None, return_components=False):
        logits, memory_probe_logits, top_query, top_memory = self._compute_logits(
            idx,
            kv_cache=kv_cache,
            memory_override=memory_override,
        )
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_components:
                return {
                    "loss": loss,
                    "logits": logits,
                    "memory_probe_logits": memory_probe_logits,
                    "memory_query": top_query,
                    "memory_read": top_memory,
                }
            return loss
        if return_components:
            return {
                "logits": logits,
                "memory_probe_logits": memory_probe_logits,
                "memory_query": top_query,
                "memory_read": top_memory,
            }
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
