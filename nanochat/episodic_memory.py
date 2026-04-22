"""
Persistent latent memory tokens for NanoChat.

This module keeps runtime memory state intentionally simple:
- a fixed set of persistent latent memory tokens per layer
- learned retrieval keys attached to those tokens
- scalar strengths used for persistence and retrieval bias
- no merge/replace heuristics or explicit token-slot bookkeeping
"""

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_base_dir


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


class PersistentMemoryTokens(nn.Module):
    """
    Runtime persistent memory represented as latent token-like vectors.

    The bank itself has no trainable parameters. Learned behavior lives in the
    model's read/write controller; this module only stores per-session state and
    exposes simple retrieval / persistence helpers.
    """

    def __init__(
        self,
        token_dim,
        key_dim,
        max_slots=256,
        top_k=8,
        decay=0.995,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.key_dim = key_dim
        self.max_slots = max_slots
        self.top_k = top_k
        self.decay = decay
        self.register_buffer("tokens", torch.zeros(max_slots, token_dim), persistent=False)
        self.register_buffer("keys", torch.zeros(max_slots, key_dim), persistent=False)
        self.register_buffer("strengths", torch.zeros(max_slots), persistent=False)
        self.register_buffer("kind_ids", torch.zeros(max_slots, dtype=torch.long), persistent=False)
        self.register_buffer("source_positions", torch.full((max_slots,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("num_slots", torch.zeros((), dtype=torch.long), persistent=False)

    def _coerce_state(self, state):
        tokens = state["tokens"]
        keys = state["keys"]
        strengths = state["strengths"]
        kind_ids = state.get("kind_ids")
        source_positions = state.get("source_positions")

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
        if strengths.dim() == 1:
            strengths = strengths.unsqueeze(0)
        if kind_ids is None:
            kind_ids = torch.zeros(tokens.size(1), device=tokens.device, dtype=torch.long)
        if kind_ids.dim() == 1:
            kind_ids = kind_ids.unsqueeze(0).expand(tokens.size(0), -1)
        kind_ids = kind_ids.to(device=tokens.device, dtype=torch.long)
        if source_positions is None:
            source_positions = torch.full(
                (tokens.size(0), tokens.size(1)),
                -1,
                device=tokens.device,
                dtype=torch.long,
            )
        if source_positions.dim() == 1:
            source_positions = source_positions.unsqueeze(0).expand(tokens.size(0), -1)
        source_positions = source_positions.to(device=tokens.device, dtype=torch.long)
        return tokens, keys, strengths, kind_ids, source_positions

    def build_state(self, tokens, keys, strengths, kind_ids=None, source_positions=None):
        tokens, keys, strengths, kind_ids, source_positions = self._coerce_state(
            {
                "tokens": tokens,
                "keys": keys,
                "strengths": strengths,
                "kind_ids": kind_ids,
                "source_positions": source_positions,
            }
        )
        strengths = strengths.clamp(min=0.0, max=1.0)
        return {
            "tokens": tokens,
            "values": tokens,
            "keys": keys,
            "strengths": strengths,
            "weights": strengths,
            "kind_ids": kind_ids,
            "source_positions": source_positions,
        }

    def live_state(self, *, batch_size=1, device=None, dtype=None):
        device = self.tokens.device if device is None else device
        dtype = self.tokens.dtype if dtype is None else dtype
        # Keep runtime buffers contiguous: some CPU diagnostics paths become
        # numerically unstable when downstream ops (notably `rms_norm`) receive
        # expanded zero-stride views.
        tokens = self.tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        keys = self.keys.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        strengths = self.strengths.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1).clone()
        kind_ids = self.kind_ids.to(device=device).unsqueeze(0).expand(batch_size, -1).clone()
        source_positions = self.source_positions.to(device=device).unsqueeze(0).expand(batch_size, -1).clone()
        return {
            "tokens": tokens,
            "values": tokens,
            "keys": keys,
            "strengths": strengths,
            "weights": strengths,
            "kind_ids": kind_ids,
            "source_positions": source_positions,
        }

    def retrieve_slots(self, query, state_override=None, slot_mask=None):
        """
        Return top retrieved persistent memory tokens.

        `query` is expected in the same latent retrieval-key space as the stored
        `keys`, i.e. shape `[B, T, key_dim]` or `[B, key_dim]`.
        """
        if state_override is None:
            state_override = self.live_state(batch_size=query.size(0), device=query.device, dtype=query.dtype)
        tokens, keys, strengths, kind_ids, source_positions = self._coerce_state(state_override)
        tokens = tokens.to(device=query.device, dtype=query.dtype)
        keys = keys.to(device=query.device, dtype=query.dtype)
        strengths = strengths.to(device=query.device, dtype=query.dtype)
        kind_ids = kind_ids.to(device=query.device)
        source_positions = source_positions.to(device=query.device)

        if query.dim() == 2:
            query = query.unsqueeze(1)
        query = _norm(query.contiguous())
        keys = _norm(keys.contiguous())

        active_mask = strengths > 1e-6
        if slot_mask is not None:
            slot_mask = slot_mask.to(device=query.device, dtype=torch.bool)
            if slot_mask.dim() == 1:
                slot_mask = slot_mask.unsqueeze(0).expand(query.size(0), -1)
            active_mask = active_mask & slot_mask

        if not active_mask.any():
            empty_scores = query.new_zeros(query.size(0), query.size(1), 0)
            empty_tokens = tokens[:, :0, :]
            empty_keys = keys[:, :0, :]
            empty_strengths = strengths[:, :0]
            empty_kinds = kind_ids[:, :0]
            empty_positions = source_positions[:, :0]
            return {
                "values": empty_tokens,
                "tokens": empty_tokens,
                "keys": empty_keys,
                "scores": empty_scores,
                "weights": empty_strengths,
                "strengths": empty_strengths,
                "kind_ids": empty_kinds,
                "source_positions": empty_positions,
            }

        safe_strengths = strengths.clamp(min=1e-6)
        scores = torch.einsum("bte,bse->bts", query, keys) + safe_strengths.log().unsqueeze(1)
        scores = scores.masked_fill(~active_mask.unsqueeze(1), float("-inf"))
        slot_count = keys.size(1)
        k = slot_count if self.top_k <= 0 else min(self.top_k, slot_count)
        if k < slot_count:
            top_scores, top_idx = scores.topk(k, dim=-1)
        else:
            top_scores = scores
            top_idx = torch.arange(slot_count, device=query.device, dtype=torch.long).view(1, 1, slot_count).expand_as(scores)

        # `gather` over expanded runtime buffers can return garbage on CPU in some
        # cases, so use explicit batched indexing here. The retrieved tensors are
        # small (top-k memory slots), and this keeps diagnostics/load smokes stable.
        batch_idx = torch.arange(query.size(0), device=query.device, dtype=torch.long).view(-1, 1, 1)
        gathered_tokens = tokens[batch_idx, top_idx]
        gathered_keys = keys[batch_idx, top_idx]
        gathered_strengths = strengths[batch_idx, top_idx]
        gathered_kinds = kind_ids[batch_idx, top_idx]
        gathered_positions = source_positions[batch_idx, top_idx]
        return {
            "values": gathered_tokens,
            "tokens": gathered_tokens,
            "keys": gathered_keys,
            "scores": top_scores,
            "weights": gathered_strengths,
            "strengths": gathered_strengths,
            "kind_ids": gathered_kinds,
            "source_positions": gathered_positions,
        }

    def read(self, query, state_override=None):
        slots = self.retrieve_slots(query, state_override=state_override)
        if slots["values"].size(-2) == 0:
            return torch.zeros(*query.shape[:-1], self.token_dim, device=query.device, dtype=query.dtype)
        att = torch.softmax(slots["scores"], dim=-1)
        return (att.unsqueeze(-1) * slots["values"]).sum(dim=-2)

    @torch.no_grad()
    def write(self, state):
        tokens, keys, strengths, kind_ids, source_positions = self._coerce_state(state)
        tokens = tokens.mean(dim=0).to(device=self.tokens.device, dtype=self.tokens.dtype)
        keys = keys.mean(dim=0).to(device=self.keys.device, dtype=self.keys.dtype)
        strengths = strengths.mean(dim=0).to(device=self.strengths.device, dtype=self.strengths.dtype).clamp(min=0.0, max=1.0)
        kind_ids = kind_ids[0].to(device=self.kind_ids.device, dtype=self.kind_ids.dtype)
        source_positions = source_positions.float().mean(dim=0).round().to(device=self.source_positions.device, dtype=self.source_positions.dtype)

        if tokens.size(0) != self.max_slots or keys.size(0) != self.max_slots:
            limit = min(tokens.size(0), self.max_slots)
            padded_tokens = self.tokens.new_zeros(self.max_slots, self.token_dim)
            padded_keys = self.keys.new_zeros(self.max_slots, self.key_dim)
            padded_strengths = self.strengths.new_zeros(self.max_slots)
            padded_kinds = self.kind_ids.new_zeros(self.max_slots)
            padded_positions = self.source_positions.new_full((self.max_slots,), -1)
            padded_tokens[:limit] = tokens[:limit]
            padded_keys[:limit] = keys[:limit]
            padded_strengths[:limit] = strengths[:limit]
            padded_kinds[:limit] = kind_ids[:limit]
            padded_positions[:limit] = source_positions[:limit]
            tokens = padded_tokens
            keys = padded_keys
            strengths = padded_strengths
            kind_ids = padded_kinds
            source_positions = padded_positions

        alpha = strengths.unsqueeze(-1)
        self.tokens.mul_(self.decay)
        self.tokens.add_(alpha * (tokens - self.tokens))
        self.keys.mul_(self.decay)
        self.keys.add_(alpha * (keys - self.keys))
        self.strengths.mul_(self.decay)
        self.strengths.copy_(torch.maximum(self.strengths, strengths).clamp_(min=0.0, max=1.0))
        self.kind_ids.copy_(kind_ids)
        self.source_positions.copy_(source_positions)
        self.num_slots.fill_(int((self.strengths > 1e-6).sum().item()))

    @torch.no_grad()
    def clear(self):
        self.tokens.zero_()
        self.keys.zero_()
        self.strengths.zero_()
        self.kind_ids.zero_()
        self.source_positions.fill_(-1)
        self.num_slots.zero_()

    @torch.no_grad()
    def export_state(self):
        return {
            "tokens": self.tokens.detach().clone().cpu(),
            "keys": self.keys.detach().clone().cpu(),
            "strengths": self.strengths.detach().clone().cpu(),
            "kind_ids": self.kind_ids.detach().clone().cpu(),
            "source_positions": self.source_positions.detach().clone().cpu(),
            "num_slots": int(self.num_slots.item()),
        }

    @torch.no_grad()
    def import_state(self, state):
        self.clear()
        if "tokens" not in state or "keys" not in state:
            return
        tokens = state["tokens"]
        keys = state["keys"]
        if tokens.shape[-1] != self.token_dim or keys.shape[-1] != self.key_dim:
            return
        limit = min(tokens.shape[0], self.max_slots)
        self.tokens[:limit].copy_(tokens[:limit].to(device=self.tokens.device, dtype=self.tokens.dtype))
        self.keys[:limit].copy_(keys[:limit].to(device=self.keys.device, dtype=self.keys.dtype))
        strengths = state.get("strengths", torch.ones(limit))
        self.strengths[:limit].copy_(strengths[:limit].to(device=self.strengths.device, dtype=self.strengths.dtype))
        kind_ids = state.get("kind_ids", torch.zeros(limit, dtype=torch.long))
        self.kind_ids[:limit].copy_(kind_ids[:limit].to(device=self.kind_ids.device, dtype=self.kind_ids.dtype))
        source_positions = state.get("source_positions", torch.full((limit,), -1, dtype=torch.long))
        self.source_positions[:limit].copy_(source_positions[:limit].to(device=self.source_positions.device, dtype=self.source_positions.dtype))
        self.num_slots.fill_(int(state.get("num_slots", int((self.strengths > 1e-6).sum().item()))))


def sanitize_session_id(session_id):
    session_id = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("._")
    return session_id or "default"


def get_session_memory_path(source, model_tag, session_id):
    session_id = sanitize_session_id(session_id)
    model_tag = sanitize_session_id(model_tag or "auto")
    base_dir = get_base_dir()
    return os.path.join(base_dir, "session_memories", source, model_tag, f"{session_id}.pt")
