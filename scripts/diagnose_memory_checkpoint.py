"""
Instrument and ablate a memory-enabled NanoChat checkpoint.

This script is for diagnosis, not training. It tries to answer:

1. Does the checkpoint actually use persistent memory on recall tasks?
2. Which path matters more on the current checkpoint?
   - top-logit memory
   - episodic residual read
   - latent workspace residual read
3. At which answer positions does memory help or hurt?
4. Are the memory paths active but weak, or largely ignored?
"""

import argparse
import contextlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine


CORE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "brief", "bullet", "can", "color",
    "compact", "concise", "current", "database", "detail", "details", "did", "do",
    "dog", "dogs", "drink", "editor", "favorite", "for", "format", "from", "i",
    "in", "is", "it", "its", "keep", "later", "like", "live", "lives", "lists",
    "me", "my", "name", "of", "on", "our", "pet", "prefer", "preferred", "prose",
    "python", "region", "remember", "remembered", "replies", "reply", "repo", "s",
    "setup", "shell", "should", "short", "supposed", "that", "the", "their", "them",
    "these", "they", "this", "those", "to", "tone", "use", "user", "uses", "version",
    "was", "we", "were", "what", "which", "you", "your",
}


def normalize_text(text):
    text = text.replace("<|assistant_end|>", " ")
    text = text.replace("<|user_start|>", " ")
    text = text.replace("<|assistant_start|>", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_core_tokens(text):
    text = text.replace("<|assistant_end|>", " ")
    text = text.replace("<|user_start|>", " ")
    text = text.replace("<|assistant_start|>", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9./-]+", " ", text)
    tokens = []
    for tok in text.split():
        tok = tok.strip(".")
        if not tok or tok in CORE_STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def core_fact_match(expected, got):
    expected_tokens = extract_core_tokens(expected)
    got_tokens = extract_core_tokens(got)
    if not expected_tokens and not got_tokens:
        return True
    return expected_tokens == got_tokens


def load_conversations(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_eval_conversation(messages):
    if len(messages) < 4:
        return None
    has_system = messages[0]["role"] == "system"
    start = 1 if has_system else 0
    pairs = []
    for i in range(start, len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append(i)
    if not pairs:
        return None
    final_idx = pairs[-1]
    prior_messages = messages[:final_idx]
    recall_user = messages[final_idx]["content"]
    expected_answer = messages[final_idx + 1]["content"]
    return prior_messages, recall_user, expected_answer


def build_turn_prefix(tokenizer, user_text):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    ids = [bos, user_start]
    ids.extend(tokenizer.encode(user_text))
    ids.append(user_end)
    ids.append(assistant_start)
    return ids


def build_user_turn_tokens(tokenizer, text):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    ids = [bos, user_start]
    ids.extend(tokenizer.encode(text))
    ids.append(user_end)
    return ids


def rank_of_token(logits, token_id):
    target_logit = logits[token_id]
    return int((logits > target_logit).sum().item()) + 1


def top_tokens(tokenizer, logits, k=5):
    top_vals, top_idx = torch.topk(logits, k=min(k, logits.numel()))
    out = []
    for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
        out.append({
            "token_id": int(idx),
            "token_text": tokenizer.decode([idx]),
            "logit": float(score),
        })
    return out


def build_empty_memory_override(model):
    device = model.get_device()
    empty = []
    for _ in model.transformer.h:
        empty.append({
            "keys": torch.zeros(0, model.config.episodic_dim, device=device, dtype=model.transformer.wte.weight.dtype),
            "values": torch.zeros(0, model.config.n_embd, device=device, dtype=model.transformer.wte.weight.dtype),
            "weights": torch.zeros(0, device=device, dtype=model.transformer.wte.weight.dtype),
            "source_positions": torch.full((0,), -1, device=device, dtype=torch.long),
            "token_ids": torch.full((0,), -1, device=device, dtype=torch.long),
        })
    return empty


@contextlib.contextmanager
def ablate_model(model, mode):
    saved = []
    if mode == "full":
        yield
        return

    def save_param(param, value):
        saved.append((param, param.detach().clone()))
        param.copy_(value)

    with torch.no_grad():
        if mode == "no_top_logits":
            pass
        elif mode == "no_workspace":
            for block in model.transformer.h:
                save_param(block.episodic_plan_score_gate, torch.full_like(block.episodic_plan_score_gate, -30.0))
                save_param(block.episodic_plan_value_gate, torch.full_like(block.episodic_plan_value_gate, -30.0))
        elif mode == "no_residual_memory":
            for block in model.transformer.h:
                save_param(block.episodic_payload_score_gate, torch.full_like(block.episodic_payload_score_gate, -30.0))
                save_param(block.episodic_payload_value_gate, torch.full_like(block.episodic_payload_value_gate, -30.0))
                save_param(block.episodic_lexical_gate, torch.full_like(block.episodic_lexical_gate, -30.0))
        elif mode == "no_token_logits":
            pass
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")
    try:
        yield
    finally:
        with torch.no_grad():
            for param, old in reversed(saved):
                param.copy_(old)


def restore_memory_state(model, state_dict, enabled=True):
    model.clear_memory_banks()
    if enabled and state_dict is not None:
        model.load_memory_state_dict(state_dict, strict=False)


def run_prompt(engine, tokenizer, user_text, max_tokens, temperature, top_k, seed):
    prefix = build_turn_prefix(tokenizer, user_text)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    out = []
    for token_column, _ in engine.generate(
        prefix,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    ):
        token = token_column[0]
        out.append(token)
        if token == assistant_end:
            break
    if out and out[-1] == assistant_end:
        out = out[:-1]
    return tokenizer.decode(out)


def teacher_forced_positions(model, tokenizer, recall_user, expected_answer, memory_override=None):
    prefix = build_turn_prefix(tokenizer, recall_user)
    answer_tokens = tokenizer.encode(expected_answer)
    rows = []
    for pos, correct_token in enumerate(answer_tokens):
        ids = prefix + answer_tokens[:pos]
        outputs = model(model._coerce_idx(ids), return_components=True, memory_override=memory_override)
        logits = outputs["logits"][0, -1]
        log_probs = F.log_softmax(logits, dim=-1)
        correct_logit = float(logits[correct_token].item())
        correct_logprob = float(log_probs[correct_token].item())
        rank = rank_of_token(logits, correct_token)
        top_id = int(torch.argmax(logits).item())
        rows.append({
            "position": pos,
            "correct_token_id": int(correct_token),
            "correct_token_text": tokenizer.decode([correct_token]),
            "predicted_token_id": top_id,
            "predicted_token_text": tokenizer.decode([top_id]),
            "correct_logit": correct_logit,
            "correct_logprob": correct_logprob,
            "correct_rank": rank,
            "memory_probe_correct_logit": float(outputs["memory_probe_logits"][0, -1, correct_token].item()),
            "top_tokens": top_tokens(tokenizer, logits, k=5),
        })
    return rows


def summarize_position_rows(rows):
    if not rows:
        return {
            "mean_correct_rank": math.nan,
            "mean_correct_logprob": math.nan,
            "positions_top1_correct": 0.0,
            "by_position": {},
        }
    by_position = defaultdict(list)
    for row in rows:
        by_position[row["position"]].append(row)
    by_position_summary = {}
    for pos, pos_rows in sorted(by_position.items()):
        by_position_summary[str(pos)] = {
            "mean_correct_rank": sum(r["correct_rank"] for r in pos_rows) / len(pos_rows),
            "mean_correct_logprob": sum(r["correct_logprob"] for r in pos_rows) / len(pos_rows),
            "positions_top1_correct": sum(int(r["predicted_token_id"] == r["correct_token_id"]) for r in pos_rows) / len(pos_rows),
            "mean_memory_probe_correct_logit": sum(r["memory_probe_correct_logit"] for r in pos_rows) / len(pos_rows),
        }
    return {
        "mean_correct_rank": sum(r["correct_rank"] for r in rows) / len(rows),
        "mean_correct_logprob": sum(r["correct_logprob"] for r in rows) / len(rows),
        "positions_top1_correct": sum(int(r["predicted_token_id"] == r["correct_token_id"]) for r in rows) / len(rows),
        "by_position": by_position_summary,
    }


def summarize_layer_diags(diags):
    out = []
    for diag in diags:
        out.append({
            "layer": diag["layer"],
            "slot_count": diag["slot_count"],
            "memory_gate": diag["memory_gate"],
            "attn_norm": diag["attn_norm"],
            "mem_norm": diag["mem_norm"],
            "mlp_norm": diag["mlp_norm"],
            "mem_entropy": diag["mem_entropy"],
            "slot_strength_mean": diag["slot_strength_mean"],
            "slot_strength_max": diag["slot_strength_max"],
        })
    return out


def evaluate_dataset(model, tokenizer, dataset_path, limit, ablation_modes, deterministic_max_tokens, failures_to_keep):
    dataset_name = Path(dataset_path).name
    rows = load_conversations(dataset_path, limit=limit)
    engine = Engine(model, tokenizer)
    aggregate = {
        mode: {
            "exact": 0,
            "core": 0,
            "num_examples": 0,
            "position_rows": [],
            "first_step_layers": [],
            "failures": [],
        }
        for mode in ablation_modes
    }
    empty_override = build_empty_memory_override(model)

    for idx, conv in enumerate(rows):
        parsed = split_eval_conversation(conv)
        if parsed is None:
            continue
        prior_messages, recall_user, expected_answer = parsed

        model.clear_memory_banks()
        for message in prior_messages:
            if message["role"] == "user":
                model.memorize(build_user_turn_tokens(tokenizer, message["content"]))
        memory_state = model.memory_state_dict()

        for mode in ablation_modes:
            restore_memory_state(model, memory_state, enabled=(mode != "no_memory_state"))
            with ablate_model(model, mode if mode != "no_memory_state" else "full"):
                response = run_prompt(
                    engine,
                    tokenizer,
                    recall_user,
                    max_tokens=deterministic_max_tokens,
                    temperature=0.0,
                    top_k=1,
                    seed=42,
                )
                exact_ok = normalize_text(response) == normalize_text(expected_answer)
                core_ok = core_fact_match(expected_answer, response)

                if mode == "no_memory_state":
                    pos_rows = teacher_forced_positions(model, tokenizer, recall_user, expected_answer, memory_override=empty_override)
                    prefix_ids = build_turn_prefix(tokenizer, recall_user)
                    layer_diags = model.collect_memory_diagnostics(prefix_ids, memory_override=empty_override)
                else:
                    pos_rows = teacher_forced_positions(model, tokenizer, recall_user, expected_answer, memory_override=None)
                    prefix_ids = build_turn_prefix(tokenizer, recall_user)
                    layer_diags = model.collect_memory_diagnostics(prefix_ids, memory_override=None)

            agg = aggregate[mode]
            agg["exact"] += int(exact_ok)
            agg["core"] += int(core_ok)
            agg["num_examples"] += 1
            agg["position_rows"].extend(pos_rows)
            agg["first_step_layers"].append(summarize_layer_diags(layer_diags))
            if not core_ok and len(agg["failures"]) < failures_to_keep:
                agg["failures"].append({
                    "recall_user": recall_user,
                    "expected_answer": expected_answer,
                    "response": response,
                    "first_positions": pos_rows[: min(4, len(pos_rows))],
                })

    summary = {
        "dataset": dataset_name,
        "modes": {},
    }
    for mode, agg in aggregate.items():
        num_examples = max(agg["num_examples"], 1)
        layer_accum = defaultdict(lambda: defaultdict(float))
        layer_counts = Counter()
        for example_layers in agg["first_step_layers"]:
            for layer_row in example_layers:
                layer = layer_row["layer"]
                layer_counts[layer] += 1
                for key, value in layer_row.items():
                    if key == "layer":
                        continue
                    layer_accum[layer][key] += float(value)
        layers = []
        for layer in sorted(layer_counts):
            count = max(layer_counts[layer], 1)
            row = {"layer": layer}
            for key, total in layer_accum[layer].items():
                row[key] = total / count
            layers.append(row)

        summary["modes"][mode] = {
            "exact_accuracy": agg["exact"] / num_examples,
            "core_accuracy": agg["core"] / num_examples,
            "teacher_forced": summarize_position_rows(agg["position_rows"]),
            "first_step_layer_summary": layers,
            "failures": agg["failures"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose a memory checkpoint with ablations and teacher-forced analysis")
    parser.add_argument("--source", type=str, default="sft")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--deterministic-max-tokens", type=int, default=64)
    parser.add_argument("--failures-to-keep", type=int, default=3)
    parser.add_argument(
        "--ablation-mode",
        action="append",
        dest="ablation_modes",
        default=None,
        choices=["full", "no_memory_state", "no_top_logits", "no_workspace", "no_residual_memory", "no_token_logits"],
        help="may be provided multiple times; defaults to a useful diagnostic set",
    )
    parser.add_argument("--save-json", type=str, default="", help="optional path to save the full JSON report")
    args = parser.parse_args()

    ablation_modes = args.ablation_modes or ["full", "no_memory_state", "no_top_logits", "no_workspace", "no_residual_memory", "no_token_logits"]
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

    report = {
        "model_tag": meta.get("_model_tag", args.model_tag),
        "step": meta.get("_step", args.step),
        "datasets": [],
    }

    for dataset in args.dataset:
        report["datasets"].append(
            evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset_path=dataset,
                limit=args.limit,
                ablation_modes=ablation_modes,
                deterministic_max_tokens=args.deterministic_max_tokens,
                failures_to_keep=args.failures_to_keep,
            )
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
