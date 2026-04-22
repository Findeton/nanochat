"""
Train NanoChat's sparse persistent memory tokens against delayed-recall tasks.

This harness is intentionally aligned with the current architecture:
- persistent latent memory tokens written from earlier context
- sparse write-time competition and sparse read-time retrieval
- no legacy workspace/copy/value branches
- memory-first training, then guarded joint finetuning

The training objective emphasizes three things:
1. fact tokens should improve when memory is enabled
2. non-fact answer positions should stay close to the no-memory baseline
3. memory writes should stay selective instead of collapsing into diffuse banks
"""

import argparse
import gc
import os
import random
import time

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)


parser = argparse.ArgumentParser(description="Train NanoChat sparse persistent memory")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="chat checkpoint tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="checkpoint step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="tag to save the memory-trained checkpoint under")
parser.add_argument("--num-iterations", type=int, default=1000, help="number of optimizer steps")
parser.add_argument("--device-batch-size", type=int, default=4, help="number of memory episodes per optimizer step")
parser.add_argument("--max-seq-len", type=int, default=None, help="sequence length for context/target rendering")
parser.add_argument("--gate-lr", type=float, default=1e-2, help="learning rate for episodic gate logits")
parser.add_argument("--memory-lr", type=float, default=2e-3, help="learning rate for the shared episodic controller")
parser.add_argument("--interface-lr", type=float, default=2e-5, help="learning rate for the guarded interface phase (lm_head + top layers)")
parser.add_argument("--train-lr", type=float, default=5e-6, help="learning rate for full-joint release phase")
parser.add_argument("--save-every", type=int, default=250, help="save every N steps (-1 disables intermediate saves)")
parser.add_argument("--reset-gates-to", type=float, default=None, help="optionally reset all episodic gate logits before training")
parser.add_argument("--skip-smoltalk", action="store_true", help="exclude SmolTalk from the training mixture")
parser.add_argument("--custom-json", action="append", default=[], help="optional custom JSONL conversation file(s) to mix in")
parser.add_argument("--custom-repeat", type=int, default=1, help="repeat each custom JSON dataset N times in the mixture")
parser.add_argument("--target-selection", type=str, default="random", choices=["random", "final"], help="which user/assistant pair to supervise on within each conversation")
parser.add_argument("--context-mode", type=str, default="full", choices=["full", "user_only"], help="whether memory is built from full prior context or only the user-side messages")
parser.add_argument("--joint-top-layers", type=int, default=2, help="number of top transformer blocks to unfreeze during guarded joint training")
parser.add_argument("--fallback-identity-span-tokens", type=int, default=4, help="fallback fact span length when explicit fact annotations are missing")
parser.add_argument("--weighted-answer-ce-weight", type=float, default=0.5, help="extra loss weight for position-aware answer CE")
parser.add_argument("--weighted-template-ce", type=float, default=0.20, help="relative CE weight for template/control positions")
parser.add_argument("--weighted-fact-ce", type=float, default=1.0, help="relative CE weight for fact-bearing positions")
parser.add_argument("--weighted-rest-ce", type=float, default=0.35, help="relative CE weight for remaining answer positions")
parser.add_argument("--fact-span-margin-loss-weight", type=float, default=0.25, help="extra loss weight that pushes exact fact tokens above hard negatives")
parser.add_argument("--fact-span-margin", type=float, default=2.5, help="target logit margin for exact fact tokens against hard negatives")
parser.add_argument("--guardrail-kl-weight", type=float, default=0.20, help="KL weight that keeps non-fact answer positions close to the no-memory baseline")
parser.add_argument("--guardrail-temperature", type=float, default=1.0, help="temperature used for the non-fact guardrail KL")
parser.add_argument("--write-diversity-loss-weight", type=float, default=0.05, help="extra loss weight that penalizes collapsed memory-token banks")
parser.add_argument("--memory-utility-loss-weight", type=float, default=0.5, help="extra loss weight that forces fact positions to improve when memory is enabled")
parser.add_argument("--memory-utility-margin", type=float, default=0.5, help="target log-probability improvement for fact positions when memory is enabled")
parser.add_argument("--phase1-steps", type=int, default=0, help="phase 1: memory-only steps with the trunk frozen")
parser.add_argument("--phase2-steps", type=int, default=0, help="phase 2: guarded joint steps (memory + interface params)")
parser.add_argument("--phase3-steps", type=int, default=0, help="phase 3: full-joint release steps")
args = parser.parse_args()
user_config = vars(args).copy()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
assert not ddp, "scripts.chat_memory currently supports single-process training only"
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb()
if not use_dummy_wandb:
    import wandb

    wandb_run = wandb.init(project="nanochat-memory", name=args.run, config=user_config)

model, tokenizer, meta = load_model("sft", device, phase="train", model_tag=args.model_tag, step=args.model_step)
loaded_model_tag = meta.get("_model_tag", args.model_tag or "auto")
args.max_seq_len = args.max_seq_len or meta.get("model_config", {}).get("sequence_len", model.config.sequence_len)
output_tag = args.output_tag or f"{loaded_model_tag}-memory"

if args.reset_gates_to is not None:
    with torch.no_grad():
        for block in model.transformer.h:
            block.episodic_memory_score_bias.fill_(args.reset_gates_to)


def top_layer_param_names(n_layer, joint_top_layers):
    first_top = max(0, n_layer - max(1, joint_top_layers))
    return {f"transformer.h.{layer_idx}." for layer_idx in range(first_top, n_layer)}


top_layer_prefixes = top_layer_param_names(model.config.n_layer, args.joint_top_layers)
memory_gate_params = []
memory_params = []
interface_params = []
trunk_params = []

for name, param in model.named_parameters():
    if name.endswith("episodic_memory_score_bias") or name == "episodic_controller.kind_bias":
        memory_gate_params.append(param)
    elif name.startswith("episodic_controller.") or ".episodic_" in name:
        memory_params.append(param)
    elif name.startswith("lm_head."):
        interface_params.append(param)
    elif any(name.startswith(prefix) for prefix in top_layer_prefixes):
        interface_params.append(param)
    elif name.startswith("value_embeds."):
        layer_name = name.split(".")[1]
        if f"transformer.h.{layer_name}." in top_layer_prefixes:
            interface_params.append(param)
        else:
            trunk_params.append(param)
    else:
        trunk_params.append(param)


def build_optimizer():
    groups = []
    if trunk_params:
        groups.append({"params": trunk_params, "lr": args.train_lr, "weight_decay": 0.01})
    if interface_params:
        groups.append({"params": interface_params, "lr": args.interface_lr, "weight_decay": 0.01})
    if memory_params:
        groups.append({"params": memory_params, "lr": args.memory_lr, "weight_decay": 0.01})
    if memory_gate_params:
        groups.append({"params": memory_gate_params, "lr": args.gate_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


optimizer = build_optimizer()

phase_total = args.phase1_steps + args.phase2_steps + args.phase3_steps
if phase_total == 0:
    args.phase1_steps = args.num_iterations
    phase_total = args.num_iterations
else:
    assert phase_total == args.num_iterations, "phase1+phase2+phase3 steps must equal num_iterations"
user_config.update(vars(args))
if not use_dummy_wandb:
    wandb_run.config.update(user_config, allow_val_change=True)


def phase_for_step(step):
    if step < args.phase1_steps:
        return "phase1_memory_only"
    if step < args.phase1_steps + args.phase2_steps:
        return "phase2_interface_joint"
    return "phase3_full_joint"


def set_phase_trainability(phase_name):
    if phase_name == "phase1_memory_only":
        enabled = (memory_gate_params, memory_params)
        disabled = (interface_params, trunk_params)
    elif phase_name == "phase2_interface_joint":
        enabled = (memory_gate_params, memory_params, interface_params)
        disabled = (trunk_params,)
    else:
        enabled = (memory_gate_params, memory_params, interface_params, trunk_params)
        disabled = ()

    for group in enabled:
        for param in group:
            param.requires_grad_(True)
    for group in disabled:
        for param in group:
            param.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


trainable_counts = {
    "memory_gate_params": sum(p.numel() for p in memory_gate_params),
    "memory_params": sum(p.numel() for p in memory_params),
    "interface_params": sum(p.numel() for p in interface_params),
    "trunk_params": sum(p.numel() for p in trunk_params),
}

print0(f"Loaded chat checkpoint: {loaded_model_tag} @ step {meta.get('_step', args.model_step)}")
print0(f"Output tag: {output_tag}")
print0(f"Memory gates: {trainable_counts['memory_gate_params']:,}")
print0(f"Memory controller params: {trainable_counts['memory_params']:,}")
print0(f"Interface params: {trainable_counts['interface_params']:,}")
print0(f"Frozen trunk params (phase 1): {trainable_counts['trunk_params']:,}")
print0("Episodic mode: sparse_unified_associative_attention")

base_dir = get_base_dir()
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON

train_tasks = []
task_labels = []
if not args.skip_smoltalk:
    from tasks.smoltalk import SmolTalk

    train_tasks.append(SmolTalk(split="train"))
    task_labels.append("SmolTalk x1")
identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
if os.path.exists(identity_path):
    train_tasks.append(CustomJSON(filepath=identity_path))
    task_labels.append("identity_conversations x1")
for filepath in args.custom_json:
    for _ in range(args.custom_repeat):
        train_tasks.append(CustomJSON(filepath=filepath))
    task_labels.append(f"{os.path.basename(filepath)} x{args.custom_repeat}")
assert train_tasks, "No training tasks configured. Provide at least one dataset."
dataset = TaskMixture(train_tasks)
print0(f"Training tasks: {', '.join(task_labels)}")
print0(f"Memory training mixture: {len(dataset):,} conversations")

rng = random.Random(42)
cursor = 0


def build_supervised_tensors(token_ids, target_mask):
    token_tensor = torch.tensor([token_ids[:args.max_seq_len + 1]], dtype=torch.long, device=device)
    inputs = token_tensor[:, :-1].contiguous()
    targets = token_tensor[:, 1:].to(dtype=torch.long).contiguous()
    mask = torch.tensor([target_mask[:args.max_seq_len + 1]], dtype=torch.bool, device=device)
    targets[~mask[:, 1:]] = -1
    return inputs, targets


def positions_from_target_mask(targets):
    return (targets[0] != -1).nonzero(as_tuple=False).flatten()


def split_template_and_identity_positions(answer_positions, anchor_target_pos, max_identity_tokens):
    if answer_positions is None or answer_positions.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=answer_positions.device if answer_positions is not None else None)
        return empty, empty
    template_positions = answer_positions[answer_positions < anchor_target_pos]
    identity_positions = answer_positions[answer_positions >= anchor_target_pos]
    if max_identity_tokens is not None and max_identity_tokens > 0:
        identity_positions = identity_positions[:max_identity_tokens]
    return template_positions, identity_positions


def encode_text_candidates(text):
    candidates = []
    for candidate_text in [text, f" {text}"]:
        candidate_ids = tokenizer.encode(candidate_text)
        if candidate_ids and candidate_ids not in candidates:
            candidates.append(candidate_ids)
    return candidates


def find_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return None
    limit = len(sequence) - len(pattern) + 1
    for i in range(limit):
        if sequence[i:i + len(pattern)] == pattern:
            return i
    return None


def target_positions_from_span_ids(rendered_ids, targets, span_ids):
    span_start = find_subsequence(rendered_ids, span_ids)
    if span_start is None:
        return None
    start_pos = max(0, span_start - 1)
    end_pos = min(targets.size(1) - 1, span_start + len(span_ids) - 2)
    positions = torch.arange(start_pos, end_pos + 1, dtype=torch.long, device=targets.device)
    positions = positions[targets[0, positions] != -1]
    if positions.numel() == 0:
        return None
    return positions


def pick_negative_token_ids(texts, target_span_len):
    token_ids = set()
    for text in texts:
        candidates = encode_text_candidates(text)
        if not candidates:
            continue
        chosen = min(candidates, key=lambda ids: (abs(len(ids) - target_span_len), -len(ids)))
        token_ids.update(int(token_id) for token_id in chosen)
    return sorted(token_ids)


def resolve_fact_groups(target_ids, targets, memory_target):
    if not memory_target:
        return []
    resolved = []
    for group in memory_target.get("fact_groups", []):
        text = group.get("text")
        if not text:
            continue
        positions = None
        chosen_span_ids = None
        for candidate_ids in encode_text_candidates(text):
            positions = target_positions_from_span_ids(target_ids, targets, candidate_ids)
            if positions is not None:
                chosen_span_ids = candidate_ids
                break
        if positions is None:
            continue
        negative_token_ids = pick_negative_token_ids(group.get("hard_negatives", []), len(chosen_span_ids))
        resolved.append(
            {
                "text": text,
                "positions": positions,
                "hard_negative_token_ids": negative_token_ids,
            }
        )
    return resolved


def build_episode(conversation):
    messages = conversation["messages"]
    memory_target = conversation.get("memory_target")
    if len(messages) < 4:
        return None

    has_system = messages[0]["role"] == "system"
    min_split = 3 if has_system else 1
    candidate_splits = []
    for split in range(min_split, len(messages) - 1):
        if messages[split]["role"] == "user" and messages[split + 1]["role"] == "assistant":
            candidate_splits.append(split)
    if not candidate_splits:
        return None

    split = candidate_splits[-1] if args.target_selection == "final" else rng.choice(candidate_splits)
    context_messages = messages[:split]
    if args.context_mode == "user_only":
        if has_system:
            context_messages = [messages[0]] + [m for m in messages[1:split] if m["role"] == "user"]
        else:
            context_messages = [m for m in messages[:split] if m["role"] == "user"]
    target_messages = messages[split:split + 2]
    if has_system:
        target_messages = [messages[0]] + target_messages

    context_ids, _ = tokenizer.render_conversation({"messages": context_messages}, max_tokens=args.max_seq_len)
    if len(context_ids) < 2:
        return None
    target_ids, target_mask = tokenizer.render_conversation({"messages": target_messages}, max_tokens=args.max_seq_len + 1)
    if len(target_ids) < 2:
        return None

    context_tensor = torch.tensor([context_ids[-args.max_seq_len:]], dtype=torch.long, device=device)
    inputs, targets = build_supervised_tensors(target_ids, target_mask)
    answer_positions = positions_from_target_mask(targets)
    if answer_positions.numel() == 0:
        return None

    first_target_pos = int(answer_positions[0].item())
    fact_group_infos = resolve_fact_groups(target_ids, targets, memory_target)
    if fact_group_infos:
        fact_positions = torch.unique(torch.cat([group["positions"] for group in fact_group_infos], dim=0))
        fact_positions, _ = torch.sort(fact_positions)
        anchor_target_pos = int(fact_positions[0].item())
        fact_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=device)
        fact_mask[fact_positions] = True
        template_positions = answer_positions[answer_positions < anchor_target_pos]
        identity_positions = fact_positions
        rest_answer_positions = answer_positions[(answer_positions >= anchor_target_pos) & ~fact_mask[answer_positions]]
    else:
        fact_positions = torch.zeros((0,), dtype=torch.long, device=device)
        anchor_target_pos = first_target_pos
        template_positions, identity_positions = split_template_and_identity_positions(
            answer_positions,
            anchor_target_pos,
            args.fallback_identity_span_tokens,
        )
        identity_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=device)
        if identity_positions.numel() > 0:
            identity_mask[identity_positions] = True
        rest_answer_positions = answer_positions[(answer_positions >= anchor_target_pos) & ~identity_mask[answer_positions]]

    return (
        context_tensor,
        inputs,
        targets,
        template_positions,
        identity_positions,
        fact_positions,
        rest_answer_positions,
        anchor_target_pos,
        fact_group_infos,
    )


def next_episode():
    global cursor
    while True:
        episode = build_episode(dataset[cursor])
        cursor = (cursor + 1) % len(dataset)
        if episode is not None:
            return episode


def build_empty_memory_override(batch_size):
    empty_idx = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    return model.build_memory_state(empty_idx)


def slot_diversity_penalty(memory_override):
    penalties = []
    for bank in memory_override:
        strengths = bank["strengths"]
        tokens = bank["tokens"]
        keys = bank["keys"]
        for batch_idx in range(tokens.size(0)):
            active = strengths[batch_idx] > 1e-6
            if int(active.sum().item()) <= 1:
                continue
            active_tokens = F.normalize(tokens[batch_idx, active].float(), dim=-1)
            active_keys = F.normalize(keys[batch_idx, active].float(), dim=-1)
            token_sim = active_tokens @ active_tokens.T
            key_sim = active_keys @ active_keys.T
            mask = ~torch.eye(token_sim.size(0), device=token_sim.device, dtype=torch.bool)
            penalties.append(0.5 * token_sim[mask].pow(2).mean() + 0.5 * key_sim[mask].pow(2).mean())
    if not penalties:
        return None
    return torch.stack(penalties).mean()


def compute_token_margin(logits_row, token_id):
    token_id = int(token_id)
    correct_logit = logits_row[token_id]
    if logits_row.numel() <= 1:
        return 0.0
    masked = logits_row.clone()
    masked[token_id] = -float("inf")
    wrong_logit = masked.max()
    return float((correct_logit - wrong_logit).item())


def active_slot_count(memory_override):
    counts = []
    for bank in memory_override:
        counts.append(float((bank["strengths"] > 1e-6).sum(dim=1).float().mean().item()))
    return sum(counts) / max(len(counts), 1)


def gate_values():
    return [torch.sigmoid(block.episodic_memory_score_bias).item() for block in model.transformer.h]


def save_memory_checkpoint(step, smooth_loss):
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)
    model.clear_memory_banks()
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        optimizer.state_dict(),
        {
            "step": step,
            "memory_loss": smooth_loss,
            "model_config": model.config.__dict__.copy(),
            "user_config": user_config,
            "base_checkpoint": {
                "model_tag": loaded_model_tag,
                "step": meta.get("_step", args.model_step),
            },
        },
        rank=ddp_rank,
    )


smooth_loss = 0.0
smooth_guardrail_kl = 0.0
smooth_weighted_answer_ce = 0.0
smooth_fact_span_margin = 0.0
smooth_write_diversity = 0.0
smooth_memory_utility = 0.0
smooth_memory_utility_margin = 0.0
smooth_anchor_margin = 0.0
smooth_active_slots = 0.0
ema_beta = 0.95
t_start = time.time()
current_phase = None

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations
    if last_step:
        save_memory_checkpoint(step, smooth_loss)
        break

    phase_name = phase_for_step(step)
    if phase_name != current_phase:
        trainable_now = set_phase_trainability(phase_name)
        model.zero_grad(set_to_none=True)
        current_phase = phase_name
        print0(f"Switched training phase: {current_phase} | trainable parameters: {trainable_now:,}")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    total_loss = 0.0
    total_guardrail_kl = 0.0
    total_weighted_answer_ce = 0.0
    total_fact_span_margin = 0.0
    total_write_diversity = 0.0
    total_memory_utility = 0.0
    total_memory_utility_margin = 0.0
    total_anchor_margin = 0.0
    total_active_slots = 0.0

    for _ in range(args.device_batch_size):
        (
            context_ids,
            inputs,
            targets,
            template_positions,
            identity_positions,
            fact_positions,
            rest_answer_positions,
            anchor_target_pos,
            fact_group_infos,
        ) = next_episode()
        model.clear_memory_banks()
        memory_override = model.build_memory_state(context_ids)
        outputs = model(inputs, targets, memory_override=memory_override, return_components=True)
        loss = outputs["loss"]

        effective_fact_positions = fact_positions if fact_positions.numel() > 0 else identity_positions
        answer_positions = positions_from_target_mask(targets)
        non_fact_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=device)
        non_fact_mask[answer_positions] = True
        if effective_fact_positions.numel() > 0:
            non_fact_mask[effective_fact_positions] = False
        non_fact_positions = non_fact_mask.nonzero(as_tuple=False).flatten()

        empty_memory_override = None
        no_memory_outputs = None
        need_no_memory = (
            args.guardrail_kl_weight > 0 and non_fact_positions.numel() > 0
        ) or (
            args.memory_utility_loss_weight > 0 and effective_fact_positions.numel() > 0
        )
        if need_no_memory:
            empty_memory_override = build_empty_memory_override(inputs.size(0))
            with torch.no_grad():
                no_memory_outputs = model(inputs, return_components=True, memory_override=empty_memory_override)

        guardrail_kl_value = 0.0
        weighted_answer_ce_value = 0.0
        fact_span_margin_value = 0.0
        write_diversity_value = 0.0
        memory_utility_loss_value = 0.0
        memory_utility_margin_value = 0.0
        anchor_margin_value = 0.0

        if args.guardrail_kl_weight > 0 and no_memory_outputs is not None and non_fact_positions.numel() > 0:
            temp = max(args.guardrail_temperature, 1e-5)
            student_logits = outputs["logits"][:, non_fact_positions, :]
            base_logits = no_memory_outputs["logits"][:, non_fact_positions, :]
            guardrail_kl = F.kl_div(
                F.log_softmax(student_logits / temp, dim=-1).reshape(-1, student_logits.size(-1)),
                F.softmax(base_logits / temp, dim=-1).reshape(-1, base_logits.size(-1)),
                reduction="batchmean",
            ) * (temp ** 2)
            guardrail_kl_value = guardrail_kl.item()
            loss = loss + args.guardrail_kl_weight * guardrail_kl

        if args.weighted_answer_ce_weight > 0:
            weighted_terms = []
            total_weight = 0.0
            for positions, zone_weight in [
                (template_positions, args.weighted_template_ce),
                (effective_fact_positions, args.weighted_fact_ce),
                (rest_answer_positions, args.weighted_rest_ce),
            ]:
                if zone_weight <= 0 or positions.numel() == 0:
                    continue
                zone_targets = targets[0, positions]
                zone_logits = outputs["logits"][:, positions, :]
                zone_ce = F.cross_entropy(
                    zone_logits.reshape(-1, zone_logits.size(-1)),
                    zone_targets.reshape(-1),
                    ignore_index=-1,
                    reduction="mean",
                )
                weighted_terms.append(zone_weight * zone_ce)
                total_weight += zone_weight
            if weighted_terms and total_weight > 0:
                weighted_answer_ce = torch.stack(weighted_terms).sum() / total_weight
                weighted_answer_ce_value = weighted_answer_ce.item()
                loss = loss + args.weighted_answer_ce_weight * weighted_answer_ce

        if args.fact_span_margin_loss_weight > 0 and fact_group_infos:
            margin_losses = []
            for group in fact_group_infos:
                neg_ids = [token_id for token_id in group["hard_negative_token_ids"] if 0 <= token_id < outputs["logits"].size(-1)]
                if not neg_ids:
                    continue
                for pos in group["positions"]:
                    correct_token = int(targets[0, pos].item())
                    if correct_token == -1:
                        continue
                    current_neg_ids = [token_id for token_id in neg_ids if token_id != correct_token]
                    if not current_neg_ids:
                        continue
                    row = outputs["logits"][0, pos]
                    correct_logit = row[correct_token]
                    negative_logit = torch.logsumexp(row[current_neg_ids], dim=0)
                    margin_losses.append(
                        torch.relu(
                            torch.tensor(args.fact_span_margin, device=device, dtype=row.dtype) - (correct_logit - negative_logit)
                        )
                    )
            if margin_losses:
                fact_span_margin_loss = torch.stack(margin_losses).mean()
                fact_span_margin_value = fact_span_margin_loss.item()
                loss = loss + args.fact_span_margin_loss_weight * fact_span_margin_loss

        if args.write_diversity_loss_weight > 0:
            write_diversity = slot_diversity_penalty(memory_override)
            if write_diversity is not None:
                write_diversity_value = write_diversity.item()
                loss = loss + args.write_diversity_loss_weight * write_diversity

        if args.memory_utility_loss_weight > 0 and no_memory_outputs is not None and effective_fact_positions.numel() > 0:
            fact_target_tokens = targets[:, effective_fact_positions]
            valid_fact_mask = fact_target_tokens.ne(-1)
            if valid_fact_mask.any():
                memory_fact_logits = outputs["logits"][:, effective_fact_positions, :]
                base_fact_logits = no_memory_outputs["logits"][:, effective_fact_positions, :]
                memory_fact_log_probs = F.log_softmax(memory_fact_logits, dim=-1)
                base_fact_log_probs = F.log_softmax(base_fact_logits, dim=-1)
                gather_index = fact_target_tokens.clamp_min(0).unsqueeze(-1)
                memory_fact_lp = memory_fact_log_probs.gather(-1, gather_index).squeeze(-1)
                base_fact_lp = base_fact_log_probs.gather(-1, gather_index).squeeze(-1)
                utility_margin = (memory_fact_lp - base_fact_lp)[valid_fact_mask]
                if utility_margin.numel() > 0:
                    memory_utility_margin_value = utility_margin.mean().item()
                    memory_utility_loss = torch.relu(
                        torch.tensor(args.memory_utility_margin, device=device, dtype=utility_margin.dtype) - utility_margin
                    ).mean()
                    memory_utility_loss_value = memory_utility_loss.item()
                    loss = loss + args.memory_utility_loss_weight * memory_utility_loss

        answer_token = targets[0, anchor_target_pos]
        if int(answer_token.item()) != -1:
            anchor_margin_value = compute_token_margin(outputs["logits"][0, anchor_target_pos], answer_token.item())

        loss_value = loss.item()
        (loss / args.device_batch_size).backward()

        total_loss += loss_value
        total_guardrail_kl += guardrail_kl_value
        total_weighted_answer_ce += weighted_answer_ce_value
        total_fact_span_margin += fact_span_margin_value
        total_write_diversity += write_diversity_value
        total_memory_utility += memory_utility_loss_value
        total_memory_utility_margin += memory_utility_margin_value
        total_anchor_margin += anchor_margin_value
        total_active_slots += active_slot_count(memory_override)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    model.clear_memory_banks()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * total_loss
    smooth_guardrail_kl = ema_beta * smooth_guardrail_kl + (1 - ema_beta) * total_guardrail_kl
    smooth_weighted_answer_ce = ema_beta * smooth_weighted_answer_ce + (1 - ema_beta) * total_weighted_answer_ce
    smooth_fact_span_margin = ema_beta * smooth_fact_span_margin + (1 - ema_beta) * total_fact_span_margin
    smooth_write_diversity = ema_beta * smooth_write_diversity + (1 - ema_beta) * total_write_diversity
    smooth_memory_utility = ema_beta * smooth_memory_utility + (1 - ema_beta) * total_memory_utility
    smooth_memory_utility_margin = ema_beta * smooth_memory_utility_margin + (1 - ema_beta) * total_memory_utility_margin
    smooth_anchor_margin = ema_beta * smooth_anchor_margin + (1 - ema_beta) * total_anchor_margin
    smooth_active_slots = ema_beta * smooth_active_slots + (1 - ema_beta) * total_active_slots

    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    debiased_guardrail_kl = smooth_guardrail_kl / (1 - ema_beta ** (step + 1))
    debiased_weighted_answer_ce = smooth_weighted_answer_ce / (1 - ema_beta ** (step + 1))
    debiased_fact_span_margin = smooth_fact_span_margin / (1 - ema_beta ** (step + 1))
    debiased_write_diversity = smooth_write_diversity / (1 - ema_beta ** (step + 1))
    debiased_memory_utility = smooth_memory_utility / (1 - ema_beta ** (step + 1))
    debiased_memory_utility_margin = smooth_memory_utility_margin / (1 - ema_beta ** (step + 1))
    debiased_anchor_margin = smooth_anchor_margin / (1 - ema_beta ** (step + 1))
    debiased_active_slots = smooth_active_slots / (1 - ema_beta ** (step + 1))
    gate_avg = sum(gate_values()) / len(model.transformer.h)
    elapsed = time.time() - t_start

    if step % 10 == 0:
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"loss: {debiased_loss:.4f} | "
            f"guardrail_kl: {debiased_guardrail_kl:.4f} | "
            f"weighted_answer_ce: {debiased_weighted_answer_ce:.4f} | "
            f"fact_span_margin: {debiased_fact_span_margin:.4f} | "
            f"write_diversity: {debiased_write_diversity:.4f} | "
            f"memory_utility: {debiased_memory_utility:.4f} | "
            f"memory_utility_margin: {debiased_memory_utility_margin:.4f} | "
            f"anchor_margin: {debiased_anchor_margin:.4f} | "
            f"active_slots: {debiased_active_slots:.2f} | "
            f"phase: {current_phase} | "
            f"gate: {gate_avg:.4f} | "
            f"elapsed: {elapsed:.0f}s"
        )

    if step % 50 == 0:
        log_dict = {
            "step": step,
            "memory/loss": debiased_loss,
            "memory/guardrail_kl": debiased_guardrail_kl,
            "memory/weighted_answer_ce": debiased_weighted_answer_ce,
            "memory/fact_span_margin": debiased_fact_span_margin,
            "memory/write_diversity": debiased_write_diversity,
            "memory/utility_loss": debiased_memory_utility,
            "memory/utility_margin": debiased_memory_utility_margin,
            "memory/anchor_margin": debiased_anchor_margin,
            "memory/active_slots": debiased_active_slots,
            "memory/gate_avg": gate_avg,
            "memory/phase": {
                "phase1_memory_only": 1,
                "phase2_interface_joint": 2,
                "phase3_full_joint": 3,
            }[current_phase],
        }
        for i, value in enumerate(gate_values()):
            log_dict[f"memory/gate_{i}"] = value
        wandb_run.log(log_dict)

    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        save_memory_checkpoint(step, smooth_loss)

print0(f"Final gate values: [{', '.join(f'{g:.4f}' for g in gate_values())}]")
wandb_run.finish()
compute_cleanup()
