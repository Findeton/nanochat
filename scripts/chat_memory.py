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
parser.add_argument("--phase2-device-batch-size", type=int, default=0, help="override device batch size during phase 2; 0 uses --device-batch-size")
parser.add_argument("--phase3-device-batch-size", type=int, default=0, help="override device batch size during phase 3; useful when the full trunk is unfrozen")
parser.add_argument("--phase38-device-batch-size", type=int, default=0, help="override device batch size during phase 3.8; 0 uses --device-batch-size")
parser.add_argument("--max-seq-len", type=int, default=None, help="sequence length for context/target rendering")
parser.add_argument("--episode-cache-size", type=int, default=65536, help="number of pre-tokenized episodes kept in RAM before training (0 disables)")
parser.add_argument("--episode-cache-reshuffle", action="store_true", help="reshuffle the pre-tokenized episode cache every cache epoch")
parser.add_argument("--episode-cache-refresh", action="store_true", help="rebuild the pre-tokenized cache from the next dataset window every cache epoch")
parser.add_argument("--gc-every", type=int, default=0, help="run Python gc every N steps; 0 disables on non-MPS devices")
parser.add_argument("--gate-lr", type=float, default=1e-2, help="learning rate for episodic gate logits")
parser.add_argument("--memory-lr", type=float, default=2e-3, help="learning rate for the shared episodic controller")
parser.add_argument("--interface-lr", type=float, default=2e-5, help="learning rate for the guarded interface phase (lm_head + top layers)")
parser.add_argument("--train-lr", type=float, default=5e-6, help="learning rate for full-joint release phase")
parser.add_argument("--save-every", type=int, default=250, help="save every N steps (-1 disables intermediate saves)")
parser.add_argument("--save-optimizer-checkpoints", action="store_true", help="also save optimizer shards; disabled by default to keep long GPU runs from filling small root disks")
parser.add_argument("--keep-last-checkpoints", type=int, default=0, help="keep only the latest N saved steps in the output tag; 0 disables pruning")
parser.add_argument("--reset-gates-to", type=float, default=None, help="optionally reset all episodic gate logits before training")
parser.add_argument("--skip-smoltalk", action="store_true", help="exclude SmolTalk from the training mixture")
parser.add_argument("--custom-json", action="append", default=[], help="optional custom JSONL conversation file(s) to mix in")
parser.add_argument("--custom-repeat", type=int, default=1, help="repeat each custom JSON dataset N times in the mixture")
parser.add_argument("--target-selection", type=str, default="random", choices=["random", "final"], help="which user/assistant pair to supervise on within each conversation")
parser.add_argument("--context-mode", type=str, default="full", choices=["full", "user_only"], help="whether memory is built from full prior context or only the user-side messages")
parser.add_argument("--memory-build-mode", type=str, default="turn", choices=["full_context", "user", "turn", "turn_recall"], help="how prior context is replayed into persistent memory during training")
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
parser.add_argument("--anchor-margin-loss-weight", type=float, default=0.25, help="extra loss weight that forces the correct fact anchor to beat the actual strongest confuser")
parser.add_argument("--anchor-margin", type=float, default=2.0, help="target logit margin for fact anchors against the strongest confuser")
parser.add_argument("--anchor-loss-tokens", type=int, default=3, help="number of early fact tokens to apply the anchor/confuser loss to")
parser.add_argument("--memory-probe-loss-weight", type=float, default=0.10, help="extra loss weight that makes the decoded memory read more lexical at fact positions")
parser.add_argument("--key-token-ce-loss-weight", type=float, default=0.75, help="direct CE weight on exact remembered value tokens")
parser.add_argument("--key-token-rank-loss-weight", type=float, default=0.75, help="loss weight that pushes exact remembered value tokens above every competing token")
parser.add_argument("--key-token-rank-margin", type=float, default=2.0, help="target top-1 logit margin for exact remembered value tokens")
parser.add_argument("--key-token-utility-loss-weight", type=float, default=1.0, help="loss weight for memory-vs-no-memory improvement on exact remembered value tokens")
parser.add_argument("--key-token-utility-margin", type=float, default=2.0, help="target log-probability gain over no-memory on exact remembered value tokens")
parser.add_argument("--deference-start-phase", type=str, default="phase2", choices=["off", "phase1", "phase2", "phase3", "phase35", "phase36", "phase37", "phase38"], help="phase where memory-deference objectives become active")
parser.add_argument("--phase2-answer-start-ce-loss-weight", type=float, default=0.50, help="phase-aware CE weight for the first assistant tokens so rollout enters a sane answer path")
parser.add_argument("--phase2-answer-start-tokens", type=int, default=8, help="number of early assistant target tokens supervised by the phase-aware answer-start loss")
parser.add_argument("--first-fact-token-multiplier", type=float, default=3.0, help="phase-aware multiplier for the first exact fact token, where fluent defaults usually win")
parser.add_argument("--habit-confuser-loss-weight", type=float, default=0.50, help="phase-aware loss weight that forces facts to beat the no-memory model's top confusers")
parser.add_argument("--habit-confuser-margin", type=float, default=3.0, help="target logit margin over no-memory habit/confuser token ids")
parser.add_argument("--habit-confuser-top-k", type=int, default=8, help="number of no-memory top tokens treated as habitual confusers at fact positions")
parser.add_argument("--span-contrast-loss-weight", type=float, default=0.35, help="phase-aware loss weight that makes whole remembered spans beat hard/same-class confusers")
parser.add_argument("--span-contrast-margin", type=float, default=1.5, help="target average log-probability margin for remembered spans over confuser spans")
parser.add_argument("--span-utility-loss-weight", type=float, default=0.25, help="phase-aware loss weight that makes whole remembered spans more likely with memory than without memory")
parser.add_argument("--span-utility-margin", type=float, default=1.0, help="target average log-probability improvement for remembered spans with memory enabled")
parser.add_argument("--span-contrast-top-k", type=int, default=4, help="number of no-memory top tokens used as per-position span confusers")
parser.add_argument("--span-contrast-max-tokens", type=int, default=12, help="maximum remembered span length included in span-level objectives")
parser.add_argument("--span-hard-token-loss-weight", type=float, default=0.0, help="loss weight for the hardest per-token remembered-span decisions against same-field/no-memory confusers")
parser.add_argument("--span-hard-token-margin", type=float, default=4.0, help="target logit margin for hard remembered-span tokens over aligned confusers")
parser.add_argument("--span-hard-token-top-k", type=int, default=2, help="number of hardest remembered-span token decisions optimized per example")
parser.add_argument("--post-key-loss-weight", type=float, default=0.50, help="phase-aware loss weight that trains the token after a remembered span and penalizes immediate key repetition")
parser.add_argument("--post-key-repeat-margin", type=float, default=3.0, help="target logit margin for the post-key token over repeated remembered-span tokens")
parser.add_argument("--post-key-max-groups", type=int, default=4, help="maximum remembered spans per example used by the post-key objective")
parser.add_argument("--rollout-loss-weight", type=float, default=0.0, help="extra loss weight for short autoregressive rollout recovery training")
parser.add_argument("--rollout-batch-frac", type=float, default=0.0, help="fraction of episodes that receive rollout-aware loss when enabled")
parser.add_argument("--rollout-steps", type=int, default=4, help="number of answer tokens to generate before rollout recovery loss")
parser.add_argument("--rollout-continuation-tokens", type=int, default=8, help="number of gold continuation tokens to supervise after the generated prefix")
parser.add_argument("--rollout-temperature", type=float, default=0.0, help="rollout sampling temperature; 0 means greedy")
parser.add_argument("--rollout-top-k", type=int, default=1, help="top-k filter for rollout sampling; 1 with temperature 0 is greedy")
parser.add_argument("--rollout-gold-start", type=str, default="first_fact", choices=["after_rollout", "first_fact"], help="where the gold recovery target starts after generated rollout tokens")
parser.add_argument("--rollout-anchor-loss-weight", type=float, default=0.25, help="anchor/confuser loss weight inside the rollout recovery objective")
parser.add_argument("--rollout-missing-key-multiplier", type=float, default=3.0, help="multiply rollout loss when the generated prefix should contain the key but does not")
parser.add_argument("--rollout-repeat-multiplier", type=float, default=2.0, help="multiply rollout loss when the generated prefix falls into a short repetition loop")
parser.add_argument("--rollout-wrong-default-multiplier", type=float, default=2.0, help="phase-aware rollout multiplier when generated tokens choose a no-memory habit/confuser before the key")
parser.add_argument("--phase3-trunk-trainable", action="store_true", help="unfreeze the full trunk during phase 3; by default phase 3 trains memory + interface only")
parser.add_argument("--phase3-only-trunk-trainable", action="store_true", help="when --phase3-trunk-trainable is set, restrict full-trunk updates to phase3_generated_recovery only")
parser.add_argument("--phase3-recovery-loss-weight", type=float, default=0.0, help="loss weight for generated-prefix recovery training")
parser.add_argument("--phase3-recovery-start-phase", type=str, default="phase3", choices=["phase2", "phase3"], help="phase where generated-prefix recovery becomes active")
parser.add_argument("--phase3-recovery-phase-only", action="store_true", help="apply generated-prefix recovery only during phase3_generated_recovery, instead of all later phases")
parser.add_argument("--phase3-recovery-batch-frac", type=float, default=0.0, help="fraction of episodes that receive generated-prefix recovery loss")
parser.add_argument("--phase3-recovery-steps", type=int, default=16, help="maximum sampled answer-prefix tokens before recovery supervision")
parser.add_argument("--phase3-recovery-temperature", type=float, default=0.2, help="temperature used to sample phase-3 generated prefixes")
parser.add_argument("--phase3-recovery-top-k", type=int, default=8, help="top-k filter used to sample phase-3 generated prefixes")
parser.add_argument("--phase3-recovery-max-target-tokens", type=int, default=12, help="maximum recovery tokens supervised from a generated prefix")
parser.add_argument("--phase3-recovery-anchor-loss-weight", type=float, default=0.50, help="anchor/confuser loss weight inside generated-prefix recovery")
parser.add_argument("--phase3-recovery-key-tail-tokens", type=int, default=2, help="number of tokens sampled after a found key to detect immediate repetition")
parser.add_argument("--phase3-recovery-post-key-tokens", type=int, default=4, help="number of gold post-key tokens used when recovering from key repetition")
parser.add_argument("--phase35-branch-loss-weight", type=float, default=0.0, help="phase 3.5 loss weight for first-token remembered-span branch decisions")
parser.add_argument("--phase35-branch-margin", type=float, default=6.0, help="target logit margin for remembered branch tokens over hard/default confusers")
parser.add_argument("--phase35-branch-tokens", type=int, default=1, help="number of leading tokens per remembered span used by branch binding")
parser.add_argument("--phase35-branch-no-memory-top-k", type=int, default=16, help="no-memory top-k tokens treated as branch confusers")
parser.add_argument("--phase35-stop-loss-weight", type=float, default=0.0, help="phase 3.5 loss weight for stopping/continuing after remembered spans")
parser.add_argument("--phase35-stop-margin", type=float, default=5.0, help="target margin for the post-span token over repeated remembered tokens")
parser.add_argument("--phase35-stop-max-groups", type=int, default=4, help="maximum remembered spans per example used by the phase 3.5 stop objective")
parser.add_argument("--phase36-branch-recovery-loss-weight", type=float, default=0.0, help="phase 3.6 loss weight for correcting sampled wrong branch/default tokens from their generated context")
parser.add_argument("--phase36-branch-recovery-batch-frac", type=float, default=0.0, help="fraction of episodes that receive generated-branch recovery loss")
parser.add_argument("--phase36-branch-recovery-steps", type=int, default=16, help="maximum sampled answer-prefix tokens before phase 3.6 branch recovery")
parser.add_argument("--phase36-branch-recovery-temperature", type=float, default=0.3, help="sampling temperature for phase 3.6 branch recovery prefixes")
parser.add_argument("--phase36-branch-recovery-top-k", type=int, default=12, help="top-k filter for phase 3.6 branch recovery prefixes")
parser.add_argument("--phase36-branch-min-prefix-tokens", type=int, default=3, help="minimum sampled answer-prefix length before no-memory default tokens can trigger branch recovery")
parser.add_argument("--phase36-branch-recovery-max-target-tokens", type=int, default=12, help="maximum gold recovery tokens supervised from a bad branch point")
parser.add_argument("--phase36-branch-recovery-post-key-tokens", type=int, default=8, help="gold post-key continuation tokens included after the corrected span")
parser.add_argument("--phase36-branch-margin-loss-weight", type=float, default=0.75, help="extra phase 3.6 loss weight forcing corrected branch tokens above the sampled bad/default token")
parser.add_argument("--phase36-branch-margin", type=float, default=5.0, help="target logit margin for phase 3.6 corrected branch tokens over bad/default confusers")
parser.add_argument("--phase36-branch-anchor-tokens", type=int, default=3, help="number of corrected recovery tokens that receive phase 3.6 branch-margin loss")
parser.add_argument("--phase36-no-memory-confuser-top-k", type=int, default=16, help="dynamic no-memory top-k tokens that can trigger/generated-context branch recovery")
parser.add_argument("--phase37-branch-loss-weight", type=float, default=0.0, help="phase 3.7 whole-span generated branch contrast loss weight")
parser.add_argument("--phase37-branch-batch-frac", type=float, default=0.0, help="fraction of episodes that receive phase 3.7 generated-branch contrast")
parser.add_argument("--phase37-branch-ce-weight", type=float, default=0.25, help="phase 3.7 CE weight on the corrected span from generated context")
parser.add_argument("--phase37-branch-margin", type=float, default=2.0, help="target average log-prob margin for the correct remembered span over wrong generated branches")
parser.add_argument("--phase37-branch-max-confusers", type=int, default=4, help="maximum wrong spans/default branches contrasted per phase 3.7 event")
parser.add_argument("--phase37-stop-loss-weight", type=float, default=0.5, help="phase 3.7 loss weight for stopping/continuing after a generated correct key")
parser.add_argument("--phase37-stop-margin", type=float, default=5.0, help="target logit margin for the post-key token over repeated remembered tokens in phase 3.7")
parser.add_argument("--phase37-sample-steps", type=int, default=16, help="maximum sampled answer-prefix tokens before phase 3.7 branch/stop selection")
parser.add_argument("--phase37-temperature", type=float, default=0.35, help="sampling temperature for phase 3.7 generated prefixes")
parser.add_argument("--phase37-top-k", type=int, default=12, help="top-k filter for phase 3.7 generated prefixes")
parser.add_argument("--phase37-min-prefix-tokens", type=int, default=3, help="minimum generated answer-prefix length before no-memory defaults can trigger phase 3.7")
parser.add_argument("--phase37-max-span-tokens", type=int, default=12, help="maximum remembered span tokens used by phase 3.7")
parser.add_argument("--phase37-post-key-tokens", type=int, default=8, help="gold post-key continuation tokens used by phase 3.7 stop training")
parser.add_argument("--phase37-no-memory-confuser-top-k", type=int, default=16, help="dynamic no-memory top-k tokens that can become phase 3.7 one-token branch confusers")
parser.add_argument("--phase38-trajectory-loss-weight", type=float, default=0.0, help="phase 3.8 sequence-level preference loss weight for gold answers over generated bad trajectories")
parser.add_argument("--phase38-trajectory-batch-frac", type=float, default=0.0, help="fraction of episodes that receive phase 3.8 generated-answer preference training")
parser.add_argument("--phase38-trajectory-margin", type=float, default=1.0, help="target average log-prob margin for gold answer trajectories over generated bad trajectories")
parser.add_argument("--phase38-trajectory-gold-ce-weight", type=float, default=0.20, help="extra CE weight on the full gold answer inside phase 3.8 trajectory training")
parser.add_argument("--phase38-sample-steps", type=int, default=24, help="maximum answer tokens sampled before phase 3.8 trajectory classification")
parser.add_argument("--phase38-temperature", type=float, default=0.25, help="sampling temperature for phase 3.8 generated bad trajectories")
parser.add_argument("--phase38-high-temperature", type=float, default=0.0, help="optional higher sampling temperature mixed into phase 3.8 for lower-probability failures")
parser.add_argument("--phase38-high-temperature-frac", type=float, default=0.0, help="fraction of phase 3.8 sampled trajectories that use --phase38-high-temperature")
parser.add_argument("--phase38-top-k", type=int, default=12, help="top-k filter for phase 3.8 generated trajectories")
parser.add_argument("--phase38-min-prefix-tokens", type=int, default=3, help="minimum generated answer-prefix length before phase 3.8 no-memory defaults can trigger")
parser.add_argument("--phase38-answer-tokens", type=int, default=32, help="maximum answer tokens used by the phase 3.8 trajectory preference objective")
parser.add_argument("--phase38-max-span-tokens", type=int, default=12, help="maximum remembered span tokens used by phase 3.8")
parser.add_argument("--phase38-post-key-tokens", type=int, default=8, help="gold post-key continuation tokens used by phase 3.8")
parser.add_argument("--phase38-no-memory-confuser-top-k", type=int, default=16, help="dynamic no-memory top-k tokens that can become phase 3.8 bad trajectory triggers")
parser.add_argument("--phase38-hard-token-loss-weight", type=float, default=0.0, help="extra phase 3.8 preference weight on the hardest factual answer-token margins")
parser.add_argument("--phase38-hard-token-margin", type=float, default=1.5, help="target log-prob margin for gold factual tokens over aligned bad-trajectory tokens")
parser.add_argument("--phase38-hard-token-top-k", type=int, default=3, help="number of hardest factual answer-token offsets used by phase 3.8 trajectory preference")
parser.add_argument("--phase38-hard-token-post-key", type=int, default=1, help="number of post-span delimiter/continuation tokens treated as decisive in phase 3.8")
parser.add_argument("--phase38-default-branch-loss-weight", type=float, default=0.0, help="extra phase 3.8 loss weight when a sampled answer chooses a no-memory/default token")
parser.add_argument("--phase38-default-branch-margin", type=float, default=3.0, help="target log-prob margin for the correct gold token over the no-memory/default token")
parser.add_argument("--phase38-default-recovery-ce-weight", type=float, default=0.25, help="short recovery CE weight from the generated default-branch prefix")
parser.add_argument("--phase38-default-recovery-tokens", type=int, default=8, help="number of gold continuation tokens supervised after a no-memory/default branch")
parser.add_argument("--phase38-probe-binding-loss-weight", type=float, default=0.0, help="phase 3.8 loss weight forcing memory-probe spans to prefer current/gold spans over stale/confuser spans")
parser.add_argument("--phase38-probe-binding-margin", type=float, default=2.0, help="target memory-probe average log-prob margin for gold spans over stale/confuser spans")
parser.add_argument("--phase38-probe-binding-max-spans", type=int, default=4, help="maximum remembered spans per example used by phase 3.8 probe binding")
parser.add_argument("--phase38-branchpoint-loss-weight", type=float, default=0.0, help="phase 3.8 loss weight for earliest sampled bad decision on the answer path")
parser.add_argument("--phase38-branchpoint-batch-frac", type=float, default=0.0, help="fraction of phase 3.8 examples that receive branch-point training")
parser.add_argument("--phase38-branchpoint-margin", type=float, default=3.0, help="target log-prob margin for the gold branch token over the sampled bad token/confusers")
parser.add_argument("--phase38-branchpoint-recovery-ce-weight", type=float, default=0.25, help="CE weight on the gold continuation after a branch-point correction")
parser.add_argument("--phase38-branchpoint-recovery-tokens", type=int, default=8, help="number of gold tokens supervised after a branch-point correction")
parser.add_argument("--phase38-branchpoint-answer-start-tokens", type=int, default=0, help="number of early answer tokens treated as structural branch points")
parser.add_argument("--phase38-branchpoint-steps", type=int, default=24, help="maximum sampled answer tokens searched for a branch-point event")
parser.add_argument("--phase38-branchpoint-temperature", type=float, default=0.30, help="sampling temperature for branch-point event discovery")
parser.add_argument("--phase38-branchpoint-top-k", type=int, default=16, help="top-k filter for branch-point event discovery")
parser.add_argument("--phase1-steps", type=int, default=0, help="phase 1: memory-only steps with the trunk frozen")
parser.add_argument("--phase2-steps", type=int, default=0, help="phase 2: guarded joint steps (memory + interface params)")
parser.add_argument("--phase3-steps", type=int, default=0, help="phase 3: generated-prefix recovery steps")
parser.add_argument("--phase35-steps", type=int, default=0, help="phase 3.5: branch binding and post-key cleanup steps")
parser.add_argument("--phase36-steps", type=int, default=0, help="phase 3.6: generated-context branch recovery steps")
parser.add_argument("--phase37-steps", type=int, default=0, help="phase 3.7: generated-context span contrast and post-key stop steps")
parser.add_argument("--phase38-steps", type=int, default=0, help="phase 3.8: trajectory-level answer preference and probe binding steps")
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

phase_total = args.phase1_steps + args.phase2_steps + args.phase3_steps + args.phase35_steps + args.phase36_steps + args.phase37_steps + args.phase38_steps
if phase_total == 0:
    args.phase1_steps = args.num_iterations
    phase_total = args.num_iterations
else:
    assert phase_total == args.num_iterations, "phase1+phase2+phase3+phase35+phase36+phase37+phase38 steps must equal num_iterations"
user_config.update(vars(args))
if not use_dummy_wandb:
    wandb_run.config.update(user_config, allow_val_change=True)


def phase_for_step(step):
    if step < args.phase1_steps:
        return "phase1_memory_only"
    if step < args.phase1_steps + args.phase2_steps:
        return "phase2_interface_joint"
    if step < args.phase1_steps + args.phase2_steps + args.phase3_steps:
        return "phase3_generated_recovery"
    if step < args.phase1_steps + args.phase2_steps + args.phase3_steps + args.phase35_steps:
        return "phase3_5_branch_binding"
    if step < args.phase1_steps + args.phase2_steps + args.phase3_steps + args.phase35_steps + args.phase36_steps:
        return "phase3_6_branch_recovery"
    if step < args.phase1_steps + args.phase2_steps + args.phase3_steps + args.phase35_steps + args.phase36_steps + args.phase37_steps:
        return "phase3_7_branch_contrast"
    if args.phase38_steps > 0:
        return "phase3_8_trajectory_preference"
    if args.phase37_steps > 0:
        return "phase3_7_branch_contrast"
    if args.phase36_steps > 0:
        return "phase3_6_branch_recovery"
    return "phase3_generated_recovery"


PHASE_RANK = {
    "phase1": 1,
    "phase2": 2,
    "phase3": 3,
    "phase35": 4,
    "phase36": 5,
    "phase37": 6,
    "phase38": 7,
    "phase1_memory_only": 1,
    "phase2_interface_joint": 2,
    "phase3_generated_recovery": 3,
    "phase3_full_joint": 3,
    "phase3_5_branch_binding": 4,
    "phase3_6_branch_recovery": 5,
    "phase3_7_branch_contrast": 6,
    "phase3_8_trajectory_preference": 7,
}


def deference_enabled(phase_name):
    if args.deference_start_phase == "off":
        return False
    return PHASE_RANK[phase_name] >= PHASE_RANK[args.deference_start_phase]


def phase35_enabled(phase_name):
    return PHASE_RANK[phase_name] >= PHASE_RANK["phase35"] and (
        args.phase35_branch_loss_weight > 0 or args.phase35_stop_loss_weight > 0
    )


def phase36_enabled(phase_name):
    return (
        PHASE_RANK[phase_name] >= PHASE_RANK["phase36"]
        and args.phase36_branch_recovery_loss_weight > 0
        and args.phase36_branch_recovery_batch_frac > 0
    )


def phase37_enabled(phase_name):
    return (
        PHASE_RANK[phase_name] >= PHASE_RANK["phase37"]
        and args.phase37_branch_loss_weight > 0
        and args.phase37_branch_batch_frac > 0
    )


def phase38_enabled(phase_name):
    return (
        phase_name == "phase3_8_trajectory_preference"
        and (
            args.phase38_trajectory_loss_weight > 0
            or args.phase38_probe_binding_loss_weight > 0
            or args.phase38_branchpoint_loss_weight > 0
        )
    )


def set_phase_trainability(phase_name):
    if phase_name == "phase1_memory_only":
        enabled = (memory_gate_params, memory_params)
        disabled = (interface_params, trunk_params)
    elif phase_name == "phase2_interface_joint":
        enabled = (memory_gate_params, memory_params, interface_params)
        disabled = (trunk_params,)
    else:
        trunk_trainable = args.phase3_trunk_trainable and (
            not args.phase3_only_trunk_trainable or phase_name == "phase3_generated_recovery"
        )
        if trunk_trainable:
            enabled = (memory_gate_params, memory_params, interface_params, trunk_params)
            disabled = ()
        else:
            enabled = (memory_gate_params, memory_params, interface_params)
            disabled = (trunk_params,)

    for group in enabled:
        for param in group:
            param.requires_grad_(True)
    for group in disabled:
        for param in group:
            param.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def device_batch_size_for_phase(phase_name):
    if phase_name == "phase2_interface_joint" and args.phase2_device_batch_size > 0:
        return args.phase2_device_batch_size
    if phase_name == "phase3_generated_recovery" and args.phase3_device_batch_size > 0:
        return args.phase3_device_batch_size
    if phase_name == "phase3_8_trajectory_preference" and args.phase38_device_batch_size > 0:
        return args.phase38_device_batch_size
    return args.device_batch_size


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
print0(f"Episodic mode: {model.config.episodic_mode}")

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
    # Clone both shifted views so masking targets never back-propagates `-1`
    # into the overlapping input slice.
    inputs = token_tensor[:, :-1].clone().contiguous()
    targets = token_tensor[:, 1:].clone().to(dtype=torch.long).contiguous()
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


def has_short_repetition(token_ids):
    if len(token_ids) < 3:
        return False
    run = 1
    previous = None
    for token_id in token_ids:
        if token_id == previous:
            run += 1
            if run >= 3:
                return True
        else:
            previous = token_id
            run = 1
    if len(token_ids) >= 4:
        for idx in range(len(token_ids) - 3):
            if token_ids[idx] == token_ids[idx + 2] and token_ids[idx + 1] == token_ids[idx + 3]:
                return True
    return False


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


def pick_negative_span_ids(texts, target_span_len):
    spans = []
    seen = set()
    for text in texts:
        candidates = encode_text_candidates(text)
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda ids: (abs(len(ids) - target_span_len), -len(ids)))
        for candidate in candidates[:2]:
            key = tuple(int(token_id) for token_id in candidate)
            if key and key not in seen:
                seen.add(key)
                spans.append(list(key))
    return spans


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
        hard_negatives = group.get("hard_negatives", [])
        negative_token_ids = pick_negative_token_ids(hard_negatives, len(chosen_span_ids))
        negative_span_ids = pick_negative_span_ids(hard_negatives, len(chosen_span_ids))
        resolved.append(
            {
                "text": text,
                "span_ids": [int(token_id) for token_id in chosen_span_ids],
                "positions": positions,
                "hard_negative_token_ids": negative_token_ids,
                "hard_negative_span_ids": negative_span_ids,
            }
        )
    return resolved


def _tensor_positions_to_list(positions):
    if positions is None:
        return []
    return [int(pos) for pos in positions.detach().cpu().flatten().tolist()]


def build_episode_record(conversation):
    messages = conversation["messages"]
    memory_target = conversation.get("memory_target")
    guardrail_only = (
        isinstance(memory_target, dict)
        and (
            memory_target.get("mode") == "guardrail_only"
            or memory_target.get("family") in {"ordinary_chat_guardrail", "smoltalk"}
        )
    )
    if len(messages) < 4 and not guardrail_only:
        return None

    has_system = messages[0]["role"] == "system"
    min_split = 3 if has_system else (0 if guardrail_only else 1)
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

    if context_messages:
        context_ids, _ = tokenizer.render_conversation({"messages": context_messages}, max_tokens=args.max_seq_len)
    else:
        # Guardrail-only two-message rows intentionally have no memory context.
        # Keep a minimal BOS context so the user/turn memory builder produces
        # an empty memory state while the final user/assistant pair still trains.
        context_ids = [int(tokenizer.get_bos_token_id())]
    if len(context_ids) < 2 and not guardrail_only:
        return None
    target_ids, target_mask = tokenizer.render_conversation({"messages": target_messages}, max_tokens=args.max_seq_len + 1)
    if len(target_ids) < 2:
        return None

    context_ids = context_ids[-args.max_seq_len:]
    mask = target_mask[:args.max_seq_len + 1]
    target_ids = target_ids[:args.max_seq_len + 1]
    if len(target_ids) < 2:
        return None
    input_ids = [int(token_id) for token_id in target_ids[:-1]]
    shifted_targets = [int(token_id) for token_id in target_ids[1:]]
    shifted_mask = list(mask[1:])
    shifted_targets = [
        token_id if bool(shifted_mask[idx]) else -1
        for idx, token_id in enumerate(shifted_targets)
    ]
    targets = torch.tensor([shifted_targets], dtype=torch.long)
    answer_positions = positions_from_target_mask(targets)
    if answer_positions.numel() == 0:
        return None

    first_target_pos = int(answer_positions[0].item())
    fact_group_infos = [] if guardrail_only else resolve_fact_groups(target_ids, targets, memory_target)
    if guardrail_only:
        fact_positions = torch.zeros((0,), dtype=torch.long, device=targets.device)
        identity_positions = torch.zeros((0,), dtype=torch.long, device=targets.device)
        anchor_target_pos = first_target_pos
        template_positions = answer_positions
        rest_answer_positions = torch.zeros((0,), dtype=torch.long, device=targets.device)
    elif fact_group_infos:
        fact_positions = torch.unique(torch.cat([group["positions"] for group in fact_group_infos], dim=0))
        fact_positions, _ = torch.sort(fact_positions)
        anchor_target_pos = int(fact_positions[0].item())
        fact_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=targets.device)
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
        identity_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=targets.device)
        if identity_positions.numel() > 0:
            identity_mask[identity_positions] = True
        rest_answer_positions = answer_positions[(answer_positions >= anchor_target_pos) & ~identity_mask[answer_positions]]

    fact_group_records = []
    for group in fact_group_infos:
        fact_group_records.append(
            {
                "text": group.get("text"),
                "span_ids": [int(token_id) for token_id in group.get("span_ids", [])],
                "positions": _tensor_positions_to_list(group.get("positions")),
                "hard_negative_token_ids": [int(token_id) for token_id in group.get("hard_negative_token_ids", [])],
                "hard_negative_span_ids": [
                    [int(token_id) for token_id in span]
                    for span in group.get("hard_negative_span_ids", [])
                ],
            }
        )

    return {
        "context_ids": [int(token_id) for token_id in context_ids],
        "input_ids": input_ids,
        "target_ids": shifted_targets,
        "template_positions": _tensor_positions_to_list(template_positions),
        "identity_positions": _tensor_positions_to_list(identity_positions),
        "fact_positions": _tensor_positions_to_list(fact_positions),
        "rest_answer_positions": _tensor_positions_to_list(rest_answer_positions),
        "anchor_target_pos": int(anchor_target_pos),
        "fact_group_infos": fact_group_records,
        "guardrail_only": bool(guardrail_only),
    }


def tensorize_positions(positions):
    return torch.tensor(positions, dtype=torch.long, device=device) if positions else torch.zeros((0,), dtype=torch.long, device=device)


def tensorize_fact_groups(groups):
    tensor_groups = []
    for group in groups:
        tensor_group = dict(group)
        tensor_group["positions"] = tensorize_positions(group.get("positions", []))
        tensor_groups.append(tensor_group)
    return tensor_groups


def build_episode(conversation):
    record = build_episode_record(conversation)
    if record is None:
        return None
    context_tensor = torch.tensor([record["context_ids"]], dtype=torch.long, device=device)
    inputs = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    targets = torch.tensor([record["target_ids"]], dtype=torch.long, device=device)
    return (
        context_tensor,
        inputs,
        targets,
        tensorize_positions(record["template_positions"]),
        tensorize_positions(record["identity_positions"]),
        tensorize_positions(record["fact_positions"]),
        tensorize_positions(record["rest_answer_positions"]),
        int(record["anchor_target_pos"]),
        tensorize_fact_groups(record["fact_group_infos"]),
    )


def build_episode_cache(start_cursor=0):
    if args.episode_cache_size <= 0:
        return None, start_cursor
    target_size = min(args.episode_cache_size, len(dataset))
    records = []
    scan_cursor = int(start_cursor) % max(len(dataset), 1)
    scanned = 0
    skipped = 0
    next_report = 10000
    start_time = time.time()
    while len(records) < target_size and scanned < len(dataset):
        record = build_episode_record(dataset[scan_cursor])
        scan_cursor = (scan_cursor + 1) % len(dataset)
        scanned += 1
        if record is not None:
            records.append(record)
        else:
            skipped += 1
        if len(records) >= next_report:
            print0(f"Pre-tokenized memory episodes: {len(records):,}/{target_size:,}")
            next_report += 10000
    print0(
        f"Pre-tokenized memory episode cache: {len(records):,} examples "
        f"(skipped {skipped:,}, scanned {scanned:,}, next source cursor {scan_cursor:,}) "
        f"in {time.time() - start_time:.1f}s"
    )
    return records, scan_cursor


episode_cache, cache_source_cursor = build_episode_cache()


def next_episode_record():
    global cursor, episode_cache, cache_source_cursor
    if episode_cache:
        cache_idx = cursor % len(episode_cache)
        if (args.episode_cache_reshuffle or args.episode_cache_refresh) and cursor > 0 and cache_idx == 0:
            if args.episode_cache_refresh:
                episode_cache, cache_source_cursor = build_episode_cache(cache_source_cursor)
                print0(f"Refreshed pre-tokenized episode cache after {cursor:,} consumed examples")
            else:
                rng.shuffle(episode_cache)
                print0(f"Reshuffled pre-tokenized episode cache after {cursor:,} examples")
        record = episode_cache[cache_idx]
        cursor += 1
        return record
    while True:
        episode = build_episode_record(dataset[cursor])
        cursor = (cursor + 1) % len(dataset)
        if episode is not None:
            return episode


def next_episode():
    record = next_episode_record()
    context_tensor = torch.tensor([record["context_ids"]], dtype=torch.long, device=device)
    inputs = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    targets = torch.tensor([record["target_ids"]], dtype=torch.long, device=device)
    return (
        context_tensor,
        inputs,
        targets,
        tensorize_positions(record["template_positions"]),
        tensorize_positions(record["identity_positions"]),
        tensorize_positions(record["fact_positions"]),
        tensorize_positions(record["rest_answer_positions"]),
        int(record["anchor_target_pos"]),
        tensorize_fact_groups(record["fact_group_infos"]),
    )


def build_empty_memory_override(batch_size):
    return model.empty_memory_override(batch_size, device=device, dtype=model.transformer.wte.weight.dtype)


PAD_TOKEN_ID = 0


def pad_token_rows(rows, pad_value=PAD_TOKEN_ID):
    batch_size = len(rows)
    max_len = max((len(row) for row in rows), default=1)
    max_len = max(max_len, 1)
    tokens = torch.full((batch_size, max_len), int(pad_value), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for batch_idx, row in enumerate(rows):
        if not row:
            continue
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        tokens[batch_idx, : row_tensor.numel()] = row_tensor
        mask[batch_idx, : row_tensor.numel()] = True
    return tokens, mask


def pad_target_rows(rows):
    batch_size = len(rows)
    max_len = max((len(row) for row in rows), default=1)
    max_len = max(max_len, 1)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    for batch_idx, row in enumerate(rows):
        if not row:
            continue
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        targets[batch_idx, : row_tensor.numel()] = row_tensor
    return targets


def next_episode_batch(batch_size=None):
    batch_size = max(1, int(batch_size or args.device_batch_size))
    records = [next_episode_record() for _ in range(batch_size)]
    context_ids, context_mask = pad_token_rows([record["context_ids"] for record in records])
    inputs, _ = pad_token_rows([record["input_ids"] for record in records])
    targets = pad_target_rows([record["target_ids"] for record in records])
    metas = []
    for record in records:
        metas.append(
            {
                "context_ids": record["context_ids"],
                "template_positions": tensorize_positions(record["template_positions"]),
                "identity_positions": tensorize_positions(record["identity_positions"]),
                "fact_positions": tensorize_positions(record["fact_positions"]),
                "rest_answer_positions": tensorize_positions(record["rest_answer_positions"]),
                "anchor_target_pos": int(record["anchor_target_pos"]),
                "fact_group_infos": tensorize_fact_groups(record["fact_group_infos"]),
                "guardrail_only": bool(record.get("guardrail_only", False)),
            }
        )
    return context_ids, context_mask, inputs, targets, metas


def slice_outputs(outputs, batch_idx):
    sliced = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > batch_idx:
            sliced[key] = value[batch_idx: batch_idx + 1]
        else:
            sliced[key] = value
    return sliced


def slice_memory_override(memory_override, batch_idx):
    return [
        {
            key: value[batch_idx: batch_idx + 1] if torch.is_tensor(value) and value.dim() > 0 else value
            for key, value in layer_state.items()
        }
        for layer_state in memory_override
    ]


def select_memory_override(memory_override, batch_indices):
    idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
    return [
        {
            key: value.index_select(0, idx) if torch.is_tensor(value) and value.dim() > 0 and value.size(0) >= idx.numel() else value
            for key, value in layer_state.items()
        }
        for layer_state in memory_override
    ]


def scatter_memory_override(memory_override, batch_indices, updated):
    idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
    for layer_state, updated_state in zip(memory_override, updated):
        for key, value in layer_state.items():
            if torch.is_tensor(value) and value.dim() > 0 and key in updated_state:
                value.index_copy_(0, idx, updated_state[key])


def batched_write_state(token_rows):
    event_tokens, event_mask = pad_token_rows(token_rows)
    block_ios = model.collect_block_ios(event_tokens, detach=False)
    return [
        block.build_episodic_state(
            key_source,
            value_source,
            model.episodic_controller,
            source_mask=event_mask,
        )
        for block, (key_source, value_source) in zip(model.transformer.h, block_ios)
    ]


def build_memory_override_for_batch(context_ids, context_mask, metas):
    if args.memory_build_mode == "full_context":
        block_ios = model.collect_block_ios(context_ids, detach=False)
        return [
            block.build_episodic_state(
                key_source,
                value_source,
                model.episodic_controller,
                source_mask=context_mask,
            )
            for block, (key_source, value_source) in zip(model.transformer.h, block_ios)
        ]

    batch_size = len(metas)
    memory_override = build_empty_memory_override(batch_size)
    event_lists = [
        model._split_memory_events(meta["context_ids"], write_mode=args.memory_build_mode)
        for meta in metas
    ]
    max_events = max((len(events) for events in event_lists), default=0)
    for event_idx in range(max_events):
        grouped = []
        batch_indices = []
        for batch_idx, events in enumerate(event_lists):
            if event_idx >= len(events):
                continue
            event = events[event_idx]
            # The common training path uses turn/user/full-context writes. Keep
            # recall-trace events on the old exact path instead of complicating
            # the hot path with per-example position gathers.
            if event.get("kind") != "memorize":
                single = torch.tensor([event["tokens"]], dtype=torch.long, device=device)
                write_state = model._build_recall_trace_state(
                    single,
                    event["query_positions"],
                    event["value_positions"],
                    reward=0.0,
                )
                current = slice_memory_override(memory_override, batch_idx)
                updated = model._update_memory_override(current, write_state)
                scatter_memory_override(memory_override, [batch_idx], updated)
                continue
            grouped.append(event["tokens"])
            batch_indices.append(batch_idx)
        if grouped:
            write_state = batched_write_state(grouped)
            current = select_memory_override(memory_override, batch_indices)
            updated = model._update_memory_override(current, write_state)
            scatter_memory_override(memory_override, batch_indices, updated)
    return memory_override


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


def anchor_margin_terms(logits_row, token_id, target_margin):
    token_id = int(token_id)
    correct_logit = logits_row[token_id]
    if logits_row.numel() <= 1:
        zero = logits_row.new_zeros(())
        return zero, float(correct_logit.item())
    masked = logits_row.clone()
    masked[token_id] = -float("inf")
    wrong_logit = masked.max()
    margin = correct_logit - wrong_logit
    loss = torch.relu(torch.tensor(target_margin, device=logits_row.device, dtype=logits_row.dtype) - margin)
    return loss, float(margin.item())


def weighted_masked_mean(values, weights, mask):
    selected_values = values[mask]
    if selected_values.numel() == 0:
        return None
    selected_weights = weights[mask].to(dtype=selected_values.dtype)
    denom = selected_weights.sum().clamp_min(1e-6)
    return (selected_values * selected_weights).sum() / denom


def first_fact_weights(fact_targets, valid, multiplier):
    weights = torch.ones_like(fact_targets, dtype=torch.float)
    if multiplier <= 1.0:
        return weights
    for batch_idx in range(valid.size(0)):
        valid_indices = valid[batch_idx].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() > 0:
            weights[batch_idx, valid_indices[0]] = multiplier
    return weights


def top_no_memory_confuser_ids(base_logits, fact_targets, top_k):
    if top_k <= 0 or base_logits.size(-1) <= 1:
        return None
    k = min(top_k, base_logits.size(-1) - 1)
    masked = base_logits.masked_fill(
        F.one_hot(fact_targets.clamp_min(0), num_classes=base_logits.size(-1)).to(dtype=torch.bool),
        -float("inf"),
    )
    return torch.topk(masked, k=k, dim=-1).indices


def key_token_objective(outputs, no_memory_outputs, targets, fact_positions, use_deference):
    if fact_positions.numel() == 0:
        return None
    if (
        args.key_token_ce_loss_weight <= 0
        and args.key_token_rank_loss_weight <= 0
        and args.key_token_utility_loss_weight <= 0
        and (not use_deference or args.habit_confuser_loss_weight <= 0)
    ):
        return None

    fact_targets = targets[:, fact_positions]
    valid = fact_targets.ne(-1)
    if not valid.any():
        return None

    logits = outputs["logits"][:, fact_positions, :]
    gather_index = fact_targets.clamp_min(0).unsqueeze(-1)
    correct_logits = logits.gather(-1, gather_index).squeeze(-1)
    fact_weights = first_fact_weights(
        fact_targets,
        valid,
        args.first_fact_token_multiplier if use_deference else 1.0,
    )
    wrong_logits = logits.masked_fill(
        F.one_hot(fact_targets.clamp_min(0), num_classes=logits.size(-1)).to(dtype=torch.bool),
        -float("inf"),
    ).max(dim=-1).values
    rank_margin = correct_logits - wrong_logits

    total = logits.new_zeros(())
    ce_value = 0.0
    rank_loss_value = 0.0
    rank_margin_value = float(rank_margin[valid].mean().item())
    utility_loss_value = 0.0
    utility_margin_value = 0.0
    habit_confuser_loss_value = 0.0
    habit_confuser_margin_value = 0.0

    if args.key_token_ce_loss_weight > 0:
        ce_terms = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            fact_targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(fact_targets)
        ce_loss = weighted_masked_mean(ce_terms, fact_weights, valid)
        if ce_loss is None:
            ce_loss = logits.new_zeros(())
        ce_value = float(ce_loss.item())
        total = total + args.key_token_ce_loss_weight * ce_loss

    if args.key_token_rank_loss_weight > 0:
        rank_terms = torch.relu(
            torch.tensor(args.key_token_rank_margin, device=device, dtype=rank_margin.dtype) - rank_margin[valid]
        )
        rank_weights = fact_weights[valid].to(dtype=rank_terms.dtype)
        rank_loss = (rank_terms * rank_weights).sum() / rank_weights.sum().clamp_min(1e-6)
        rank_loss_value = float(rank_loss.item())
        total = total + args.key_token_rank_loss_weight * rank_loss

    if args.key_token_utility_loss_weight > 0 and no_memory_outputs is not None:
        memory_lp = F.log_softmax(logits, dim=-1).gather(-1, gather_index).squeeze(-1)
        base_logits = no_memory_outputs["logits"][:, fact_positions, :]
        base_lp = F.log_softmax(base_logits, dim=-1).gather(-1, gather_index).squeeze(-1)
        utility_margin = (memory_lp - base_lp)[valid]
        if utility_margin.numel() > 0:
            utility_margin_value = float(utility_margin.mean().item())
            utility_terms = torch.relu(
                torch.tensor(args.key_token_utility_margin, device=device, dtype=utility_margin.dtype) - utility_margin
            )
            utility_weights = fact_weights[valid].to(dtype=utility_terms.dtype)
            utility_loss = (utility_terms * utility_weights).sum() / utility_weights.sum().clamp_min(1e-6)
            utility_loss_value = float(utility_loss.item())
            total = total + args.key_token_utility_loss_weight * utility_loss

    if use_deference and args.habit_confuser_loss_weight > 0 and no_memory_outputs is not None:
        base_logits = no_memory_outputs["logits"][:, fact_positions, :]
        confuser_ids = top_no_memory_confuser_ids(base_logits, fact_targets, args.habit_confuser_top_k)
        if confuser_ids is not None:
            memory_confuser_logits = logits.gather(-1, confuser_ids).max(dim=-1).values
            habit_margin = correct_logits - memory_confuser_logits
            habit_confuser_margin_value = float(habit_margin[valid].mean().item())
            habit_terms = torch.relu(
                torch.tensor(args.habit_confuser_margin, device=device, dtype=habit_margin.dtype) - habit_margin[valid]
            )
            habit_weights = fact_weights[valid].to(dtype=habit_terms.dtype)
            habit_loss = (habit_terms * habit_weights).sum() / habit_weights.sum().clamp_min(1e-6)
            habit_confuser_loss_value = float(habit_loss.item())
            total = total + args.habit_confuser_loss_weight * habit_loss

    return {
        "loss": total,
        "ce": ce_value,
        "rank_loss": rank_loss_value,
        "rank_margin": rank_margin_value,
        "utility_loss": utility_loss_value,
        "utility_margin": utility_margin_value,
        "habit_confuser_loss": habit_confuser_loss_value,
        "habit_confuser_margin": habit_confuser_margin_value,
    }


def contiguous_position_groups(positions):
    if positions is None or positions.numel() == 0:
        return []
    sorted_positions, _ = torch.sort(positions)
    groups = []
    start = 0
    for idx in range(1, int(sorted_positions.numel())):
        if int(sorted_positions[idx].item()) != int(sorted_positions[idx - 1].item()) + 1:
            groups.append(sorted_positions[start:idx])
            start = idx
    groups.append(sorted_positions[start:])
    return groups


def resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=None):
    spans = []
    if fact_group_infos:
        all_span_ids = [group.get("span_ids", []) for group in fact_group_infos]
        for idx, group in enumerate(fact_group_infos):
            positions = group.get("positions")
            if positions is None or positions.numel() == 0:
                continue
            negatives = list(group.get("hard_negative_span_ids", []))
            for other_idx, other_span in enumerate(all_span_ids):
                if other_idx != idx and other_span:
                    negatives.append(other_span)
            spans.append({"positions": positions, "negative_span_ids": negatives})
        return spans
    # For unannotated long structured answers, the legacy fallback marks the
    # first few assistant tokens as "identity" (e.g. "We should use the"),
    # which is not a real remembered span. Use fallback spans only for short
    # direct answers where the assistant output is essentially the value.
    if answer_positions is not None and int(answer_positions.numel()) > args.fallback_identity_span_tokens + 2:
        return []
    for positions in contiguous_position_groups(effective_fact_positions):
        spans.append({"positions": positions, "negative_span_ids": []})
    return spans


def valid_span_positions(targets, positions, max_tokens):
    if positions is None or positions.numel() == 0:
        return None
    if max_tokens > 0:
        positions = positions[:max_tokens]
    valid = targets[0, positions].ne(-1)
    positions = positions[valid]
    if positions.numel() == 0:
        return None
    return positions


def span_confuser_lp(log_probs, positions, candidate_ids):
    if not candidate_ids:
        return None, None
    usable = min(int(positions.numel()), len(candidate_ids))
    if usable <= 0:
        return None, None
    ids = torch.tensor(candidate_ids[:usable], dtype=torch.long, device=log_probs.device)
    pos = positions[:usable]
    vocab = log_probs.size(-1)
    valid = (ids >= 0) & (ids < vocab)
    if not valid.any():
        return None, None
    ids = ids[valid]
    pos = pos[valid]
    lp = log_probs[0, pos, :].gather(-1, ids.view(-1, 1)).squeeze(-1)
    return lp.mean(), int(ids.numel())


def span_level_objective(outputs, no_memory_outputs, targets, answer_positions, fact_group_infos, effective_fact_positions, use_deference):
    if not use_deference:
        return None
    if (
        args.span_contrast_loss_weight <= 0
        and args.span_utility_loss_weight <= 0
        and args.span_hard_token_loss_weight <= 0
    ):
        return None

    spans = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    if not spans:
        return None

    logits = outputs["logits"]
    log_probs = F.log_softmax(logits, dim=-1)
    base_log_probs = None
    if no_memory_outputs is not None:
        base_log_probs = F.log_softmax(no_memory_outputs["logits"], dim=-1)

    contrast_losses = []
    contrast_margins = []
    utility_losses = []
    utility_margins = []
    hard_token_losses = []
    hard_token_margins = []

    for span in spans:
        positions = valid_span_positions(targets, span["positions"], args.span_contrast_max_tokens)
        if positions is None:
            continue
        gold_ids = targets[0, positions].clamp_min(0)
        gold_lp = log_probs[0, positions, :].gather(-1, gold_ids.view(-1, 1)).squeeze(-1).mean()

        if args.span_utility_loss_weight > 0 and base_log_probs is not None:
            base_gold_lp = base_log_probs[0, positions, :].gather(-1, gold_ids.view(-1, 1)).squeeze(-1).mean()
            utility_margin = gold_lp - base_gold_lp
            utility_margins.append(float(utility_margin.item()))
            utility_losses.append(
                torch.relu(
                    torch.tensor(args.span_utility_margin, device=logits.device, dtype=utility_margin.dtype) - utility_margin
                )
            )

        if args.span_hard_token_loss_weight > 0:
            for offset, pos in enumerate(positions):
                gold_id = int(targets[0, pos].item())
                if gold_id < 0 or gold_id >= logits.size(-1):
                    continue

                confuser_ids = set()
                for negative_span_ids in span.get("negative_span_ids", []):
                    if not negative_span_ids:
                        continue
                    if offset < len(negative_span_ids):
                        confuser_ids.add(int(negative_span_ids[offset]))
                    # Most branch failures happen on the first arbitrary token,
                    # so keep the first competing token active even when the
                    # aligned offset is a continuation piece.
                    if offset == 0:
                        confuser_ids.add(int(negative_span_ids[0]))

                if no_memory_outputs is not None and args.span_contrast_top_k > 0:
                    base_row = no_memory_outputs["logits"][0, pos]
                    masked = base_row.clone()
                    masked[gold_id] = -float("inf")
                    k = min(args.span_contrast_top_k, masked.numel() - 1)
                    if k > 0:
                        confuser_ids.update(int(token_id) for token_id in torch.topk(masked, k=k).indices.tolist())

                confuser_ids = sorted(
                    token_id
                    for token_id in confuser_ids
                    if 0 <= token_id < logits.size(-1) and token_id != gold_id
                )
                if not confuser_ids:
                    continue

                row = logits[0, pos]
                confuser_tensor = torch.tensor(confuser_ids, dtype=torch.long, device=row.device)
                confuser_logit = torch.logsumexp(row[confuser_tensor], dim=0)
                hard_margin = row[gold_id] - confuser_logit
                hard_token_margins.append(float(hard_margin.item()))
                hard_token_losses.append(
                    torch.relu(
                        torch.tensor(args.span_hard_token_margin, device=row.device, dtype=row.dtype) - hard_margin
                    )
                )

        if args.span_contrast_loss_weight <= 0:
            continue

        negative_lps = []
        for negative_span_ids in span.get("negative_span_ids", []):
            neg_lp, _ = span_confuser_lp(log_probs, positions, negative_span_ids)
            if neg_lp is not None:
                negative_lps.append(neg_lp)

        if base_log_probs is not None and args.span_contrast_top_k > 0:
            base_rows = no_memory_outputs["logits"][:, positions, :]
            masked = base_rows.masked_fill(
                F.one_hot(gold_ids.view(1, -1), num_classes=base_rows.size(-1)).to(dtype=torch.bool),
                -float("inf"),
            )
            k = min(args.span_contrast_top_k, base_rows.size(-1) - 1)
            top_ids = torch.topk(masked, k=k, dim=-1).indices
            memory_confuser_lp = log_probs[:, positions, :].gather(-1, top_ids).max(dim=-1).values
            negative_lps.append(memory_confuser_lp[0].mean())

        if not negative_lps:
            continue

        confuser_lp = torch.stack(negative_lps).max()
        contrast_margin = gold_lp - confuser_lp
        contrast_margins.append(float(contrast_margin.item()))
        contrast_losses.append(
            torch.relu(
                torch.tensor(args.span_contrast_margin, device=logits.device, dtype=contrast_margin.dtype) - contrast_margin
            )
        )

    if not contrast_losses and not utility_losses and not hard_token_losses:
        return None

    total = logits.new_zeros(())
    contrast_loss_value = 0.0
    contrast_margin_value = 0.0
    utility_loss_value = 0.0
    utility_margin_value = 0.0
    hard_token_loss_value = 0.0
    hard_token_margin_value = 0.0

    if contrast_losses:
        contrast_loss = torch.stack(contrast_losses).mean()
        contrast_loss_value = float(contrast_loss.item())
        contrast_margin_value = sum(contrast_margins) / len(contrast_margins)
        total = total + args.span_contrast_loss_weight * contrast_loss

    if utility_losses:
        utility_loss = torch.stack(utility_losses).mean()
        utility_loss_value = float(utility_loss.item())
        utility_margin_value = sum(utility_margins) / len(utility_margins)
        total = total + args.span_utility_loss_weight * utility_loss

    if hard_token_losses:
        hard_terms = torch.stack(hard_token_losses)
        k = min(max(1, args.span_hard_token_top_k), hard_terms.numel())
        hard_token_loss = torch.topk(hard_terms, k=k).values.mean()
        hard_token_loss_value = float(hard_token_loss.item())
        hard_token_margin_value = sum(hard_token_margins) / len(hard_token_margins)
        total = total + args.span_hard_token_loss_weight * hard_token_loss

    return {
        "loss": total,
        "contrast_loss": contrast_loss_value,
        "contrast_margin": contrast_margin_value,
        "utility_loss": utility_loss_value,
        "utility_margin": utility_margin_value,
        "hard_token_loss": hard_token_loss_value,
        "hard_token_margin": hard_token_margin_value,
    }


def post_key_objective(outputs, targets, answer_positions, fact_group_infos, effective_fact_positions, use_deference):
    if not use_deference or args.post_key_loss_weight <= 0:
        return None
    spans = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    if not spans:
        return None

    answer_position_set = {int(pos.item()) for pos in answer_positions}
    logits = outputs["logits"]
    losses = []
    ce_values = []
    repeat_margins = []

    for span in spans[: max(1, args.post_key_max_groups)]:
        positions = valid_span_positions(targets, span["positions"], args.span_contrast_max_tokens)
        if positions is None:
            continue
        post_pos = int(positions[-1].item()) + 1
        if post_pos >= targets.size(1) or post_pos not in answer_position_set:
            continue
        target_id = int(targets[0, post_pos].item())
        if target_id == -1:
            continue
        row = logits[0, post_pos]
        ce = F.cross_entropy(row.view(1, -1), torch.tensor([target_id], dtype=torch.long, device=row.device))
        ce_values.append(float(ce.item()))
        post_loss = ce

        repeat_ids = torch.unique(targets[0, positions].clamp_min(0))
        repeat_ids = repeat_ids[(repeat_ids >= 0) & (repeat_ids < row.numel()) & (repeat_ids != target_id)]
        if repeat_ids.numel() > 0:
            repeat_logit = torch.logsumexp(row[repeat_ids], dim=0)
            margin = row[target_id] - repeat_logit
            repeat_margins.append(float(margin.item()))
            post_loss = post_loss + torch.relu(
                torch.tensor(args.post_key_repeat_margin, device=row.device, dtype=row.dtype) - margin
            )
        losses.append(post_loss)

    if not losses:
        return None

    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "ce": sum(ce_values) / len(ce_values) if ce_values else 0.0,
        "repeat_margin": sum(repeat_margins) / len(repeat_margins) if repeat_margins else 0.0,
    }


def phase35_branch_and_stop_objective(
    outputs,
    no_memory_outputs,
    targets,
    answer_positions,
    fact_group_infos,
    effective_fact_positions,
    phase_name,
):
    if not phase35_enabled(phase_name):
        return None

    spans = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    if not spans:
        return None

    logits = outputs["logits"]
    vocab_size = logits.size(-1)
    answer_position_set = {int(pos.item()) for pos in answer_positions}

    branch_losses = []
    branch_margins = []
    stop_losses = []
    stop_margins = []

    for span in spans:
        positions = valid_span_positions(targets, span["positions"], args.span_contrast_max_tokens)
        if positions is None:
            continue

        if args.phase35_branch_loss_weight > 0 and args.phase35_branch_tokens > 0:
            branch_positions = positions[: args.phase35_branch_tokens]
            for offset, pos in enumerate(branch_positions):
                gold_id = int(targets[0, pos].item())
                if gold_id < 0 or gold_id >= vocab_size:
                    continue

                confuser_ids = set()
                for negative_span_ids in span.get("negative_span_ids", []):
                    if not negative_span_ids:
                        continue
                    # Use the aligned token and the first token. The first token
                    # is where most arbitrary bindings branch (Pepper/Juniper,
                    # Clover/Miso, auth/chat).
                    if offset < len(negative_span_ids):
                        confuser_ids.add(int(negative_span_ids[offset]))
                    confuser_ids.add(int(negative_span_ids[0]))

                if no_memory_outputs is not None and args.phase35_branch_no_memory_top_k > 0:
                    base_row = no_memory_outputs["logits"][0, pos]
                    masked = base_row.clone()
                    if 0 <= gold_id < masked.numel():
                        masked[gold_id] = -float("inf")
                    k = min(args.phase35_branch_no_memory_top_k, masked.numel() - 1)
                    if k > 0:
                        confuser_ids.update(int(token_id) for token_id in torch.topk(masked, k=k).indices.tolist())

                confuser_ids = sorted(token_id for token_id in confuser_ids if 0 <= token_id < vocab_size and token_id != gold_id)
                if not confuser_ids:
                    continue

                row = logits[0, pos]
                confuser_tensor = torch.tensor(confuser_ids, dtype=torch.long, device=row.device)
                confuser_logit = torch.logsumexp(row[confuser_tensor], dim=0)
                margin = row[gold_id] - confuser_logit
                branch_margins.append(float(margin.item()))
                branch_losses.append(
                    torch.relu(
                        torch.tensor(args.phase35_branch_margin, device=row.device, dtype=row.dtype) - margin
                    )
                )

        if args.phase35_stop_loss_weight > 0 and len(stop_losses) < max(1, args.phase35_stop_max_groups):
            post_pos = int(positions[-1].item()) + 1
            if post_pos >= targets.size(1) or post_pos not in answer_position_set:
                continue
            target_id = int(targets[0, post_pos].item())
            if target_id < 0 or target_id >= vocab_size:
                continue
            row = logits[0, post_pos]
            post_loss = F.cross_entropy(row.view(1, -1), torch.tensor([target_id], dtype=torch.long, device=row.device))

            repeat_ids = set(int(token_id.item()) for token_id in torch.unique(targets[0, positions].clamp_min(0)))
            for negative_span_ids in span.get("negative_span_ids", []):
                repeat_ids.update(int(token_id) for token_id in negative_span_ids)
            repeat_ids = sorted(token_id for token_id in repeat_ids if 0 <= token_id < vocab_size and token_id != target_id)
            if repeat_ids:
                repeat_tensor = torch.tensor(repeat_ids, dtype=torch.long, device=row.device)
                repeat_logit = torch.logsumexp(row[repeat_tensor], dim=0)
                margin = row[target_id] - repeat_logit
                stop_margins.append(float(margin.item()))
                post_loss = post_loss + torch.relu(
                    torch.tensor(args.phase35_stop_margin, device=row.device, dtype=row.dtype) - margin
                )
            stop_losses.append(post_loss)

    if not branch_losses and not stop_losses:
        return None

    total = logits.new_zeros(())
    branch_loss_value = 0.0
    branch_margin_value = 0.0
    stop_loss_value = 0.0
    stop_margin_value = 0.0

    if branch_losses:
        branch_loss = torch.stack(branch_losses).mean()
        branch_loss_value = float(branch_loss.item())
        branch_margin_value = sum(branch_margins) / len(branch_margins)
        total = total + args.phase35_branch_loss_weight * branch_loss

    if stop_losses:
        stop_loss = torch.stack(stop_losses).mean()
        stop_loss_value = float(stop_loss.item())
        stop_margin_value = sum(stop_margins) / len(stop_margins) if stop_margins else 0.0
        total = total + args.phase35_stop_loss_weight * stop_loss

    return {
        "loss": total,
        "branch_loss": branch_loss_value,
        "branch_margin": branch_margin_value,
        "stop_loss": stop_loss_value,
        "stop_margin": stop_margin_value,
        "applied": 1.0,
    }


def sample_next_token_from_logits(logits, temperature, top_k):
    logits = logits.float()
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        if temperature <= 0:
            return int(indices[torch.argmax(values)].item())
        probs = F.softmax(values / max(temperature, 1e-5), dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(indices[sampled.item()].item())
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def suffix_prefix_len(sequence, pattern):
    if not sequence or not pattern:
        return 0
    max_len = min(len(sequence), len(pattern) - 1)
    for length in range(max_len, 0, -1):
        if sequence[-length:] == pattern[:length]:
            return length
    return 0


def first_subsequence_end(sequence, pattern):
    start = find_subsequence(sequence, pattern)
    if start is None:
        return None
    return start + len(pattern)


@torch.no_grad()
def rollout_prefix_tokens(prefix_tokens, memory_override, steps):
    generated = []
    rollout_tokens = list(prefix_tokens)
    for _ in range(steps):
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(logits, args.rollout_temperature, args.rollout_top_k)
        generated.append(next_token)
        rollout_tokens.append(next_token)
    return generated


@torch.no_grad()
def sampled_prefix_tokens(prefix_tokens, memory_override, steps, temperature, top_k, spans=None, key_tail_tokens=0):
    generated = []
    rollout_tokens = list(prefix_tokens)
    found_key_tail = None
    for _ in range(steps):
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(logits, temperature, top_k)
        generated.append(next_token)
        rollout_tokens.append(next_token)

        if spans:
            if generated_prefix_has_wrong_span(generated, spans) or has_short_repetition(generated):
                break
            if found_key_tail is None:
                for span in spans:
                    end = first_subsequence_end(generated, span["ids"])
                    if end is not None:
                        found_key_tail = end + max(0, key_tail_tokens)
                        break
            if found_key_tail is not None and len(generated) >= found_key_tail:
                break
    return generated


def key_is_missing(generated_tokens, key_tokens):
    if not generated_tokens or not key_tokens:
        return False
    if len(key_tokens) <= len(generated_tokens):
        return find_subsequence(generated_tokens, key_tokens) is None
    return key_tokens[0] not in generated_tokens


def answer_start_objective(outputs, targets, answer_positions, use_deference):
    if not use_deference or args.phase2_answer_start_ce_loss_weight <= 0 or args.phase2_answer_start_tokens <= 0:
        return None
    if answer_positions.numel() == 0:
        return None
    start_positions = answer_positions[: args.phase2_answer_start_tokens]
    start_targets = targets[:, start_positions]
    valid = start_targets.ne(-1)
    if not valid.any():
        return None
    start_logits = outputs["logits"][:, start_positions, :]
    ce_terms = F.cross_entropy(
        start_logits.reshape(-1, start_logits.size(-1)),
        start_targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(start_targets)
    loss = ce_terms[valid].mean()
    return {"loss": loss, "ce": float(loss.item())}


def generated_hits_no_memory_confuser(generated_tokens, answer_tokens, answer_positions, fact_answer_indices, no_memory_outputs):
    if no_memory_outputs is None or not generated_tokens or not fact_answer_indices:
        return False
    first_fact_answer_idx = fact_answer_indices[0]
    if first_fact_answer_idx >= int(answer_tokens.numel()):
        return False
    fact_pos = int(answer_positions[first_fact_answer_idx].item())
    correct_token = int(answer_tokens[first_fact_answer_idx].item())
    row = no_memory_outputs["logits"][0, fact_pos]
    masked = row.clone()
    masked[correct_token] = -float("inf")
    k = min(max(args.habit_confuser_top_k, 1), row.numel() - 1)
    confuser_ids = torch.topk(masked, k=k).indices.tolist()
    gold_answer_ids = {int(token.item()) for token in answer_tokens if int(token.item()) != -1}
    confuser_ids = {int(token_id) for token_id in confuser_ids if int(token_id) not in gold_answer_ids}
    if not confuser_ids:
        return False
    return any(int(token_id) in confuser_ids for token_id in generated_tokens)


def phase3_recovery_enabled(phase_name):
    if args.phase3_recovery_loss_weight <= 0 or args.phase3_recovery_batch_frac <= 0:
        return False
    if args.phase3_recovery_phase_only and phase_name != "phase3_generated_recovery":
        return False
    return PHASE_RANK[phase_name] >= PHASE_RANK[args.phase3_recovery_start_phase]


def answer_tokens_after_position(targets, answer_positions, position, max_tokens):
    if max_tokens <= 0:
        return []
    positions = answer_positions[answer_positions > position]
    if positions.numel() == 0:
        return []
    positions = positions[:max_tokens]
    ids = targets[0, positions].tolist()
    return [int(token_id) for token_id in ids if int(token_id) != -1]


def phase3_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions):
    spans = []
    resolved = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    for span in resolved:
        positions = valid_span_positions(targets, span["positions"], args.phase3_recovery_max_target_tokens)
        if positions is None:
            continue
        ids = [int(token_id) for token_id in targets[0, positions].tolist() if int(token_id) != -1]
        if not ids:
            continue
        post_ids = answer_tokens_after_position(
            targets,
            answer_positions,
            int(positions[-1].item()),
            args.phase3_recovery_post_key_tokens,
        )
        spans.append(
            {
                "ids": ids,
                "positions": positions,
                "post_ids": post_ids,
                "negative_span_ids": span.get("negative_span_ids", []),
            }
        )
    return spans


def phase36_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions):
    spans = []
    resolved = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    for span in resolved:
        positions = valid_span_positions(targets, span["positions"], args.phase36_branch_recovery_max_target_tokens)
        if positions is None:
            continue
        ids = [int(token_id) for token_id in targets[0, positions].tolist() if int(token_id) != -1]
        if not ids:
            continue
        post_ids = answer_tokens_after_position(
            targets,
            answer_positions,
            int(positions[-1].item()),
            args.phase36_branch_recovery_post_key_tokens,
        )
        spans.append(
            {
                "ids": ids,
                "positions": positions,
                "post_ids": post_ids,
                "negative_span_ids": span.get("negative_span_ids", []),
            }
        )
    return spans


def phase37_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions):
    spans = []
    resolved = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    for span in resolved:
        positions = valid_span_positions(targets, span["positions"], args.phase37_max_span_tokens)
        if positions is None:
            continue
        ids = [int(token_id) for token_id in targets[0, positions].tolist() if int(token_id) != -1]
        if not ids:
            continue
        post_ids = answer_tokens_after_position(
            targets,
            answer_positions,
            int(positions[-1].item()),
            args.phase37_post_key_tokens,
        )
        spans.append(
            {
                "ids": ids,
                "positions": positions,
                "post_ids": post_ids,
                "negative_span_ids": span.get("negative_span_ids", []),
            }
        )
    return spans


def phase38_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions):
    spans = []
    resolved = resolved_fact_spans(fact_group_infos, effective_fact_positions, answer_positions=answer_positions)
    for span in resolved:
        positions = valid_span_positions(targets, span["positions"], args.phase38_max_span_tokens)
        if positions is None:
            continue
        ids = [int(token_id) for token_id in targets[0, positions].tolist() if int(token_id) != -1]
        if not ids:
            continue
        post_ids = answer_tokens_after_position(
            targets,
            answer_positions,
            int(positions[-1].item()),
            args.phase38_post_key_tokens,
        )
        spans.append(
            {
                "ids": ids,
                "positions": positions,
                "post_ids": post_ids,
                "negative_span_ids": span.get("negative_span_ids", []),
            }
        )
    return spans


def generated_prefix_has_wrong_span(generated_tokens, spans):
    for span in spans:
        for negative_ids in span.get("negative_span_ids", []):
            if negative_ids and find_subsequence(generated_tokens, negative_ids) is not None:
                return True
    return False


def select_phase3_recovery(generated_tokens, spans):
    if not generated_tokens or not spans:
        return None

    repeated = has_short_repetition(generated_tokens)
    wrong_span = False
    found_span = None
    found_span_end = None
    first_missing_span = None

    for span in spans:
        end = first_subsequence_end(generated_tokens, span["ids"])
        if end is None:
            if first_missing_span is None:
                first_missing_span = span
        elif found_span is None or end < found_span_end:
            found_span = span
            found_span_end = end
        for negative_ids in span.get("negative_span_ids", []):
            if negative_ids and find_subsequence(generated_tokens, negative_ids) is not None:
                wrong_span = True

    if found_span is not None:
        tail = generated_tokens[found_span_end:]
        tail_repeats_key = bool(tail and any(token_id in set(found_span["ids"]) for token_id in tail))
        if repeated or tail_repeats_key:
            recovery_ids = found_span["post_ids"]
            if not recovery_ids and first_missing_span is not None:
                recovery_ids = first_missing_span["ids"]
            if recovery_ids:
                return {
                    "reason": "repeated",
                    "recovery_ids": recovery_ids,
                    "key_missing": False,
                    "partial_key": False,
                    "wrong_span": wrong_span,
                    "repeated": True,
                }
        if first_missing_span is None:
            return None

    for span in spans:
        partial_len = suffix_prefix_len(generated_tokens, span["ids"])
        if partial_len > 0:
            recovery_ids = span["ids"][partial_len:] + span["post_ids"]
            if recovery_ids:
                return {
                    "reason": "partial_key",
                    "recovery_ids": recovery_ids,
                    "key_missing": False,
                    "partial_key": True,
                    "wrong_span": wrong_span,
                    "repeated": repeated,
                }

    target_span = first_missing_span or spans[0]
    reason = "wrong_span" if wrong_span else "repeated" if repeated else "key_missing"
    return {
        "reason": reason,
        "recovery_ids": target_span["ids"] + target_span["post_ids"],
        "key_missing": first_missing_span is not None,
        "partial_key": False,
        "wrong_span": wrong_span,
        "repeated": repeated,
    }


def first_missing_span(generated_tokens, spans):
    for span in spans:
        if find_subsequence(generated_tokens, span["ids"]) is None:
            return span
    return None


def negative_remainders_for_prefix(prefix_len, negative_span_ids):
    remainders = []
    for negative_ids in negative_span_ids:
        if prefix_len < len(negative_ids):
            remainders.append([int(token_id) for token_id in negative_ids[prefix_len:]])
    return remainders


@torch.no_grad()
def no_memory_top_ids_for_context(rollout_tokens, no_memory_override, gold_answer_ids, top_k):
    if no_memory_override is None or top_k <= 0:
        return set()
    rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
    logits = model(rollout_idx, memory_override=no_memory_override)[0, -1].float().clone()
    for token_id in gold_answer_ids:
        if 0 <= token_id < logits.numel():
            logits[token_id] = -float("inf")
    k = min(top_k, logits.numel() - 1)
    if k <= 0:
        return set()
    return {int(token_id) for token_id in torch.topk(logits, k=k).indices.tolist()}


@torch.no_grad()
def sampled_branch_recovery_event(prefix_tokens, memory_override, no_memory_override, spans, gold_answer_ids):
    if not spans:
        return None

    generated = []
    rollout_tokens = list(prefix_tokens)
    min_prefix = max(0, args.phase36_branch_min_prefix_tokens)

    for _ in range(args.phase36_branch_recovery_steps):
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(
            logits,
            args.phase36_branch_recovery_temperature,
            args.phase36_branch_recovery_top_k,
        )

        no_memory_top_ids = set()
        if len(generated) >= min_prefix:
            no_memory_top_ids = no_memory_top_ids_for_context(
                rollout_tokens,
                no_memory_override,
                gold_answer_ids,
                args.phase36_no_memory_confuser_top_k,
            )

        # If the model has started a remembered span but is about to diverge,
        # train the remaining span from the generated prefix before the bad token.
        for span in spans:
            partial_len = suffix_prefix_len(generated, span["ids"])
            if 0 < partial_len < len(span["ids"]) and next_token != span["ids"][partial_len]:
                recovery_ids = span["ids"][partial_len:] + span["post_ids"]
                if recovery_ids:
                    return {
                        "reason": "partial_wrong",
                        "span": span,
                        "generated_prefix": list(generated),
                        "bad_token": int(next_token),
                        "no_memory_top_ids": no_memory_top_ids,
                        "recovery_ids": recovery_ids,
                    }

        candidate = generated + [next_token]

        # Directly catch known same-class negatives/current-vs-old confusions.
        for span in spans:
            for negative_ids in span.get("negative_span_ids", []):
                if not negative_ids:
                    continue
                if next_token in negative_ids or find_subsequence(candidate, negative_ids) is not None:
                    recovery_ids = span["ids"] + span["post_ids"]
                    if recovery_ids:
                        return {
                            "reason": "wrong_span",
                            "span": span,
                            "generated_prefix": list(generated),
                            "bad_token": int(next_token),
                            "no_memory_top_ids": no_memory_top_ids,
                            "recovery_ids": recovery_ids,
                        }

        # Catch no-memory/default habits from the same generated context. This is
        # intentionally gated by a short prefix so fluent-but-different answer
        # openers do not consume the branch objective before a fact decision.
        missing_span = first_missing_span(generated, spans)
        if (
            missing_span is not None
            and len(generated) >= min_prefix
            and next_token in no_memory_top_ids
            and next_token not in gold_answer_ids
        ):
            recovery_ids = missing_span["ids"] + missing_span["post_ids"]
            if recovery_ids:
                return {
                    "reason": "no_memory_default",
                    "span": missing_span,
                    "generated_prefix": list(generated),
                    "bad_token": int(next_token),
                    "no_memory_top_ids": no_memory_top_ids,
                    "recovery_ids": recovery_ids,
                }

        # If the model already emitted the correct key and is about to re-enter
        # the same key, teach the post-key continuation from the actual rollout.
        for span in spans:
            span_end = first_subsequence_end(generated, span["ids"])
            if span_end is not None and span["post_ids"] and next_token in set(span["ids"]):
                return {
                    "reason": "repeated_after_key",
                    "span": span,
                    "generated_prefix": list(generated),
                    "bad_token": int(next_token),
                    "no_memory_top_ids": no_memory_top_ids,
                    "recovery_ids": span["post_ids"],
                }

        generated.append(int(next_token))
        rollout_tokens.append(int(next_token))

        if has_short_repetition(generated):
            missing_span = first_missing_span(generated, spans) or spans[0]
            recovery_ids = missing_span["ids"] + missing_span["post_ids"]
            if recovery_ids:
                return {
                    "reason": "repeated",
                    "span": missing_span,
                    "generated_prefix": list(generated),
                    "bad_token": int(next_token),
                    "no_memory_top_ids": no_memory_top_ids,
                    "recovery_ids": recovery_ids,
                }

    return None


@torch.no_grad()
def sampled_phase37_event(prefix_tokens, memory_override, no_memory_override, spans, gold_answer_ids):
    if not spans:
        return None

    generated = []
    rollout_tokens = list(prefix_tokens)
    min_prefix = max(0, args.phase37_min_prefix_tokens)

    for _ in range(args.phase37_sample_steps):
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(
            logits,
            args.phase37_temperature,
            args.phase37_top_k,
        )

        no_memory_top_ids = set()
        if len(generated) >= min_prefix:
            no_memory_top_ids = no_memory_top_ids_for_context(
                rollout_tokens,
                no_memory_override,
                gold_answer_ids,
                args.phase37_no_memory_confuser_top_k,
            )

        # If a correct value has already been generated, the next decision is
        # not another branch decision. It is a stop/continue decision. This is
        # what catches "Biscuit. Biscuit..." and "Figaro" after the correct
        # "Fig" prefix.
        for span in spans:
            span_end = first_subsequence_end(generated, span["ids"])
            if span_end is None or not span["post_ids"]:
                continue
            repeat_ids = set(int(token_id) for token_id in span["ids"])
            repeat_ids.add(int(next_token))
            for negative_ids in span.get("negative_span_ids", []):
                repeat_ids.update(int(token_id) for token_id in negative_ids)
                neg_prefix_len = suffix_prefix_len(generated, negative_ids)
                if 0 < neg_prefix_len < len(negative_ids) and next_token == int(negative_ids[neg_prefix_len]):
                    return {
                        "kind": "stop",
                        "reason": "wrong_continuation",
                        "span": span,
                        "generated_prefix": list(generated),
                        "bad_token": int(next_token),
                        "target_ids": span["post_ids"],
                        "repeat_ids": repeat_ids,
                    }
            if next_token in set(span["ids"]):
                return {
                    "kind": "stop",
                    "reason": "repeated_after_key",
                    "span": span,
                    "generated_prefix": list(generated),
                    "bad_token": int(next_token),
                    "target_ids": span["post_ids"],
                    "repeat_ids": repeat_ids,
                }

        # If the model started a correct span and is about to diverge, train
        # the remaining gold span against the actual bad continuation and any
        # aligned same-field negatives.
        for span in spans:
            partial_len = suffix_prefix_len(generated, span["ids"])
            if 0 < partial_len < len(span["ids"]) and next_token != int(span["ids"][partial_len]):
                target_ids = [int(token_id) for token_id in span["ids"][partial_len:]]
                confusers = [[int(next_token)]]
                confusers.extend(negative_remainders_for_prefix(partial_len, span.get("negative_span_ids", [])))
                return {
                    "kind": "branch",
                    "reason": "partial_wrong",
                    "span": span,
                    "generated_prefix": list(generated),
                    "bad_token": int(next_token),
                    "target_ids": target_ids,
                    "confuser_span_ids": confusers,
                }

        candidate = generated + [next_token]

        # Direct same-field/current-vs-old branch confusion.
        for span in spans:
            for negative_ids in span.get("negative_span_ids", []):
                if not negative_ids:
                    continue
                if next_token in negative_ids or find_subsequence(candidate, negative_ids) is not None:
                    confusers = [[int(token_id) for token_id in negative_ids]]
                    for other_negative in span.get("negative_span_ids", []):
                        if other_negative and other_negative != negative_ids:
                            confusers.append([int(token_id) for token_id in other_negative])
                    return {
                        "kind": "branch",
                        "reason": "wrong_span",
                        "span": span,
                        "generated_prefix": list(generated),
                        "bad_token": int(next_token),
                        "target_ids": [int(token_id) for token_id in span["ids"]],
                        "confuser_span_ids": confusers,
                    }

        missing_span = first_missing_span(generated, spans)
        if (
            missing_span is not None
            and len(generated) >= min_prefix
            and next_token in no_memory_top_ids
            and next_token not in gold_answer_ids
        ):
            confusers = [[int(next_token)]]
            confusers.extend([int(token_id)] for token_id in no_memory_top_ids if int(token_id) != int(next_token))
            confusers.extend([int(token_id) for token_id in negative_ids] for negative_ids in missing_span.get("negative_span_ids", []))
            return {
                "kind": "branch",
                "reason": "no_memory_default",
                "span": missing_span,
                "generated_prefix": list(generated),
                "bad_token": int(next_token),
                "target_ids": [int(token_id) for token_id in missing_span["ids"]],
                "confuser_span_ids": confusers,
            }

        generated.append(int(next_token))
        rollout_tokens.append(int(next_token))

        if has_short_repetition(generated):
            for span in spans:
                if first_subsequence_end(generated, span["ids"]) is not None and span["post_ids"]:
                    repeat_ids = set(int(token_id) for token_id in span["ids"])
                    for negative_ids in span.get("negative_span_ids", []):
                        repeat_ids.update(int(token_id) for token_id in negative_ids)
                    return {
                        "kind": "stop",
                        "reason": "repeated",
                        "span": span,
                        "generated_prefix": list(generated),
                        "bad_token": int(next_token),
                        "target_ids": span["post_ids"],
                        "repeat_ids": repeat_ids,
                    }
            missing_span = first_missing_span(generated, spans) or spans[0]
            return {
                "kind": "branch",
                "reason": "repeated_without_key",
                "span": missing_span,
                "generated_prefix": list(generated),
                "bad_token": int(next_token),
                "target_ids": [int(token_id) for token_id in missing_span["ids"]],
                "confuser_span_ids": [[int(next_token)]],
            }

    return None


def generated_contains_any_negative_span(generated_tokens, spans):
    for span in spans:
        for negative_ids in span.get("negative_span_ids", []):
            if negative_ids and find_subsequence(generated_tokens, negative_ids) is not None:
                return True
    return False


def generated_has_wrong_post_key_continuation(generated_tokens, spans):
    for span in spans:
        span_end = first_subsequence_end(generated_tokens, span["ids"])
        if span_end is None or span_end >= len(generated_tokens):
            continue
        if not span.get("post_ids"):
            continue
        expected = int(span["post_ids"][0])
        observed = int(generated_tokens[span_end])
        if observed != expected:
            return True
    return False


def generated_partial_wrong_span(generated_tokens, spans):
    if not generated_tokens:
        return False
    prefix_without_last = generated_tokens[:-1]
    observed = int(generated_tokens[-1])
    for span in spans:
        partial_len = suffix_prefix_len(prefix_without_last, span["ids"])
        if 0 < partial_len < len(span["ids"]) and observed != int(span["ids"][partial_len]):
            return True
    return False


@torch.no_grad()
def sampled_phase38_trajectory_event(prefix_tokens, memory_override, no_memory_override, spans, gold_answer_ids):
    if not spans:
        return None

    generated = []
    rollout_tokens = list(prefix_tokens)
    min_prefix = max(0, args.phase38_min_prefix_tokens)
    sample_temperature = args.phase38_temperature
    if (
        args.phase38_high_temperature > 0
        and args.phase38_high_temperature_frac > 0
        and rng.random() < args.phase38_high_temperature_frac
    ):
        sample_temperature = args.phase38_high_temperature

    for _ in range(args.phase38_sample_steps):
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(
            logits,
            sample_temperature,
            args.phase38_top_k,
        )

        no_memory_top_ids = set()
        if len(generated) >= min_prefix:
            no_memory_top_ids = no_memory_top_ids_for_context(
                rollout_tokens,
                no_memory_override,
                gold_answer_ids,
                args.phase38_no_memory_confuser_top_k,
            )

        candidate = generated + [int(next_token)]
        reason = None

        if generated_partial_wrong_span(candidate, spans):
            reason = "partial_wrong"
        elif generated_contains_any_negative_span(candidate, spans):
            reason = "wrong_span"
        elif generated_has_wrong_post_key_continuation(candidate, spans):
            reason = "wrong_after_key"
        elif (
            len(generated) >= min_prefix
            and int(next_token) in no_memory_top_ids
            and int(next_token) not in gold_answer_ids
            and first_missing_span(generated, spans) is not None
        ):
            reason = "no_memory_default"

        generated.append(int(next_token))
        rollout_tokens.append(int(next_token))

        if reason is None and has_short_repetition(generated):
            reason = "repeated"

        if reason is not None:
            return {"reason": reason, "generated_answer": list(generated)}

    if first_missing_span(generated, spans) is not None:
        return {"reason": "key_missing", "generated_answer": list(generated)}
    if generated_contains_any_negative_span(generated, spans):
        return {"reason": "wrong_span", "generated_answer": list(generated)}
    if generated_has_wrong_post_key_continuation(generated, spans):
        return {"reason": "wrong_after_key", "generated_answer": list(generated)}
    if has_short_repetition(generated):
        return {"reason": "repeated", "generated_answer": list(generated)}
    return None


def sequence_token_logprobs(prefix_tokens, candidate_ids, memory_override):
    candidate_ids = [int(token_id) for token_id in candidate_ids if int(token_id) >= 0]
    if not candidate_ids:
        return None, None
    sequence = list(prefix_tokens) + candidate_ids
    if len(sequence) < 2:
        return None, None
    rollout_inputs = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    logits = model(rollout_inputs, memory_override=memory_override, return_components=False)
    start = len(prefix_tokens) - 1
    end = min(start + len(candidate_ids), logits.size(1))
    if start < 0 or start >= end:
        return None, None
    usable = end - start
    ids = torch.tensor(candidate_ids[:usable], dtype=torch.long, device=device)
    vocab_size = logits.size(-1)
    valid = (ids >= 0) & (ids < vocab_size)
    if not valid.any():
        return None, {"logits": logits}
    log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
    token_lps = log_probs[valid].gather(-1, ids[valid].view(-1, 1)).squeeze(-1)
    return token_lps, {"logits": logits}


def sequence_mean_logprob(prefix_tokens, candidate_ids, memory_override):
    token_lps, outputs = sequence_token_logprobs(prefix_tokens, candidate_ids, memory_override)
    if token_lps is None:
        return None, outputs
    return token_lps.mean(), outputs


def phase38_decisive_answer_offsets(answer_positions, targets, spans, max_answer_tokens):
    answer_offset_by_pos = {
        int(pos.item()): idx
        for idx, pos in enumerate(answer_positions[:max_answer_tokens])
    }
    offsets = set()

    for span in spans:
        positions = valid_span_positions(targets, span["positions"], args.phase38_max_span_tokens)
        if positions is None:
            continue
        for pos in positions:
            offset = answer_offset_by_pos.get(int(pos.item()))
            if offset is not None:
                offsets.add(offset)

        post_start = int(positions[-1].item()) + 1
        for delta in range(max(0, args.phase38_hard_token_post_key)):
            offset = answer_offset_by_pos.get(post_start + delta)
            if offset is not None:
                offsets.add(offset)

    return sorted(offset for offset in offsets if 0 <= offset < max_answer_tokens)


def phase38_hard_token_preference(
    prefix_tokens,
    gold_answer_ids,
    generated_answer_ids,
    answer_positions,
    targets,
    spans,
    memory_override,
):
    if args.phase38_hard_token_loss_weight <= 0:
        return None

    decisive_offsets = phase38_decisive_answer_offsets(
        answer_positions,
        targets,
        spans,
        min(len(gold_answer_ids), max(1, args.phase38_answer_tokens)),
    )
    usable_offsets = [
        offset
        for offset in decisive_offsets
        if offset < len(generated_answer_ids)
        and offset < len(gold_answer_ids)
        and int(generated_answer_ids[offset]) != int(gold_answer_ids[offset])
    ]
    if not usable_offsets:
        return None

    gold_lps, _ = sequence_token_logprobs(prefix_tokens, gold_answer_ids, memory_override)
    bad_lps, _ = sequence_token_logprobs(prefix_tokens, generated_answer_ids, memory_override)
    if gold_lps is None or bad_lps is None:
        return None

    usable_offsets = [
        offset
        for offset in usable_offsets
        if offset < int(gold_lps.numel()) and offset < int(bad_lps.numel())
    ]
    if not usable_offsets:
        return None

    offset_tensor = torch.tensor(usable_offsets, dtype=torch.long, device=gold_lps.device)
    margins = gold_lps[offset_tensor] - bad_lps[offset_tensor]
    terms = F.softplus(
        torch.tensor(args.phase38_hard_token_margin, device=margins.device, dtype=margins.dtype) - margins
    )
    k = min(max(1, args.phase38_hard_token_top_k), terms.numel())
    hard_loss = torch.topk(terms, k=k).values.mean()
    hard_margin = torch.topk(-margins, k=k).values.mul(-1).mean()
    return hard_loss, float(hard_margin.item())


def phase38_default_branch_objective(
    prefix_tokens,
    gold_answer_ids,
    generated_answer_ids,
    memory_override,
):
    if args.phase38_default_branch_loss_weight <= 0:
        return None
    if not gold_answer_ids or not generated_answer_ids:
        return None

    # The no-memory/default detector fires on the newest generated token.
    # Train from the generated prefix that led to that choice, so the model
    # learns to recover before the default branch becomes a full trajectory.
    branch_offset = len(generated_answer_ids) - 1
    if branch_offset < 0 or branch_offset >= len(gold_answer_ids):
        return None

    default_token = int(generated_answer_ids[branch_offset])
    gold_token = int(gold_answer_ids[branch_offset])
    if default_token == gold_token:
        return None
    if default_token < 0 or default_token >= model.config.vocab_size:
        return None
    if gold_token < 0 or gold_token >= model.config.vocab_size:
        return None

    branch_prefix = list(prefix_tokens) + [int(token_id) for token_id in generated_answer_ids[:branch_offset]]
    if not branch_prefix:
        return None

    branch_inputs = torch.tensor([branch_prefix], dtype=torch.long, device=device)
    branch_logits = model(branch_inputs, memory_override=memory_override, return_components=False)
    row = branch_logits[0, -1]
    log_probs = F.log_softmax(row, dim=-1)
    margin = log_probs[gold_token] - log_probs[default_token]
    margin_target = torch.tensor(
        args.phase38_default_branch_margin,
        device=margin.device,
        dtype=margin.dtype,
    )
    branch_loss = F.softplus(margin_target - margin)
    total = branch_loss

    recovery_ce_value = 0.0
    recovery_tokens = max(1, args.phase38_default_recovery_tokens)
    recovery_ids = [
        int(token_id)
        for token_id in gold_answer_ids[branch_offset : branch_offset + recovery_tokens]
        if 0 <= int(token_id) < model.config.vocab_size
    ]
    if args.phase38_default_recovery_ce_weight > 0 and recovery_ids:
        recovery_lps, _ = sequence_token_logprobs(branch_prefix, recovery_ids, memory_override)
        if recovery_lps is not None:
            recovery_ce = -recovery_lps.mean()
            recovery_ce_value = float(recovery_ce.item())
            total = total + args.phase38_default_recovery_ce_weight * recovery_ce

    return {
        "loss": total,
        "branch_loss": float(branch_loss.item()),
        "branch_margin": float(margin.item()),
        "recovery_ce": recovery_ce_value,
    }


def phase38_branchpoint_offsets(answer_positions, targets, spans, max_answer_tokens):
    offsets = set()
    for offset in range(min(max(0, args.phase38_branchpoint_answer_start_tokens), max_answer_tokens)):
        offsets.add(offset)
    offsets.update(phase38_decisive_answer_offsets(answer_positions, targets, spans, max_answer_tokens))
    return offsets


@torch.no_grad()
def sampled_phase38_branchpoint_event(
    prefix_tokens,
    memory_override,
    no_memory_override,
    spans,
    gold_answer_ids,
    answer_positions,
    targets,
):
    if not spans or not gold_answer_ids:
        return None

    max_answer_tokens = min(len(gold_answer_ids), max(1, args.phase38_answer_tokens))
    targeted_offsets = phase38_branchpoint_offsets(answer_positions, targets, spans, max_answer_tokens)
    if not targeted_offsets:
        return None

    rollout_tokens = list(prefix_tokens)
    rollin_answer_ids = []
    gold_answer_set = {int(token_id) for token_id in gold_answer_ids}

    for offset in range(min(max_answer_tokens, max(1, args.phase38_branchpoint_steps))):
        gold_token = int(gold_answer_ids[offset])
        rollout_idx = torch.tensor([rollout_tokens], dtype=torch.long, device=device)
        logits = model(rollout_idx, memory_override=memory_override)[0, -1]
        next_token = sample_next_token_from_logits(
            logits,
            args.phase38_branchpoint_temperature,
            args.phase38_branchpoint_top_k,
        )

        no_memory_top_ids = no_memory_top_ids_for_context(
            rollout_tokens,
            no_memory_override,
            gold_answer_set,
            args.phase38_no_memory_confuser_top_k,
        )
        candidate = rollin_answer_ids + [int(next_token)]
        reason = None

        if int(next_token) == gold_token:
            rollin_answer_ids.append(gold_token)
            rollout_tokens.append(gold_token)
            continue

        if generated_partial_wrong_span(candidate, spans):
            reason = "partial_wrong"
        elif generated_contains_any_negative_span(candidate, spans):
            reason = "wrong_span"
        elif generated_has_wrong_post_key_continuation(candidate, spans):
            reason = "wrong_after_key"
        elif has_short_repetition(candidate):
            reason = "repeated"
        elif (
            int(next_token) in no_memory_top_ids
            and int(next_token) not in gold_answer_set
            and first_missing_span(rollin_answer_ids, spans) is not None
        ):
            reason = "no_memory_default"
        elif offset in targeted_offsets:
            reason = "answer_start" if offset < args.phase38_branchpoint_answer_start_tokens else "fact_branch"

        if reason is not None:
            target_ids = [
                int(token_id)
                for token_id in gold_answer_ids[offset : offset + max(1, args.phase38_branchpoint_recovery_tokens)]
                if 0 <= int(token_id) < model.config.vocab_size
            ]
            if not target_ids:
                return None
            return {
                "reason": reason,
                "offset": offset,
                "rollin_answer_ids": list(rollin_answer_ids),
                "bad_token": int(next_token),
                "gold_token": gold_token,
                "target_ids": target_ids,
                "no_memory_top_ids": set(int(token_id) for token_id in no_memory_top_ids),
            }

        # Keep non-targeted, non-factual deviations from consuming the curriculum.
        # This lets us search for the memory-critical branch while remaining on a
        # sane gold path, similar to scheduled sampling with immediate correction.
        rollin_answer_ids.append(gold_token)
        rollout_tokens.append(gold_token)

    return None


def phase38_branchpoint_objective(
    prefix_tokens,
    gold_answer_ids,
    answer_positions,
    targets,
    spans,
    memory_override,
    no_memory_override,
):
    if args.phase38_branchpoint_loss_weight <= 0 or args.phase38_branchpoint_batch_frac <= 0:
        return None
    if rng.random() > args.phase38_branchpoint_batch_frac:
        return None

    event = sampled_phase38_branchpoint_event(
        prefix_tokens,
        memory_override,
        no_memory_override,
        spans,
        gold_answer_ids,
        answer_positions,
        targets,
    )
    if event is None:
        return None

    branch_prefix = list(prefix_tokens) + [int(token_id) for token_id in event["rollin_answer_ids"]]
    if not branch_prefix:
        return None

    branch_inputs = torch.tensor([branch_prefix], dtype=torch.long, device=device)
    branch_logits = model(branch_inputs, memory_override=memory_override, return_components=False)
    row = branch_logits[0, -1]
    gold_token = int(event["gold_token"])
    if gold_token < 0 or gold_token >= row.numel():
        return None

    confuser_ids = set(int(token_id) for token_id in event.get("no_memory_top_ids", set()))
    confuser_ids.add(int(event["bad_token"]))
    confuser_ids = sorted(
        token_id
        for token_id in confuser_ids
        if 0 <= token_id < row.numel() and token_id != gold_token
    )

    log_probs = F.log_softmax(row.float(), dim=-1)
    if confuser_ids:
        confuser_tensor = torch.tensor(confuser_ids, dtype=torch.long, device=row.device)
        margin = log_probs[gold_token] - torch.logsumexp(log_probs[confuser_tensor], dim=0)
    else:
        masked = log_probs.clone()
        masked[gold_token] = -float("inf")
        margin = log_probs[gold_token] - masked.max()
    branch_loss = F.softplus(
        torch.tensor(args.phase38_branchpoint_margin, device=margin.device, dtype=margin.dtype) - margin
    )

    recovery_ce_value = 0.0
    total = branch_loss
    recovery_lps, _ = sequence_token_logprobs(branch_prefix, event["target_ids"], memory_override)
    if recovery_lps is not None and args.phase38_branchpoint_recovery_ce_weight > 0:
        recovery_ce = -recovery_lps.mean()
        recovery_ce_value = float(recovery_ce.item())
        total = total + args.phase38_branchpoint_recovery_ce_weight * recovery_ce

    return {
        "loss": total,
        "branch_loss": float(branch_loss.item()),
        "branch_margin": float(margin.item()),
        "recovery_ce": recovery_ce_value,
        "applied": 1.0,
        "answer_start": event["reason"] == "answer_start",
        "fact_branch": event["reason"] == "fact_branch",
        "no_memory_default": event["reason"] == "no_memory_default",
        "wrong_span": event["reason"] == "wrong_span",
        "partial_wrong": event["reason"] == "partial_wrong",
        "wrong_after_key": event["reason"] == "wrong_after_key",
        "repeated": event["reason"] == "repeated",
    }


def phase3_generated_prefix_recovery_loss(inputs, targets, answer_positions, fact_group_infos, effective_fact_positions, memory_override, phase_name):
    if not phase3_recovery_enabled(phase_name):
        return None
    if rng.random() > args.phase3_recovery_batch_frac:
        return None
    if answer_positions.numel() == 0:
        return None

    spans = phase3_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions)
    if not spans:
        return None

    first_answer_pos = int(answer_positions[0].item())
    prefix_tokens = inputs[0, : first_answer_pos + 1].tolist()
    generated_tokens = sampled_prefix_tokens(
        prefix_tokens,
        memory_override,
        args.phase3_recovery_steps,
        args.phase3_recovery_temperature,
        args.phase3_recovery_top_k,
        spans=spans,
        key_tail_tokens=args.phase3_recovery_key_tail_tokens,
    )
    decision = select_phase3_recovery(generated_tokens, spans)
    if decision is None:
        return None
    recovery_ids = [int(token_id) for token_id in decision["recovery_ids"][: args.phase3_recovery_max_target_tokens]]
    if not recovery_ids:
        return None

    rollout_sequence = prefix_tokens + generated_tokens + recovery_ids
    if len(rollout_sequence) < 2:
        return None
    rollout_inputs = torch.tensor([rollout_sequence[:-1]], dtype=torch.long, device=device)
    rollout_targets = torch.tensor([rollout_sequence[1:]], dtype=torch.long, device=device)
    supervised_targets = torch.full_like(rollout_targets, -1)
    target_start = len(prefix_tokens) + len(generated_tokens) - 1
    target_end = min(target_start + len(recovery_ids), supervised_targets.size(1))
    if target_start < 0 or target_start >= target_end:
        return None
    supervised_targets[:, target_start:target_end] = rollout_targets[:, target_start:target_end]

    recovery_outputs = model(
        rollout_inputs,
        supervised_targets,
        memory_override=memory_override,
        return_components=True,
    )
    ce_loss = recovery_outputs["loss"]
    anchor_losses = []
    anchor_margins = []
    for offset, token_id in enumerate(recovery_ids[: max(1, args.anchor_loss_tokens)]):
        pos = target_start + offset
        if pos >= recovery_outputs["logits"].size(1):
            continue
        anchor_loss_term, margin_value = anchor_margin_terms(
            recovery_outputs["logits"][0, pos],
            token_id,
            args.anchor_margin,
        )
        anchor_losses.append(anchor_loss_term)
        anchor_margins.append(margin_value)

    anchor_loss = rollout_inputs.new_zeros((), dtype=recovery_outputs["logits"].dtype)
    if anchor_losses:
        anchor_loss = torch.stack(anchor_losses).mean()
    combined = ce_loss + args.phase3_recovery_anchor_loss_weight * anchor_loss
    return {
        "loss": combined,
        "ce": float(ce_loss.item()),
        "anchor_margin": sum(anchor_margins) / len(anchor_margins) if anchor_margins else 0.0,
        "key_missing": decision["key_missing"],
        "partial_key": decision["partial_key"],
        "wrong_span": decision["wrong_span"],
        "repeated": decision["repeated"],
        "applied": 1.0,
    }


def phase36_generated_branch_recovery_loss(
    inputs,
    targets,
    answer_positions,
    fact_group_infos,
    effective_fact_positions,
    memory_override,
    no_memory_override,
    phase_name,
):
    if not phase36_enabled(phase_name):
        return None
    if rng.random() > args.phase36_branch_recovery_batch_frac:
        return None
    if answer_positions.numel() == 0:
        return None

    spans = phase36_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions)
    if not spans:
        return None

    answer_token_ids = [
        int(token_id)
        for token_id in targets[0, answer_positions].tolist()
        if int(token_id) != -1
    ]
    gold_answer_ids = set(answer_token_ids)
    first_answer_pos = int(answer_positions[0].item())
    prefix_tokens = inputs[0, : first_answer_pos + 1].tolist()

    event = sampled_branch_recovery_event(
        prefix_tokens,
        memory_override,
        no_memory_override,
        spans,
        gold_answer_ids,
    )
    if event is None:
        return None

    recovery_ids = [int(token_id) for token_id in event["recovery_ids"][: args.phase36_branch_recovery_max_target_tokens]]
    if not recovery_ids:
        return None

    generated_prefix = [int(token_id) for token_id in event["generated_prefix"]]
    rollout_sequence = prefix_tokens + generated_prefix + recovery_ids
    if len(rollout_sequence) < 2:
        return None

    rollout_inputs = torch.tensor([rollout_sequence[:-1]], dtype=torch.long, device=device)
    rollout_targets = torch.tensor([rollout_sequence[1:]], dtype=torch.long, device=device)
    supervised_targets = torch.full_like(rollout_targets, -1)
    target_start = len(prefix_tokens) + len(generated_prefix) - 1
    target_end = min(target_start + len(recovery_ids), supervised_targets.size(1))
    if target_start < 0 or target_start >= target_end:
        return None
    supervised_targets[:, target_start:target_end] = rollout_targets[:, target_start:target_end]

    recovery_outputs = model(
        rollout_inputs,
        supervised_targets,
        memory_override=memory_override,
        return_components=True,
    )
    ce_loss = recovery_outputs["loss"]

    branch_losses = []
    branch_margins = []
    span = event["span"]
    vocab_size = recovery_outputs["logits"].size(-1)
    for offset, token_id in enumerate(recovery_ids[: max(1, args.phase36_branch_anchor_tokens)]):
        pos = target_start + offset
        if pos >= recovery_outputs["logits"].size(1):
            continue
        if token_id < 0 or token_id >= vocab_size:
            continue

        confuser_ids = set()
        if offset == 0:
            confuser_ids.add(int(event["bad_token"]))
            confuser_ids.update(int(token_id) for token_id in event.get("no_memory_top_ids", set()))
        for negative_ids in span.get("negative_span_ids", []):
            if not negative_ids:
                continue
            if offset < len(negative_ids):
                confuser_ids.add(int(negative_ids[offset]))
            if offset == 0:
                confuser_ids.add(int(negative_ids[0]))
        gold_id = int(recovery_ids[offset])
        confuser_ids = sorted(confuser_id for confuser_id in confuser_ids if 0 <= confuser_id < vocab_size and confuser_id != gold_id)
        if not confuser_ids:
            continue

        row = recovery_outputs["logits"][0, pos]
        confuser_tensor = torch.tensor(confuser_ids, dtype=torch.long, device=row.device)
        confuser_logit = torch.logsumexp(row[confuser_tensor], dim=0)
        margin = row[int(recovery_ids[offset])] - confuser_logit
        branch_margins.append(float(margin.item()))
        branch_losses.append(
            torch.relu(
                torch.tensor(args.phase36_branch_margin, device=row.device, dtype=row.dtype) - margin
            )
        )

    branch_loss = rollout_inputs.new_zeros((), dtype=recovery_outputs["logits"].dtype)
    if branch_losses:
        branch_loss = torch.stack(branch_losses).mean()
    combined = ce_loss + args.phase36_branch_margin_loss_weight * branch_loss
    reason = event["reason"]
    return {
        "loss": combined,
        "ce": float(ce_loss.item()),
        "branch_loss": float(branch_loss.item()),
        "branch_margin": sum(branch_margins) / len(branch_margins) if branch_margins else 0.0,
        "applied": 1.0,
        "no_memory_default": reason == "no_memory_default",
        "wrong_span": reason == "wrong_span",
        "partial_wrong": reason == "partial_wrong",
        "repeated": reason in {"repeated", "repeated_after_key"},
    }


def phase37_generated_span_contrast_loss(
    inputs,
    targets,
    answer_positions,
    fact_group_infos,
    effective_fact_positions,
    memory_override,
    no_memory_override,
    phase_name,
):
    if not phase37_enabled(phase_name):
        return None
    if rng.random() > args.phase37_branch_batch_frac:
        return None
    if answer_positions.numel() == 0:
        return None

    spans = phase37_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions)
    if not spans:
        return None

    answer_token_ids = [
        int(token_id)
        for token_id in targets[0, answer_positions].tolist()
        if int(token_id) != -1
    ]
    gold_answer_ids = set(answer_token_ids)
    first_answer_pos = int(answer_positions[0].item())
    prefix_tokens = inputs[0, : first_answer_pos + 1].tolist()

    event = sampled_phase37_event(
        prefix_tokens,
        memory_override,
        no_memory_override,
        spans,
        gold_answer_ids,
    )
    if event is None:
        return None

    generated_prefix = [int(token_id) for token_id in event["generated_prefix"]]
    event_prefix = prefix_tokens + generated_prefix
    total = inputs.new_zeros((), dtype=model.transformer.wte.weight.dtype)

    branch_ce_value = 0.0
    branch_loss_value = 0.0
    branch_margin_value = 0.0
    stop_loss_value = 0.0
    stop_margin_value = 0.0
    applied = 0.0

    if event["kind"] == "branch" and args.phase37_branch_loss_weight > 0:
        target_ids = [int(token_id) for token_id in event["target_ids"][: args.phase37_max_span_tokens]]
        gold_lp, _ = sequence_mean_logprob(event_prefix, target_ids, memory_override)
        if gold_lp is not None:
            branch_ce = -gold_lp
            branch_ce_value = float(branch_ce.item())

            neg_lps = []
            seen = set()
            for candidate in event.get("confuser_span_ids", []):
                candidate_ids = [
                    int(token_id)
                    for token_id in candidate[: args.phase37_max_span_tokens]
                    if 0 <= int(token_id) < model.config.vocab_size
                ]
                key = tuple(candidate_ids)
                if not key or key in seen or key == tuple(target_ids):
                    continue
                seen.add(key)
                neg_lp, _ = sequence_mean_logprob(event_prefix, candidate_ids, memory_override)
                if neg_lp is not None:
                    neg_lps.append(neg_lp)
                if len(neg_lps) >= max(1, args.phase37_branch_max_confusers):
                    break

            branch_loss = branch_ce.new_zeros(())
            if neg_lps:
                confuser_lp = torch.logsumexp(torch.stack(neg_lps), dim=0)
                branch_margin = gold_lp - confuser_lp
                branch_margin_value = float(branch_margin.item())
                branch_loss = torch.relu(
                    torch.tensor(args.phase37_branch_margin, device=branch_margin.device, dtype=branch_margin.dtype)
                    - branch_margin
                )
                branch_loss_value = float(branch_loss.item())

            total = total + args.phase37_branch_loss_weight * (
                branch_loss + args.phase37_branch_ce_weight * branch_ce
            )
            applied = 1.0

    if event["kind"] == "stop" and args.phase37_stop_loss_weight > 0:
        target_ids = [int(token_id) for token_id in event["target_ids"][: args.phase37_post_key_tokens]]
        post_lp, _ = sequence_mean_logprob(event_prefix, target_ids, memory_override)
        if post_lp is not None and target_ids:
            stop_loss = -post_lp
            stop_loss_value = float(stop_loss.item())

            stop_inputs = torch.tensor([event_prefix], dtype=torch.long, device=device)
            stop_outputs = model(stop_inputs, memory_override=memory_override, return_components=True)
            row = stop_outputs["logits"][0, -1]
            target_id = int(target_ids[0])
            repeat_ids = set(int(token_id) for token_id in event.get("repeat_ids", set()))
            repeat_ids.add(int(event.get("bad_token", -1)))
            repeat_ids = sorted(
                token_id
                for token_id in repeat_ids
                if 0 <= token_id < row.numel() and token_id != target_id
            )
            if repeat_ids and 0 <= target_id < row.numel():
                repeat_tensor = torch.tensor(repeat_ids, dtype=torch.long, device=row.device)
                repeat_logit = torch.logsumexp(row[repeat_tensor], dim=0)
                margin = row[target_id] - repeat_logit
                stop_margin_value = float(margin.item())
                stop_loss = stop_loss + torch.relu(
                    torch.tensor(args.phase37_stop_margin, device=row.device, dtype=row.dtype) - margin
                )
                stop_loss_value = float(stop_loss.item())

            total = total + args.phase37_stop_loss_weight * stop_loss
            applied = 1.0

    if applied == 0.0:
        return None

    reason = event["reason"]
    return {
        "loss": total,
        "branch_ce": branch_ce_value,
        "branch_loss": branch_loss_value,
        "branch_margin": branch_margin_value,
        "stop_loss": stop_loss_value,
        "stop_margin": stop_margin_value,
        "applied": applied,
        "no_memory_default": reason == "no_memory_default",
        "wrong_span": reason == "wrong_span",
        "partial_wrong": reason == "partial_wrong",
        "repeated": reason in {"repeated", "repeated_after_key", "wrong_continuation"},
    }


def phase38_probe_binding_objective(outputs, targets, spans):
    if args.phase38_probe_binding_loss_weight <= 0 or not spans:
        return None
    if "memory_probe_logits" not in outputs:
        return None

    probe_log_probs = F.log_softmax(outputs["memory_probe_logits"], dim=-1)
    losses = []
    margins = []

    for span in spans[: max(1, args.phase38_probe_binding_max_spans)]:
        positions = valid_span_positions(targets, span["positions"], args.phase38_max_span_tokens)
        if positions is None:
            continue
        gold_ids = targets[0, positions].clamp_min(0)
        gold_lp = probe_log_probs[0, positions, :].gather(-1, gold_ids.view(-1, 1)).squeeze(-1).mean()

        negative_lps = []
        for negative_ids in span.get("negative_span_ids", []):
            neg_lp, _ = span_confuser_lp(probe_log_probs, positions, negative_ids)
            if neg_lp is not None:
                negative_lps.append(neg_lp)
        if not negative_lps:
            continue

        confuser_lp = torch.logsumexp(torch.stack(negative_lps), dim=0)
        margin = gold_lp - confuser_lp
        margins.append(float(margin.item()))
        losses.append(
            torch.relu(
                torch.tensor(args.phase38_probe_binding_margin, device=margin.device, dtype=margin.dtype) - margin
            )
        )

    if not losses:
        return None

    loss = torch.stack(losses).mean()
    return {
        "loss": args.phase38_probe_binding_loss_weight * loss,
        "probe_loss": float(loss.item()),
        "probe_margin": sum(margins) / len(margins),
    }


def phase38_trajectory_preference_loss(
    inputs,
    targets,
    answer_positions,
    fact_group_infos,
    effective_fact_positions,
    memory_override,
    no_memory_override,
    outputs,
    phase_name,
):
    if not phase38_enabled(phase_name):
        return None
    if answer_positions.numel() == 0:
        return None

    spans = phase38_recovery_spans(targets, answer_positions, fact_group_infos, effective_fact_positions)
    if not spans:
        return None

    total = inputs.new_zeros((), dtype=model.transformer.wte.weight.dtype)
    probe_loss_value = 0.0
    probe_margin_value = 0.0
    trajectory_loss_value = 0.0
    trajectory_margin_value = 0.0
    gold_ce_value = 0.0
    default_branch_loss_value = 0.0
    default_branch_margin_value = 0.0
    default_recovery_ce_value = 0.0
    default_branch_applied = 0.0
    branchpoint_loss_value = 0.0
    branchpoint_margin_value = 0.0
    branchpoint_recovery_ce_value = 0.0
    branchpoint_applied_value = 0.0
    branchpoint_answer_start_value = 0.0
    branchpoint_fact_branch_value = 0.0
    branchpoint_no_memory_default_value = 0.0
    branchpoint_wrong_span_value = 0.0
    branchpoint_partial_wrong_value = 0.0
    branchpoint_wrong_after_key_value = 0.0
    branchpoint_repeated_value = 0.0
    applied = 0.0
    reason = "none"

    probe_metrics = phase38_probe_binding_objective(outputs, targets, spans)
    if probe_metrics is not None:
        total = total + probe_metrics["loss"]
        probe_loss_value = probe_metrics["probe_loss"]
        probe_margin_value = probe_metrics["probe_margin"]
        applied = 1.0

    answer_token_ids = [
        int(token_id)
        for token_id in targets[0, answer_positions].tolist()
        if int(token_id) != -1
    ]
    gold_answer_ids = answer_token_ids[: max(1, args.phase38_answer_tokens)]
    first_answer_pos = int(answer_positions[0].item())
    prefix_tokens = inputs[0, : first_answer_pos + 1].tolist()

    branchpoint_metrics = phase38_branchpoint_objective(
        prefix_tokens,
        gold_answer_ids,
        answer_positions,
        targets,
        spans,
        memory_override,
        no_memory_override,
    )
    if branchpoint_metrics is not None:
        total = total + args.phase38_branchpoint_loss_weight * branchpoint_metrics["loss"]
        branchpoint_loss_value = branchpoint_metrics["branch_loss"]
        branchpoint_margin_value = branchpoint_metrics["branch_margin"]
        branchpoint_recovery_ce_value = branchpoint_metrics["recovery_ce"]
        branchpoint_applied_value = branchpoint_metrics["applied"]
        branchpoint_answer_start_value = 1.0 if branchpoint_metrics["answer_start"] else 0.0
        branchpoint_fact_branch_value = 1.0 if branchpoint_metrics["fact_branch"] else 0.0
        branchpoint_no_memory_default_value = 1.0 if branchpoint_metrics["no_memory_default"] else 0.0
        branchpoint_wrong_span_value = 1.0 if branchpoint_metrics["wrong_span"] else 0.0
        branchpoint_partial_wrong_value = 1.0 if branchpoint_metrics["partial_wrong"] else 0.0
        branchpoint_wrong_after_key_value = 1.0 if branchpoint_metrics["wrong_after_key"] else 0.0
        branchpoint_repeated_value = 1.0 if branchpoint_metrics["repeated"] else 0.0
        applied = 1.0

    if args.phase38_trajectory_loss_weight > 0 and args.phase38_trajectory_batch_frac > 0:
        if rng.random() <= args.phase38_trajectory_batch_frac:

            event = sampled_phase38_trajectory_event(
                prefix_tokens,
                memory_override,
                no_memory_override,
                spans,
                set(answer_token_ids),
            )
            if event is not None:
                generated_answer_ids = [
                    int(token_id)
                    for token_id in event["generated_answer"][: max(1, args.phase38_answer_tokens)]
                    if 0 <= int(token_id) < model.config.vocab_size
                ]
                if gold_answer_ids and generated_answer_ids and generated_answer_ids != gold_answer_ids[:len(generated_answer_ids)]:
                    gold_lp, _ = sequence_mean_logprob(prefix_tokens, gold_answer_ids, memory_override)
                    bad_lp, _ = sequence_mean_logprob(prefix_tokens, generated_answer_ids, memory_override)
                    if gold_lp is not None and bad_lp is not None:
                        trajectory_margin = gold_lp - bad_lp
                        trajectory_margin_value = float(trajectory_margin.item())
                        preference_loss = F.softplus(
                            torch.tensor(
                                args.phase38_trajectory_margin,
                                device=trajectory_margin.device,
                                dtype=trajectory_margin.dtype,
                            )
                            - trajectory_margin
                        )
                        gold_ce = -gold_lp
                        gold_ce_value = float(gold_ce.item())
                        trajectory_loss = preference_loss + args.phase38_trajectory_gold_ce_weight * gold_ce
                        hard_metrics = phase38_hard_token_preference(
                            prefix_tokens,
                            gold_answer_ids,
                            generated_answer_ids,
                            answer_positions,
                            targets,
                            spans,
                            memory_override,
                        )
                        if hard_metrics is not None:
                            hard_loss, hard_margin_value = hard_metrics
                            trajectory_loss = trajectory_loss + args.phase38_hard_token_loss_weight * hard_loss
                            # Prefer the more diagnostic margin in logs: this is
                            # the worst factual decision boundary, not the easy
                            # average over the whole sentence.
                            trajectory_margin_value = hard_margin_value
                        if event["reason"] == "no_memory_default":
                            default_metrics = phase38_default_branch_objective(
                                prefix_tokens,
                                gold_answer_ids,
                                generated_answer_ids,
                                memory_override,
                            )
                            if default_metrics is not None:
                                trajectory_loss = trajectory_loss + (
                                    args.phase38_default_branch_loss_weight * default_metrics["loss"]
                                )
                                default_branch_loss_value = default_metrics["branch_loss"]
                                default_branch_margin_value = default_metrics["branch_margin"]
                                default_recovery_ce_value = default_metrics["recovery_ce"]
                                default_branch_applied = 1.0
                        trajectory_loss_value = float(trajectory_loss.item())
                        total = total + args.phase38_trajectory_loss_weight * trajectory_loss
                        applied = 1.0
                        reason = event["reason"]

    if applied == 0.0:
        return None

    return {
        "loss": total,
        "trajectory_loss": trajectory_loss_value,
        "trajectory_margin": trajectory_margin_value,
        "gold_ce": gold_ce_value,
        "default_branch_loss": default_branch_loss_value,
        "default_branch_margin": default_branch_margin_value,
        "default_recovery_ce": default_recovery_ce_value,
        "default_branch_applied": default_branch_applied,
        "branchpoint_loss": branchpoint_loss_value,
        "branchpoint_margin": branchpoint_margin_value,
        "branchpoint_recovery_ce": branchpoint_recovery_ce_value,
        "branchpoint_applied": branchpoint_applied_value,
        "branchpoint_answer_start": branchpoint_answer_start_value,
        "branchpoint_fact_branch": branchpoint_fact_branch_value,
        "branchpoint_no_memory_default": branchpoint_no_memory_default_value,
        "branchpoint_wrong_span": branchpoint_wrong_span_value,
        "branchpoint_partial_wrong": branchpoint_partial_wrong_value,
        "branchpoint_wrong_after_key": branchpoint_wrong_after_key_value,
        "branchpoint_repeated": branchpoint_repeated_value,
        "probe_loss": probe_loss_value,
        "probe_margin": probe_margin_value,
        "applied": applied,
        "key_missing": reason == "key_missing",
        "wrong_span": reason == "wrong_span",
        "wrong_after_key": reason == "wrong_after_key",
        "partial_wrong": reason == "partial_wrong",
        "no_memory_default": reason == "no_memory_default",
        "repeated": reason == "repeated",
    }


def rollout_recovery_loss(inputs, targets, answer_positions, fact_positions, memory_override, no_memory_outputs, use_deference):
    if args.rollout_loss_weight <= 0 or args.rollout_steps <= 0 or args.rollout_continuation_tokens <= 0:
        return None
    if args.rollout_batch_frac <= 0 or rng.random() > args.rollout_batch_frac:
        return None
    if answer_positions.numel() <= 1:
        return None

    answer_tokens = targets[0, answer_positions]
    valid_answer_mask = answer_tokens.ne(-1)
    if not valid_answer_mask.all():
        answer_positions = answer_positions[valid_answer_mask]
        answer_tokens = answer_tokens[valid_answer_mask]
    if answer_tokens.numel() <= 1:
        return None

    answer_index_by_position = {int(pos.item()): idx for idx, pos in enumerate(answer_positions)}
    fact_answer_indices = [
        answer_index_by_position[int(pos.item())]
        for pos in fact_positions
        if int(pos.item()) in answer_index_by_position
    ]
    fact_answer_indices = sorted(set(fact_answer_indices))

    rollout_steps = min(args.rollout_steps, int(answer_tokens.numel()) - 1)
    recovery_start = rollout_steps
    if args.rollout_gold_start == "first_fact" and fact_answer_indices:
        recovery_start = min(fact_answer_indices)
    recovery_start = min(max(0, recovery_start), int(answer_tokens.numel()) - 1)
    continuation_len = min(args.rollout_continuation_tokens, int(answer_tokens.numel()) - recovery_start)
    if rollout_steps <= 0 or continuation_len <= 0:
        return None

    first_answer_pos = int(answer_positions[0].item())
    prefix_tokens = inputs[0, : first_answer_pos + 1].tolist()
    generated_tokens = rollout_prefix_tokens(prefix_tokens, memory_override, rollout_steps)
    gold_continuation = [int(t.item()) for t in answer_tokens[recovery_start: recovery_start + continuation_len]]
    rollout_sequence = prefix_tokens + generated_tokens + gold_continuation
    if len(rollout_sequence) < 2:
        return None

    rollout_inputs = torch.tensor([rollout_sequence[:-1]], dtype=torch.long, device=device)
    rollout_targets = torch.tensor([rollout_sequence[1:]], dtype=torch.long, device=device)
    supervised_targets = torch.full_like(rollout_targets, -1)
    target_start = len(prefix_tokens) + rollout_steps - 1
    target_end = min(target_start + continuation_len, supervised_targets.size(1))
    if target_start < 0 or target_start >= target_end:
        return None
    supervised_targets[:, target_start:target_end] = rollout_targets[:, target_start:target_end]

    rollout_outputs = model(
        rollout_inputs,
        supervised_targets,
        memory_override=memory_override,
        return_components=True,
    )
    ce_loss = rollout_outputs["loss"]

    anchor_losses = []
    anchor_margins = []
    for answer_idx in fact_answer_indices:
        if answer_idx < recovery_start or answer_idx >= recovery_start + continuation_len:
            continue
        rollout_pos = target_start + (answer_idx - recovery_start)
        if rollout_pos >= rollout_outputs["logits"].size(1):
            continue
        token_id = int(answer_tokens[answer_idx].item())
        if token_id == -1:
            continue
        anchor_loss_term, margin_value = anchor_margin_terms(
            rollout_outputs["logits"][0, rollout_pos],
            token_id,
            args.anchor_margin,
        )
        anchor_losses.append(anchor_loss_term)
        anchor_margins.append(margin_value)

    anchor_loss = rollout_inputs.new_zeros((), dtype=rollout_outputs["logits"].dtype)
    if anchor_losses:
        anchor_loss = torch.stack(anchor_losses).mean()
    anchor_margin_value = sum(anchor_margins) / len(anchor_margins) if anchor_margins else 0.0

    multiplier = 1.0
    key_missing = False
    if fact_answer_indices:
        key_tokens = [int(answer_tokens[idx].item()) for idx in fact_answer_indices]
        key_missing = key_is_missing(generated_tokens, key_tokens)
        if key_missing:
            multiplier *= max(args.rollout_missing_key_multiplier, 1.0)
    repeated = has_short_repetition(generated_tokens)
    if repeated:
        multiplier *= max(args.rollout_repeat_multiplier, 1.0)
    wrong_default = False
    if use_deference and args.rollout_wrong_default_multiplier > 1.0:
        wrong_default = generated_hits_no_memory_confuser(
            generated_tokens,
            answer_tokens,
            answer_positions,
            fact_answer_indices,
            no_memory_outputs,
        )
        if wrong_default:
            multiplier *= args.rollout_wrong_default_multiplier

    combined = multiplier * (ce_loss + args.rollout_anchor_loss_weight * anchor_loss)
    return {
        "loss": combined,
        "ce": float(ce_loss.item()),
        "anchor_margin": anchor_margin_value,
        "multiplier": multiplier,
        "key_missing": key_missing,
        "repeated": repeated,
        "wrong_default": wrong_default,
        "applied": 1.0,
    }


def active_slot_count(memory_override):
    counts = []
    for bank in memory_override:
        counts.append(float((bank["strengths"] > 1e-6).sum(dim=1).float().mean().item()))
    return sum(counts) / max(len(counts), 1)


def gate_values():
    return [torch.sigmoid(block.episodic_memory_score_bias).item() for block in model.transformer.h]


def prune_saved_checkpoints(checkpoint_dir, keep_last):
    if keep_last <= 0 or ddp_rank != 0 or not os.path.isdir(checkpoint_dir):
        return
    steps = set()
    for name in os.listdir(checkpoint_dir):
        if name.startswith(("model_", "meta_")):
            stem = name.split("_", 1)[1].split(".", 1)[0]
        elif name.startswith("optim_"):
            stem = name.split("_", 2)[1]
        else:
            continue
        if stem.isdigit():
            steps.add(int(stem))
    delete_steps = sorted(steps)[:-keep_last]
    for old_step in delete_steps:
        for name in (
            f"model_{old_step:06d}.pt",
            f"meta_{old_step:06d}.json",
            f"optim_{old_step:06d}_rank{ddp_rank:d}.pt",
        ):
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                os.remove(path)
                print0(f"Pruned old checkpoint file: {path}")


def save_memory_checkpoint(step, smooth_loss):
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)
    model.clear_memory_banks()
    prune_saved_checkpoints(checkpoint_dir, max(args.keep_last_checkpoints - 1, 0))
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        optimizer.state_dict() if args.save_optimizer_checkpoints else None,
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
    prune_saved_checkpoints(checkpoint_dir, args.keep_last_checkpoints)


smooth_loss = 0.0
smooth_guardrail_kl = 0.0
smooth_weighted_answer_ce = 0.0
smooth_fact_span_margin = 0.0
smooth_write_diversity = 0.0
smooth_memory_utility = 0.0
smooth_memory_utility_margin = 0.0
smooth_anchor_margin = 0.0
smooth_anchor_loss = 0.0
smooth_memory_probe_ce = 0.0
smooth_answer_start_ce = 0.0
smooth_key_token_ce = 0.0
smooth_key_token_rank_loss = 0.0
smooth_key_token_rank_margin = 0.0
smooth_key_token_utility_loss = 0.0
smooth_key_token_utility_margin = 0.0
smooth_habit_confuser_loss = 0.0
smooth_habit_confuser_margin = 0.0
smooth_span_contrast_loss = 0.0
smooth_span_contrast_margin = 0.0
smooth_span_utility_loss = 0.0
smooth_span_utility_margin = 0.0
smooth_span_hard_token_loss = 0.0
smooth_span_hard_token_margin = 0.0
smooth_post_key_ce = 0.0
smooth_post_key_repeat_margin = 0.0
smooth_rollout_ce = 0.0
smooth_rollout_anchor_margin = 0.0
smooth_rollout_multiplier = 0.0
smooth_rollout_applied = 0.0
smooth_rollout_key_missing = 0.0
smooth_rollout_repeated = 0.0
smooth_rollout_wrong_default = 0.0
smooth_phase3_recovery_ce = 0.0
smooth_phase3_recovery_anchor_margin = 0.0
smooth_phase3_recovery_applied = 0.0
smooth_phase3_key_missing = 0.0
smooth_phase3_partial_key = 0.0
smooth_phase3_wrong_span = 0.0
smooth_phase3_repeated = 0.0
smooth_phase35_branch_loss = 0.0
smooth_phase35_branch_margin = 0.0
smooth_phase35_stop_loss = 0.0
smooth_phase35_stop_margin = 0.0
smooth_phase35_applied = 0.0
smooth_phase36_ce = 0.0
smooth_phase36_branch_loss = 0.0
smooth_phase36_branch_margin = 0.0
smooth_phase36_applied = 0.0
smooth_phase36_no_memory_default = 0.0
smooth_phase36_wrong_span = 0.0
smooth_phase36_partial_wrong = 0.0
smooth_phase36_repeated = 0.0
smooth_phase37_branch_ce = 0.0
smooth_phase37_branch_loss = 0.0
smooth_phase37_branch_margin = 0.0
smooth_phase37_stop_loss = 0.0
smooth_phase37_stop_margin = 0.0
smooth_phase37_applied = 0.0
smooth_phase37_no_memory_default = 0.0
smooth_phase37_wrong_span = 0.0
smooth_phase37_partial_wrong = 0.0
smooth_phase37_repeated = 0.0
smooth_phase38_trajectory_loss = 0.0
smooth_phase38_trajectory_margin = 0.0
smooth_phase38_gold_ce = 0.0
smooth_phase38_default_branch_loss = 0.0
smooth_phase38_default_branch_margin = 0.0
smooth_phase38_default_recovery_ce = 0.0
smooth_phase38_default_branch_applied = 0.0
smooth_phase38_branchpoint_loss = 0.0
smooth_phase38_branchpoint_margin = 0.0
smooth_phase38_branchpoint_recovery_ce = 0.0
smooth_phase38_branchpoint_applied = 0.0
smooth_phase38_branchpoint_answer_start = 0.0
smooth_phase38_branchpoint_fact_branch = 0.0
smooth_phase38_branchpoint_no_memory_default = 0.0
smooth_phase38_branchpoint_wrong_span = 0.0
smooth_phase38_branchpoint_partial_wrong = 0.0
smooth_phase38_branchpoint_wrong_after_key = 0.0
smooth_phase38_branchpoint_repeated = 0.0
smooth_phase38_probe_loss = 0.0
smooth_phase38_probe_margin = 0.0
smooth_phase38_applied = 0.0
smooth_phase38_key_missing = 0.0
smooth_phase38_wrong_span = 0.0
smooth_phase38_wrong_after_key = 0.0
smooth_phase38_partial_wrong = 0.0
smooth_phase38_no_memory_default = 0.0
smooth_phase38_repeated = 0.0
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
        phase_batch_size = device_batch_size_for_phase(current_phase)
        print0(
            f"Switched training phase: {current_phase} | "
            f"trainable parameters: {trainable_now:,} | device_batch_size: {phase_batch_size}"
        )

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
    total_anchor_loss = 0.0
    total_memory_probe_ce = 0.0
    total_answer_start_ce = 0.0
    total_key_token_ce = 0.0
    total_key_token_rank_loss = 0.0
    total_key_token_rank_margin = 0.0
    total_key_token_utility_loss = 0.0
    total_key_token_utility_margin = 0.0
    total_habit_confuser_loss = 0.0
    total_habit_confuser_margin = 0.0
    total_span_contrast_loss = 0.0
    total_span_contrast_margin = 0.0
    total_span_utility_loss = 0.0
    total_span_utility_margin = 0.0
    total_span_hard_token_loss = 0.0
    total_span_hard_token_margin = 0.0
    total_post_key_ce = 0.0
    total_post_key_repeat_margin = 0.0
    total_rollout_ce = 0.0
    total_rollout_anchor_margin = 0.0
    total_rollout_multiplier = 0.0
    total_rollout_applied = 0.0
    total_rollout_key_missing = 0.0
    total_rollout_repeated = 0.0
    total_rollout_wrong_default = 0.0
    total_phase3_recovery_ce = 0.0
    total_phase3_recovery_anchor_margin = 0.0
    total_phase3_recovery_applied = 0.0
    total_phase3_key_missing = 0.0
    total_phase3_partial_key = 0.0
    total_phase3_wrong_span = 0.0
    total_phase3_repeated = 0.0
    total_phase35_branch_loss = 0.0
    total_phase35_branch_margin = 0.0
    total_phase35_stop_loss = 0.0
    total_phase35_stop_margin = 0.0
    total_phase35_applied = 0.0
    total_phase36_ce = 0.0
    total_phase36_branch_loss = 0.0
    total_phase36_branch_margin = 0.0
    total_phase36_applied = 0.0
    total_phase36_no_memory_default = 0.0
    total_phase36_wrong_span = 0.0
    total_phase36_partial_wrong = 0.0
    total_phase36_repeated = 0.0
    total_phase37_branch_ce = 0.0
    total_phase37_branch_loss = 0.0
    total_phase37_branch_margin = 0.0
    total_phase37_stop_loss = 0.0
    total_phase37_stop_margin = 0.0
    total_phase37_applied = 0.0
    total_phase37_no_memory_default = 0.0
    total_phase37_wrong_span = 0.0
    total_phase37_partial_wrong = 0.0
    total_phase37_repeated = 0.0
    total_phase38_trajectory_loss = 0.0
    total_phase38_trajectory_margin = 0.0
    total_phase38_gold_ce = 0.0
    total_phase38_default_branch_loss = 0.0
    total_phase38_default_branch_margin = 0.0
    total_phase38_default_recovery_ce = 0.0
    total_phase38_default_branch_applied = 0.0
    total_phase38_branchpoint_loss = 0.0
    total_phase38_branchpoint_margin = 0.0
    total_phase38_branchpoint_recovery_ce = 0.0
    total_phase38_branchpoint_applied = 0.0
    total_phase38_branchpoint_answer_start = 0.0
    total_phase38_branchpoint_fact_branch = 0.0
    total_phase38_branchpoint_no_memory_default = 0.0
    total_phase38_branchpoint_wrong_span = 0.0
    total_phase38_branchpoint_partial_wrong = 0.0
    total_phase38_branchpoint_wrong_after_key = 0.0
    total_phase38_branchpoint_repeated = 0.0
    total_phase38_probe_loss = 0.0
    total_phase38_probe_margin = 0.0
    total_phase38_applied = 0.0
    total_phase38_key_missing = 0.0
    total_phase38_wrong_span = 0.0
    total_phase38_wrong_after_key = 0.0
    total_phase38_partial_wrong = 0.0
    total_phase38_no_memory_default = 0.0
    total_phase38_repeated = 0.0
    total_active_slots = 0.0

    (
        batch_context_ids,
        batch_context_mask,
        batch_inputs,
        batch_targets,
        batch_metas,
    ) = next_episode_batch(device_batch_size_for_phase(phase_name))
    batch_size = batch_inputs.size(0)
    model.clear_memory_banks()
    batch_memory_override = build_memory_override_for_batch(batch_context_ids, batch_context_mask, batch_metas)
    batch_outputs = model(batch_inputs, batch_targets, memory_override=batch_memory_override, return_components=True)
    batch_base_loss = batch_outputs["loss"]
    extra_loss = batch_base_loss.new_zeros(())
    total_loss = float(batch_base_loss.item()) * batch_size
    use_deference = deference_enabled(phase_name)

    batch_has_effective_facts = any(
        meta["fact_positions"].numel() > 0 or meta["identity_positions"].numel() > 0
        for meta in batch_metas
    )
    batch_has_non_fact = any(positions_from_target_mask(batch_targets[b:b + 1]).numel() > 0 for b in range(batch_size))
    need_no_memory = (
        args.guardrail_kl_weight > 0 and batch_has_non_fact
    ) or (
        args.memory_utility_loss_weight > 0 and batch_has_effective_facts
    ) or (
        args.key_token_utility_loss_weight > 0 and batch_has_effective_facts
    ) or (
        use_deference and args.habit_confuser_loss_weight > 0 and batch_has_effective_facts
    ) or (
        use_deference
        and batch_has_effective_facts
        and (
            args.span_utility_loss_weight > 0
            or args.span_contrast_loss_weight > 0
            or args.span_hard_token_loss_weight > 0
        )
    ) or (
        use_deference and args.rollout_loss_weight > 0 and args.rollout_wrong_default_multiplier > 1.0
    ) or (
        phase35_enabled(phase_name)
        and args.phase35_branch_loss_weight > 0
        and args.phase35_branch_no_memory_top_k > 0
    ) or (
        phase36_enabled(phase_name)
        and args.phase36_no_memory_confuser_top_k > 0
    ) or (
        phase37_enabled(phase_name)
        and args.phase37_no_memory_confuser_top_k > 0
    ) or (
        phase38_enabled(phase_name)
        and args.phase38_no_memory_confuser_top_k > 0
    )
    batch_empty_memory_override = None
    batch_no_memory_outputs = None
    if need_no_memory:
        batch_empty_memory_override = build_empty_memory_override(batch_size)
        with torch.no_grad():
            batch_no_memory_outputs = model(batch_inputs, return_components=True, memory_override=batch_empty_memory_override)

    for batch_idx, meta in enumerate(batch_metas):
        inputs = batch_inputs[batch_idx: batch_idx + 1]
        targets = batch_targets[batch_idx: batch_idx + 1]
        template_positions = meta["template_positions"]
        identity_positions = meta["identity_positions"]
        fact_positions = meta["fact_positions"]
        rest_answer_positions = meta["rest_answer_positions"]
        anchor_target_pos = meta["anchor_target_pos"]
        fact_group_infos = meta["fact_group_infos"]
        guardrail_only = bool(meta.get("guardrail_only", False))
        memory_override = slice_memory_override(batch_memory_override, batch_idx)
        empty_memory_override = slice_memory_override(batch_empty_memory_override, batch_idx) if batch_empty_memory_override is not None else None
        outputs = slice_outputs(batch_outputs, batch_idx)
        no_memory_outputs = slice_outputs(batch_no_memory_outputs, batch_idx) if batch_no_memory_outputs is not None else None
        loss = batch_base_loss.new_zeros(())

        effective_fact_positions = fact_positions if fact_positions.numel() > 0 else identity_positions
        answer_positions = positions_from_target_mask(targets)
        non_fact_mask = torch.zeros(targets.size(1), dtype=torch.bool, device=device)
        non_fact_mask[answer_positions] = True
        if effective_fact_positions.numel() > 0:
            non_fact_mask[effective_fact_positions] = False
        non_fact_positions = non_fact_mask.nonzero(as_tuple=False).flatten()

        guardrail_kl_value = 0.0
        weighted_answer_ce_value = 0.0
        fact_span_margin_value = 0.0
        write_diversity_value = 0.0
        memory_utility_loss_value = 0.0
        memory_utility_margin_value = 0.0
        anchor_margin_value = 0.0
        anchor_loss_value = 0.0
        memory_probe_ce_value = 0.0
        answer_start_ce_value = 0.0
        key_token_ce_value = 0.0
        key_token_rank_loss_value = 0.0
        key_token_rank_margin_value = 0.0
        key_token_utility_loss_value = 0.0
        key_token_utility_margin_value = 0.0
        habit_confuser_loss_value = 0.0
        habit_confuser_margin_value = 0.0
        span_contrast_loss_value = 0.0
        span_contrast_margin_value = 0.0
        span_utility_loss_value = 0.0
        span_utility_margin_value = 0.0
        span_hard_token_loss_value = 0.0
        span_hard_token_margin_value = 0.0
        post_key_ce_value = 0.0
        post_key_repeat_margin_value = 0.0
        rollout_ce_value = 0.0
        rollout_anchor_margin_value = 0.0
        rollout_multiplier_value = 0.0
        rollout_applied_value = 0.0
        rollout_key_missing_value = 0.0
        rollout_repeated_value = 0.0
        rollout_wrong_default_value = 0.0
        phase3_recovery_ce_value = 0.0
        phase3_recovery_anchor_margin_value = 0.0
        phase3_recovery_applied_value = 0.0
        phase3_key_missing_value = 0.0
        phase3_partial_key_value = 0.0
        phase3_wrong_span_value = 0.0
        phase3_repeated_value = 0.0
        phase35_branch_loss_value = 0.0
        phase35_branch_margin_value = 0.0
        phase35_stop_loss_value = 0.0
        phase35_stop_margin_value = 0.0
        phase35_applied_value = 0.0
        phase36_ce_value = 0.0
        phase36_branch_loss_value = 0.0
        phase36_branch_margin_value = 0.0
        phase36_applied_value = 0.0
        phase36_no_memory_default_value = 0.0
        phase36_wrong_span_value = 0.0
        phase36_partial_wrong_value = 0.0
        phase36_repeated_value = 0.0
        phase37_branch_ce_value = 0.0
        phase37_branch_loss_value = 0.0
        phase37_branch_margin_value = 0.0
        phase37_stop_loss_value = 0.0
        phase37_stop_margin_value = 0.0
        phase37_applied_value = 0.0
        phase37_no_memory_default_value = 0.0
        phase37_wrong_span_value = 0.0
        phase37_partial_wrong_value = 0.0
        phase37_repeated_value = 0.0
        phase38_trajectory_loss_value = 0.0
        phase38_trajectory_margin_value = 0.0
        phase38_gold_ce_value = 0.0
        phase38_default_branch_loss_value = 0.0
        phase38_default_branch_margin_value = 0.0
        phase38_default_recovery_ce_value = 0.0
        phase38_default_branch_applied_value = 0.0
        phase38_branchpoint_loss_value = 0.0
        phase38_branchpoint_margin_value = 0.0
        phase38_branchpoint_recovery_ce_value = 0.0
        phase38_branchpoint_applied_value = 0.0
        phase38_branchpoint_answer_start_value = 0.0
        phase38_branchpoint_fact_branch_value = 0.0
        phase38_branchpoint_no_memory_default_value = 0.0
        phase38_branchpoint_wrong_span_value = 0.0
        phase38_branchpoint_partial_wrong_value = 0.0
        phase38_branchpoint_wrong_after_key_value = 0.0
        phase38_branchpoint_repeated_value = 0.0
        phase38_probe_loss_value = 0.0
        phase38_probe_margin_value = 0.0
        phase38_applied_value = 0.0
        phase38_key_missing_value = 0.0
        phase38_wrong_span_value = 0.0
        phase38_wrong_after_key_value = 0.0
        phase38_partial_wrong_value = 0.0
        phase38_no_memory_default_value = 0.0
        phase38_repeated_value = 0.0

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

        if effective_fact_positions.numel() > 0:
            anchor_positions = effective_fact_positions[: max(1, args.anchor_loss_tokens)]
        elif guardrail_only:
            anchor_positions = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            anchor_positions = answer_positions[:1]
        if anchor_positions.numel() > 0:
            anchor_losses = []
            anchor_margins = []
            for pos in anchor_positions:
                answer_token = targets[0, pos]
                if int(answer_token.item()) == -1:
                    continue
                row = outputs["logits"][0, pos]
                anchor_loss_term, margin_value = anchor_margin_terms(row, answer_token.item(), args.anchor_margin)
                anchor_losses.append(anchor_loss_term)
                anchor_margins.append(margin_value)
            if anchor_margins:
                anchor_margin_value = sum(anchor_margins) / len(anchor_margins)
            if anchor_losses and args.anchor_margin_loss_weight > 0:
                anchor_loss = torch.stack(anchor_losses).mean()
                anchor_loss_value = anchor_loss.item()
                loss = loss + args.anchor_margin_loss_weight * anchor_loss

        if args.memory_probe_loss_weight > 0 and effective_fact_positions.numel() > 0:
            probe_targets = targets[:, effective_fact_positions]
            valid_probe = probe_targets.ne(-1)
            if valid_probe.any():
                probe_logits = outputs["memory_probe_logits"][:, effective_fact_positions, :]
                probe_loss = F.cross_entropy(
                    probe_logits.reshape(-1, probe_logits.size(-1)),
                    probe_targets.reshape(-1),
                    ignore_index=-1,
                    reduction="mean",
                )
                memory_probe_ce_value = probe_loss.item()
                loss = loss + args.memory_probe_loss_weight * probe_loss

        answer_start_metrics = answer_start_objective(outputs, targets, answer_positions, use_deference)
        if answer_start_metrics is not None:
            loss = loss + args.phase2_answer_start_ce_loss_weight * answer_start_metrics["loss"]
            answer_start_ce_value = answer_start_metrics["ce"]

        key_token_metrics = key_token_objective(outputs, no_memory_outputs, targets, effective_fact_positions, use_deference)
        if key_token_metrics is not None:
            loss = loss + key_token_metrics["loss"]
            key_token_ce_value = key_token_metrics["ce"]
            key_token_rank_loss_value = key_token_metrics["rank_loss"]
            key_token_rank_margin_value = key_token_metrics["rank_margin"]
            key_token_utility_loss_value = key_token_metrics["utility_loss"]
            key_token_utility_margin_value = key_token_metrics["utility_margin"]
            habit_confuser_loss_value = key_token_metrics["habit_confuser_loss"]
            habit_confuser_margin_value = key_token_metrics["habit_confuser_margin"]

        span_metrics = span_level_objective(
            outputs,
            no_memory_outputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            use_deference,
        )
        if span_metrics is not None:
            loss = loss + span_metrics["loss"]
            span_contrast_loss_value = span_metrics["contrast_loss"]
            span_contrast_margin_value = span_metrics["contrast_margin"]
            span_utility_loss_value = span_metrics["utility_loss"]
            span_utility_margin_value = span_metrics["utility_margin"]
            span_hard_token_loss_value = span_metrics["hard_token_loss"]
            span_hard_token_margin_value = span_metrics["hard_token_margin"]

        post_key_metrics = post_key_objective(
            outputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            use_deference,
        )
        if post_key_metrics is not None:
            loss = loss + args.post_key_loss_weight * post_key_metrics["loss"]
            post_key_ce_value = post_key_metrics["ce"]
            post_key_repeat_margin_value = post_key_metrics["repeat_margin"]

        rollout_metrics = rollout_recovery_loss(
            inputs,
            targets,
            answer_positions,
            effective_fact_positions,
            memory_override,
            no_memory_outputs,
            use_deference,
        )
        if rollout_metrics is not None:
            loss = loss + args.rollout_loss_weight * rollout_metrics["loss"]
            rollout_ce_value = rollout_metrics["ce"]
            rollout_anchor_margin_value = rollout_metrics["anchor_margin"]
            rollout_multiplier_value = rollout_metrics["multiplier"]
            rollout_applied_value = rollout_metrics["applied"]
            rollout_key_missing_value = 1.0 if rollout_metrics["key_missing"] else 0.0
            rollout_repeated_value = 1.0 if rollout_metrics["repeated"] else 0.0
            rollout_wrong_default_value = 1.0 if rollout_metrics["wrong_default"] else 0.0

        phase3_recovery_metrics = phase3_generated_prefix_recovery_loss(
            inputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            memory_override,
            phase_name,
        )
        if phase3_recovery_metrics is not None:
            loss = loss + args.phase3_recovery_loss_weight * phase3_recovery_metrics["loss"]
            phase3_recovery_ce_value = phase3_recovery_metrics["ce"]
            phase3_recovery_anchor_margin_value = phase3_recovery_metrics["anchor_margin"]
            phase3_recovery_applied_value = phase3_recovery_metrics["applied"]
            phase3_key_missing_value = 1.0 if phase3_recovery_metrics["key_missing"] else 0.0
            phase3_partial_key_value = 1.0 if phase3_recovery_metrics["partial_key"] else 0.0
            phase3_wrong_span_value = 1.0 if phase3_recovery_metrics["wrong_span"] else 0.0
            phase3_repeated_value = 1.0 if phase3_recovery_metrics["repeated"] else 0.0

        phase35_metrics = phase35_branch_and_stop_objective(
            outputs,
            no_memory_outputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            phase_name,
        )
        if phase35_metrics is not None:
            loss = loss + phase35_metrics["loss"]
            phase35_branch_loss_value = phase35_metrics["branch_loss"]
            phase35_branch_margin_value = phase35_metrics["branch_margin"]
            phase35_stop_loss_value = phase35_metrics["stop_loss"]
            phase35_stop_margin_value = phase35_metrics["stop_margin"]
            phase35_applied_value = phase35_metrics["applied"]

        phase36_metrics = phase36_generated_branch_recovery_loss(
            inputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            memory_override,
            empty_memory_override,
            phase_name,
        )
        if phase36_metrics is not None:
            loss = loss + args.phase36_branch_recovery_loss_weight * phase36_metrics["loss"]
            phase36_ce_value = phase36_metrics["ce"]
            phase36_branch_loss_value = phase36_metrics["branch_loss"]
            phase36_branch_margin_value = phase36_metrics["branch_margin"]
            phase36_applied_value = phase36_metrics["applied"]
            phase36_no_memory_default_value = 1.0 if phase36_metrics["no_memory_default"] else 0.0
            phase36_wrong_span_value = 1.0 if phase36_metrics["wrong_span"] else 0.0
            phase36_partial_wrong_value = 1.0 if phase36_metrics["partial_wrong"] else 0.0
            phase36_repeated_value = 1.0 if phase36_metrics["repeated"] else 0.0

        phase37_metrics = phase37_generated_span_contrast_loss(
            inputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            memory_override,
            empty_memory_override,
            phase_name,
        )
        if phase37_metrics is not None:
            loss = loss + phase37_metrics["loss"]
            phase37_branch_ce_value = phase37_metrics["branch_ce"]
            phase37_branch_loss_value = phase37_metrics["branch_loss"]
            phase37_branch_margin_value = phase37_metrics["branch_margin"]
            phase37_stop_loss_value = phase37_metrics["stop_loss"]
            phase37_stop_margin_value = phase37_metrics["stop_margin"]
            phase37_applied_value = phase37_metrics["applied"]
            phase37_no_memory_default_value = 1.0 if phase37_metrics["no_memory_default"] else 0.0
            phase37_wrong_span_value = 1.0 if phase37_metrics["wrong_span"] else 0.0
            phase37_partial_wrong_value = 1.0 if phase37_metrics["partial_wrong"] else 0.0
            phase37_repeated_value = 1.0 if phase37_metrics["repeated"] else 0.0

        phase38_metrics = phase38_trajectory_preference_loss(
            inputs,
            targets,
            answer_positions,
            fact_group_infos,
            effective_fact_positions,
            memory_override,
            empty_memory_override,
            outputs,
            phase_name,
        )
        if phase38_metrics is not None:
            loss = loss + phase38_metrics["loss"]
            phase38_trajectory_loss_value = phase38_metrics["trajectory_loss"]
            phase38_trajectory_margin_value = phase38_metrics["trajectory_margin"]
            phase38_gold_ce_value = phase38_metrics["gold_ce"]
            phase38_default_branch_loss_value = phase38_metrics["default_branch_loss"]
            phase38_default_branch_margin_value = phase38_metrics["default_branch_margin"]
            phase38_default_recovery_ce_value = phase38_metrics["default_recovery_ce"]
            phase38_default_branch_applied_value = phase38_metrics["default_branch_applied"]
            phase38_branchpoint_loss_value = phase38_metrics["branchpoint_loss"]
            phase38_branchpoint_margin_value = phase38_metrics["branchpoint_margin"]
            phase38_branchpoint_recovery_ce_value = phase38_metrics["branchpoint_recovery_ce"]
            phase38_branchpoint_applied_value = phase38_metrics["branchpoint_applied"]
            phase38_branchpoint_answer_start_value = phase38_metrics["branchpoint_answer_start"]
            phase38_branchpoint_fact_branch_value = phase38_metrics["branchpoint_fact_branch"]
            phase38_branchpoint_no_memory_default_value = phase38_metrics["branchpoint_no_memory_default"]
            phase38_branchpoint_wrong_span_value = phase38_metrics["branchpoint_wrong_span"]
            phase38_branchpoint_partial_wrong_value = phase38_metrics["branchpoint_partial_wrong"]
            phase38_branchpoint_wrong_after_key_value = phase38_metrics["branchpoint_wrong_after_key"]
            phase38_branchpoint_repeated_value = phase38_metrics["branchpoint_repeated"]
            phase38_probe_loss_value = phase38_metrics["probe_loss"]
            phase38_probe_margin_value = phase38_metrics["probe_margin"]
            phase38_applied_value = phase38_metrics["applied"]
            phase38_key_missing_value = 1.0 if phase38_metrics["key_missing"] else 0.0
            phase38_wrong_span_value = 1.0 if phase38_metrics["wrong_span"] else 0.0
            phase38_wrong_after_key_value = 1.0 if phase38_metrics["wrong_after_key"] else 0.0
            phase38_partial_wrong_value = 1.0 if phase38_metrics["partial_wrong"] else 0.0
            phase38_no_memory_default_value = 1.0 if phase38_metrics["no_memory_default"] else 0.0
            phase38_repeated_value = 1.0 if phase38_metrics["repeated"] else 0.0

        extra_loss = extra_loss + loss
        loss_value = loss.item()
        total_loss += loss_value
        total_guardrail_kl += guardrail_kl_value
        total_weighted_answer_ce += weighted_answer_ce_value
        total_fact_span_margin += fact_span_margin_value
        total_write_diversity += write_diversity_value
        total_memory_utility += memory_utility_loss_value
        total_memory_utility_margin += memory_utility_margin_value
        total_anchor_margin += anchor_margin_value
        total_anchor_loss += anchor_loss_value
        total_memory_probe_ce += memory_probe_ce_value
        total_answer_start_ce += answer_start_ce_value
        total_key_token_ce += key_token_ce_value
        total_key_token_rank_loss += key_token_rank_loss_value
        total_key_token_rank_margin += key_token_rank_margin_value
        total_key_token_utility_loss += key_token_utility_loss_value
        total_key_token_utility_margin += key_token_utility_margin_value
        total_habit_confuser_loss += habit_confuser_loss_value
        total_habit_confuser_margin += habit_confuser_margin_value
        total_span_contrast_loss += span_contrast_loss_value
        total_span_contrast_margin += span_contrast_margin_value
        total_span_utility_loss += span_utility_loss_value
        total_span_utility_margin += span_utility_margin_value
        total_span_hard_token_loss += span_hard_token_loss_value
        total_span_hard_token_margin += span_hard_token_margin_value
        total_post_key_ce += post_key_ce_value
        total_post_key_repeat_margin += post_key_repeat_margin_value
        total_rollout_ce += rollout_ce_value
        total_rollout_anchor_margin += rollout_anchor_margin_value
        total_rollout_multiplier += rollout_multiplier_value
        total_rollout_applied += rollout_applied_value
        total_rollout_key_missing += rollout_key_missing_value
        total_rollout_repeated += rollout_repeated_value
        total_rollout_wrong_default += rollout_wrong_default_value
        total_phase3_recovery_ce += phase3_recovery_ce_value
        total_phase3_recovery_anchor_margin += phase3_recovery_anchor_margin_value
        total_phase3_recovery_applied += phase3_recovery_applied_value
        total_phase3_key_missing += phase3_key_missing_value
        total_phase3_partial_key += phase3_partial_key_value
        total_phase3_wrong_span += phase3_wrong_span_value
        total_phase3_repeated += phase3_repeated_value
        total_phase35_branch_loss += phase35_branch_loss_value
        total_phase35_branch_margin += phase35_branch_margin_value
        total_phase35_stop_loss += phase35_stop_loss_value
        total_phase35_stop_margin += phase35_stop_margin_value
        total_phase35_applied += phase35_applied_value
        total_phase36_ce += phase36_ce_value
        total_phase36_branch_loss += phase36_branch_loss_value
        total_phase36_branch_margin += phase36_branch_margin_value
        total_phase36_applied += phase36_applied_value
        total_phase36_no_memory_default += phase36_no_memory_default_value
        total_phase36_wrong_span += phase36_wrong_span_value
        total_phase36_partial_wrong += phase36_partial_wrong_value
        total_phase36_repeated += phase36_repeated_value
        total_phase37_branch_ce += phase37_branch_ce_value
        total_phase37_branch_loss += phase37_branch_loss_value
        total_phase37_branch_margin += phase37_branch_margin_value
        total_phase37_stop_loss += phase37_stop_loss_value
        total_phase37_stop_margin += phase37_stop_margin_value
        total_phase37_applied += phase37_applied_value
        total_phase37_no_memory_default += phase37_no_memory_default_value
        total_phase37_wrong_span += phase37_wrong_span_value
        total_phase37_partial_wrong += phase37_partial_wrong_value
        total_phase37_repeated += phase37_repeated_value
        total_phase38_trajectory_loss += phase38_trajectory_loss_value
        total_phase38_trajectory_margin += phase38_trajectory_margin_value
        total_phase38_gold_ce += phase38_gold_ce_value
        total_phase38_default_branch_loss += phase38_default_branch_loss_value
        total_phase38_default_branch_margin += phase38_default_branch_margin_value
        total_phase38_default_recovery_ce += phase38_default_recovery_ce_value
        total_phase38_default_branch_applied += phase38_default_branch_applied_value
        total_phase38_branchpoint_loss += phase38_branchpoint_loss_value
        total_phase38_branchpoint_margin += phase38_branchpoint_margin_value
        total_phase38_branchpoint_recovery_ce += phase38_branchpoint_recovery_ce_value
        total_phase38_branchpoint_applied += phase38_branchpoint_applied_value
        total_phase38_branchpoint_answer_start += phase38_branchpoint_answer_start_value
        total_phase38_branchpoint_fact_branch += phase38_branchpoint_fact_branch_value
        total_phase38_branchpoint_no_memory_default += phase38_branchpoint_no_memory_default_value
        total_phase38_branchpoint_wrong_span += phase38_branchpoint_wrong_span_value
        total_phase38_branchpoint_partial_wrong += phase38_branchpoint_partial_wrong_value
        total_phase38_branchpoint_wrong_after_key += phase38_branchpoint_wrong_after_key_value
        total_phase38_branchpoint_repeated += phase38_branchpoint_repeated_value
        total_phase38_probe_loss += phase38_probe_loss_value
        total_phase38_probe_margin += phase38_probe_margin_value
        total_phase38_applied += phase38_applied_value
        total_phase38_key_missing += phase38_key_missing_value
        total_phase38_wrong_span += phase38_wrong_span_value
        total_phase38_wrong_after_key += phase38_wrong_after_key_value
        total_phase38_partial_wrong += phase38_partial_wrong_value
        total_phase38_no_memory_default += phase38_no_memory_default_value
        total_phase38_repeated += phase38_repeated_value
        total_active_slots += active_slot_count(memory_override)

    combined_loss = batch_base_loss + extra_loss / max(batch_size, 1)
    if not torch.isfinite(combined_loss):
        print0(f"Skipping step {step}: non-finite combined loss")
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model.clear_memory_banks()
        continue
    combined_loss.backward()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    nonfinite_grad_tensors = 0
    for param in trainable_params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            nonfinite_grad_tensors += 1
            param.grad.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
    if nonfinite_grad_tensors > 0 and step % 10 == 0:
        print0(f"Sanitized non-finite gradients at step {step}: {nonfinite_grad_tensors} tensors")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=False)
    if not torch.isfinite(grad_norm):
        print0(f"Skipping step {step}: non-finite grad norm ({grad_norm})")
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model.clear_memory_banks()
        continue
    optimizer.step()
    model.zero_grad(set_to_none=True)
    model.clear_memory_banks()
    if device.type == "mps":
        gc.collect()
        torch.mps.empty_cache()
    elif args.gc_every > 0 and step > 0 and step % args.gc_every == 0:
        gc.collect()

    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * total_loss
    smooth_guardrail_kl = ema_beta * smooth_guardrail_kl + (1 - ema_beta) * total_guardrail_kl
    smooth_weighted_answer_ce = ema_beta * smooth_weighted_answer_ce + (1 - ema_beta) * total_weighted_answer_ce
    smooth_fact_span_margin = ema_beta * smooth_fact_span_margin + (1 - ema_beta) * total_fact_span_margin
    smooth_write_diversity = ema_beta * smooth_write_diversity + (1 - ema_beta) * total_write_diversity
    smooth_memory_utility = ema_beta * smooth_memory_utility + (1 - ema_beta) * total_memory_utility
    smooth_memory_utility_margin = ema_beta * smooth_memory_utility_margin + (1 - ema_beta) * total_memory_utility_margin
    smooth_anchor_margin = ema_beta * smooth_anchor_margin + (1 - ema_beta) * total_anchor_margin
    smooth_anchor_loss = ema_beta * smooth_anchor_loss + (1 - ema_beta) * total_anchor_loss
    smooth_memory_probe_ce = ema_beta * smooth_memory_probe_ce + (1 - ema_beta) * total_memory_probe_ce
    smooth_answer_start_ce = ema_beta * smooth_answer_start_ce + (1 - ema_beta) * total_answer_start_ce
    smooth_key_token_ce = ema_beta * smooth_key_token_ce + (1 - ema_beta) * total_key_token_ce
    smooth_key_token_rank_loss = ema_beta * smooth_key_token_rank_loss + (1 - ema_beta) * total_key_token_rank_loss
    smooth_key_token_rank_margin = ema_beta * smooth_key_token_rank_margin + (1 - ema_beta) * total_key_token_rank_margin
    smooth_key_token_utility_loss = ema_beta * smooth_key_token_utility_loss + (1 - ema_beta) * total_key_token_utility_loss
    smooth_key_token_utility_margin = ema_beta * smooth_key_token_utility_margin + (1 - ema_beta) * total_key_token_utility_margin
    smooth_habit_confuser_loss = ema_beta * smooth_habit_confuser_loss + (1 - ema_beta) * total_habit_confuser_loss
    smooth_habit_confuser_margin = ema_beta * smooth_habit_confuser_margin + (1 - ema_beta) * total_habit_confuser_margin
    smooth_span_contrast_loss = ema_beta * smooth_span_contrast_loss + (1 - ema_beta) * total_span_contrast_loss
    smooth_span_contrast_margin = ema_beta * smooth_span_contrast_margin + (1 - ema_beta) * total_span_contrast_margin
    smooth_span_utility_loss = ema_beta * smooth_span_utility_loss + (1 - ema_beta) * total_span_utility_loss
    smooth_span_utility_margin = ema_beta * smooth_span_utility_margin + (1 - ema_beta) * total_span_utility_margin
    smooth_span_hard_token_loss = ema_beta * smooth_span_hard_token_loss + (1 - ema_beta) * total_span_hard_token_loss
    smooth_span_hard_token_margin = ema_beta * smooth_span_hard_token_margin + (1 - ema_beta) * total_span_hard_token_margin
    smooth_post_key_ce = ema_beta * smooth_post_key_ce + (1 - ema_beta) * total_post_key_ce
    smooth_post_key_repeat_margin = ema_beta * smooth_post_key_repeat_margin + (1 - ema_beta) * total_post_key_repeat_margin
    smooth_rollout_ce = ema_beta * smooth_rollout_ce + (1 - ema_beta) * total_rollout_ce
    smooth_rollout_anchor_margin = ema_beta * smooth_rollout_anchor_margin + (1 - ema_beta) * total_rollout_anchor_margin
    smooth_rollout_multiplier = ema_beta * smooth_rollout_multiplier + (1 - ema_beta) * total_rollout_multiplier
    smooth_rollout_applied = ema_beta * smooth_rollout_applied + (1 - ema_beta) * total_rollout_applied
    smooth_rollout_key_missing = ema_beta * smooth_rollout_key_missing + (1 - ema_beta) * total_rollout_key_missing
    smooth_rollout_repeated = ema_beta * smooth_rollout_repeated + (1 - ema_beta) * total_rollout_repeated
    smooth_rollout_wrong_default = ema_beta * smooth_rollout_wrong_default + (1 - ema_beta) * total_rollout_wrong_default
    smooth_phase3_recovery_ce = ema_beta * smooth_phase3_recovery_ce + (1 - ema_beta) * total_phase3_recovery_ce
    smooth_phase3_recovery_anchor_margin = ema_beta * smooth_phase3_recovery_anchor_margin + (1 - ema_beta) * total_phase3_recovery_anchor_margin
    smooth_phase3_recovery_applied = ema_beta * smooth_phase3_recovery_applied + (1 - ema_beta) * total_phase3_recovery_applied
    smooth_phase3_key_missing = ema_beta * smooth_phase3_key_missing + (1 - ema_beta) * total_phase3_key_missing
    smooth_phase3_partial_key = ema_beta * smooth_phase3_partial_key + (1 - ema_beta) * total_phase3_partial_key
    smooth_phase3_wrong_span = ema_beta * smooth_phase3_wrong_span + (1 - ema_beta) * total_phase3_wrong_span
    smooth_phase3_repeated = ema_beta * smooth_phase3_repeated + (1 - ema_beta) * total_phase3_repeated
    smooth_phase35_branch_loss = ema_beta * smooth_phase35_branch_loss + (1 - ema_beta) * total_phase35_branch_loss
    smooth_phase35_branch_margin = ema_beta * smooth_phase35_branch_margin + (1 - ema_beta) * total_phase35_branch_margin
    smooth_phase35_stop_loss = ema_beta * smooth_phase35_stop_loss + (1 - ema_beta) * total_phase35_stop_loss
    smooth_phase35_stop_margin = ema_beta * smooth_phase35_stop_margin + (1 - ema_beta) * total_phase35_stop_margin
    smooth_phase35_applied = ema_beta * smooth_phase35_applied + (1 - ema_beta) * total_phase35_applied
    smooth_phase36_ce = ema_beta * smooth_phase36_ce + (1 - ema_beta) * total_phase36_ce
    smooth_phase36_branch_loss = ema_beta * smooth_phase36_branch_loss + (1 - ema_beta) * total_phase36_branch_loss
    smooth_phase36_branch_margin = ema_beta * smooth_phase36_branch_margin + (1 - ema_beta) * total_phase36_branch_margin
    smooth_phase36_applied = ema_beta * smooth_phase36_applied + (1 - ema_beta) * total_phase36_applied
    smooth_phase36_no_memory_default = ema_beta * smooth_phase36_no_memory_default + (1 - ema_beta) * total_phase36_no_memory_default
    smooth_phase36_wrong_span = ema_beta * smooth_phase36_wrong_span + (1 - ema_beta) * total_phase36_wrong_span
    smooth_phase36_partial_wrong = ema_beta * smooth_phase36_partial_wrong + (1 - ema_beta) * total_phase36_partial_wrong
    smooth_phase36_repeated = ema_beta * smooth_phase36_repeated + (1 - ema_beta) * total_phase36_repeated
    smooth_phase37_branch_ce = ema_beta * smooth_phase37_branch_ce + (1 - ema_beta) * total_phase37_branch_ce
    smooth_phase37_branch_loss = ema_beta * smooth_phase37_branch_loss + (1 - ema_beta) * total_phase37_branch_loss
    smooth_phase37_branch_margin = ema_beta * smooth_phase37_branch_margin + (1 - ema_beta) * total_phase37_branch_margin
    smooth_phase37_stop_loss = ema_beta * smooth_phase37_stop_loss + (1 - ema_beta) * total_phase37_stop_loss
    smooth_phase37_stop_margin = ema_beta * smooth_phase37_stop_margin + (1 - ema_beta) * total_phase37_stop_margin
    smooth_phase37_applied = ema_beta * smooth_phase37_applied + (1 - ema_beta) * total_phase37_applied
    smooth_phase37_no_memory_default = ema_beta * smooth_phase37_no_memory_default + (1 - ema_beta) * total_phase37_no_memory_default
    smooth_phase37_wrong_span = ema_beta * smooth_phase37_wrong_span + (1 - ema_beta) * total_phase37_wrong_span
    smooth_phase37_partial_wrong = ema_beta * smooth_phase37_partial_wrong + (1 - ema_beta) * total_phase37_partial_wrong
    smooth_phase37_repeated = ema_beta * smooth_phase37_repeated + (1 - ema_beta) * total_phase37_repeated
    smooth_phase38_trajectory_loss = ema_beta * smooth_phase38_trajectory_loss + (1 - ema_beta) * total_phase38_trajectory_loss
    smooth_phase38_trajectory_margin = ema_beta * smooth_phase38_trajectory_margin + (1 - ema_beta) * total_phase38_trajectory_margin
    smooth_phase38_gold_ce = ema_beta * smooth_phase38_gold_ce + (1 - ema_beta) * total_phase38_gold_ce
    smooth_phase38_default_branch_loss = ema_beta * smooth_phase38_default_branch_loss + (1 - ema_beta) * total_phase38_default_branch_loss
    smooth_phase38_default_branch_margin = ema_beta * smooth_phase38_default_branch_margin + (1 - ema_beta) * total_phase38_default_branch_margin
    smooth_phase38_default_recovery_ce = ema_beta * smooth_phase38_default_recovery_ce + (1 - ema_beta) * total_phase38_default_recovery_ce
    smooth_phase38_default_branch_applied = ema_beta * smooth_phase38_default_branch_applied + (1 - ema_beta) * total_phase38_default_branch_applied
    smooth_phase38_branchpoint_loss = ema_beta * smooth_phase38_branchpoint_loss + (1 - ema_beta) * total_phase38_branchpoint_loss
    smooth_phase38_branchpoint_margin = ema_beta * smooth_phase38_branchpoint_margin + (1 - ema_beta) * total_phase38_branchpoint_margin
    smooth_phase38_branchpoint_recovery_ce = ema_beta * smooth_phase38_branchpoint_recovery_ce + (1 - ema_beta) * total_phase38_branchpoint_recovery_ce
    smooth_phase38_branchpoint_applied = ema_beta * smooth_phase38_branchpoint_applied + (1 - ema_beta) * total_phase38_branchpoint_applied
    smooth_phase38_branchpoint_answer_start = ema_beta * smooth_phase38_branchpoint_answer_start + (1 - ema_beta) * total_phase38_branchpoint_answer_start
    smooth_phase38_branchpoint_fact_branch = ema_beta * smooth_phase38_branchpoint_fact_branch + (1 - ema_beta) * total_phase38_branchpoint_fact_branch
    smooth_phase38_branchpoint_no_memory_default = ema_beta * smooth_phase38_branchpoint_no_memory_default + (1 - ema_beta) * total_phase38_branchpoint_no_memory_default
    smooth_phase38_branchpoint_wrong_span = ema_beta * smooth_phase38_branchpoint_wrong_span + (1 - ema_beta) * total_phase38_branchpoint_wrong_span
    smooth_phase38_branchpoint_partial_wrong = ema_beta * smooth_phase38_branchpoint_partial_wrong + (1 - ema_beta) * total_phase38_branchpoint_partial_wrong
    smooth_phase38_branchpoint_wrong_after_key = ema_beta * smooth_phase38_branchpoint_wrong_after_key + (1 - ema_beta) * total_phase38_branchpoint_wrong_after_key
    smooth_phase38_branchpoint_repeated = ema_beta * smooth_phase38_branchpoint_repeated + (1 - ema_beta) * total_phase38_branchpoint_repeated
    smooth_phase38_probe_loss = ema_beta * smooth_phase38_probe_loss + (1 - ema_beta) * total_phase38_probe_loss
    smooth_phase38_probe_margin = ema_beta * smooth_phase38_probe_margin + (1 - ema_beta) * total_phase38_probe_margin
    smooth_phase38_applied = ema_beta * smooth_phase38_applied + (1 - ema_beta) * total_phase38_applied
    smooth_phase38_key_missing = ema_beta * smooth_phase38_key_missing + (1 - ema_beta) * total_phase38_key_missing
    smooth_phase38_wrong_span = ema_beta * smooth_phase38_wrong_span + (1 - ema_beta) * total_phase38_wrong_span
    smooth_phase38_wrong_after_key = ema_beta * smooth_phase38_wrong_after_key + (1 - ema_beta) * total_phase38_wrong_after_key
    smooth_phase38_partial_wrong = ema_beta * smooth_phase38_partial_wrong + (1 - ema_beta) * total_phase38_partial_wrong
    smooth_phase38_no_memory_default = ema_beta * smooth_phase38_no_memory_default + (1 - ema_beta) * total_phase38_no_memory_default
    smooth_phase38_repeated = ema_beta * smooth_phase38_repeated + (1 - ema_beta) * total_phase38_repeated
    mean_active_slots = total_active_slots / max(batch_size, 1)
    smooth_active_slots = ema_beta * smooth_active_slots + (1 - ema_beta) * mean_active_slots

    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    debiased_guardrail_kl = smooth_guardrail_kl / (1 - ema_beta ** (step + 1))
    debiased_weighted_answer_ce = smooth_weighted_answer_ce / (1 - ema_beta ** (step + 1))
    debiased_fact_span_margin = smooth_fact_span_margin / (1 - ema_beta ** (step + 1))
    debiased_write_diversity = smooth_write_diversity / (1 - ema_beta ** (step + 1))
    debiased_memory_utility = smooth_memory_utility / (1 - ema_beta ** (step + 1))
    debiased_memory_utility_margin = smooth_memory_utility_margin / (1 - ema_beta ** (step + 1))
    debiased_anchor_margin = smooth_anchor_margin / (1 - ema_beta ** (step + 1))
    debiased_anchor_loss = smooth_anchor_loss / (1 - ema_beta ** (step + 1))
    debiased_memory_probe_ce = smooth_memory_probe_ce / (1 - ema_beta ** (step + 1))
    debiased_answer_start_ce = smooth_answer_start_ce / (1 - ema_beta ** (step + 1))
    debiased_key_token_ce = smooth_key_token_ce / (1 - ema_beta ** (step + 1))
    debiased_key_token_rank_loss = smooth_key_token_rank_loss / (1 - ema_beta ** (step + 1))
    debiased_key_token_rank_margin = smooth_key_token_rank_margin / (1 - ema_beta ** (step + 1))
    debiased_key_token_utility_loss = smooth_key_token_utility_loss / (1 - ema_beta ** (step + 1))
    debiased_key_token_utility_margin = smooth_key_token_utility_margin / (1 - ema_beta ** (step + 1))
    debiased_habit_confuser_loss = smooth_habit_confuser_loss / (1 - ema_beta ** (step + 1))
    debiased_habit_confuser_margin = smooth_habit_confuser_margin / (1 - ema_beta ** (step + 1))
    debiased_span_contrast_loss = smooth_span_contrast_loss / (1 - ema_beta ** (step + 1))
    debiased_span_contrast_margin = smooth_span_contrast_margin / (1 - ema_beta ** (step + 1))
    debiased_span_utility_loss = smooth_span_utility_loss / (1 - ema_beta ** (step + 1))
    debiased_span_utility_margin = smooth_span_utility_margin / (1 - ema_beta ** (step + 1))
    debiased_span_hard_token_loss = smooth_span_hard_token_loss / (1 - ema_beta ** (step + 1))
    debiased_span_hard_token_margin = smooth_span_hard_token_margin / (1 - ema_beta ** (step + 1))
    debiased_post_key_ce = smooth_post_key_ce / (1 - ema_beta ** (step + 1))
    debiased_post_key_repeat_margin = smooth_post_key_repeat_margin / (1 - ema_beta ** (step + 1))
    debiased_rollout_ce = smooth_rollout_ce / (1 - ema_beta ** (step + 1))
    debiased_rollout_anchor_margin = smooth_rollout_anchor_margin / (1 - ema_beta ** (step + 1))
    debiased_rollout_multiplier = smooth_rollout_multiplier / (1 - ema_beta ** (step + 1))
    debiased_rollout_applied = smooth_rollout_applied / (1 - ema_beta ** (step + 1))
    debiased_rollout_key_missing = smooth_rollout_key_missing / (1 - ema_beta ** (step + 1))
    debiased_rollout_repeated = smooth_rollout_repeated / (1 - ema_beta ** (step + 1))
    debiased_rollout_wrong_default = smooth_rollout_wrong_default / (1 - ema_beta ** (step + 1))
    debiased_phase3_recovery_ce = smooth_phase3_recovery_ce / (1 - ema_beta ** (step + 1))
    debiased_phase3_recovery_anchor_margin = smooth_phase3_recovery_anchor_margin / (1 - ema_beta ** (step + 1))
    debiased_phase3_recovery_applied = smooth_phase3_recovery_applied / (1 - ema_beta ** (step + 1))
    debiased_phase3_key_missing = smooth_phase3_key_missing / (1 - ema_beta ** (step + 1))
    debiased_phase3_partial_key = smooth_phase3_partial_key / (1 - ema_beta ** (step + 1))
    debiased_phase3_wrong_span = smooth_phase3_wrong_span / (1 - ema_beta ** (step + 1))
    debiased_phase3_repeated = smooth_phase3_repeated / (1 - ema_beta ** (step + 1))
    debiased_phase35_branch_loss = smooth_phase35_branch_loss / (1 - ema_beta ** (step + 1))
    debiased_phase35_branch_margin = smooth_phase35_branch_margin / (1 - ema_beta ** (step + 1))
    debiased_phase35_stop_loss = smooth_phase35_stop_loss / (1 - ema_beta ** (step + 1))
    debiased_phase35_stop_margin = smooth_phase35_stop_margin / (1 - ema_beta ** (step + 1))
    debiased_phase35_applied = smooth_phase35_applied / (1 - ema_beta ** (step + 1))
    debiased_phase36_ce = smooth_phase36_ce / (1 - ema_beta ** (step + 1))
    debiased_phase36_branch_loss = smooth_phase36_branch_loss / (1 - ema_beta ** (step + 1))
    debiased_phase36_branch_margin = smooth_phase36_branch_margin / (1 - ema_beta ** (step + 1))
    debiased_phase36_applied = smooth_phase36_applied / (1 - ema_beta ** (step + 1))
    debiased_phase36_no_memory_default = smooth_phase36_no_memory_default / (1 - ema_beta ** (step + 1))
    debiased_phase36_wrong_span = smooth_phase36_wrong_span / (1 - ema_beta ** (step + 1))
    debiased_phase36_partial_wrong = smooth_phase36_partial_wrong / (1 - ema_beta ** (step + 1))
    debiased_phase36_repeated = smooth_phase36_repeated / (1 - ema_beta ** (step + 1))
    debiased_phase37_branch_ce = smooth_phase37_branch_ce / (1 - ema_beta ** (step + 1))
    debiased_phase37_branch_loss = smooth_phase37_branch_loss / (1 - ema_beta ** (step + 1))
    debiased_phase37_branch_margin = smooth_phase37_branch_margin / (1 - ema_beta ** (step + 1))
    debiased_phase37_stop_loss = smooth_phase37_stop_loss / (1 - ema_beta ** (step + 1))
    debiased_phase37_stop_margin = smooth_phase37_stop_margin / (1 - ema_beta ** (step + 1))
    debiased_phase37_applied = smooth_phase37_applied / (1 - ema_beta ** (step + 1))
    debiased_phase37_no_memory_default = smooth_phase37_no_memory_default / (1 - ema_beta ** (step + 1))
    debiased_phase37_wrong_span = smooth_phase37_wrong_span / (1 - ema_beta ** (step + 1))
    debiased_phase37_partial_wrong = smooth_phase37_partial_wrong / (1 - ema_beta ** (step + 1))
    debiased_phase37_repeated = smooth_phase37_repeated / (1 - ema_beta ** (step + 1))
    debiased_phase38_trajectory_loss = smooth_phase38_trajectory_loss / (1 - ema_beta ** (step + 1))
    debiased_phase38_trajectory_margin = smooth_phase38_trajectory_margin / (1 - ema_beta ** (step + 1))
    debiased_phase38_gold_ce = smooth_phase38_gold_ce / (1 - ema_beta ** (step + 1))
    debiased_phase38_default_branch_loss = smooth_phase38_default_branch_loss / (1 - ema_beta ** (step + 1))
    debiased_phase38_default_branch_margin = smooth_phase38_default_branch_margin / (1 - ema_beta ** (step + 1))
    debiased_phase38_default_recovery_ce = smooth_phase38_default_recovery_ce / (1 - ema_beta ** (step + 1))
    debiased_phase38_default_branch_applied = smooth_phase38_default_branch_applied / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_loss = smooth_phase38_branchpoint_loss / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_margin = smooth_phase38_branchpoint_margin / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_recovery_ce = smooth_phase38_branchpoint_recovery_ce / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_applied = smooth_phase38_branchpoint_applied / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_answer_start = smooth_phase38_branchpoint_answer_start / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_fact_branch = smooth_phase38_branchpoint_fact_branch / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_no_memory_default = smooth_phase38_branchpoint_no_memory_default / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_wrong_span = smooth_phase38_branchpoint_wrong_span / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_partial_wrong = smooth_phase38_branchpoint_partial_wrong / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_wrong_after_key = smooth_phase38_branchpoint_wrong_after_key / (1 - ema_beta ** (step + 1))
    debiased_phase38_branchpoint_repeated = smooth_phase38_branchpoint_repeated / (1 - ema_beta ** (step + 1))
    debiased_phase38_probe_loss = smooth_phase38_probe_loss / (1 - ema_beta ** (step + 1))
    debiased_phase38_probe_margin = smooth_phase38_probe_margin / (1 - ema_beta ** (step + 1))
    debiased_phase38_applied = smooth_phase38_applied / (1 - ema_beta ** (step + 1))
    debiased_phase38_key_missing = smooth_phase38_key_missing / (1 - ema_beta ** (step + 1))
    debiased_phase38_wrong_span = smooth_phase38_wrong_span / (1 - ema_beta ** (step + 1))
    debiased_phase38_wrong_after_key = smooth_phase38_wrong_after_key / (1 - ema_beta ** (step + 1))
    debiased_phase38_partial_wrong = smooth_phase38_partial_wrong / (1 - ema_beta ** (step + 1))
    debiased_phase38_no_memory_default = smooth_phase38_no_memory_default / (1 - ema_beta ** (step + 1))
    debiased_phase38_repeated = smooth_phase38_repeated / (1 - ema_beta ** (step + 1))
    debiased_active_slots = smooth_active_slots / (1 - ema_beta ** (step + 1))
    gate_avg = sum(gate_values()) / len(model.transformer.h)
    elapsed = time.time() - t_start

    if step % 10 == 0:
        print0(
            f"\nstep {step:05d}/{args.num_iterations} | "
            f"loss: {debiased_loss:.4f} | "
            f"guardrail_kl: {debiased_guardrail_kl:.4f} | "
            f"weighted_answer_ce: {debiased_weighted_answer_ce:.4f} | "
            f"fact_span_margin: {debiased_fact_span_margin:.4f} | "
            f"write_diversity: {debiased_write_diversity:.4f} | "
            f"memory_utility: {debiased_memory_utility:.4f} | "
            f"memory_utility_margin: {debiased_memory_utility_margin:.4f} | "
            f"anchor_margin: {debiased_anchor_margin:.4f} | "
            f"anchor_loss: {debiased_anchor_loss:.4f} | "
            f"memory_probe_ce: {debiased_memory_probe_ce:.4f} | "
            f"answer_start_ce: {debiased_answer_start_ce:.4f} | "
            f"key_token_ce: {debiased_key_token_ce:.4f} | "
            f"key_token_rank_margin: {debiased_key_token_rank_margin:.4f} | "
            f"key_token_utility_margin: {debiased_key_token_utility_margin:.4f} | "
            f"habit_confuser_margin: {debiased_habit_confuser_margin:.4f} | "
            f"span_contrast_margin: {debiased_span_contrast_margin:.4f} | "
            f"span_utility_margin: {debiased_span_utility_margin:.4f} | "
            f"span_hard_token_margin: {debiased_span_hard_token_margin:.4f} | "
            f"post_key_ce: {debiased_post_key_ce:.4f} | "
            f"post_key_repeat_margin: {debiased_post_key_repeat_margin:.4f} | "
            f"rollout_ce: {debiased_rollout_ce:.4f} | "
            f"rollout_anchor_margin: {debiased_rollout_anchor_margin:.4f} | "
            f"rollout_multiplier: {debiased_rollout_multiplier:.2f} | "
            f"rollout_applied: {debiased_rollout_applied:.2f} | "
            f"rollout_key_missing: {debiased_rollout_key_missing:.2f} | "
            f"rollout_repeated: {debiased_rollout_repeated:.2f} | "
            f"rollout_wrong_default: {debiased_rollout_wrong_default:.2f} | "
            f"phase3_recovery_ce: {debiased_phase3_recovery_ce:.4f} | "
            f"phase3_anchor_margin: {debiased_phase3_recovery_anchor_margin:.4f} | "
            f"phase3_applied: {debiased_phase3_recovery_applied:.2f} | "
            f"phase3_key_missing: {debiased_phase3_key_missing:.2f} | "
            f"phase3_partial_key: {debiased_phase3_partial_key:.2f} | "
            f"phase3_wrong_span: {debiased_phase3_wrong_span:.2f} | "
            f"phase3_repeated: {debiased_phase3_repeated:.2f} | "
            f"phase35_branch_loss: {debiased_phase35_branch_loss:.4f} | "
            f"phase35_branch_margin: {debiased_phase35_branch_margin:.4f} | "
            f"phase35_stop_loss: {debiased_phase35_stop_loss:.4f} | "
            f"phase35_stop_margin: {debiased_phase35_stop_margin:.4f} | "
            f"phase35_applied: {debiased_phase35_applied:.2f} | "
            f"phase36_ce: {debiased_phase36_ce:.4f} | "
            f"phase36_branch_loss: {debiased_phase36_branch_loss:.4f} | "
            f"phase36_branch_margin: {debiased_phase36_branch_margin:.4f} | "
            f"phase36_applied: {debiased_phase36_applied:.2f} | "
            f"phase36_no_memory_default: {debiased_phase36_no_memory_default:.2f} | "
            f"phase36_wrong_span: {debiased_phase36_wrong_span:.2f} | "
            f"phase36_partial_wrong: {debiased_phase36_partial_wrong:.2f} | "
            f"phase36_repeated: {debiased_phase36_repeated:.2f} | "
            f"phase37_branch_ce: {debiased_phase37_branch_ce:.4f} | "
            f"phase37_branch_loss: {debiased_phase37_branch_loss:.4f} | "
            f"phase37_branch_margin: {debiased_phase37_branch_margin:.4f} | "
            f"phase37_stop_loss: {debiased_phase37_stop_loss:.4f} | "
            f"phase37_stop_margin: {debiased_phase37_stop_margin:.4f} | "
            f"phase37_applied: {debiased_phase37_applied:.2f} | "
            f"phase37_no_memory_default: {debiased_phase37_no_memory_default:.2f} | "
            f"phase37_wrong_span: {debiased_phase37_wrong_span:.2f} | "
            f"phase37_partial_wrong: {debiased_phase37_partial_wrong:.2f} | "
            f"phase37_repeated: {debiased_phase37_repeated:.2f} | "
            f"phase38_traj_loss: {debiased_phase38_trajectory_loss:.4f} | "
            f"phase38_traj_margin: {debiased_phase38_trajectory_margin:.4f} | "
            f"phase38_gold_ce: {debiased_phase38_gold_ce:.4f} | "
            f"phase38_default_loss: {debiased_phase38_default_branch_loss:.4f} | "
            f"phase38_default_margin: {debiased_phase38_default_branch_margin:.4f} | "
            f"phase38_default_recovery_ce: {debiased_phase38_default_recovery_ce:.4f} | "
            f"phase38_default_applied: {debiased_phase38_default_branch_applied:.2f} | "
            f"phase38_branchpoint_loss: {debiased_phase38_branchpoint_loss:.4f} | "
            f"phase38_branchpoint_margin: {debiased_phase38_branchpoint_margin:.4f} | "
            f"phase38_branchpoint_recovery_ce: {debiased_phase38_branchpoint_recovery_ce:.4f} | "
            f"phase38_branchpoint_applied: {debiased_phase38_branchpoint_applied:.2f} | "
            f"phase38_branchpoint_answer_start: {debiased_phase38_branchpoint_answer_start:.2f} | "
            f"phase38_branchpoint_fact_branch: {debiased_phase38_branchpoint_fact_branch:.2f} | "
            f"phase38_branchpoint_no_memory_default: {debiased_phase38_branchpoint_no_memory_default:.2f} | "
            f"phase38_branchpoint_wrong_span: {debiased_phase38_branchpoint_wrong_span:.2f} | "
            f"phase38_branchpoint_partial_wrong: {debiased_phase38_branchpoint_partial_wrong:.2f} | "
            f"phase38_branchpoint_wrong_after_key: {debiased_phase38_branchpoint_wrong_after_key:.2f} | "
            f"phase38_branchpoint_repeated: {debiased_phase38_branchpoint_repeated:.2f} | "
            f"phase38_probe_loss: {debiased_phase38_probe_loss:.4f} | "
            f"phase38_probe_margin: {debiased_phase38_probe_margin:.4f} | "
            f"phase38_applied: {debiased_phase38_applied:.2f} | "
            f"phase38_key_missing: {debiased_phase38_key_missing:.2f} | "
            f"phase38_wrong_span: {debiased_phase38_wrong_span:.2f} | "
            f"phase38_wrong_after_key: {debiased_phase38_wrong_after_key:.2f} | "
            f"phase38_partial_wrong: {debiased_phase38_partial_wrong:.2f} | "
            f"phase38_no_memory_default: {debiased_phase38_no_memory_default:.2f} | "
            f"phase38_repeated: {debiased_phase38_repeated:.2f} | "
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
            "memory/anchor_loss": debiased_anchor_loss,
            "memory/memory_probe_ce": debiased_memory_probe_ce,
            "memory/answer_start_ce": debiased_answer_start_ce,
            "memory/key_token_ce": debiased_key_token_ce,
            "memory/key_token_rank_loss": debiased_key_token_rank_loss,
            "memory/key_token_rank_margin": debiased_key_token_rank_margin,
            "memory/key_token_utility_loss": debiased_key_token_utility_loss,
            "memory/key_token_utility_margin": debiased_key_token_utility_margin,
            "memory/habit_confuser_loss": debiased_habit_confuser_loss,
            "memory/habit_confuser_margin": debiased_habit_confuser_margin,
            "memory/span_contrast_loss": debiased_span_contrast_loss,
            "memory/span_contrast_margin": debiased_span_contrast_margin,
            "memory/span_utility_loss": debiased_span_utility_loss,
            "memory/span_utility_margin": debiased_span_utility_margin,
            "memory/span_hard_token_loss": debiased_span_hard_token_loss,
            "memory/span_hard_token_margin": debiased_span_hard_token_margin,
            "memory/post_key_ce": debiased_post_key_ce,
            "memory/post_key_repeat_margin": debiased_post_key_repeat_margin,
            "memory/rollout_ce": debiased_rollout_ce,
            "memory/rollout_anchor_margin": debiased_rollout_anchor_margin,
            "memory/rollout_multiplier": debiased_rollout_multiplier,
            "memory/rollout_applied": debiased_rollout_applied,
            "memory/rollout_key_missing": debiased_rollout_key_missing,
            "memory/rollout_repeated": debiased_rollout_repeated,
            "memory/rollout_wrong_default": debiased_rollout_wrong_default,
            "memory/phase3_recovery_ce": debiased_phase3_recovery_ce,
            "memory/phase3_recovery_anchor_margin": debiased_phase3_recovery_anchor_margin,
            "memory/phase3_recovery_applied": debiased_phase3_recovery_applied,
            "memory/phase3_key_missing": debiased_phase3_key_missing,
            "memory/phase3_partial_key": debiased_phase3_partial_key,
            "memory/phase3_wrong_span": debiased_phase3_wrong_span,
            "memory/phase3_repeated": debiased_phase3_repeated,
            "memory/phase35_branch_loss": debiased_phase35_branch_loss,
            "memory/phase35_branch_margin": debiased_phase35_branch_margin,
            "memory/phase35_stop_loss": debiased_phase35_stop_loss,
            "memory/phase35_stop_margin": debiased_phase35_stop_margin,
            "memory/phase35_applied": debiased_phase35_applied,
            "memory/phase36_ce": debiased_phase36_ce,
            "memory/phase36_branch_loss": debiased_phase36_branch_loss,
            "memory/phase36_branch_margin": debiased_phase36_branch_margin,
            "memory/phase36_applied": debiased_phase36_applied,
            "memory/phase36_no_memory_default": debiased_phase36_no_memory_default,
            "memory/phase36_wrong_span": debiased_phase36_wrong_span,
            "memory/phase36_partial_wrong": debiased_phase36_partial_wrong,
            "memory/phase36_repeated": debiased_phase36_repeated,
            "memory/phase37_branch_ce": debiased_phase37_branch_ce,
            "memory/phase37_branch_loss": debiased_phase37_branch_loss,
            "memory/phase37_branch_margin": debiased_phase37_branch_margin,
            "memory/phase37_stop_loss": debiased_phase37_stop_loss,
            "memory/phase37_stop_margin": debiased_phase37_stop_margin,
            "memory/phase37_applied": debiased_phase37_applied,
            "memory/phase37_no_memory_default": debiased_phase37_no_memory_default,
            "memory/phase37_wrong_span": debiased_phase37_wrong_span,
            "memory/phase37_partial_wrong": debiased_phase37_partial_wrong,
            "memory/phase37_repeated": debiased_phase37_repeated,
            "memory/phase38_trajectory_loss": debiased_phase38_trajectory_loss,
            "memory/phase38_trajectory_margin": debiased_phase38_trajectory_margin,
            "memory/phase38_gold_ce": debiased_phase38_gold_ce,
            "memory/phase38_default_branch_loss": debiased_phase38_default_branch_loss,
            "memory/phase38_default_branch_margin": debiased_phase38_default_branch_margin,
            "memory/phase38_default_recovery_ce": debiased_phase38_default_recovery_ce,
            "memory/phase38_default_branch_applied": debiased_phase38_default_branch_applied,
            "memory/phase38_branchpoint_loss": debiased_phase38_branchpoint_loss,
            "memory/phase38_branchpoint_margin": debiased_phase38_branchpoint_margin,
            "memory/phase38_branchpoint_recovery_ce": debiased_phase38_branchpoint_recovery_ce,
            "memory/phase38_branchpoint_applied": debiased_phase38_branchpoint_applied,
            "memory/phase38_branchpoint_answer_start": debiased_phase38_branchpoint_answer_start,
            "memory/phase38_branchpoint_fact_branch": debiased_phase38_branchpoint_fact_branch,
            "memory/phase38_branchpoint_no_memory_default": debiased_phase38_branchpoint_no_memory_default,
            "memory/phase38_branchpoint_wrong_span": debiased_phase38_branchpoint_wrong_span,
            "memory/phase38_branchpoint_partial_wrong": debiased_phase38_branchpoint_partial_wrong,
            "memory/phase38_branchpoint_wrong_after_key": debiased_phase38_branchpoint_wrong_after_key,
            "memory/phase38_branchpoint_repeated": debiased_phase38_branchpoint_repeated,
            "memory/phase38_probe_loss": debiased_phase38_probe_loss,
            "memory/phase38_probe_margin": debiased_phase38_probe_margin,
            "memory/phase38_applied": debiased_phase38_applied,
            "memory/phase38_key_missing": debiased_phase38_key_missing,
            "memory/phase38_wrong_span": debiased_phase38_wrong_span,
            "memory/phase38_wrong_after_key": debiased_phase38_wrong_after_key,
            "memory/phase38_partial_wrong": debiased_phase38_partial_wrong,
            "memory/phase38_no_memory_default": debiased_phase38_no_memory_default,
            "memory/phase38_repeated": debiased_phase38_repeated,
            "memory/active_slots": debiased_active_slots,
            "memory/gate_avg": gate_avg,
            "memory/phase": {
                "phase1_memory_only": 1,
                "phase2_interface_joint": 2,
                "phase3_generated_recovery": 3,
                "phase3_full_joint": 3,
                "phase3_5_branch_binding": 4,
                "phase3_6_branch_recovery": 5,
                "phase3_7_branch_contrast": 6,
                "phase3_8_trajectory_preference": 7,
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
