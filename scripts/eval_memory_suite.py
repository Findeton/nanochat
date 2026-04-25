"""
Broader evaluation for episodic-memory NanoChat.

This script evaluates a checkpoint on whole datasets of two-session-style recall
tasks. It mirrors chat behavior more honestly than the narrow one-off script:

- prior user turns are written into persistent episodic memory one turn at a time
- memory is saved and reloaded between the write and recall phases
- the final user turn is asked from a fresh conversation

It reports both deterministic (greedy) and sampled recall accuracy.
"""

import argparse
import json
import os
import re
from pathlib import Path

from nanochat.common import autodetect_device_type, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.episodic_memory import get_session_memory_path


def normalize_text(text):
    text = text.replace("<|assistant_end|>", " ")
    text = text.replace("<|user_start|>", " ")
    text = text.replace("<|assistant_start|>", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    conversations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            conversations.append(item.get("messages", item) if isinstance(item, dict) else item)
            if limit is not None and len(conversations) >= limit:
                break
    return conversations


def split_eval_conversation(messages):
    if len(messages) < 4:
        return None
    has_system = messages[0]["role"] == "system"
    pairs = []
    start = 1 if has_system else 0
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


def build_user_turn_tokens(tokenizer, text):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(text))
    tokens.append(user_end)
    return tokens


def run_prompt(tokenizer, engine, user_text, max_tokens, temperature, top_k, seed):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(user_text))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)
    out = []
    for token_column, _ in engine.generate(
        conversation_tokens,
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


def evaluate_dataset(
    model,
    tokenizer,
    dataset_path,
    source,
    model_tag,
    deterministic_max_tokens,
    sampled_max_tokens,
    sampled_temperature,
    sampled_top_k,
    sampled_seeds,
    limit=None,
    failures_to_print=3,
):
    conversations = load_conversations(dataset_path, limit=limit)
    engine = Engine(model, tokenizer)
    dataset_name = Path(dataset_path).name
    det_correct = 0
    det_core_correct = 0
    samp_correct = 0
    samp_core_correct = 0
    sampled_total = 0
    det_failures = []
    det_soft_successes = []
    samp_failures = []
    samp_soft_successes = []

    for idx, conv in enumerate(conversations):
        parsed = split_eval_conversation(conv)
        if parsed is None:
            continue
        prior_messages, recall_user, expected_answer = parsed
        session_id = f"eval-{dataset_name}-{idx}"
        session_path = get_session_memory_path(source, model_tag, session_id)
        if os.path.exists(session_path):
            os.remove(session_path)

        model.clear_memory_banks()
        for message in prior_messages:
            if message["role"] == "user":
                model.memorize(build_user_turn_tokens(tokenizer, message["content"]))
        model.save_memory_state(session_path)

        model.clear_memory_banks()
        model.load_memory_state(session_path, strict=False)
        det_response = run_prompt(
            tokenizer,
            engine,
            recall_user,
            max_tokens=deterministic_max_tokens,
            temperature=0.0,
            top_k=1,
            seed=42,
        )
        det_ok = normalize_text(det_response) == normalize_text(expected_answer)
        det_core_ok = core_fact_match(expected_answer, det_response)
        det_correct += int(det_ok)
        det_core_correct += int(det_core_ok)
        if not det_ok and len(det_failures) < failures_to_print:
            det_failures.append((recall_user, expected_answer, det_response))
        if det_core_ok and not det_ok and len(det_soft_successes) < failures_to_print:
            det_soft_successes.append((recall_user, expected_answer, det_response))

        for seed in sampled_seeds:
            model.clear_memory_banks()
            model.load_memory_state(session_path, strict=False)
            samp_response = run_prompt(
                tokenizer,
                engine,
                recall_user,
                max_tokens=sampled_max_tokens,
                temperature=sampled_temperature,
                top_k=sampled_top_k,
                seed=seed,
            )
            samp_ok = normalize_text(samp_response) == normalize_text(expected_answer)
            samp_core_ok = core_fact_match(expected_answer, samp_response)
            samp_correct += int(samp_ok)
            samp_core_correct += int(samp_core_ok)
            sampled_total += 1
            if not samp_ok and len(samp_failures) < failures_to_print:
                samp_failures.append((recall_user, expected_answer, samp_response))
            if samp_core_ok and not samp_ok and len(samp_soft_successes) < failures_to_print:
                samp_soft_successes.append((recall_user, expected_answer, samp_response))

        if os.path.exists(session_path):
            os.remove(session_path)

    n = len(conversations)
    return {
        "dataset": dataset_name,
        "num_examples": n,
        "det_accuracy": det_correct / max(n, 1),
        "det_core_accuracy": det_core_correct / max(n, 1),
        "sampled_accuracy": samp_correct / max(sampled_total, 1),
        "sampled_core_accuracy": samp_core_correct / max(sampled_total, 1),
        "sampled_total": sampled_total,
        "det_failures": det_failures,
        "det_soft_successes": det_soft_successes,
        "sampled_failures": samp_failures,
        "sampled_soft_successes": samp_soft_successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate episodic memory over datasets")
    parser.add_argument("--source", type=str, default="sft")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dataset", action="append", required=True, help="JSONL dataset path; may be provided multiple times")
    parser.add_argument("--limit", type=int, default=None, help="optional per-dataset example cap")
    parser.add_argument("--deterministic-max-tokens", type=int, default=96)
    parser.add_argument("--sampled-max-tokens", type=int, default=96)
    parser.add_argument("--sampled-temperature", type=float, default=0.6)
    parser.add_argument("--sampled-top-k", type=int, default=50)
    parser.add_argument("--sampled-seeds", type=str, default="1,2,3,4,5", help="comma-separated seed list for sampled eval")
    parser.add_argument("--failures-to-print", type=int, default=3)
    parser.add_argument("--json-output", type=str, default=None, help="optional path to write the full evaluation summary as JSON")
    args = parser.parse_args()

    sampled_seeds = [int(s) for s in args.sampled_seeds.split(",") if s.strip()]
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    loaded_model_tag = meta.get("_model_tag", args.model_tag)

    results = []
    for dataset_path in args.dataset:
        results.append(
            evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                source=args.source,
                model_tag=loaded_model_tag,
                deterministic_max_tokens=args.deterministic_max_tokens,
                sampled_max_tokens=args.sampled_max_tokens,
                sampled_temperature=args.sampled_temperature,
                sampled_top_k=args.sampled_top_k,
                sampled_seeds=sampled_seeds,
                limit=args.limit,
                failures_to_print=args.failures_to_print,
            )
        )

    total_examples = sum(r["num_examples"] for r in results)
    weighted_det = sum(r["det_accuracy"] * r["num_examples"] for r in results) / max(total_examples, 1)
    weighted_det_core = sum(r["det_core_accuracy"] * r["num_examples"] for r in results) / max(total_examples, 1)
    weighted_samp_num = sum(r["sampled_accuracy"] * r["sampled_total"] for r in results)
    weighted_samp_core_num = sum(r["sampled_core_accuracy"] * r["sampled_total"] for r in results)
    weighted_samp_den = sum(r["sampled_total"] for r in results)
    weighted_samp = weighted_samp_num / max(weighted_samp_den, 1)
    weighted_samp_core = weighted_samp_core_num / max(weighted_samp_den, 1)
    summary = {
        "model_tag": loaded_model_tag,
        "step": meta.get("_step", args.step),
        "limit_per_dataset": args.limit,
        "sampled_temperature": args.sampled_temperature,
        "sampled_top_k": args.sampled_top_k,
        "sampled_seeds": sampled_seeds,
        "overall_det_accuracy": weighted_det,
        "overall_det_core_accuracy": weighted_det_core,
        "overall_sampled_accuracy": weighted_samp,
        "overall_sampled_core_accuracy": weighted_samp_core,
        "results": results,
    }

    print(f"MODEL: {loaded_model_tag} @ step {meta.get('_step', args.step)}")
    print(f"LIMIT_PER_DATASET: {args.limit}")
    print(f"SAMPLED: temperature={args.sampled_temperature}, top_k={args.sampled_top_k}, seeds={sampled_seeds}")
    print()
    for r in results:
        print(
            f"DATASET {r['dataset']}: "
            f"examples={r['num_examples']} "
            f"det_accuracy={r['det_accuracy']:.3f} "
            f"det_core_accuracy={r['det_core_accuracy']:.3f} "
            f"sampled_accuracy={r['sampled_accuracy']:.3f} "
            f"sampled_core_accuracy={r['sampled_core_accuracy']:.3f}"
        )
        for recall_user, expected, got in r["det_soft_successes"]:
            print("  DET_CORE_ONLY")
            print(f"    Q: {recall_user}")
            print(f"    EXPECTED: {expected}")
            print(f"    GOT: {got}")
        for recall_user, expected, got in r["det_failures"]:
            print("  DET_FAIL")
            print(f"    Q: {recall_user}")
            print(f"    EXPECTED: {expected}")
            print(f"    GOT: {got}")
        for recall_user, expected, got in r["sampled_soft_successes"]:
            print("  SAMP_CORE_ONLY")
            print(f"    Q: {recall_user}")
            print(f"    EXPECTED: {expected}")
            print(f"    GOT: {got}")
        for recall_user, expected, got in r["sampled_failures"]:
            print("  SAMP_FAIL")
            print(f"    Q: {recall_user}")
            print(f"    EXPECTED: {expected}")
            print(f"    GOT: {got}")
        print()

    print(f"OVERALL_DETERMINISTIC_ACCURACY: {weighted_det:.3f}")
    print(f"OVERALL_DETERMINISTIC_CORE_ACCURACY: {weighted_det_core:.3f}")
    print(f"OVERALL_SAMPLED_ACCURACY: {weighted_samp:.3f}")
    print(f"OVERALL_SAMPLED_CORE_ACCURACY: {weighted_samp_core:.3f}")
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
