"""
Dataset-side analysis for NanoChat memory experiments.

This script is meant to answer questions like:
- Are train and eval answers distributed similarly?
- Are we mostly training on one-token / one-fact answers?
- How much longer / more compositional are the broader eval targets?
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init


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


def load_conversations(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def summarize_lengths(values):
    values = sorted(values)
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    n = len(values)
    return {
        "min": values[0],
        "p50": values[n // 2],
        "p90": values[min(n - 1, int(0.9 * (n - 1)))],
        "max": values[-1],
        "mean": sum(values) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze memory dataset structure")
    parser.add_argument("--source", type=str, default="sft")
    parser.add_argument("--model-tag", type=str, default="d12")
    parser.add_argument("--step", type=int, default=967)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    del model

    report = []
    core_vocab_by_dataset = {}
    answer_vocab_by_dataset = {}

    for dataset in args.dataset:
        rows = load_conversations(dataset)
        recall_lengths = []
        answer_lengths = []
        answer_core_lengths = []
        prior_user_turns = []
        first_answer_tokens = Counter()
        first_core_tokens = Counter()
        answer_prefixes = Counter()
        core_vocab = Counter()
        answer_vocab = Counter()

        for conv in rows:
            parsed = split_eval_conversation(conv)
            if parsed is None:
                continue
            prior_messages, recall_user, expected_answer = parsed
            prior_user_turns.append(sum(1 for m in prior_messages if m["role"] == "user"))
            recall_tok = tokenizer.encode(recall_user)
            answer_tok = tokenizer.encode(expected_answer)
            recall_lengths.append(len(recall_tok))
            answer_lengths.append(len(answer_tok))
            if answer_tok:
                first_answer_tokens[tokenizer.decode([answer_tok[0]])] += 1
                answer_prefixes[tokenizer.decode(answer_tok[: min(4, len(answer_tok))])] += 1
            core_tokens = extract_core_tokens(expected_answer)
            answer_core_lengths.append(len(core_tokens))
            if core_tokens:
                first_core_tokens[core_tokens[0]] += 1
            for tok in core_tokens:
                core_vocab[tok] += 1
            answer_vocab[expected_answer] += 1

        core_vocab_by_dataset[Path(dataset).name] = set(core_vocab)
        answer_vocab_by_dataset[Path(dataset).name] = set(answer_vocab)
        report.append({
            "dataset": Path(dataset).name,
            "num_examples": len(rows),
            "prior_user_turns": summarize_lengths(prior_user_turns),
            "recall_token_lengths": summarize_lengths(recall_lengths),
            "answer_token_lengths": summarize_lengths(answer_lengths),
            "answer_core_token_lengths": summarize_lengths(answer_core_lengths),
            "top_first_answer_tokens": first_answer_tokens.most_common(10),
            "top_first_core_tokens": first_core_tokens.most_common(10),
            "top_answer_prefixes": answer_prefixes.most_common(10),
            "unique_answers": len(answer_vocab),
            "unique_core_tokens": len(core_vocab),
        })

    overlaps = []
    dataset_names = [Path(d).name for d in args.dataset]
    for i, left in enumerate(dataset_names):
        for right in dataset_names[i + 1:]:
            core_overlap = len(core_vocab_by_dataset[left] & core_vocab_by_dataset[right])
            answer_overlap = len(answer_vocab_by_dataset[left] & answer_vocab_by_dataset[right])
            overlaps.append({
                "left": left,
                "right": right,
                "shared_core_tokens": core_overlap,
                "shared_exact_answers": answer_overlap,
            })

    print(json.dumps({"datasets": report, "overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()
