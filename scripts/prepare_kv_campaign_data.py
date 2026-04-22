"""
Prepare staged training/eval data for larger-scale K/V-memory campaigns.

This script builds three shuffled JSONL mixtures from:

- a larger, higher-quality generated K/V corpus
- selected existing memory datasets for continuity with earlier work

The resulting stages are:

- stage1_identity: exact identity + template-bearing recall
- stage2_structured: longer templated and multi-fact recall
- stage3_mixed: broad mixed continuation used for later fine-tuning

It also writes a machine-readable manifest with:
- output dataset paths
- per-source counts
- recommended training defaults for each stage
- recommended eval datasets for campaign evaluation

Example:
python -m scripts.prepare_kv_campaign_data
"""

import argparse
import json
import math
import random
from pathlib import Path

from scripts.generate_kv_campaign_corpus import DEFAULT_OUTPUT_DIR as DEFAULT_GENERATED_DATA_DIR
from scripts.generate_kv_campaign_corpus import DEFAULT_SIZES as GENERATED_DEFAULT_SIZES
from scripts.generate_kv_campaign_corpus import generate_corpus

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "kv_campaign"
DEFAULT_GENERATED_DIR = DEFAULT_GENERATED_DATA_DIR

DEFAULT_GUARDRAIL_ARGS = {
    "distill_weight": 0.04,
    "distill_temperature": 1.0,
    "distill_mask": "non_target",
}


STAGE_SPECS = {
    "stage1_identity": {
        "description": "Identity-first K/V training focused on template-bearing recall, exact filler identity, paraphrase robustness, and guarded fine-tuning.",
        "size": 260_000,
        "sources": [
            {"name": "kv_stage1_large_train.jsonl", "weight": 8, "role": "large generated identity-first K/V corpus", "generated": True},
            {"name": "memory_dogname_focus.jsonl", "weight": 2, "role": "template-bearing dog-name recall"},
            {"name": "memory_dogname_nameonly.jsonl", "weight": 1, "role": "direct filler emission"},
            {"name": "memory_train_r1.jsonl", "weight": 1, "role": "simple recall prompts across domains"},
        ],
        "recommended_num_iterations": 24_000,
        "recommended_save_every": 2_000,
        "recommended_eval_every": 4_000,
        "recommended_intermediate_eval_limit": 256,
        "recommended_train_args": {
            "enable_kv_injection": True,
            "skip_smoltalk": True,
            "target_selection": "final",
            "context_mode": "full",
            "max_seq_len": 192,
            "device_batch_size": 8,
            "template_distill_weight": 0.10,
            "identity_token_loss_weight": 0.20,
            "identity_anchor_margin_loss_weight": 0.10,
            "identity_span_tokens": 4,
            "weighted_answer_ce_weight": 0.50,
            "weighted_template_ce": 0.25,
            "weighted_fact_ce": 1.00,
            "weighted_rest_ce": 0.50,
            "fact_span_margin_loss_weight": 0.20,
            "fact_span_margin": 2.50,
            **DEFAULT_GUARDRAIL_ARGS,
        },
    },
    "stage2_structured": {
        "description": "Structured-answer K/V training with longer templates, multi-fact responses, update behavior, and teacher-preservation guardrails.",
        "size": 340_000,
        "sources": [
            {"name": "kv_stage2_large_train.jsonl", "weight": 8, "role": "large generated structured K/V corpus", "generated": True},
            {"name": "memory_train_hq.jsonl", "weight": 2, "role": "high-quality multi-fact structured answers"},
            {"name": "memory_train_binding.jsonl", "weight": 2, "role": "identity binding under light distractors"},
            {"name": "memory_dogname_focus.jsonl", "weight": 1, "role": "template-bearing identity anchor retention"},
            {"name": "memory_train_r1.jsonl", "weight": 1, "role": "simple recall variety"},
        ],
        "recommended_num_iterations": 32_000,
        "recommended_save_every": 2_000,
        "recommended_eval_every": 4_000,
        "recommended_intermediate_eval_limit": 256,
        "recommended_train_args": {
            "enable_kv_injection": True,
            "skip_smoltalk": True,
            "target_selection": "final",
            "context_mode": "full",
            "max_seq_len": 256,
            "device_batch_size": 8,
            "template_distill_weight": 0.08,
            "identity_token_loss_weight": 0.25,
            "identity_anchor_margin_loss_weight": 0.10,
            "identity_span_tokens": 4,
            "weighted_answer_ce_weight": 0.40,
            "weighted_template_ce": 0.30,
            "weighted_fact_ce": 1.00,
            "weighted_rest_ce": 0.60,
            "fact_span_margin_loss_weight": 0.15,
            "fact_span_margin": 2.25,
            "span_token_loss_weight": 0.10,
            "span_value_loss_weight": 0.05,
            "answer_span_tokens": 8,
            **DEFAULT_GUARDRAIL_ARGS,
        },
    },
    "stage3_mixed": {
        "description": "Broad mixed continuation that keeps identity/template pressure, uses larger generated coverage, and preserves behavior with teacher guardrails.",
        "size": 440_000,
        "sources": [
            {"name": "kv_stage3_large_train.jsonl", "weight": 8, "role": "large generated broad K/V corpus", "generated": True},
            {"name": "memory_train_hq.jsonl", "weight": 2, "role": "high-quality structured recall"},
            {"name": "memory_train_binding.jsonl", "weight": 2, "role": "binding and distractor resistance"},
            {"name": "memory_dogname_focus.jsonl", "weight": 2, "role": "template-bearing focused recall"},
            {"name": "memory_dogname_nameonly.jsonl", "weight": 1, "role": "direct identity anchors"},
            {"name": "memory_train_r1.jsonl", "weight": 1, "role": "simple recall coverage"},
        ],
        "recommended_num_iterations": 40_000,
        "recommended_save_every": 2_000,
        "recommended_eval_every": 5_000,
        "recommended_intermediate_eval_limit": 512,
        "recommended_train_args": {
            "enable_kv_injection": True,
            "skip_smoltalk": True,
            "target_selection": "final",
            "context_mode": "full",
            "max_seq_len": 256,
            "device_batch_size": 8,
            "template_distill_weight": 0.06,
            "identity_token_loss_weight": 0.20,
            "identity_anchor_margin_loss_weight": 0.08,
            "identity_span_tokens": 4,
            "weighted_answer_ce_weight": 0.35,
            "weighted_template_ce": 0.35,
            "weighted_fact_ce": 1.00,
            "weighted_rest_ce": 0.70,
            "fact_span_margin_loss_weight": 0.10,
            "fact_span_margin": 2.00,
            "span_token_loss_weight": 0.10,
            "span_value_loss_weight": 0.05,
            "answer_span_tokens": 10,
            "distill_weight": 0.05,
            "distill_temperature": 1.0,
            "distill_mask": "non_target",
        },
    },
}


EVAL_DATASETS = [
    {"name": "kv_stage1_generated", "path": "kv_stage1_large_eval.jsonl", "role": "generated identity-focused held-out eval", "generated": True},
    {"name": "kv_stage2_generated", "path": "kv_stage2_large_eval.jsonl", "role": "generated structured held-out eval", "generated": True},
    {"name": "kv_stage3_generated", "path": "kv_stage3_large_eval.jsonl", "role": "generated broad held-out eval", "generated": True},
    {"name": "focus", "path": "memory_dogname_focus_eval.jsonl", "role": "templated focused identity recall"},
    {"name": "binding", "path": "memory_eval_binding.jsonl", "role": "binding and distractor generalization"},
    {"name": "hq", "path": "memory_eval_hq.jsonl", "role": "multi-fact structured recall"},
    {"name": "r1", "path": "memory_eval_r1.jsonl", "role": "simple recall coverage"},
]


def load_jsonl_lines(path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(raw)
    return lines


def allocate_counts(total, weighted_sources):
    total_weight = sum(src["weight"] for src in weighted_sources)
    raw = [total * src["weight"] / total_weight for src in weighted_sources]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weighted_sources)),
        key=lambda i: (raw[i] - counts[i], weighted_sources[i]["weight"]),
        reverse=True,
    )
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def sample_lines(lines, count, rng):
    if not lines:
        return []
    if count <= len(lines):
        return rng.sample(lines, count)
    sampled = []
    full_passes, remainder = divmod(count, len(lines))
    for _ in range(full_passes):
        chunk = list(lines)
        rng.shuffle(chunk)
        sampled.extend(chunk)
    if remainder:
        sampled.extend(rng.sample(lines, remainder))
    return sampled


def resolve_source_path(source, data_dir, generated_data_dir):
    base_dir = generated_data_dir if source.get("generated") else data_dir
    return base_dir / source["name"]


def load_generated_manifest(generated_data_dir):
    manifest_path = generated_data_dir / "kv_large_manifest.json"
    if not manifest_path.exists():
        return None, None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f), str(manifest_path)


def build_stage(stage_name, spec, data_dir, generated_data_dir, output_dir, rng, size_override=None):
    target_size = size_override if size_override is not None else spec["size"]
    weighted_sources = []
    for source in spec["sources"]:
        path = resolve_source_path(source, data_dir, generated_data_dir)
        lines = load_jsonl_lines(path)
        weighted_sources.append(
            {
                **source,
                "path": str(path),
                "available_examples": len(lines),
                "lines": lines,
            }
        )

    counts = allocate_counts(target_size, weighted_sources)
    mixed_lines = []
    source_summary = []
    selected_generated = 0
    selected_legacy = 0
    for source, count in zip(weighted_sources, counts):
        sampled = sample_lines(source["lines"], count, rng)
        mixed_lines.extend(sampled)
        if source.get("generated"):
            selected_generated += count
        else:
            selected_legacy += count
        source_summary.append(
            {
                "name": source["name"],
                "path": source["path"],
                "role": source["role"],
                "weight": source["weight"],
                "generated": bool(source.get("generated", False)),
                "available_examples": source["available_examples"],
                "selected_examples": count,
            }
        )

    rng.shuffle(mixed_lines)
    output_path = output_dir / f"{stage_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for line in mixed_lines:
            f.write(line)
            f.write("\n")

    return {
        "name": stage_name,
        "description": spec["description"],
        "path": str(output_path),
        "num_examples": len(mixed_lines),
        "recommended_num_iterations": spec["recommended_num_iterations"],
        "recommended_save_every": spec["recommended_save_every"],
        "recommended_eval_every": spec["recommended_eval_every"],
        "recommended_intermediate_eval_limit": spec["recommended_intermediate_eval_limit"],
        "recommended_train_args": spec["recommended_train_args"],
        "generated_examples": selected_generated,
        "legacy_examples": selected_legacy,
        "sources": source_summary,
    }


def build_manifest(output_dir, data_dir, generated_data_dir, stage_results, seed):
    generated_manifest, generated_manifest_path = load_generated_manifest(generated_data_dir)
    eval_sets = []
    for dataset in EVAL_DATASETS:
        base_dir = generated_data_dir if dataset.get("generated") else data_dir
        eval_sets.append(
            {
                **dataset,
                "path": str(base_dir / dataset["path"]),
            }
        )
    return {
        "seed": seed,
        "data_dir": str(data_dir),
        "generated_data_dir": str(generated_data_dir),
        "generated_manifest_path": generated_manifest_path,
        "generated_corpus_summary": generated_manifest,
        "output_dir": str(output_dir),
        "train_stages": stage_results,
        "eval_sets": eval_sets,
        "guardrails": {
            "teacher_preservation_distillation": True,
            "default_distill_args": DEFAULT_GUARDRAIL_ARGS,
            "teacher_strategy": "fixed_base_teacher_by_default",
        },
        "notes": [
            "stage1_identity is the safest first K/V run from r18",
            "stage2_structured broadens answer planning and multi-fact exposure",
            "stage3_mixed is the longer continuation once stage1/2 are stable",
            "the campaign now uses fixed teacher-preservation guardrails by default",
            "the staged mixtures now pull primarily from a much larger generated K/V corpus",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare staged datasets for large-scale K/V campaigns")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generated-data-dir", type=str, default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--generate-large-corpus", action="store_true", help="generate the larger staged K/V corpus before preparing the campaign mixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-size", type=int, default=STAGE_SPECS["stage1_identity"]["size"])
    parser.add_argument("--stage2-size", type=int, default=STAGE_SPECS["stage2_structured"]["size"])
    parser.add_argument("--stage3-size", type=int, default=STAGE_SPECS["stage3_mixed"]["size"])
    parser.add_argument("--kv-stage1-large-train", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage1_large_train.jsonl"])
    parser.add_argument("--kv-stage1-large-eval", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage1_large_eval.jsonl"])
    parser.add_argument("--kv-stage2-large-train", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage2_large_train.jsonl"])
    parser.add_argument("--kv-stage2-large-eval", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage2_large_eval.jsonl"])
    parser.add_argument("--kv-stage3-large-train", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage3_large_train.jsonl"])
    parser.add_argument("--kv-stage3-large-eval", type=int, default=GENERATED_DEFAULT_SIZES["kv_stage3_large_eval.jsonl"])
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    generated_data_dir = Path(args.generated_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_large_corpus:
        large_sizes = {
            "kv_stage1_large_train.jsonl": args.kv_stage1_large_train,
            "kv_stage1_large_eval.jsonl": args.kv_stage1_large_eval,
            "kv_stage2_large_train.jsonl": args.kv_stage2_large_train,
            "kv_stage2_large_eval.jsonl": args.kv_stage2_large_eval,
            "kv_stage3_large_train.jsonl": args.kv_stage3_large_train,
            "kv_stage3_large_eval.jsonl": args.kv_stage3_large_eval,
        }
        generate_corpus(generated_data_dir, large_sizes, args.seed)

    stage_results = []
    size_overrides = {
        "stage1_identity": args.stage1_size,
        "stage2_structured": args.stage2_size,
        "stage3_mixed": args.stage3_size,
    }
    for stage_name, spec in STAGE_SPECS.items():
        stage_results.append(build_stage(stage_name, spec, data_dir, generated_data_dir, output_dir, rng, size_override=size_overrides[stage_name]))

    manifest = build_manifest(output_dir, data_dir, generated_data_dir, stage_results, args.seed)
    manifest_path = output_dir / "kv_campaign_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote staged K/V campaign data to {output_dir}")
    for stage in stage_results:
        print(f"- {stage['name']}: {stage['num_examples']} examples -> {stage['path']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
