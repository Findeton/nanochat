"""
Run a staged K/V-memory training and evaluation campaign.

This script is intended for a stronger machine than the local development box.
It consumes the manifest written by `scripts.prepare_kv_campaign_data` and
executes a reproducible three-stage campaign:

1. stage1_identity
2. stage2_structured
3. stage3_mixed

Each stage:
- trains from the previous checkpoint
- writes a stage-specific output tag
- evaluates on the standard memory eval family
- stores machine-readable eval JSON under the campaign directory

Example:
python -m scripts.run_kv_campaign \
  --base-model-tag d12-episodic-proposal-a-r18-long1000 \
  --base-model-step 1000 \
  --prepare-data
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "data" / "kv_campaign" / "kv_campaign_manifest.json"
DEFAULT_CAMPAIGN_GUARDRAILS = {
    "distill_weight": 0.04,
    "distill_temperature": 1.0,
    "distill_mask": "non_target",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a staged K/V memory training/eval campaign")
    parser.add_argument("--campaign-manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prepare-data", action="store_true", help="regenerate staged campaign data before training")
    parser.add_argument("--generate-large-data", action="store_true", help="generate the larger staged K/V corpus before preparing the campaign mixture")
    parser.add_argument("--base-model-tag", type=str, required=True)
    parser.add_argument("--base-model-step", type=int, default=None)
    parser.add_argument("--teacher-model-tag", type=str, default=None, help="defaults to --base-model-tag")
    parser.add_argument("--teacher-model-step", type=int, default=None, help="defaults to --base-model-step")
    parser.add_argument("--campaign-name", type=str, default="kv-campaign")
    parser.add_argument("--device-type", type=str, default="cuda")
    parser.add_argument("--source", type=str, default="sft")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--eval-limit", type=int, default=None, help="optional per-dataset eval limit")
    parser.add_argument("--intermediate-eval-limit", type=int, default=None, help="optional per-dataset eval limit for intermediate checkpoint evals")
    parser.add_argument("--sampled-temperature", type=float, default=0.6)
    parser.add_argument("--sampled-top-k", type=int, default=50)
    parser.add_argument("--sampled-seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-intermediate-evals", action="store_true", help="only run the final eval for each stage")
    parser.add_argument("--only-stage", type=str, default=None, choices=["stage1_identity", "stage2_structured", "stage3_mixed"])
    parser.add_argument("--stage1-iterations", type=int, default=None)
    parser.add_argument("--stage2-iterations", type=int, default=None)
    parser.add_argument("--stage3-iterations", type=int, default=None)
    parser.add_argument("--save-every-override", type=int, default=None, help="override all stage checkpoint save cadences")
    parser.add_argument("--eval-every-override", type=int, default=None, help="override all stage intermediate eval cadences")
    return parser.parse_args()


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def maybe_run(command, *, dry_run):
    print(shell_join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data_if_needed(args):
    manifest_path = Path(args.campaign_manifest)
    if not args.prepare_data and manifest_path.exists():
        return
    output_dir = manifest_path.parent
    command = [
        args.python_bin,
        "-m",
        "scripts.prepare_kv_campaign_data",
        "--output-dir",
        str(output_dir),
    ]
    if args.generate_large_data:
        command.append("--generate-large-corpus")
    maybe_run(command, dry_run=args.dry_run)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Campaign manifest {manifest_path} does not exist yet. "
            "Run scripts.prepare_kv_campaign_data first, or rerun without --dry-run."
        )


def build_train_command(args, stage, model_tag, model_step, output_tag, run_name):
    train_args = {**DEFAULT_CAMPAIGN_GUARDRAILS, **stage["recommended_train_args"]}
    save_every = args.save_every_override if args.save_every_override is not None else stage["recommended_save_every"]
    command = [
        args.python_bin,
        "-m",
        "scripts.chat_memory",
        "--run",
        run_name,
        "--device-type",
        args.device_type,
        "--model-tag",
        model_tag,
        "--output-tag",
        output_tag,
        "--custom-json",
        stage["path"],
        "--num-iterations",
        str(stage["recommended_num_iterations"]),
        "--save-every",
        str(save_every),
    ]
    if model_step is not None:
        command.extend(["--model-step", str(model_step)])

    teacher_model_tag = args.teacher_model_tag or args.base_model_tag
    teacher_model_step = args.teacher_model_step if args.teacher_model_step is not None else args.base_model_step
    if teacher_model_tag is not None:
        command.extend(["--teacher-model-tag", teacher_model_tag])
    if teacher_model_step is not None:
        command.extend(["--teacher-model-step", str(teacher_model_step)])

    for key, value in train_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def build_eval_command(args, manifest, model_tag, model_step, json_output_path, eval_limit):
    command = [
        args.python_bin,
        "-m",
        "scripts.eval_memory_suite",
        "--source",
        args.source,
        "--model-tag",
        model_tag,
        "--device-type",
        args.device_type,
        "--sampled-temperature",
        str(args.sampled_temperature),
        "--sampled-top-k",
        str(args.sampled_top_k),
        "--sampled-seeds",
        args.sampled_seeds,
        "--failures-to-print",
        "0",
        "--json-output",
        str(json_output_path),
    ]
    if model_step is not None:
        command.extend(["--step", str(model_step)])
    if eval_limit is not None:
        command.extend(["--limit", str(eval_limit)])
    for dataset in manifest["eval_sets"]:
        command.extend(["--dataset", dataset["path"]])
    return command


def stage_iteration_override(args, stage_name):
    mapping = {
        "stage1_identity": args.stage1_iterations,
        "stage2_structured": args.stage2_iterations,
        "stage3_mixed": args.stage3_iterations,
    }
    return mapping[stage_name]


def compute_eval_steps(total_steps, eval_every, skip_intermediate):
    if skip_intermediate or eval_every is None or eval_every <= 0:
        return [total_steps]
    steps = list(range(eval_every, total_steps + 1, eval_every))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def main():
    args = parse_args()
    prepare_data_if_needed(args)
    manifest = load_manifest(args.campaign_manifest)

    campaign_dir = Path(manifest["output_dir"]) / args.campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "campaign_name": args.campaign_name,
        "campaign_dir": str(campaign_dir),
        "base_model_tag": args.base_model_tag,
        "base_model_step": args.base_model_step,
        "teacher_model_tag": args.teacher_model_tag or args.base_model_tag,
        "teacher_model_step": args.teacher_model_step if args.teacher_model_step is not None else args.base_model_step,
        "stages": [],
    }

    previous_model_tag = args.base_model_tag
    previous_model_step = args.base_model_step

    for stage in manifest["train_stages"]:
        stage_name = stage["name"]
        if args.only_stage and stage_name != args.only_stage:
            continue

        override = stage_iteration_override(args, stage_name)
        if override is not None:
            stage["recommended_num_iterations"] = override

        save_every = args.save_every_override if args.save_every_override is not None else stage["recommended_save_every"]
        eval_every = args.eval_every_override if args.eval_every_override is not None else stage["recommended_eval_every"]
        eval_steps = compute_eval_steps(stage["recommended_num_iterations"], eval_every, args.skip_intermediate_evals)
        output_tag = f"{args.campaign_name}-{stage_name}"
        run_name = output_tag
        train_command = build_train_command(args, stage, previous_model_tag, previous_model_step, output_tag, run_name)
        eval_entries = []
        for eval_step in eval_steps:
            is_final = eval_step == stage["recommended_num_iterations"]
            eval_limit = args.eval_limit if is_final else (
                args.intermediate_eval_limit
                if args.intermediate_eval_limit is not None
                else stage["recommended_intermediate_eval_limit"]
            )
            eval_json_path = campaign_dir / f"{stage_name}_step{eval_step:06d}_eval.json"
            eval_entries.append(
                {
                    "step": eval_step,
                    "limit": eval_limit,
                    "is_final": is_final,
                    "json_path": str(eval_json_path),
                    "command": build_eval_command(args, manifest, output_tag, eval_step, eval_json_path, eval_limit),
                }
            )

        plan["stages"].append(
            {
                "name": stage_name,
                "description": stage["description"],
                "train_dataset": stage["path"],
                "num_examples": stage["num_examples"],
                "input_model_tag": previous_model_tag,
                "input_model_step": previous_model_step,
                "output_model_tag": output_tag,
                "output_model_step": stage["recommended_num_iterations"],
                "save_every": save_every,
                "eval_every": eval_every,
                "eval_steps": eval_steps,
                "train_command": train_command,
                "evals": eval_entries,
            }
        )

        maybe_run(train_command, dry_run=args.dry_run)
        for eval_entry in eval_entries:
            maybe_run(eval_entry["command"], dry_run=args.dry_run)

        previous_model_tag = output_tag
        previous_model_step = stage["recommended_num_iterations"]

    plan_path = campaign_dir / "campaign_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"Wrote campaign plan to {plan_path}")


if __name__ == "__main__":
    main()
