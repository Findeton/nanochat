import json
import random
from pathlib import Path
from types import SimpleNamespace

from tasks.customjson import CustomJSON
from scripts.generate_kv_campaign_corpus import generate_corpus
from scripts.prepare_kv_campaign_data import STAGE_SPECS, build_manifest, build_stage
from scripts.run_kv_campaign import build_train_command, compute_eval_steps


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def minimal_conversation(label):
    return [
        {"role": "user", "content": f"remember {label}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "what was it?"},
        {"role": "assistant", "content": label},
    ]


def ensure_stage_sources(data_dir, stage_name):
    spec = STAGE_SPECS[stage_name]
    for source in spec["sources"]:
        rows = [minimal_conversation(f"{source['name']}-a"), minimal_conversation(f"{source['name']}-b")]
        write_jsonl(data_dir / source["name"], rows)


def test_build_stage_writes_expected_number_of_examples(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    output_dir.mkdir()
    ensure_stage_sources(data_dir, "stage1_identity")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    for source in STAGE_SPECS["stage1_identity"]["sources"]:
        if source.get("generated"):
            write_jsonl(generated_dir / source["name"], [minimal_conversation("g-a"), minimal_conversation("g-b")])

    stage = build_stage(
        "stage1_identity",
        STAGE_SPECS["stage1_identity"],
        data_dir,
        generated_dir,
        output_dir,
        random.Random(7),
        size_override=14,
    )

    assert stage["num_examples"] == 14
    assert sum(source["selected_examples"] for source in stage["sources"]) == 14
    output_path = Path(stage["path"])
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as f:
        assert sum(1 for _ in f) == 14


def test_build_manifest_contains_stage_and_eval_metadata(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    generated_dir = tmp_path / "generated"
    output_dir.mkdir()
    data_dir.mkdir()
    generated_dir.mkdir()
    stage_results = [
        {
            "name": "stage1_identity",
            "description": "x",
            "path": str(output_dir / "stage1_identity.jsonl"),
            "num_examples": 10,
            "recommended_num_iterations": 20,
            "recommended_save_every": 5,
            "recommended_eval_every": 10,
            "recommended_intermediate_eval_limit": 3,
            "recommended_train_args": {"enable_kv_injection": True},
            "sources": [],
        }
    ]

    manifest = build_manifest(output_dir, data_dir, generated_dir, stage_results, seed=123)

    assert manifest["seed"] == 123
    assert manifest["train_stages"][0]["name"] == "stage1_identity"
    assert any(dataset["name"] == "focus" for dataset in manifest["eval_sets"])
    assert manifest["guardrails"]["teacher_preservation_distillation"] is True


def test_build_train_command_wires_stage_defaults():
    args = SimpleNamespace(
        python_bin="python",
        device_type="cuda",
        base_model_tag="base-tag",
        base_model_step=1000,
        teacher_model_tag=None,
        teacher_model_step=None,
        save_every_override=None,
    )
    stage = {
        "path": "/tmp/stage1.jsonl",
        "recommended_num_iterations": 50,
        "recommended_save_every": 10,
        "recommended_train_args": {
            "enable_kv_injection": True,
            "skip_smoltalk": True,
            "template_distill_weight": 0.1,
        },
    }

    command = build_train_command(
        args=args,
        stage=stage,
        model_tag="input-tag",
        model_step=100,
        output_tag="output-tag",
        run_name="run-name",
    )

    assert "-m" in command and "scripts.chat_memory" in command
    assert "--model-tag" in command and "input-tag" in command
    assert "--model-step" in command and "100" in command
    assert "--output-tag" in command and "output-tag" in command
    assert "--save-every" in command and "10" in command
    assert "--enable-kv-injection" in command
    assert "--skip-smoltalk" in command
    assert "--template-distill-weight" in command
    assert "--distill-weight" in command
    assert "--teacher-model-tag" in command and "base-tag" in command


def test_compute_eval_steps_includes_final_checkpoint():
    assert compute_eval_steps(24, 8, False) == [8, 16, 24]
    assert compute_eval_steps(25, 8, False) == [8, 16, 24, 25]
    assert compute_eval_steps(25, 8, True) == [25]


def test_generate_corpus_writes_all_stage_files(tmp_path):
    output_dir = tmp_path / "generated"
    sizes = {
        "kv_stage1_large_train.jsonl": 4,
        "kv_stage1_large_eval.jsonl": 2,
        "kv_stage2_large_train.jsonl": 4,
        "kv_stage2_large_eval.jsonl": 2,
        "kv_stage3_large_train.jsonl": 4,
        "kv_stage3_large_eval.jsonl": 2,
    }
    manifest_path = generate_corpus(output_dir, sizes, seed=7)

    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["files"]) == 6
    for item in manifest["files"]:
        path = Path(item["path"])
        assert path.exists()
        assert "family_counts" in item
        with path.open("r", encoding="utf-8") as f:
            row = json.loads(next(f))
        assert "messages" in row
        assert "memory_target" in row
        assert row["memory_target"]["fact_groups"]
        assert len(row["messages"]) >= 4
        for i, message in enumerate(row["messages"]):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role


def test_customjson_accepts_object_rows_with_metadata(tmp_path):
    path = tmp_path / "annotated.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "messages": minimal_conversation("tuna"),
                    "memory_target": {
                        "fact_groups": [{"text": "tuna", "hard_negatives": ["luna", "spot"]}],
                    },
                }
            )
        )
        f.write("\n")

    dataset = CustomJSON(filepath=str(path))
    example = dataset[0]

    assert "messages" in example
    assert "memory_target" in example
    assert example["memory_target"]["fact_groups"][0]["text"] == "tuna"
