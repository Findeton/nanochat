import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig


def build_tiny_model():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        ve_gate_channels=8,
        episodic_dim=32,
        episodic_slots=8,
        episodic_top_k=4,
    )
    model = GPT(config)
    model.init_weights()
    return model


def enable_memory(model):
    with torch.no_grad():
        model.episodic_controller.kind_bias.zero_()
        for block in model.transformer.h:
            block.episodic_memory_score_bias.zero_()
    return model


def test_missing_episodic_params_are_reinitialized_and_legacy_keys_dropped():
    model = build_tiny_model()
    model_data = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "episodic_" not in k and "episodic_controller" not in k
    }
    model_data["transformer.h.0.episodic_plan_score_gate"] = torch.zeros(1)
    _patch_missing_keys(model_data, model)

    assert "transformer.h.0.episodic_plan_score_gate" not in model_data
    assert "episodic_controller.write_key_proj.weight" in model_data
    assert "episodic_controller.write_surprise_gate.weight" in model_data
    assert "episodic_controller.no_write_bias" in model_data
    assert "episodic_controller.write_recency_bias" in model_data
    assert "episodic_controller.recall_query_proj.weight" in model_data
    assert "transformer.h.0.episodic_memory_score_bias" in model_data


def test_memory_state_round_trip():
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))

    model.clear_memory_banks()
    model.memorize(tokens)
    state = model.memory_state_dict()

    assert state["version"] == 3
    assert any(bank["num_slots"] > 0 for bank in state["banks"])
    assert state["episodic_dim"] == model.config.episodic_dim

    model.clear_memory_banks()
    model.load_memory_state_dict(state)
    restored = model.memory_state_dict()

    assert torch.allclose(restored["banks"][0]["tokens"], state["banks"][0]["tokens"])
    assert torch.allclose(restored["banks"][0]["keys"], state["banks"][0]["keys"])
    assert torch.allclose(restored["banks"][0]["strengths"], state["banks"][0]["strengths"])


def test_memory_state_export_is_detached_from_live_buffers():
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))

    model.clear_memory_banks()
    model.memorize(tokens)
    state = model.memory_state_dict()
    saved_tokens = state["banks"][0]["tokens"].clone()
    saved_keys = state["banks"][0]["keys"].clone()
    saved_strengths = state["banks"][0]["strengths"].clone()

    model.clear_memory_banks()

    assert torch.allclose(state["banks"][0]["tokens"], saved_tokens)
    assert torch.allclose(state["banks"][0]["keys"], saved_keys)
    assert torch.allclose(state["banks"][0]["strengths"], saved_strengths)


def test_memory_state_file_round_trip(tmp_path):
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))
    path = tmp_path / "memory.pt"

    model.memorize(tokens)
    model.save_memory_state(path)
    model.clear_memory_banks()
    model.load_memory_state(path)

    state = model.memory_state_dict()
    assert any(bank["num_slots"] > 0 for bank in state["banks"])


def test_memory_metadata_tracks_chat_roles_and_next_tokens():
    model = build_tiny_model()
    model._special_token_ids = {
        "user_start": 120,
        "user_end": 121,
        "assistant_start": 122,
        "assistant_end": 123,
    }
    tokens = torch.tensor([[120, 5, 6, 121, 122, 7, 8, 123]], dtype=torch.long)

    metadata = model._derive_memory_metadata(tokens)

    assert torch.equal(metadata["role_ids"][0], torch.tensor([0, 1, 1, 0, 0, 2, 2, 0]))
    assert torch.equal(metadata["group_ids"][0], torch.tensor([-1, 0, 0, -1, -1, 1, 1, -1]))
    assert torch.equal(metadata["next_token_ids"][0], torch.tensor([-1, 6, -1, -1, -1, 8, -1, -1]))


def test_memory_override_backprop_reaches_controller():
    model = enable_memory(build_tiny_model())
    context = torch.randint(0, model.config.vocab_size, (1, 10))
    target = torch.randint(0, model.config.vocab_size, (1, 10))

    memory_override = model.build_memory_state(context)
    outputs = model(target[:, :-1], target[:, 1:], memory_override=memory_override, return_components=True)
    loss = outputs["loss"] + 0.1 * F.cross_entropy(
        outputs["memory_probe_logits"].reshape(-1, outputs["memory_probe_logits"].size(-1)),
        target[:, 1:].reshape(-1),
        ignore_index=-1,
    )
    loss.backward()

    assert model.episodic_controller.write_key_proj.weight.grad is not None
    assert model.episodic_controller.write_surprise_gate.weight.grad is not None
    assert model.episodic_controller.no_write_bias.grad is not None
    assert model.episodic_controller.write_value_proj.in_proj.weight.grad is not None
    assert model.episodic_controller.recall_query_proj.weight.grad is not None
    assert model.episodic_controller.write_key_proj.weight.grad.abs().sum().item() > 0


def test_forward_without_memory_runs():
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    logits = model(tokens)
    assert logits.shape == (1, 8, model.config.vocab_size)


def test_write_salience_loss_reaches_write_gate():
    model = build_tiny_model()
    context = torch.randint(0, model.config.vocab_size, (1, 10))
    block_ios = model.collect_block_ios(context, detach=False)
    key_source, _ = block_ios[0]
    scores = model.transformer.h[0].compute_write_salience(key_source, model.episodic_controller)
    loss = F.cross_entropy(scores, torch.tensor([3], dtype=torch.long))
    loss.backward()

    gate_grad = model.episodic_controller.write_gate.weight.grad
    assert gate_grad is not None
    assert gate_grad.abs().sum().item() > 0


def test_retrieve_slots_returns_latent_memory_tokens():
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))
    model.clear_memory_banks()
    model.memorize(tokens)

    block = model.transformer.h[0]
    query = torch.randn(1, 3, model.config.episodic_dim)
    slots = block.episodic_memory.retrieve_slots(query)

    assert slots["keys"].shape[:3] == (1, 3, model.config.episodic_top_k)
    assert slots["values"].shape[-1] == model.config.n_embd


def test_top_memory_read_is_finite_and_decodable():
    model = enable_memory(build_tiny_model())
    context = torch.randint(0, model.config.vocab_size, (1, 10))
    inputs = torch.randint(0, model.config.vocab_size, (1, 8))
    memory_override = model.build_memory_state(context)

    outputs = model(inputs, memory_override=memory_override, return_components=True)
    assert torch.isfinite(outputs["memory_read"]).all()
    assert torch.isfinite(outputs["memory_probe_logits"]).all()


def test_collect_memory_diagnostics_reports_layerwise_activity():
    model = build_tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))
    model.clear_memory_banks()
    model.memorize(tokens[:, :8])

    diags = model.collect_memory_diagnostics(tokens[:, 8:])

    assert len(diags) == model.config.n_layer
    assert diags[0]["layer"] == 0
    assert "attn_norm" in diags[0]
    assert "mem_norm" in diags[0]
    assert "mem_attention" in diags[0]
    assert "slot_strength_mean" in diags[0]


def test_write_recall_trace_populates_memory():
    model = build_tiny_model()
    episode = torch.randint(0, model.config.vocab_size, (1, 12))

    model.clear_memory_banks()
    model.write_recall_trace(episode, key_positions=[2, 3, 4], value_positions=[7, 8, 9], reward=0.2)
    state = model.memory_state_dict()

    assert any(bank["num_slots"] > 0 for bank in state["banks"])


def test_competitive_write_state_uses_multiple_slots_with_finite_strengths():
    model = build_tiny_model()
    context = torch.randint(0, model.config.vocab_size, (1, 12))

    state = model.build_memory_state(context)
    strengths = state[0]["strengths"][0]

    assert torch.isfinite(strengths).all()
    assert int((strengths > 1e-6).sum().item()) > 1


def test_sparse_write_budget_limits_active_slots():
    model = build_tiny_model()
    context = torch.randint(0, model.config.vocab_size, (1, 12))

    state = model.build_memory_state(context)
    active = int((state[0]["strengths"][0] > 1e-6).sum().item())
    max_active = model.episodic_controller.summary_budget + model.episodic_controller.anchor_budget

    assert active <= max_active


def test_fixed_size_update_can_decay_stale_slots():
    model = build_tiny_model()
    block = model.transformer.h[0]
    current = block.episodic_memory.build_state(
        torch.ones(1, model.config.episodic_slots, model.config.n_embd),
        torch.ones(1, model.config.episodic_slots, model.config.episodic_dim),
        torch.ones(1, model.config.episodic_slots),
    )
    write = block.episodic_memory.empty_state(batch_size=1)
    updated = block.episodic_memory.updated_state(current, write)

    assert torch.all(updated["strengths"] < current["strengths"])
    assert torch.allclose(updated["strengths"], current["strengths"] * block.episodic_memory.decay)


def test_turn_replay_build_matches_live_online_writes():
    model = build_tiny_model()
    model._special_token_ids = {
        "user_start": 120,
        "user_end": 121,
        "assistant_start": 122,
        "assistant_end": 123,
    }
    conversation = torch.tensor(
        [[
            model.config.vocab_size - 1,
            120, 5, 6, 121, 122, 7, 8, 123,
            120, 9, 10, 121, 122, 11, 12, 123,
        ]],
        dtype=torch.long,
    )

    expected_state = model.build_memory_state(conversation, write_mode="turn")
    model.clear_memory_banks()
    model.replay_memory_sequence(conversation, write_mode="turn")
    live_state = model.memory_state_dict()

    for layer_idx in range(model.config.n_layer):
        assert torch.allclose(expected_state[layer_idx]["tokens"][0].cpu(), live_state["banks"][layer_idx]["tokens"])
        assert torch.allclose(expected_state[layer_idx]["keys"][0].cpu(), live_state["banks"][layer_idx]["keys"])
        assert torch.allclose(expected_state[layer_idx]["strengths"][0].cpu(), live_state["banks"][layer_idx]["strengths"])


def test_recall_mask_anchors_to_prefix_before_current_answer():
    model = build_tiny_model()
    model._special_token_ids = {
        "user_start": 120,
        "user_end": 121,
        "assistant_start": 122,
        "assistant_end": 123,
    }
    tokens = torch.tensor([[1, 120, 5, 121, 122, 7, 8, 9]], dtype=torch.long)

    mask = model._build_recall_mask(tokens)

    assert torch.equal(mask[0], torch.tensor([True, True, True, True, True, False, False, False]))
