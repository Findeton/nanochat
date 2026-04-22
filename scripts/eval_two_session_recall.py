"""
Deterministic two-session recall evaluation for episodic-memory NanoChat.

This script:
1. loads a chat checkpoint
2. runs a first session that should write to episodic memory
3. saves memory state to disk
4. reloads the model fresh
5. asks a recall question from scratch in the same session
"""

import argparse
import os

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.episodic_memory import get_session_memory_path


def run_prompt(model, tokenizer, engine, user_text, max_tokens=128):
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
        temperature=0.0,
        top_k=1,
        seed=42,
    ):
        token = token_column[0]
        out.append(token)
        if token == assistant_end:
            break
    if out and out[-1] == assistant_end:
        out = out[:-1]
    return tokenizer.decode(out)


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-session recall")
    parser.add_argument("--source", type=str, default="sft")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--session-id", type=str, default="two-session-eval")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--write-prompt", type=str, required=True)
    parser.add_argument("--recall-prompt", type=str, required=True)
    parser.add_argument("--write-mode", type=str, default="user", choices=["user", "turn"], help="what gets written into episodic memory in the first session")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    session_path = get_session_memory_path(args.source, meta.get("_model_tag", args.model_tag), args.session_id)
    if os.path.exists(session_path):
        os.remove(session_path)

    response1 = run_prompt(model, tokenizer, engine, args.write_prompt)
    bos = tokenizer.get_bos_token_id()
    if args.write_mode == "user":
        memory_tokens = [bos, tokenizer.encode_special("<|user_start|>")]
        memory_tokens.extend(tokenizer.encode(args.write_prompt))
        memory_tokens.append(tokenizer.encode_special("<|user_end|>"))
    else:
        memory_tokens = [bos, tokenizer.encode_special("<|user_start|>")]
        memory_tokens.extend(tokenizer.encode(args.write_prompt))
        memory_tokens.append(tokenizer.encode_special("<|user_end|>"))
        memory_tokens.append(tokenizer.encode_special("<|assistant_start|>"))
        memory_tokens.extend(tokenizer.encode(response1))
        memory_tokens.append(tokenizer.encode_special("<|assistant_end|>"))
    model.memorize(memory_tokens)
    model.save_memory_state(session_path)

    model2, tokenizer2, meta2 = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model2.load_memory_state(session_path, strict=False)
    engine2 = Engine(model2, tokenizer2)
    response2 = run_prompt(model2, tokenizer2, engine2, args.recall_prompt)

    print("WRITE_PROMPT:", args.write_prompt)
    print("WRITE_RESPONSE:", response1)
    print("RECALL_PROMPT:", args.recall_prompt)
    print("RECALL_RESPONSE:", response2)
    print("SESSION_PATH:", session_path)
    print("LAYER0_SLOTS:", int(model2.transformer.h[0].episodic_memory.num_slots.item()))


if __name__ == "__main__":
    main()
