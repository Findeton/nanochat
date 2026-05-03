"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""

import argparse
import os

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from nanochat.episodic_memory import get_session_memory_path


parser = argparse.ArgumentParser(description="Chat with the model")
parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--prompt", type=str, default="", help="Prompt the model, get a single response back")
parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Temperature for generation")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
parser.add_argument("--session-id", type=str, default="", help="Optional persistent episodic-memory session id")
parser.add_argument("--forget-session", action="store_true", help="Delete any persisted episodic memory for this session before starting")
parser.add_argument("--write-mode", type=str, default="turn", choices=["none", "user", "turn"], help="how to write live session memory; use none for read-only recall probes")
parser.add_argument("--reconsolidate-recall", action="store_true", help="After each assistant reply, write a latent retrieval trace keyed by the user query and valued by the assistant answer")
parser.add_argument("--reconsolidate-reward", type=float, default=0.15, help="Reward/strength assigned to reconsolidated retrieval traces")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type for evaluation: cuda|cpu|mps. empty => autodetect")
args = parser.parse_args()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

loaded_model_tag = meta.get("_model_tag", args.model_tag or "auto")
session_path = None
if args.session_id:
    session_path = get_session_memory_path(args.source, loaded_model_tag, args.session_id)
    if args.forget_session and os.path.exists(session_path):
        os.remove(session_path)
    model.clear_memory_banks()
    if os.path.exists(session_path):
        model.load_memory_state(session_path, strict=False)

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
if session_path:
    print("Type 'forget' to clear persisted episodic memory for this session")
print("-" * 50)

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if session_path and user_input.lower() == "forget":
        conversation_tokens = [bos]
        model.clear_memory_banks()
        if os.path.exists(session_path):
            os.remove(session_path)
        print("Session memory cleared.")
        continue

    if not user_input:
        continue

    conversation_tokens.append(user_start)
    user_content_tokens = tokenizer.encode(user_input)
    conversation_tokens.extend(user_content_tokens)
    conversation_tokens.append(user_end)
    if session_path and args.write_mode == "user":
        # Match the training-time memory format from tokenizer.render_conversation:
        # each remembered episode starts with BOS, followed by the rendered user turn.
        user_turn_tokens = [bos, user_start]
        user_turn_tokens.extend(user_content_tokens)
        user_turn_tokens.append(user_end)
        model.memorize(user_turn_tokens)
        model.save_memory_state(session_path)
    conversation_tokens.append(assistant_start)

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    # Persistent episodic memory is not compatible with KV-cache decoding yet.
    # Be conservative for session-backed chats: even if a freshly loaded bank has
    # tiny strengths, memory slots may still be active and must use the no-KV path.
    generator = engine._generate_without_kv_cache if session_path else engine.generate
    for token_column, token_masks in generator(conversation_tokens, **generate_kwargs):
        token = token_column[0]
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()

    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
    answer_content_tokens = response_tokens[:-1] if response_tokens and response_tokens[-1] == assistant_end else response_tokens

    if session_path and args.write_mode == "turn" and answer_content_tokens:
        turn_tokens = [bos, user_start]
        turn_tokens.extend(user_content_tokens)
        turn_tokens.append(user_end)
        turn_tokens.append(assistant_start)
        turn_tokens.extend(answer_content_tokens)
        turn_tokens.append(assistant_end)
        model.memorize(turn_tokens)
        model.save_memory_state(session_path)

    if session_path and args.reconsolidate_recall:
        if answer_content_tokens:
            recall_tokens = [bos, user_start]
            recall_tokens.extend(user_content_tokens)
            recall_tokens.append(user_end)
            recall_tokens.append(assistant_start)
            answer_start = len(recall_tokens)
            recall_tokens.extend(answer_content_tokens)
            recall_tokens.append(assistant_end)
            query_positions = list(range(2, 2 + len(user_content_tokens)))
            value_positions = list(range(answer_start, answer_start + len(answer_content_tokens)))
            model.write_recall_trace(recall_tokens, query_positions, value_positions, reward=args.reconsolidate_reward)
            model.save_memory_state(session_path)

    if args.prompt:
        break
