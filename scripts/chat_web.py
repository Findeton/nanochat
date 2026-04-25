#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.
"""

import argparse
import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.episodic_memory import get_session_memory_path


MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description="NanoChat Web Server")
parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Default temperature for generation")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Default top-k sampling parameter")
parser.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max tokens for generation")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument("--write-mode", type=str, default="turn", choices=["user", "turn"], help="how to write live session memory")
parser.add_argument("--reconsolidate-recall", action="store_true", help="After each assistant reply, write a latent retrieval trace keyed by the user query and valued by the assistant answer")
parser.add_argument("--reconsolidate-reward", type=float, default=0.15, help="Reward/strength assigned to reconsolidated retrieval traces")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type for evaluation: cuda|cpu|mps. empty => autodetect")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    model_tag: str


class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type)
                print(f"Loading model on {device_type}...")

            model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                model_tag=meta.get("_model_tag", model_tag or "auto"),
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    session_id: Optional[str] = None


def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request")

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message")
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed")

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role. Must be 'user' or 'assistant'")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';",
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")


async def generate_stream(worker: Worker, tokens, temperature=None, max_new_tokens=None, top_k=None) -> AsyncGenerator[str, None]:
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith("�"):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)

    logger.info("=" * 20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-" * 20)

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        session_path = None
        worker.engine.model.clear_memory_banks()
        if request.session_id:
            session_path = get_session_memory_path(args.source, worker.model_tag, request.session_id)
            if os.path.exists(session_path):
                worker.engine.model.load_memory_state(session_path, strict=False)
        last_user = next((message for message in reversed(request.messages) if message.role == "user"), None)

        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)
        if session_path and last_user is not None and args.write_mode == "user":
            # Keep runtime writes aligned with the training distribution:
            # episodic memories are stored as BOS + rendered user turn.
            user_turn_tokens = [bos, user_start]
            user_turn_tokens.extend(worker.tokenizer.encode(last_user.content))
            user_turn_tokens.append(user_end)
            worker.engine.model.memorize(user_turn_tokens)
            worker.engine.model.save_memory_state(session_path)

        response_chunks = []

        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                ):
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_chunks.append(chunk_data["token"])
                    yield chunk
            finally:
                full_response = "".join(response_chunks)
                if session_path and last_user is not None and args.write_mode == "turn" and full_response:
                    turn_tokens = [bos, user_start]
                    turn_tokens.extend(worker.tokenizer.encode(last_user.content))
                    turn_tokens.append(user_end)
                    turn_tokens.append(assistant_start)
                    turn_tokens.extend(worker.tokenizer.encode(full_response))
                    turn_tokens.append(assistant_end)
                    worker.engine.model.memorize(turn_tokens)
                    worker.engine.model.save_memory_state(session_path)
                if session_path and last_user is not None and args.reconsolidate_recall and full_response:
                    user_content_tokens = worker.tokenizer.encode(last_user.content)
                    answer_content_tokens = worker.tokenizer.encode(full_response)
                    if user_content_tokens and answer_content_tokens:
                        recall_tokens = [bos, user_start]
                        recall_tokens.extend(user_content_tokens)
                        recall_tokens.append(user_end)
                        recall_tokens.append(assistant_start)
                        answer_start = len(recall_tokens)
                        recall_tokens.extend(answer_content_tokens)
                        recall_tokens.append(assistant_end)
                        query_positions = list(range(2, 2 + len(user_content_tokens)))
                        value_positions = list(range(answer_start, answer_start + len(answer_content_tokens)))
                        worker.engine.model.write_recall_trace(
                            recall_tokens,
                            query_positions,
                            value_positions,
                            reward=args.reconsolidate_reward,
                        )
                        worker.engine.model.save_memory_state(session_path)
                worker.engine.model.clear_memory_banks()
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("=" * 20)
                await worker_pool.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")
    except Exception as e:
        worker.engine.model.clear_memory_banks()
        await worker_pool.release_worker(worker)
        raise e


@app.get("/health")
async def health():
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
    }


@app.get("/stats")
async def stats():
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in worker_pool.workers],
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
