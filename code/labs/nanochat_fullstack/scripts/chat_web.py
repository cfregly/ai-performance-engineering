#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m labs.nanochat_fullstack.scripts.chat_web

- 4 GPUs
python -m labs.nanochat_fullstack.scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 1-200
  - Max tokens clamped to 1-4096
"""

from __future__ import annotations

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager, nullcontext, suppress
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--reuse-ids-buffer', action='store_true', default=False, help='Reuse GPU buffer for next-token IDs during inference (micro-optimization, default off to match original behavior)')
parser.add_argument('--enable-batch-decode', action='store_true', default=False, help='Enable padded attention + batched decode path')
parser.add_argument('--enable-dynamic-batching', action='store_true', default=False, help='Enable request queue + dynamic batching (experimental)')
parser.add_argument('--batch-size', type=int, default=4, help='Maximum requests to batch together when dynamic batching is enabled')
parser.add_argument('--batch-timeout-ms', type=int, default=10, help='Batch collection timeout in milliseconds for dynamic batching')
args = parser.parse_args()

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

@dataclass
class BatchItem:
    tokens: list
    request: ChatRequest
    queue: asyncio.Queue
    response_tokens: list

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(
                model,
                tokenizer,
                reuse_ids_buffer=args.reuse_ids_buffer,
                enable_batch_decode=args.enable_batch_decode or args.enable_dynamic_batching
            )
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def build_conversation_tokens(tokenizer, messages):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)
    return conversation_tokens

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    if args.enable_dynamic_batching:
        app.state.batch_queue = asyncio.Queue()
        app.state.batcher_task = asyncio.create_task(batching_loop(app))
    print(f"Server ready at http://localhost:{args.port}")
    yield
    if args.enable_dynamic_batching:
        app.state.batcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.batcher_task

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
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]

            # Stopping criteria
            if token == assistant_end or token == bos:
                break

            # Append the token to sequence
            accumulated_tokens.append(token)
            # Decode all accumulated tokens to get proper UTF-8 handling
            # Note that decode is a quite efficient operation, basically table lookup and string concat
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # Only emit text if it doesn't end with a replacement character
            # This ensures we don't emit incomplete UTF-8 sequences
            if not current_text.endswith('�'):
                # Extract only the new text since last clean decode
                new_text = current_text[len(last_clean_text):]
                if new_text:  # Only yield if there's new content
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

async def process_batch(app: FastAPI, worker: Worker, batch_items: List[BatchItem]):
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    queues = [item.queue for item in batch_items]
    accum_tokens = [[] for _ in batch_items]
    last_clean_text = [""] * len(batch_items)
    done = [False] * len(batch_items)
    response_tokens = [item.response_tokens for item in batch_items]

    temperatures = [
        item.request.temperature if item.request.temperature is not None else args.temperature
        for item in batch_items
    ]
    top_ks = [
        item.request.top_k if item.request.top_k is not None else args.top_k
        for item in batch_items
    ]
    max_tokens_list = [
        item.request.max_tokens if item.request.max_tokens is not None else args.max_tokens
        for item in batch_items
    ]
    tokens_batch = [item.tokens for item in batch_items]

    try:
        with worker.autocast_ctx:
            for token_column, token_masks in worker.engine.generate_batched(
                tokens_batch,
                max_tokens=max_tokens_list,
                temperature=temperatures,
                top_k=top_ks,
                seed=random.randint(0, 2**31 - 1)
            ):
                for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                    if mask == -1:
                        done[i] = True
                        continue
                    if done[i]:
                        continue
                    if token == assistant_end or token == bos:
                        done[i] = True
                        continue

                    accum_tokens[i].append(token)
                    current_text = worker.tokenizer.decode(accum_tokens[i])
                    if not current_text.endswith('�'):
                        new_text = current_text[len(last_clean_text[i]):]
                        if new_text:
                            response_tokens[i].append(new_text)
                            last_clean_text[i] = current_text
                            payload = json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)
                            await queues[i].put(f"data: {payload}\n\n")
                if all(done):
                    break
    except Exception as e:
        error_payload = json.dumps({'error': str(e)}, ensure_ascii=False)
        for q in queues:
            await q.put(f"data: {error_payload}\n\n")
    finally:
        for i, q in enumerate(queues):
            await q.put(f"data: {json.dumps({'done': True})}\n\n")
            await q.put(None)
            full_response = "".join(response_tokens[i])
            logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}) [batch {i+1}/{len(batch_items)}]: {full_response}")
        logger.info("="*20)
        await app.state.worker_pool.release_worker(worker)

async def batching_loop(app: FastAPI):
    while True:
        try:
            first_item = await app.state.batch_queue.get()
        except asyncio.CancelledError:
            break

        batch = [first_item]
        try:
            while len(batch) < args.batch_size:
                next_item = await asyncio.wait_for(app.state.batch_queue.get(), timeout=args.batch_timeout_ms / 1000)
                batch.append(next_item)
        except asyncio.TimeoutError:
            pass

        worker = await app.state.worker_pool.acquire_worker()
        asyncio.create_task(process_batch(app, worker, batch))

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    conversation_tokens = build_conversation_tokens(
        app.state.worker_pool.workers[0].tokenizer,
        request.messages
    )

    # Dynamic batching path queues the request instead of acquiring a worker immediately
    if args.enable_dynamic_batching:
        response_queue: asyncio.Queue = asyncio.Queue()
        batch_item = BatchItem(
            tokens=conversation_tokens,
            request=request,
            queue=response_queue,
            response_tokens=[]
        )
        await app.state.batch_queue.put(batch_item)

        async def stream_from_queue():
            while True:
                chunk = await response_queue.get()
                if chunk is None:
                    break
                yield chunk
            logger.info("="*20)

        return StreamingResponse(
            stream_from_queue(),
            media_type="text/event-stream"
        )

    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
