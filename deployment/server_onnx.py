import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.stream import BlueskyStreamer
from .inference_onnx import RegiBERTONNX
from .knn_transform import NeighborProjector

post_queue = asyncio.Queue(maxsize=50)
MODEL_STATE = {}
CONNECTED_CLIENTS = set()

COLOR_SOUTENU = (220, 38, 38)
COLOR_COURANT = (37, 99, 235)
COLOR_FAMILIER = (16, 185, 129)


def compute_rgb(probs: dict) -> list:
    p_s = probs.get("soutenu", 0.0)
    p_c = probs.get("courant", 0.0)
    p_f = probs.get("familier", 0.0)

    r = int(p_s * COLOR_SOUTENU[0] + p_c * COLOR_COURANT[0] + p_f * COLOR_FAMILIER[0])
    g = int(p_s * COLOR_SOUTENU[1] + p_c * COLOR_COURANT[1] + p_f * COLOR_FAMILIER[1])
    b = int(p_s * COLOR_SOUTENU[2] + p_c * COLOR_COURANT[2] + p_f * COLOR_FAMILIER[2])
    return [r, g, b]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LOG: Starting server lifespan...", flush=True)

    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    
    print("LOG: Loading numpy targets and projections...", flush=True)
    targets = np.load(checkpoints_dir / "static_targets.npy")
    projections = np.load(checkpoints_dir / "static_projections.npy")

    print("LOG: Building static points array...", flush=True)
    static_points = []
    for i in range(len(projections)):
        x = float(np.nan_to_num(projections[i][0]))
        y = float(np.nan_to_num(projections[i][1]))
        z = float(np.nan_to_num(projections[i][2]))

        if len(targets) > i:
            target_val = targets[i]
            if isinstance(target_val, np.ndarray) and len(target_val) >= 3:
                p_s, p_c, p_f = float(target_val[0]), float(target_val[1]), float(target_val[2])
            else:
                idx = int(target_val)
                p_s = 1.0 if idx == 0 else 0.0
                p_c = 1.0 if idx == 1 else 0.0
                p_f = 1.0 if idx == 2 else 0.0
        else:
            p_s, p_c, p_f = 0.5, 0.5, 0.5

        r = (p_s * 0.862) + (p_c * 0.145) + (p_f * 0.063)
        g = (p_s * 0.149) + (p_c * 0.388) + (p_f * 0.725)
        b = (p_s * 0.149) + (p_c * 0.922) + (p_f * 0.506)

        static_points.append([x, y, z, r, g, b])

    print("LOG: Initializing ONNX Engine...", flush=True)
    tokenizer_dir = checkpoints_dir / "tokenizer"
    engine = RegiBERTONNX(
        model_path=str(checkpoints_dir / "regibert_int8.onnx"),
        tokenizer_name_or_path=str(tokenizer_dir) if tokenizer_dir.exists() else "camembert-base"
    )

    print("LOG: Initializing NeighborProjector (k-NN)...", flush=True)
    projector = NeighborProjector(checkpoints_dir, k=15)

    MODEL_STATE.update({
        "engine": engine,
        "projector": projector,
        "static_projections": static_points,
    })

    print("LOG: Starting background Bluesky streamer...", flush=True)
    streamer = BlueskyStreamer(output_queue=post_queue, interval_seconds=10.0)
    asyncio.create_task(streamer.start())
    asyncio.create_task(inference_worker())

    print("LOG: Server startup complete and operational!", flush=True)
    yield

    print("Shutting down server...")
    MODEL_STATE.clear()


app = FastAPI(
    title="RegiBERT 3D",
    description="fastapi server for live inference and 3D visualization of Bluesky posts in french (ONNX Runtime)",
    lifespan=lifespan,
)

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RegiBERT 3D ONNX Server is running"}

# TODO: restrict to your actual frontend origin(s), instead of "*".
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def inference_worker():
    while True:
        item = await post_queue.get()
        text = item["text"]

        engine = MODEL_STATE["engine"]
        projector = MODEL_STATE["projector"]

        probs, pooled_embedding = engine.predict_with_embedding(text)
        coords_3d = projector.transform(pooled_embedding)
        rgb = compute_rgb(probs)

        post_url = f"https://bsky.app/profile/{item.get('author')}/post/{item.get('rkey')}"

        payload = {
            "type": "new_post",
            "id": item["id"],
            "author": item.get("author", ""),
            "created_at": item.get("created_at", ""),
            "text": text,
            "coords": coords_3d,
            "probs": probs,
            "rgb": rgb,
            "url": post_url,
        }

        if CONNECTED_CLIENTS:
            await asyncio.gather(*[ws.send_json(payload) for ws in CONNECTED_CLIENTS])

        post_queue.task_done()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    CONNECTED_CLIENTS.add(websocket)
    print(f"New WS client connected. Total: {len(CONNECTED_CLIENTS)}")

    try:
        await websocket.send_json({
            "type": "init_background",
            "points": MODEL_STATE["static_projections"],
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"WS client disconnected. Remaining: {len(CONNECTED_CLIENTS)}")