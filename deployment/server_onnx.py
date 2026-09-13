"""
ONNX-based rewrite of backend/server.py, for lightweight deployment (no
PyTorch, no CUDA — suitable for Northflank's free tier).

Key differences from backend/server.py:
  - No torch. Inference runs through onnxruntime via RegiBERTONNX
    (see inference_onnx.py).
  - The redundant second forward pass (output_hidden_states=True to fetch
    hidden_states[best_layer]) has been removed: model.forward()'s own
    `embeddings` output already IS the mean-pooled last layer, which is
    hidden_states[12] — and best_layer was found to be 12. If you retrain
    and best_layer changes, this shortcut breaks; see the warning in
    inference_onnx.py's predict_with_embedding().

Expected file layout at runtime (see Dockerfile for how this is assembled):
  /app/config.py
  /app/backend/__init__.py, /app/backend/stream.py
  /app/server_onnx.py         (this file)
  /app/inference_onnx.py
  /app/checkpoints/regibert.onnx
  /app/checkpoints/umap_3d.joblib
Run in production via (from the project root, or see Dockerfile):
    uvicorn deployment.server_onnx:app --host 0.0.0.0 --port 8000
(requires deployment/__init__.py to exist, since this file uses a relative
import for inference_onnx — running "python server_onnx.py" directly will
NOT work; always launch it as a package through uvicorn as shown above.)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
import gc
sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from backend.stream import BlueskyStreamer
from .inference_onnx import RegiBERTONNX

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
    print("Démarrage du serveur (ONNX Runtime, CPU) ...")

    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    
    reducer_path = checkpoints_dir / "umap_reducer.joblib"
    proj_path = checkpoints_dir / "static_projections.npy"
    targets_path = checkpoints_dir / "static_targets.npy"

    if not reducer_path.exists() or not proj_path.exists():
        raise FileNotFoundError(f"Fichiers légers UMAP introuvables dans {checkpoints_dir}")

    reducer_data = joblib.load(reducer_path)
    reducer = reducer_data["umap_model"] if isinstance(reducer_data, dict) else reducer_data

    static_projections = np.load(proj_path)
    targets = np.load(targets_path) if targets_path.exists() else []

    static_points = []
    for i in range(len(static_projections)):
        x = float(np.nan_to_num(static_projections[i][0]))
        y = float(np.nan_to_num(static_projections[i][1]))
        z = float(np.nan_to_num(static_projections[i][2]))
        
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

    del static_projections
    del targets
    gc.collect()

    onnx_path = checkpoints_dir / "regibert_int8.onnx"
    tokenizer_dir = checkpoints_dir / "tokenizer"

    if not onnx_path.exists():
        onnx_path = checkpoints_dir / "regibert.onnx"

    engine = RegiBERTONNX(
        onnx_path=str(onnx_path),
        tokenizer_dir=str(tokenizer_dir) if tokenizer_dir.exists() else "camembert-base"
    )

    MODEL_STATE.update({
        "engine": engine,
        "reducer": reducer,
        "static_projections": static_points,
    })

    streamer = BlueskyStreamer(output_queue=post_queue, interval_seconds=10.0)
    asyncio.create_task(streamer.start())
    asyncio.create_task(inference_worker())

    yield

    print("Shutting down server and freeing up resources...")
    MODEL_STATE.clear()


app = FastAPI(
    title="RegiBERT 3D",
    description="fastapi server for live inference and 3D visualization of Bluesky posts in french (ONNX Runtime)",
    lifespan=lifespan,
)

# TODO: restrict to your actual frontend origin(s) before/after going live,
# e.g. ["https://your-username.github.io"], instead of "*".
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def inference_worker():
    while True:
        item = await post_queue.get()
        text = item["text"]

        engine = MODEL_STATE["engine"]
        reducer = MODEL_STATE["reducer"]

        probs, pooled_embedding = engine.predict_with_embedding(text)

        coords_3d = reducer.transform(pooled_embedding.reshape(1, -1))[0].tolist()
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