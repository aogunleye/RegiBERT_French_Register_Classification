import sys
from pathlib import Path
from contextlib import asynccontextmanager
sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
import torch
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import config
from src.model import RegiBERT
from backend.stream import BlueskyStreamer

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

def mean_pooling_layer(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).float().cpu().numpy()

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = config.DEVICE
    print(f"🚀 Démarrage du serveur sur : {device}")

    umap_path = Path(config.UMAP_SAVE_PATH)
    if not umap_path.exists():
        raise FileNotFoundError(f"UMAP file not found: {umap_path}")

    umap_data = joblib.load(umap_path)
    reducer = umap_data["umap_model"]
    static_projections = umap_data["projections"]
    targets = umap_data.get("targets", [])
    best_layer = umap_data.get("best_layer", 12)

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

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = RegiBERT()
    model_path = Path(config.MODEL_SAVE_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    MODEL_STATE.update({
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "reducer": reducer,
        "best_layer": best_layer,
        "static_projections": static_points
    })

    streamer = BlueskyStreamer(output_queue=post_queue, interval_seconds=10.0)
    asyncio.create_task(streamer.start())
    asyncio.create_task(inference_worker())

    yield

    print("Shutting down server and freeing up resources...")
    MODEL_STATE.clear()


app = FastAPI(
    title="RegiBERT 3D", 
    description="fastapi server for live inference and 3D visualization of Bluesky posts in french",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], )

async def inference_worker():
    while True:
        item = await post_queue.get()
        text = item["text"]

        device = MODEL_STATE["device"]
        tokenizer = MODEL_STATE["tokenizer"]
        model = MODEL_STATE["model"]
        reducer = MODEL_STATE["reducer"]
        best_layer = MODEL_STATE["best_layer"]

        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.USE_AMP):
                logprobs, _ = model(input_ids, attention_mask)
                
                outputs = model.camembert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                layer_state = outputs.hidden_states[best_layer]

            probs_tensor = torch.exp(logprobs).squeeze(0).cpu()

            pooled_embedding = mean_pooling_layer(layer_state, attention_mask)

        coords_3d = reducer.transform(pooled_embedding)[0].tolist()

        probs = {"soutenu": float(probs_tensor[0]), "courant": float(probs_tensor[1]), "familier": float(probs_tensor[2])}

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
            "url": post_url
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
            "points": MODEL_STATE["static_projections"]
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"WS client disconnected. Remaining: {len(CONNECTED_CLIENTS)}")