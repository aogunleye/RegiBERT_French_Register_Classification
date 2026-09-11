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
from src.model import TremoloClassifier
from backend.stream import BlueskyStreamer

post_queue = asyncio.Queue(maxsize=50)  # Queue pour stocker les posts à traiter
MODEL_STATE = {}
CONNECTED_CLIENTS = set()

# rgb colors for the three classes
COLOR_SOUTENU = (220, 38, 38)    # red (#DC2626)
COLOR_COURANT = (37, 99, 235)    # blue (#2563EB)
COLOR_FAMILIER = (16, 185, 129)  # green (#10B981)

def compute_rgb(probs: dict) -> list:
    """calculates the RGB color based on the probabilities of the three classes"""
    p_s = probs.get("soutenu", 0.0)
    p_c = probs.get("courant", 0.0)
    p_f = probs.get("familier", 0.0)

    r = int(p_s * COLOR_SOUTENU[0] + p_c * COLOR_COURANT[0] + p_f * COLOR_FAMILIER[0])
    g = int(p_s * COLOR_SOUTENU[1] + p_c * COLOR_COURANT[1] + p_f * COLOR_FAMILIER[1])
    b = int(p_s * COLOR_SOUTENU[2] + p_c * COLOR_COURANT[2] + p_f * COLOR_FAMILIER[2])
    return [r, g, b]

def mean_pooling_layer(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """ajusting the mean pooling to handle attention masks correctly"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).float().cpu().numpy()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP : Chargement unique des artefacts au démarrage ---
    device = config.DEVICE
    print(f"🚀 Démarrage du serveur sur : {device}")

    # 1. Chargement de la projection UMAP et du nuage statique
    umap_path = Path(config.UMAP_SAVE_PATH)
    if not umap_path.exists():
        raise FileNotFoundError(f"Fichier UMAP introuvable : {umap_path}")

    umap_data = joblib.load(umap_path)
    reducer = umap_data["umap_model"]
    static_projections = umap_data["projections"]
    targets = umap_data.get("targets", [])
    best_layer = umap_data.get("best_layer", 12)

    # 2. Préparation des 45k points avec leurs VRAIES couleurs [x, y, z, r, g, b]
    static_points = []
    for i in range(len(static_projections)):
        x = float(np.nan_to_num(static_projections[i][0]))
        y = float(np.nan_to_num(static_projections[i][1]))
        z = float(np.nan_to_num(static_projections[i][2]))
        
        # Récupération des probabilités d'origine pour colorer fidèlement
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
            p_s, p_c, p_f = 0.5, 0.5, 0.5  # Gris par défaut si erreur
            
        # Normalisation RGB entre 0 et 1 pour Three.js
        r = (p_s * 0.862) + (p_c * 0.145) + (p_f * 0.063)
        g = (p_s * 0.149) + (p_c * 0.388) + (p_f * 0.725)
        b = (p_s * 0.149) + (p_c * 0.922) + (p_f * 0.506)
        
        static_points.append([x, y, z, r, g, b])

    # 3. Chargement du Tokenizer et du Modèle TremoloClassifier
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = TremoloClassifier()
    model_path = Path(config.MODEL_SAVE_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Poids du modèle introuvables : {model_path}")

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

    # 4. Lancement des tâches asynchrones en arrière-plan
    streamer = BlueskyStreamer(output_queue=post_queue, interval_seconds=10.0)
    asyncio.create_task(streamer.start())
    asyncio.create_task(inference_worker())

    yield  # L'application FastAPI fonctionne ici

    # --- SHUTDOWN : Nettoyage des ressources lors de l'arrêt ---
    print("🛑 Fermeture du serveur et libération des ressources...")
    MODEL_STATE.clear()


app = FastAPI(
    title="Tremolo 3D - API Inférence & WebSocket", 
    description="Serveur FastAPI pour l'inférence temps réel et la visualisation 3D des posts Bluesky en français.",
    lifespan=lifespan
)

# Activation CORS pour la communication avec le frontend Three.js
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], )

async def inference_worker():
    """ does the inference on new posts from the queue and sends results to connected WebSocket clients """
    while True:
        item = await post_queue.get()
        text = item["text"]

        device = MODEL_STATE["device"]
        tokenizer = MODEL_STATE["tokenizer"]
        model = MODEL_STATE["model"]
        reducer = MODEL_STATE["reducer"]
        best_layer = MODEL_STATE["best_layer"]

        # Encodage du texte avec le Tokenizer CamemBERT
        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            # Pass en FP16 (AMP) aligné sur config.USE_AMP et config.DEVICE
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.USE_AMP):
                logprobs, _ = model(input_ids, attention_mask)
                
                # Extraction des hidden_states de la meilleure couche
                outputs = model.camembert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                layer_state = outputs.hidden_states[best_layer]

            probs_tensor = torch.exp(logprobs).squeeze(0).cpu()

            # Vectorisation 768D par Mean Pooling (converti en float32 pour UMAP)
            pooled_embedding = mean_pooling_layer(layer_state, attention_mask)

        # Projection instantanée 3D via UMAP
        coords_3d = reducer.transform(pooled_embedding)[0].tolist()

        probs = {"soutenu": float(probs_tensor[0]), "courant": float(probs_tensor[1]), "familier": float(probs_tensor[2])}

        rgb = compute_rgb(probs)

        # Construction de l'URL exacte vers le post
        post_url = f"https://bsky.app/profile/{item.get('author')}/post/{item.get('rkey')}"

        # Vers la fin de inference_worker() :
        payload = {
            "type": "new_post", 
            "id": item["id"],
            "author": item.get("author", ""),          # <-- NOUVEAU
            "created_at": item.get("created_at", ""),  # <-- NOUVEAU
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
    print(f"🔌 Nouveau client WS connecté. Total : {len(CONNECTED_CLIENTS)}")

    try:
        # Envoi initial du fond de carte (points statiques + couleurs réelles)
        await websocket.send_json({
            "type": "init_background",
            "points": MODEL_STATE["static_projections"]
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"❌ Client WS déconnecté. Restants : {len(CONNECTED_CLIENTS)}")