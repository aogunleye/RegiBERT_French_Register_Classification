"""
Drop-in ONNX inference wrapper for the FastAPI real-time pipeline.

In your existing FastAPI app (the file that currently loads the PyTorch
model and calls it on each incoming Bluesky post), replace:

    model = TremoloClassifier()
    model.load_state_dict(torch.load(...))
    ...
    logprobs, _ = model(input_ids, attention_mask)

with:

    from inference_onnx import RegiBERTONNX
    engine = RegiBERTONNX()   # load once, at startup
    ...
    probs = engine.predict(text)   # {"soutenu": .., "courant": .., "familier": ..}
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_PATH = str(Path(__file__).resolve().parent / "checkpoints" / "regibert.onnx")
# use "regibert_int8.onnx" instead if you ran quantize_onnx.py
TOKENIZER_NAME = "camembert-base"
MAX_LENGTH = 128


class RegiBERTONNX:
    def __init__(self, model_path: str = MODEL_PATH, tokenizer_name_or_path: str = TOKENIZER_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

    def predict(self, text: str) -> dict:
        probs, _ = self.predict_with_embedding(text)
        return probs

    def predict_with_embedding(self, text: str):
        """
        Returns (probs: dict, pooled_embedding: np.ndarray of shape (768,)).
        The embedding is the mean-pooled LAST hidden layer of camembert-base
        (exactly what model.py's forward() returns as `embeddings`).

        IMPORTANT: this equals hidden_states[12] only because probing found
        best_layer = 12 (the final layer) for this trained model. If you
        retrain and best_layer changes, this export no longer matches the
        layer used for UMAP calibration — re-check probe_layers.py's result
        and adjust model.py / re-export accordingly.
        """
        encoding = self.tokenizer(
            text, return_tensors="np", truncation=True, padding="max_length", max_length=MAX_LENGTH
        )
        outputs = self.session.run(
            None,
            {
                "input_ids": encoding["input_ids"].astype(np.int64),
                "attention_mask": encoding["attention_mask"].astype(np.int64),
            },
        )
        log_probs = outputs[0][0]          # shape (3,)
        pooled_embedding = outputs[1][0]   # shape (768,)
        probs_arr = np.exp(log_probs)

        # Order confirmed against dataset.py's label_cols = ["Soutenu", "Courant", "Familier"]
        probs = {
            "soutenu": float(probs_arr[0]),
            "courant": float(probs_arr[1]),
            "familier": float(probs_arr[2]),
        }
        return probs, pooled_embedding


if __name__ == "__main__":
    engine = RegiBERTONNX()
    probs, embedding = engine.predict_with_embedding("Salut ça va ou quoi frérot ?")
    print(probs)
    print(f"Embedding shape: {embedding.shape}")