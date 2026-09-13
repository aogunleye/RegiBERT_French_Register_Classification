"""
Sanity check: compares PyTorch and ONNX Runtime outputs on the same inputs
to confirm the export is numerically correct before deploying.

Run from the project root, after export_to_onnx.py:
    python verify_onnx.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.model import RegiBERT


def main():
    device = torch.device("cpu")

    # --- PyTorch reference ---
    model = RegiBERT()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test_sentences = [
        "Salut ça va ou quoi frérot ?",
        "Bonjour, je vous prie de bien vouloir excuser mon retard.",
        "Il fait beau aujourd'hui à Paris.",
    ]

    for text in test_sentences:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            torch_log_probs, _ = model(input_ids, attention_mask)
        torch_probs = torch.exp(torch_log_probs).numpy()

        # --- ONNX Runtime ---
        onnx_path = Path(__file__).resolve().parent / "checkpoints" / "regibert.onnx"
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            },
        )
        onnx_log_probs = onnx_outputs[0]
        onnx_probs = np.exp(onnx_log_probs)

        max_diff = np.abs(torch_probs - onnx_probs).max()
        print(f"\nText: {text}")
        print(f"  PyTorch probs: {torch_probs.round(4)}")
        print(f"  ONNX probs:    {onnx_probs.round(4)}")
        print(f"  Max abs diff:  {max_diff:.6f}  {'OK' if max_diff < 1e-4 else 'MISMATCH — investigate'}")


if __name__ == "__main__":
    main()