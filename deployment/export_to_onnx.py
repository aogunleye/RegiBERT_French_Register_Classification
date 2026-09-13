import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.model import RegiBERT


def main():
    device = torch.device("cpu")  # export sur CPU pour une meilleure portabilité

    print("Loading trained model...")
    model = RegiBERT()
    
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    if hasattr(model, "camembert"):
        model.camembert.config.attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    dummy_text = "Ceci est un exemple de tweet pour tracer le graphe ONNX."
    encoding = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    output_path = Path(__file__).resolve().parent / "checkpoints" / "regibert.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_path} ...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["log_probs", "pooled_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "log_probs": {0: "batch"},
            "pooled_embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("Export complete.")


if __name__ == "__main__":
    main()