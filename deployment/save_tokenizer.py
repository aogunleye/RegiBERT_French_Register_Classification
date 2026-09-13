from pathlib import Path
from transformers import AutoTokenizer

tokenizer_dir = Path(__file__).resolve().parent / "checkpoints" / "tokenizer"
tokenizer_dir.mkdir(parents=True, exist_ok=True)

print("Chargement et sauvegarde du tokenizer camembert-base...")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
tokenizer.save_pretrained(tokenizer_dir)

print(f"Tokenizer sauvegardé dans : {tokenizer_dir}")