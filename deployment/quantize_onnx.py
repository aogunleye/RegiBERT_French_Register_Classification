from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    input_path = checkpoints_dir / "regibert.onnx"
    output_path = checkpoints_dir / "regibert_int8.onnx"

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    before = input_path.stat().st_size / (1024 * 1024)
    after = output_path.stat().st_size / (1024 * 1024)
    print(f"Original size: {before:.1f} MB")
    print(f"Quantized size: {after:.1f} MB")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()