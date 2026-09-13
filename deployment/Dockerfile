# Production image for Northflank — no PyTorch, no CUDA, minimal footprint.
#
# IMPORTANT: build with the PROJECT ROOT as context, not this folder, e.g.:
#     docker build -f deployment/Dockerfile -t regibert-backend .
# (run from RegiBERT_French_Register_Classification/, not from deployment/)

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements-inference.txt ./deployment/requirements-inference.txt
RUN pip install --no-cache-dir -r deployment/requirements-inference.txt

# Preserve the repo's nesting: config.py and backend/ at root, deployment/
# (with the ONNX model and server) one level below — this matters because
# server_onnx.py resolves paths relative to its own location.
COPY config.py .
COPY backend/ ./backend/
COPY deployment/ ./deployment/

# Large binary files (ONNX model + embeddings) are NOT stored in git — they
# exceed GitHub's 100MB limit. They're downloaded here instead, from a
# Hugging Face Hub repo, so the git repo only ever holds code.
# CHANGE THIS to match the REPO_ID used in upload_to_hf.py
ARG HF_REPO_ID=aogunleye/regibert-deployment
RUN mkdir -p deployment/checkpoints && \
    curl -L -o deployment/checkpoints/regibert_int8.onnx \
        "https://huggingface.co/${HF_REPO_ID}/resolve/main/regibert_int8.onnx" && \
    curl -L -o deployment/checkpoints/static_embeddings.npy \
        "https://huggingface.co/${HF_REPO_ID}/resolve/main/static_embeddings.npy" && \
    curl -L -o deployment/checkpoints/static_projections.npy \
        "https://huggingface.co/${HF_REPO_ID}/resolve/main/static_projections.npy" && \
    curl -L -o deployment/checkpoints/static_targets.npy \
        "https://huggingface.co/${HF_REPO_ID}/resolve/main/static_targets.npy"

# The UMAP calibration file lives in the root checkpoints/ folder in the
# repo (produced by src/fit_umap.py), separate from deployment/checkpoints/
# which only holds the exported ONNX model.
COPY checkpoints/umap_3d.joblib ./checkpoints/umap_3d.joblib

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.server_onnx:app --host 0.0.0.0 --port ${PORT}"]