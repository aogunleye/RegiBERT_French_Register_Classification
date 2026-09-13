FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements-inference.txt ./deployment/requirements-inference.txt
RUN pip install --no-cache-dir -r deployment/requirements-inference.txt

COPY config.py .
COPY backend/ ./backend/
COPY deployment/ ./deployment/

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

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.server_onnx:app --host 0.0.0.0 --port ${PORT}"]