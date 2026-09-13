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

# Téléchargement propre de tous les fichiers (modèle + projections + tokenizer)
ARG HF_REPO_ID=aogunleye/regibert-deployment
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${HF_REPO_ID}', local_dir='deployment/checkpoints')"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.server_onnx:app --host 0.0.0.0 --port ${PORT}"]