FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements-inference.txt ./deployment/requirements-inference.txt
RUN pip install --no-cache-dir -r deployment/requirements-inference.txt

COPY config.py .
COPY backend/ ./backend/
COPY deployment/ ./deployment/

COPY deployment/checkpoints/umap_3d.joblib ./checkpoints/umap_3d.joblib

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.server_onnx:app --host 0.0.0.0 --port ${PORT}"]