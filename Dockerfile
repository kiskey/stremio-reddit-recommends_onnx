# (STAGE 1 remains the same)
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# ===================================================================
# STAGE 2: The "Final" - The lean production image
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and assets
COPY src/ /app/src # <-- ADD THIS LINE
COPY main.py .
COPY config.ini .
COPY recommendations.db .
COPY model.onnx .

# (The rest of the file remains the same)
ENV WORKERS=2
ENV LOG_LEVEL=info
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8000/manifest.json || exit 1
CMD gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --log-level $LOG_LEVEL
