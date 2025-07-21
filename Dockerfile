# ===================================================================
# STAGE 1: The "Builder"
# ===================================================================
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt

# ===================================================================
# STAGE 2: The "Final" - Production Image
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY onnx_model/ /app/onnx_model/
COPY main.py .
COPY config.ini .
COPY recommendations.db .
COPY healthcheck.py .
COPY imdb_lookup.db .

# --- Set environment variables ---
ENV WORKERS=2
ENV LOG_LEVEL=info
ENV SIMILAR_POST_COUNT=5
ENV MAX_RESULTS=100
ENV KID_FRIENDLY_MODE=false
ENV MIN_IMDB_RATING=5.2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD [ "python", "healthcheck.py" ]

# The CMD line to run the app remains the same
CMD gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --log-level $LOG_LEVEL
