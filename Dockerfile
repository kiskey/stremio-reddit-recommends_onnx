# ===================================================================
# STAGE 1: The "Builder" - Our temporary workshop
# ===================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install all build-time dependencies
COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt

# This stage doesn't need to do anything else. All asset creation
# is now handled by the GitHub Actions workflow directly.

# ===================================================================
# STAGE 2: The "Final" - Our slim, production-ready shipping crate
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

# Install only the lightweight, app-specific runtime dependencies
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# --- Copy all the necessary assets directly into the final image ---
# These files were created by the 'build-assets' job and checked into Git.
# The 'build-and-push-docker' job checks them out before running this Dockerfile.

# Copy the ONNX model directory (contains model.onnx, tokenizer.json, etc.)
COPY onnx_model/ /app/onnx_model/

# Copy our application code and the database
COPY src/ /app/src/
COPY main.py .
COPY config.ini .
COPY recommendations.db .

# --- Set environment variables for the final image ---
ENV WORKERS=2
ENV LOG_LEVEL=info
ENV SIMILAR_POST_COUNT=5
ENV MAX_RESULTS=100

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8000/manifest.json || exit 1

# The CMD line to run the lightweight server
CMD gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --log-level $LOG_LEVEL
