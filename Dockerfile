# ===================================================================
# STAGE 1: The "Builder" stage - only used to cache build dependencies
# This stage isn't strictly necessary with the new workflow but is
# good practice if you ever need to compile complex packages.
# For simplicity, we can remove it, but let's keep it for best practice.
# ===================================================================
FROM python:3.11-slim as builder

# This stage is just a placeholder in our new workflow, as all asset
# creation happens outside the Docker build. We could use it to
# pre-install dependencies if needed, but we will do that in the final stage.

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
# Note: We don't need to copy src/ as it's not used by main.py
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
