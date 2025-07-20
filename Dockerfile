# ===================================================================
# STAGE 1: The "Builder"
# ===================================================================
FROM python:3.11-slim as builder

# This stage is now only used for caching build dependencies in the GHA runner.
# The multi-stage build isn't strictly necessary for the final image size
# in this new script-based setup, but it remains a good practice.
WORKDIR /app
COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt


# ===================================================================
# STAGE 2: The "Final" - Production Image
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

# Install only the lightweight runtime dependencies
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy the ONNX model directory
COPY onnx_model/ /app/onnx_model/

# Copy the application code, config, and database
COPY main.py .
COPY config.ini .
COPY recommendations.db .

# --- ENHANCEMENT: Copy the dedicated health check script ---
COPY healthcheck.py .

# Set environment variables
ENV WORKERS=2
ENV LOG_LEVEL=info
ENV SIMILAR_POST_COUNT=5
ENV MAX_RESULTS=100

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD [ "python", "healthcheck.py" ]

# The CMD line to run the app remains the same
CMD gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --log-level $LOG_LEVEL
