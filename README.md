# Project 2: Stremio Reddit Vibe Recommender (ONNX Edition)

[![Process Data and Publish to GHCR](https://github.com/your-username/your-repo-name/actions/workflows/main.yml/badge.svg)](https://github.com/your-username/your-repo-name/actions/workflows/main.yml)

This is a hyper-optimized version of the original "Vibe Recommender" addon. It provides the exact same intelligent, "vibe-based" search functionality but with a final Docker image that is **~90% smaller** and even more performant at runtime.

The core innovation is the use of the **ONNX (Open Neural Network Exchange)** format. We convert the large PyTorch AI model into a single, compact `.onnx` file. The final Docker container runs this model using a small, specialized ONNX Runtime, completely eliminating the need for the multi-gigabyte PyTorch library.

## Key Improvements

*   **Drastically Smaller Image:** Final Docker image size is reduced from **~3.2 GB to ~400-500 MB**.
*   **Faster Inference:** ONNX Runtimes are often faster for CPU-based inference than native PyTorch.
*   **Framework Agnostic:** The `.onnx` model is a universal standard, making the core AI asset future-proof.

## Architecture Overview

1.  **Offline Processor (GitHub Actions):** This is the engine room. A scheduled workflow automatically:
    *   Builds a local IMDb lookup database.
    *   Scans popular posts in movie subreddits and filters for quality.
    *   Generates the `recommendations.db` with movie suggestions linked to post "vibes".
    *   Converts the SentenceTransformer AI model into a lean `model.onnx` file.
    *   Builds and pushes the final, lightweight Docker image to the GitHub Container Registry.

2.  **Online Addon (Docker & FastAPI):** The public-facing component.
    *   A lightweight server built from a multi-stage Dockerfile for minimum size.
    *   Contains NO PyTorch. It only uses the small `onnxruntime` library.
    *   Loads the `recommendations.db` and `model.onnx` to serve user requests instantly.

## Setup & Deployment

The setup is identical to the original project, but the underlying technology is far more efficient.

1.  **Fork this repository.**
2.  **Add Reddit API Secrets** to your repository settings (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, etc.).
3.  **Run the First Workflow:** Go to the **Actions** tab, find "Process Data and Publish to GHCR", and run it manually. This will generate your `recommendations.db`, your `model.onnx`, and publish the first lean Docker image.
4.  **Deploy with Docker Compose & Watchtower** using the same `docker-compose.yml` structure as the original project. Remember to update the `image:` path in your compose file to point to the new image name (e.g., `ghcr.io/your-username/your-repo-name:latest`).

*(See the original project's README for detailed deployment instructions if needed.)*
