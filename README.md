# Stremio Reddit Vibe Recommender

[![Process Data, Convert Model, and Publish to GHCR](https://github.com/kiskey/stremio-reddit-recommends_onnx/actions/workflows/main.yml/badge.svg)](https://github.com/kiskey/stremio-reddit-recommends_onnx/actions/workflows/main.yml)

This is an intelligent, self-updating recommendation engine that provides movie suggestions based on the collective "vibe" of Reddit's movie communities. It is built on the philosophy of **"Ingest Broadly, Filter Intelligently."** The addon gathers a comprehensive database of all valid movie suggestions and then allows you, the user, to apply powerful filters at runtime to tailor the experience to your exact needs.

Instead of searching for genres, you can search for feelings, moods, or abstract concepts like *"a cozy mystery for a rainy day"* or *"mind-bending movies that will make me question reality."*

## Core Features

*   **Vibe-Based Semantic Search:** Uses an AI model to understand the meaning behind your search query.
*   **Fully Automated:** A GitHub Action runs daily to ingest new suggestions, build the comprehensive database, and publish a new Docker image.
*   **Complete IMDb Master Database:** Builds and uses a local database of all IMDb movie entries, including titles, genres, IMDb ratings, and US content ratings (G, PG, R).
*   **Powerful Runtime Filtering:** The addon's behavior can be changed instantly by setting environment variables, no rebuild required.
    *   **Kid-Friendly Mode:** A simple toggle to filter out all R and NC-17 rated movies.
    *   **Minimum IMDb Rating:** Set a quality floor to hide movies below a certain rating.
*   **Integrated GHCR Deployment:** Automatically builds and pushes the latest Docker image to the GitHub Container Registry.
*   **Highly Efficient:** The addon is extremely fast, using a pre-converted ONNX model and loading all necessary data into memory on startup for near-instant response times.

## Setup & Deployment

### Step 1: Fork and Configure Secrets
(This section remains the same)

### Step 2: Run the First Workflow
(This section remains the same)

### Step 3: Deploy with Docker Compose & Watchtower

This is the recommended method for a "set it and forget it" deployment.

1.  **Create a Personal Access Token (PAT) for Watchtower:**
    *   Go to your GitHub `Settings > Developer settings > Personal access tokens > Tokens (classic)`.
    *   Generate a new token with the **`read:packages`** scope. Copy the token.

2.  **Create a `docker-compose.yml` file** on your server. **Replace the placeholder values.**

    ```yaml
    version: "3.8"
    services:
      stremio-vibe-addon:
        # --- IMPORTANT: Replace with your details ---
        image: ghcr.io/your-github-username/your-repo-name:latest
        container_name: stremio-vibe-addon
        restart: unless-stopped
        ports:
          - "8000:8000"
        environment:
          # --- POWERFUL RUNTIME CONTROLS ---
          - KID_FRIENDLY_MODE=true
          - MIN_IMDB_RATING=6.5
          - WORKERS=2
          - LOG_LEVEL=info
          - SIMILAR_POST_COUNT=5
          - MAX_RESULTS=100 # This is the page size

      watchtower:
        image: containrrr/watchtower
        container_name: watchtower
        restart: unless-stopped
        volumes:
          - /var/run/docker.sock:/var/run/docker.sock
        environment:
          # --- IMPORTANT: Give Watchtower read-only access to GHCR ---
          - REPO_USER=your-github-username
          - REPO_PASS=your-personal-access-token-you-just-copied
        command: --interval 86400 # Check for new images once a day
    ```

3.  **Run the deployment:**
    ```bash
    docker-compose up -d
    ```

Your addon is now live! Install it in Stremio using the URL `http://your-server-ip:8000/manifest.json`.

## Runtime Configuration

Control the addon's behavior by setting environment variables in your `docker-compose.yml` file.

| Variable Name         | Description                                                                                             | Default |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------- |
| `MIN_IMDB_RATING`     | Filters out any movie with an IMDb rating below this value. Set to `0` to disable.                      | `5.2`   |
| `KID_FRIENDLY_MODE`   | If `true`, filters out movies rated R or NC-17.                                                         | `false` |
| `WORKERS`             | The number of Gunicorn worker processes.                                                                | `2`     |
| `LOG_LEVEL`           | The verbosity of the server logs. Options: `debug`, `info`, `warning`, `error`.                         | `info`  |
| `SIMILAR_POST_COUNT`  | How many Reddit threads to use for search results. Higher = more variety.                               | `5`     |
| `MAX_RESULTS`         | The number of items to show per page (page size).                                                       | `100`   |
