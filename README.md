# Stremio Reddit Vibe Recommender [ONNX Edition]

[![Process Data, Convert Model, and Publish to GHCR](https://github.com/kiskey/stremio-reddit-recommends_onnx/actions/workflows/main.yml/badge.svg)](https://github.com/kiskey/stremio-reddit-recommends_onnx/actions/workflows/main.yml)

This isn't just another movie addon. This is a self-updating, intelligent recommendation engine that provides movie suggestions based on the collective "vibe" of Reddit's movie communities.

Instead of searching for genres, you can search for feelings, moods, or abstract concepts like *"a cozy mystery for a rainy day"* or *"mind-bending movies that will make me question reality."* The addon finds real discussions on Reddit that match your mood and recommends the movies that the community agreed were the best fit.

The core philosophy is **quality over quantity**. A rigorous filtering process ensures that every recommendation is a high-quality, community-vetted suggestion, guaranteed to be a valid movie that Stremio can display.

## Core Features

*   **Vibe-Based Semantic Search:** Uses a powerful AI model to understand the meaning behind your search query, not just keywords.
*   **Fully Automated:** A GitHub Action runs daily to fetch new suggestions, process them, and build an updated Docker image.
*   **Offline IMDb Matching:** Uses a local, fast IMDb database to reliably match movie titles to `tt` IDs for perfect integration with Stremio's metadata. No external API calls are made at runtime.
*   **Integrated GHCR Deployment:** Automatically builds and pushes the latest Docker image to the GitHub Container Registry, making deployment seamless.
*   **Highly Efficient:** The live addon is extremely lightweight and fast. All heavy processing is done offline. The addon's only job is to perform a quick vector search, resulting in near-instant response times.
*   **Runtime Configuration:** Easily configurable at runtime using environment variables to tune performance and search behavior without rebuilding the image.

## Architecture Overview

The system works in two distinct parts:

1.  **The Offline Processor (GitHub Actions):** This is the engine room. A scheduled workflow automatically:
    *   Builds a local IMDb lookup database from the latest public datasets.
    *   Scans popular posts in movie-related subreddits.
    *   Applies strict quality filters to posts and comments.
    *   Finds `tt` IDs for high-quality movie suggestions.
    *   Generates AI "vibe" vectors for the source posts.
    *   Commits the updated `recommendations.db` back to the repository.

2.  **The Online Addon (Docker & FastAPI):** This is the public-facing component.
    *   A lightweight server that is packaged with the latest `recommendations.db`.
    *   Does **zero** scraping or slow processing.
    *   Loads the entire database into memory for lightning-fast lookups.
    *   Provides a default catalog of the all-time best recommendations.
    *   Responds to user searches by finding the most similar "vibes" in its database and returning the associated movies.

## Setup & Deployment

### Step 1: Fork and Configure Secrets

1.  **Fork this repository** to your own GitHub account.

2.  **Create Reddit API Credentials:**
    *   Go to Reddit's [apps preferences page](https://www.reddit.com/prefs/apps).
    *   Click "are you a developer? create an app...".
    *   Name it, select the `script` type, and set the `redirect uri` to `http://localhost:8080`.
    *   You will get a **client ID** (under the app name) and a **client secret**.

3.  **Add Reddit Secrets to GitHub:** In your forked repository, go to `Settings > Secrets and variables > Actions` and create the following repository secrets:
    *   `REDDIT_CLIENT_ID`: The client ID from the previous step.
    *   `REDDIT_CLIENT_SECRET`: The client secret from the previous step.
    *   `REDDIT_USER_AGENT`: A unique identifier, e.g., `VibeAddon/1.0 by u/YourRedditUsername`.
    *   `REDDIT_USERNAME`: Your Reddit username.
    *   `REDDIT_PASSWORD`: Your Reddit password.

### Step 2: Run the First Workflow

Before you can deploy the addon, you need the `recommendations.db` file to be generated.

1.  Go to the **Actions** tab in your repository.
2.  Click on the **Process Data and Publish to GHCR** workflow in the sidebar.
3.  Click the **Run workflow** dropdown, then the **Run workflow** button.

This will take 10-15 minutes. It will process the data, create the database, and push the very first version of your Docker image to the GitHub Container Registry.

### Step 3: Deploy with Docker Compose & Watchtower

This is the recommended method for a "set it and forget it" deployment. It will automatically pull and restart your addon every time a new image is published by the GitHub Action.

1.  **Create a Personal Access Token (PAT) for Watchtower:**
    *   Go to your GitHub `Settings > Developer settings > Personal access tokens > Tokens (classic)`.
    *   Click **Generate new token**.
    *   Give it a note (e.g., "Watchtower Access") and set an expiration.
    *   Check the **`read:packages`** scope. This is the only permission needed.
    *   Generate the token and **copy it immediately**.

2.  **Create a `docker-compose.yml` file** on your server with the following content. **Replace the placeholder values.**

    ```yaml
    version: "3.8"
    services:
      stremio-vibe-addon:
        # --- IMPORTANT: Replace with your details ---
        image: ghcr.io/your-github-username/your-repo-name:latest
        container_name: stremio-vibe-addon
        restart: unless-stopped
        ports:
          - "8000:8000" # You can change the host port if needed
        environment:
          # --- Optional: See runtime configuration below ---
          - WORKERS=2
          - LOG_LEVEL=info
          - SIMILAR_POST_COUNT=5
          - MAX_RESULTS=100

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

You can customize the running addon's behavior by passing environment variables in your `docker-compose.yml` file.

| Variable Name         | Description                                                                                             | Default |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------- |
| `WORKERS`             | The number of Gunicorn worker processes. More workers handle more concurrent requests but use more RAM. | `2`     |
| `LOG_LEVEL`           | The verbosity of the server logs. Options: `debug`, `info`, `warning`, `error`.                         | `info`  |
| `SIMILAR_POST_COUNT`  | How many different Reddit threads to use when building search results. Higher = more variety, less precision. | `5`     |
| `MAX_RESULTS`         | The maximum number of movies to return in any catalog or search result.                                 | `100`   |
