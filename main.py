import sqlite3
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import configparser
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import re

# --- Load Config and Environment Variables ---
config = configparser.ConfigParser()
config.read('config.ini')

SIMILAR_POST_COUNT = int(os.getenv('SIMILAR_POST_COUNT', config['NLP']['similar_post_count']))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 100)) # This will now act as our "page size"
RECS_DB_FILE = config['DATABASE']['recommendations_database_file']

print("--- Vibe Recommender (ONNX Edition) v1.1 ---")
print(f"Runtime Config: SIMILAR_POST_COUNT={SIMILAR_POST_COUNT}, MAX_RESULTS (Page Size)={MAX_RESULTS}")

# --- Initialize empty data structures ---
post_vectors, post_titles = {}, {}
suggestions_by_post = defaultdict(list)
DEFAULT_CATALOG_IDS = []
tokenizer, onnx_session = None, None

# --- Load Models and Data on Startup ---
try:
    print("Loading ONNX model and tokenizer from 'onnx_model' directory...")
    onnx_session = ort.InferenceSession('onnx_model/model.onnx')
    tokenizer = AutoTokenizer.from_pretrained('onnx_model')
    print("ONNX session and tokenizer loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load AI model or tokenizer: {e}")

# --- Helper functions for ONNX inference and post-processing ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def encode_text_onnx(text: str) -> np.ndarray:
    if not tokenizer or not onnx_session: return np.array([])
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="np")
    raw_output = onnx_session.run(None, {key: val for key, val in inputs.items()})
    pooled = mean_pooling(raw_output, inputs['attention_mask'])
    return normalize(pooled)

# --- Load pre-computed data from the recommendations database ---
print(f"Loading recommendations database: {RECS_DB_FILE}")
try:
    conn = sqlite3.connect(f'file:{RECS_DB_FILE}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT post_id, post_title, post_vector FROM posts")
    posts_data = cursor.fetchall()
    post_vectors = {row['post_id']: np.frombuffer(row['post_vector'], dtype=np.float32).reshape(1, -1) for row in posts_data}
    post_titles = {row['post_id']: row['post_title'] for row in posts_data}
    cursor.execute("SELECT post_id, tt_id, upvotes FROM suggestions")
    suggestions_data = cursor.fetchall()
    default_catalog_scores = defaultdict(int)
    for row in suggestions_data:
        suggestions_by_post[row['post_id']].append((row['tt_id'], row['upvotes']))
        default_catalog_scores[row['tt_id']] += row['upvotes']
    conn.close()
    sorted_default_catalog = sorted(default_catalog_scores.items(), key=lambda item: item[1], reverse=True)
    DEFAULT_CATALOG_IDS = [item[0] for item in sorted_default_catalog]
    print(f"Successfully loaded {len(post_vectors)} post vectors and {len(suggestions_data)} suggestions.")
except Exception as e:
    print(f"CRITICAL: Database not found or failed to load: {e}.")

app = FastAPI()

@app.get("/manifest.json")
async def get_manifest():
    return {
        "id": "com.mjlan.reddit-vibe-recommender-onnx",
        "version": "1.1.0", 
        "name": "Reddit Vibe (ONNX)",
        "description": "Reddit vibes based Hyper-optimized movie recommendations",
        "resources": ["catalog"],
        "types": ["movie"],
        "catalogs": [
            {
                "type": "movie",
                "id": "reddit-vibe-catalog",
                "name": "Reddit Vibe Search",
                # --- ENHANCEMENT: Added "extra" property to support pagination ---
                "extra": [
                    {"name": "search", "isRequired": False},
                    {"name": "skip", "isRequired": False}
                ]
            }
        ]
    }

# --- ENHANCEMENT: Consolidated all catalog routes into one powerful endpoint ---
@app.get("/catalog/movie/{catalog_id}/{extra_props}.json")
async def get_catalog(catalog_id: str, extra_props: str):
    if catalog_id != "reddit-vibe-catalog":
        return JSONResponse(status_code=404, content={"error": "Catalog not found"})

    # --- ENHANCEMENT: Parse the extra properties for search and skip values ---
    search_query = None
    skip = 0
    
    search_match = re.search(r'search=([^&]+)', extra_props)
    if search_match:
        search_query = search_match.group(1)

    skip_match = re.search(r'skip=(\d+)', extra_props)
    if skip_match:
        skip = int(skip_match.group(1))

    final_tt_ids = []

    if search_query:
        # (Search logic is identical to before)
        if not tokenizer or not onnx_session:
            return JSONResponse(status_code=503, content={"error": "AI models not loaded"})
        if not post_vectors: return {"metas": []}
        print(f"Handling search query: '{search_query}', skipping: {skip}")
        query_vector = encode_text_onnx(search_query)
        post_ids, all_vectors = list(post_vectors.keys()), np.vstack(list(post_vectors.values()))
        similarities = cosine_similarity(query_vector, all_vectors)[0]
        top_indices = np.argsort(similarities)[-SIMILAR_POST_COUNT:][::-1]
        similar_post_ids = [post_ids[i] for i in top_indices]
        weighted_suggestions = defaultdict(int)
        for post_id in similar_post_ids:
            for tt_id, upvotes in suggestions_by_post.get(post_id, []):
                weighted_suggestions[tt_id] += upvotes
        sorted_suggestions = sorted(weighted_suggestions.items(), key=lambda item: item[1], reverse=True)
        final_tt_ids = [item[0] for item in sorted_suggestions]
    else:
        # Serve the default catalog
        print(f"Serving default catalog, skipping: {skip}")
        final_tt_ids = DEFAULT_CATALOG_IDS

    # --- ENHANCEMENT: Apply pagination slicing to the final list of IDs ---
    paginated_ids = final_tt_ids[skip : skip + MAX_RESULTS]
    
    # --- ENHANCEMENT: Create rich meta objects with posters ---
    metas = []
    for tt_id in paginated_ids:
        metas.append({
            "id": tt_id,
            "type": "movie",
            "name": "Reddit Vibe Movie", # Name is optional, Stremio will fetch the real one
            "poster": f"https://images.metahub.space/poster/medium/{tt_id}/img",
            "posterShape": "poster"
        })

    return {"metas": metas}

@app.get("/")
async def root():
    return {"message": "Stremio Reddit Vibe Recommender (ONNX Edition) v1.1 is running."}
