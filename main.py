import sqlite3
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import configparser
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import re
import io

# --- Config and Env Vars ---
config = configparser.ConfigParser(); config.read('config.ini')
SIMILAR_POST_COUNT = int(os.getenv('SIMILAR_POST_COUNT', config['NLP']['similar_post_count']))
PAGE_SIZE = int(os.getenv('MAX_RESULTS', 100))
RECS_DB_FILE = config['DATABASE']['recommendations_database_file']
print("--- Vibe Recommender (ONNX Edition) v1.6 (Final) ---")

# --- Initialize data structures ---
post_vectors, post_titles = {}, {}; suggestions_by_post = defaultdict(list)
DEFAULT_CATALOG = []; tokenizer, onnx_session = None, None

# --- BUG FIX: Define the correct data converter function for loading ---
def npy_blob_to_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    # Use np.load, the correct counterpart to np.save
    return np.load(out)

# --- Load Models ---
try:
    onnx_session = ort.InferenceSession('onnx_model/model.onnx')
    tokenizer = AutoTokenizer.from_pretrained('onnx_model')
    print("ONNX session and tokenizer loaded successfully.")
except Exception as e: print(f"CRITICAL: Failed to load AI model or tokenizer: {e}")

# --- Helper functions ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]; input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1); sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask
def normalize(embeddings): return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
def encode_text_onnx(text: str) -> np.ndarray:
    if not tokenizer or not onnx_session: return np.array([])
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="np")
    onnx_inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
    raw_output = onnx_session.run(None, onnx_inputs)
    pooled = mean_pooling(raw_output, inputs['attention_mask']); return normalize(pooled)

# --- Load Database ---
print(f"Loading recommendations database: {RECS_DB_FILE}")
try:
    conn = sqlite3.connect(f'file:{RECS_DB_FILE}?mode=ro', uri=True); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT post_id, post_title, post_vector FROM posts"); posts_data = cursor.fetchall()
    
    # --- BUG FIX: Use the correct conversion logic when loading vectors ---
    post_vectors = {row['post_id']: npy_blob_to_array(row['post_vector']) for row in posts_data}
    post_titles = {row['post_id']: row['post_title'] for row in posts_data}
    
    cursor.execute("SELECT post_id, tt_id, title, upvotes FROM suggestions"); suggestions_data = cursor.fetchall()
    temp_scores = {}
    for row in suggestions_data:
        suggestions_by_post[row['post_id']].append({"id": row['tt_id'], "title": row['title'], "upvotes": row['upvotes']})
        if row['tt_id'] not in temp_scores: temp_scores[row['tt_id']] = {'score': 0, 'title': row['title']}
        temp_scores[row['tt_id']]['score'] += row['upvotes']
    conn.close()
    sorted_default_catalog = sorted(temp_scores.items(), key=lambda item: item[1]['score'], reverse=True)
    DEFAULT_CATALOG = [{"id": item[0], "title": item[1]['title']} for item in sorted_default_catalog]
    print(f"Successfully loaded {len(post_vectors)} post vectors.")
    print(f"Default catalog created with {len(DEFAULT_CATALOG)} unique movies.")
except Exception as e: print(f"CRITICAL: Database not found or failed to load: {e}.")

app = FastAPI()

# --- Manifest ---
@app.get("/manifest.json")
async def get_manifest():
    return {
        "id": "com.mjlan.reddit-vibe-recommender-onnx", "version": "1.6.0", "name": "Reddit Vibe (ONNX)",
        "description": "Reddit vibes based Hyper-optimized movie recommendations", "resources": ["catalog"], "types": ["movie"],
        "catalogs": [{"type": "movie", "id": "reddit-vibe-catalog", "name": "Reddit Vibe Search", "extra": [{"name": "search", "isRequired": False}, {"name": "skip", "isRequired": False}]}]
    }

# --- Central logic function ---
async def _get_catalog_logic(search_query: str = None, skip: int = 0):
    final_items = []
    if search_query:
        if not tokenizer or not onnx_session: return JSONResponse(status_code=503, content={"error": "AI models not loaded"})
        if not post_vectors: return {"metas": []}
        print(f"Handling search query: '{search_query}', skipping: {skip}")
        query_vector = encode_text_onnx(search_query)
        
        # --- BUG FIX: Ensure vectors are correctly shaped for cosine_similarity ---
        post_ids = list(post_vectors.keys())
        # vstack expects a list of arrays, which post_vectors.values() already is
        all_vectors = np.vstack(list(post_vectors.values()))
        
        similarities = cosine_similarity(query_vector, all_vectors)[0]
        top_indices = np.argsort(similarities)[-SIMILAR_POST_COUNT:][::-1]
        similar_post_ids = [post_ids[i] for i in top_indices]
        weighted_suggestions = defaultdict(lambda: {'score': 0, 'title': ''})
        for post_id in similar_post_ids:
            for suggestion in suggestions_by_post.get(post_id, []):
                entry = weighted_suggestions[suggestion['id']]; entry['score'] += suggestion['upvotes']; entry['title'] = suggestion['title']
        sorted_suggestions = sorted(weighted_suggestions.items(), key=lambda item: item[1]['score'], reverse=True)
        final_items = [{"id": item[0], "title": item[1]['title']} for item in sorted_suggestions]
    else:
        print(f"Serving default catalog, skipping: {skip}")
        final_items = DEFAULT_CATALOG
    paginated_items = final_items[skip : skip + PAGE_SIZE]
    metas = [{"id": item['id'], "type": "movie", "name": item['title'], "poster": f"https://images.metahub.space/poster/medium/{item['id']}/img", "posterShape": "poster"} for item in paginated_items]
    return {"metas": metas}

# --- Routing ---
@app.get("/catalog/movie/{catalog_id}.json")
async def get_default_catalog(catalog_id: str):
    if catalog_id != "reddit-vibe-catalog": return JSONResponse(status_code=404, content={"error": "Catalog not found"})
    return await _get_catalog_logic()

@app.get("/catalog/movie/{catalog_id}/{extra_props}.json")
async def get_catalog_with_extras(catalog_id: str, extra_props: str):
    if catalog_id != "reddit-vibe-catalog": return JSONResponse(status_code=404, content={"error": "Catalog not found"})
    search_query, skip = None, 0; search_match = re.search(r'search=([^&]+)', extra_props); skip_match = re.search(r'skip=(\d+)', extra_props)
    if search_match: search_query = search_match.group(1)
    if skip_match: skip = int(skip_match.group(1))
    return await _get_catalog_logic(search_query=search_query, skip=skip)

@app.get("/")
async def root(): return {"message": "Stremio Reddit Vibe Recommender (ONNX Edition) v1.6 is running."}
