import sqlite3
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import configparser
import os
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

# --- Load Config and Environment Variables ---
config = configparser.ConfigParser()
config.read('config.ini')

SIMILAR_POST_COUNT = int(os.getenv('SIMILAR_POST_COUNT', config['NLP']['similar_post_count']))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 100))
MODEL_NAME = config['NLP']['sentence_transformer_model']
RECS_DB_FILE = config['DATABASE']['recommendations_database_file']

print("--- Vibe Recommender (ONNX Edition) ---")
print(f"Runtime Config: SIMILAR_POST_COUNT={SIMILAR_POST_COUNT}, MAX_RESULTS={MAX_RESULTS}")

# --- Load Models and Data on Startup ---
print("Initializing ONNX runtime session for model.onnx...")
onnx_session = ort.InferenceSession('model.onnx')
print("ONNX session loaded.")

print(f"Loading tokenizer from model: {MODEL_NAME}")
# We still need the original tokenizer to convert text to the correct input format for the ONNX model.
# The tokenizer is very small and has no heavy dependencies.
tokenizer = SentenceTransformer(MODEL_NAME).tokenizer
print("Tokenizer loaded.")

# --- Helper function for ONNX inference ---
def encode_text_onnx(text: str) -> np.ndarray:
    inputs = tokenizer(
        [text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="np" # Return numpy arrays for ONNX
    )
    onnx_inputs = {key: val for key, val in inputs.items()}
    embedding = onnx_session.run(None, onnx_inputs)[0]
    return embedding

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
    suggestions_by_post = defaultdict(list)
    default_catalog_scores = defaultdict(int)
    for row in suggestions_data:
        suggestions_by_post[row['post_id']].append((row['tt_id'], row['upvotes']))
        default_catalog_scores[row['tt_id']] += row['upvotes']
    
    conn.close()
    
    sorted_default_catalog = sorted(default_catalog_scores.items(), key=lambda item: item[1], reverse=True)
    DEFAULT_CATALOG_IDS = [item[0] for item in sorted_default_catalog]
    print(f"Loaded {len(post_vectors)} post vectors and {len(suggestions_data)} suggestions.")

except Exception as e:
    print(f"Database not found or failed to load: {e}. Please run the GitHub Action to generate it.")
    post_vectors, DEFAULT_CATALOG_IDS, suggestions_by_post, post_titles = {}, [], defaultdict(list), {}

app = FastAPI()

@app.get("/manifest.json")
async def get_manifest():
    return {
        "id": "com.yourname.reddit-vibe-recommender-onnx",
        "version": "1.0.0",
        "name": "Reddit Vibe (ONNX)",
        "description": "Hyper-optimized movie recommendations based on Reddit vibes.",
        "resources": ["catalog"],
        "types": ["movie"],
        "catalogs": [
            {
                "type": "movie",
                "id": "reddit-vibe-catalog",
                "name": "Reddit Vibe Search",
                "extra": [{"name": "search", "isRequired": False}]
            }
        ]
    }

@app.get("/catalog/movie/{catalog_id}.json")
@app.get("/catalog/movie/{catalog_id}/search={search_query}.json")
async def get_catalog(request: Request, catalog_id: str, search_query: str = None):
    if catalog_id != "reddit-vibe-catalog":
        return JSONResponse(status_code=404, content={"error": "Catalog not found"})
    if not post_vectors:
        return {"metas": []}
    
    final_tt_ids = []

    if search_query:
        print(f"Handling search query: '{search_query}'")
        query_vector = encode_text_onnx(search_query)
        
        post_ids = list(post_vectors.keys())
        all_vectors = np.array(list(post_vectors.values())).squeeze()
        similarities = cosine_similarity(query_vector, all_vectors)[0]
        
        top_indices = np.argsort(similarities)[-SIMILAR_POST_COUNT:][::-1]
        similar_post_ids = [post_ids[i] for i in top_indices]
        
        print("Found similar Reddit posts:")
        for post_id in similar_post_ids:
            print(f"  - {post_titles.get(post_id, 'Unknown Title')}")

        weighted_suggestions = {}
        for post_id in similar_post_ids:
            for tt_id, upvotes in suggestions_by_post.get(post_id, []):
                weighted_suggestions[tt_id] = weighted_suggestions.get(tt_id, 0) + upvotes
        
        sorted_suggestions = sorted(weighted_suggestions.items(), key=lambda item: item[1], reverse=True)
        final_tt_ids = [item[0] for item in sorted_suggestions]
    else:
        print("Serving default catalog.")
        final_tt_ids = DEFAULT_CATALOG_IDS

    metas = [{"id": tt_id, "type": "movie"} for tt_id in final_tt_ids[:MAX_RESULTS]]
    return {"metas": metas}

@app.get("/")
async def root():
    return {"message": "Stremio Reddit Vibe Recommender (ONNX Edition) is running."}
