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

# --- Load Config and Environment Variables ---
config = configparser.ConfigParser()
config.read('config.ini')

SIMILAR_POST_COUNT = int(os.getenv('SIMILAR_POST_COUNT', config['NLP']['similar_post_count']))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 100))
MODEL_NAME = config['NLP']['sentence_transformer_model']
RECS_DB_FILE = config['DATABASE']['recommendations_database_file']

print("--- Vibe Recommender (ONNX Edition) ---")
print(f"Runtime Config: SIMILAR_POST_COUNT={SIMILAR_POST_COUNT}, MAX_RESULTS={MAX_RESULTS}")

# --- Initialize empty data structures for robust startup ---
post_vectors = {}
post_titles = {}
suggestions_by_post = defaultdict(list)
DEFAULT_CATALOG_IDS = []
tokenizer = None
onnx_session = None

# --- Load Models and Data on Startup ---
try:
    print("Initializing ONNX runtime session for model.onnx...")
    onnx_session = ort.InferenceSession('model.onnx')
    print("ONNX session loaded.")

    print(f"Loading tokenizer directly from pretrained model: {MODEL_NAME}")
    # We load the tokenizer directly from the Hugging Face hub.
    # This is more robust and decouples us from the sentence-transformers library at runtime.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.")

except Exception as e:
    print(f"CRITICAL: Failed to load AI model or tokenizer: {e}")
    print("The addon will not be able to perform searches.")

# --- Helper function for ONNX inference ---
def encode_text_onnx(text: str) -> np.ndarray:
    if not tokenizer or not onnx_session:
        # Return an empty array if models aren't loaded, to prevent crashes.
        return np.array([])

    inputs = tokenizer(
        [text], padding='max_length', truncation=True, max_length=128, return_tensors="np"
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
    print("The addon will run but will return empty results until the database is created.")

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
    
    final_tt_ids = []

    if search_query:
        if not tokenizer or not onnx_session:
            return JSONResponse(status_code=503, content={"error": "AI models not loaded"})
        if not post_vectors:
             return {"metas": []}

        print(f"Handling search query: '{search_query}'")
        query_vector = encode_text_onnx(search_query)
        
        post_ids = list(post_vectors.keys())
        
        # This is the more robust way to create the vector matrix
        all_vectors = np.vstack(list(post_vectors.values()))
        
        similarities = cosine_similarity(query_vector, all_vectors)[0]
        
        top_indices = np.argsort(similarities)[-SIMILAR_POST_COUNT:][::-1]
        similar_post_ids = [post_ids[i] for i in top_indices]
        
        print("Found similar Reddit posts:")
        for post_id in similar_post_ids:
            print(f"  - {post_titles.get(post_id, 'Unknown Title')}")

        weighted_suggestions = defaultdict(int)
        for post_id in similar_post_ids:
            for tt_id, upvotes in suggestions_by_post.get(post_id, []):
                weighted_suggestions[tt_id] += upvotes
        
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
