import praw
import sqlite3
import configparser
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import io

# --- Helper functions ---
def adapt_array(arr):
    out = io.BytesIO(); np.save(out, arr); out.seek(0)
    return sqlite3.Binary(out.read())
def convert_array(text):
    out = io.BytesIO(text); out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def process_data():
    config = configparser.ConfigParser(); config.read('config.ini')
    SUBREDDITS = config['REDDIT']['subreddits'].split(',')
    POST_LIMIT = int(config['REDDIT']['post_limit'])
    COMMENT_SCORE_THRESH = int(config['REDDIT']['comment_score_threshold'])
    MODEL_NAME = config['NLP']['sentence_transformer_model']
    IMDB_DB_FILE = Path(config['DATABASE']['imdb_database_file'])
    RECS_DB_FILE = Path(config['DATABASE']['recommendations_database_file'])
    FILTER_GENRES = {genre.strip().lower() for genre in config['FILTERS']['filter_genres'].split(',') if genre.strip()}
    
    print("--- Data Processing (Comprehensive Ingestion) ---")
    print(f"Comment Score Threshold: {COMMENT_SCORE_THRESH}")
    print(f"Filtering Build-Time Genres: {FILTER_GENRES if FILTER_GENRES else 'None'}")

    model = SentenceTransformer(MODEL_NAME)
    reddit = praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'], client_secret=os.environ['REDDIT_CLIENT_SECRET'], user_agent=os.environ['REDDIT_USER_AGENT'], username=os.environ['REDDIT_USERNAME'], password=os.environ['REDDIT_PASSWORD'])
    
    imdb_conn = sqlite3.connect(IMDB_DB_FILE)
    recs_conn = sqlite3.connect(RECS_DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    recs_cursor = recs_conn.cursor()

    recs_cursor.execute('CREATE TABLE IF NOT EXISTS posts (post_id TEXT PRIMARY KEY, post_title TEXT, post_vector array)');
    recs_cursor.execute('CREATE TABLE IF NOT EXISTS suggestions (suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT, post_id TEXT, tt_id TEXT, title TEXT, upvotes INTEGER, FOREIGN KEY(post_id) REFERENCES posts(post_id))');

    for sub in SUBREDDITS:
        print(f"\nProcessing subreddit: r/{sub}")
        subreddit = reddit.subreddit(sub)
        try:
            for post in subreddit.hot(limit=POST_LIMIT):
                if not post.is_self or post.stickied: continue
                
                print(f"\nProcessing Post: '{post.title[:50]}...' (Score: {post.score})")
                post_vector = model.encode(post.title)
                recs_cursor.execute("INSERT OR IGNORE INTO posts (post_id, post_title, post_vector) VALUES (?, ?, ?)", (post.id, post.title, post_vector))
                
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if comment.score >= COMMENT_SCORE_THRESH:
                        movie_titles = [line.strip() for line in comment.body.split('\n') if line.strip()]
                        for title_text in movie_titles:
                            cleaned_title = title_text.lower().strip().replace('*', '').replace('"', '')
                            imdb_cursor = imdb_conn.cursor()
                            imdb_cursor.execute("SELECT tconst, primaryTitle, genres FROM movies WHERE cleaned_title = ?", (cleaned_title,))
                            result = imdb_cursor.fetchone()
                            
                            if result:
                                tt_id, movie_title, genres_str = result
                                
                                # Apply the optional build-time genre filter
                                movie_genres = {g.strip().lower() for g in genres_str.split(',')}
                                if movie_genres.intersection(FILTER_GENRES):
                                    print(f"  [FILTERED] '{movie_title}' contains a blocked genre.")
                                    continue
                                    
                                print(f"  [INGESTED] '{title_text}' -> {tt_id} ({movie_title})")
                                recs_cursor.execute("INSERT INTO suggestions (post_id, tt_id, title, upvotes) VALUES (?, ?, ?, ?)", (post.id, tt_id, movie_title, comment.score))
        except Exception as e:
            print(f"An error occurred while processing r/{sub}: {e}")
            continue

    print("\nCommitting changes and closing connections.")
    recs_conn.commit(); recs_conn.close(); imdb_conn.close()
    print("Processing complete.")

if __name__ == "__main__":
    process_data()
