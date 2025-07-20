import pandas as pd
import sqlite3
import configparser
from pathlib import Path

def build_db():
    config = configparser.ConfigParser()
    config.read('config.ini')

    DB_FILE = Path(config['DATABASE']['imdb_database_file'])
    MIN_VOTES = int(config['IMDB']['min_votes'])
    BASICS_FILE = Path('./data/title.basics.tsv.gz')
    RATINGS_FILE = Path('./data/title.ratings.tsv.gz')
    
    if DB_FILE.exists():
        DB_FILE.unlink()

    print("Reading IMDb ratings data...")
    ratings_df = pd.read_csv(RATINGS_FILE, sep='\t', usecols=['tconst', 'numVotes'])
    
    print(f"Filtering ratings for movies with at least {MIN_VOTES} votes...")
    popular_movies = ratings_df[ratings_df['numVotes'] >= MIN_VOTES]

    print("Reading IMDb basics data...")
    basics_df = pd.read_csv(BASICS_FILE, sep='\t', usecols=['tconst', 'primaryTitle', 'titleType', 'startYear'], low_memory=False)
    
    print("Filtering for movies only...")
    movies_df = basics_df[basics_df['titleType'] == 'movie']

    print("Merging popular movies with titles...")
    merged_df = pd.merge(popular_movies, movies_df, on='tconst')
    
    merged_df['cleaned_title'] = merged_df['primaryTitle'].str.lower().str.strip()
    
    final_df = merged_df[['tconst', 'cleaned_title', 'primaryTitle', 'startYear']]

    print(f"Connecting to SQLite database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    
    print(f"Writing {len(final_df)} movies to the database...")
    final_df.to_sql('movies', conn, if_exists='replace', index=False)

    print("Creating index on 'cleaned_title' for fast lookups...")
    conn.execute("CREATE INDEX idx_cleaned_title ON movies (cleaned_title);")
    
    conn.close()
    print("IMDb lookup database built successfully.")

if __name__ == "__main__":
    build_db()
