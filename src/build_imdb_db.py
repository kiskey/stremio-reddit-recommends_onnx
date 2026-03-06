import pandas as pd
import sqlite3
import configparser
from pathlib import Path

def build_db():
    config = configparser.ConfigParser()
    config.read('config.ini')

    DB_FILE = Path(config['DATABASE']['imdb_database_file'])
    BASICS_FILE = Path(config['IMDB_DATA']['basics_file'])
    RATINGS_FILE = Path(config['IMDB_DATA']['ratings_file'])
    AKAS_FILE = Path(config['IMDB_DATA']['akas_file'])
    
    # 1. Read Genres to filter from config
    filter_genres = [g.strip().lower() for g in config['FILTERS'].get('filter_genres', '').split(',') if g.strip()]
    
    if DB_FILE.exists():
        DB_FILE.unlink()

    print("Reading IMDb basics data...")
    basics_df = pd.read_csv(BASICS_FILE, sep='\t', usecols=['tconst', 'primaryTitle', 'titleType', 'startYear', 'genres'], low_memory=False)
    
    # Filter for movies only
    movies_df = basics_df[basics_df['titleType'] == 'movie'].copy()
    
    # 2. FIX: Implement Genre Filtering if defined in config
    if filter_genres:
        print(f"Applying genre filters: {filter_genres}")
        for genre in filter_genres:
            # Removes movies that contain any of the filtered genres
            movies_df = movies_df[~movies_df['genres'].str.lower().str.contains(genre, na=False)]

    movies_df.set_index('tconst', inplace=True)
    print(f"Loaded {len(movies_df)} movie entries after filtering.")

    print("Reading IMDb ratings data...")
    ratings_df = pd.read_csv(RATINGS_FILE, sep='\t', usecols=['tconst', 'averageRating'])
    ratings_df.set_index('tconst', inplace=True)

    print("Reading IMDb AKAS data...")
    akas_df = pd.read_csv(AKAS_FILE, sep='\t', usecols=['titleId', 'region', 'attributes'], low_memory=False)
    us_ratings = akas_df[(akas_df['region'] == 'US') & (akas_df['attributes'].notna())].copy()
    us_ratings = us_ratings[us_ratings['attributes'].isin(['G', 'PG', 'PG-13', 'R', 'NC-17'])]
    us_ratings = us_ratings.drop_duplicates(subset='titleId', keep='first').set_index('titleId')
    us_ratings.rename(columns={'attributes': 'contentRating'}, inplace=True)

    print("Merging data sources...")
    master_df = movies_df.join(ratings_df, how='left')
    master_df = master_df.join(us_ratings['contentRating'], how='left')

    master_df.reset_index(inplace=True)
    
    # 3. PERMANENT FIX: Explicit assignment instead of inplace=True
    # This ensures the SQLite database will NOT have NULL values
    master_df['averageRating'] = master_df['averageRating'].fillna(0.0)
    master_df['contentRating'] = master_df['contentRating'].fillna('NR')
    
    master_df['cleaned_title'] = master_df['primaryTitle'].str.lower().str.strip()
    
    final_df = master_df[['tconst', 'cleaned_title', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'contentRating']]

    print(f"Writing to {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    final_df.to_sql('movies', conn, if_exists='replace', index=False)

    print("Creating database index...")
    conn.execute("CREATE INDEX idx_cleaned_title ON movies (cleaned_title);")
    conn.execute("CREATE INDEX idx_tconst ON movies (tconst);") # Added for faster joins
    
    conn.close()
    print("Database built successfully. All NULLs have been purged.")

if __name__ == "__main__":
    build_db()
