import pandas as pd
import sqlite3
import configparser
from pathlib import Path

def build_db():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # --- THE FIX: Read file paths from config and remove the old 'MIN_VOTES' line ---
    DB_FILE = Path(config['DATABASE']['imdb_database_file'])
    BASICS_FILE = Path(config['IMDB_DATA']['basics_file'])
    RATINGS_FILE = Path(config['IMDB_DATA']['ratings_file'])
    AKAS_FILE = Path(config['IMDB_DATA']['akas_file'])
    
    if DB_FILE.exists():
        DB_FILE.unlink()

    print("Reading IMDb basics data (titles, years, genres)...")
    basics_df = pd.read_csv(BASICS_FILE, sep='\t', usecols=['tconst', 'primaryTitle', 'titleType', 'startYear', 'genres'], low_memory=False)
    movies_df = basics_df[basics_df['titleType'] == 'movie'].copy()
    movies_df.set_index('tconst', inplace=True)
    print(f"Loaded {len(movies_df)} movie entries.")

    print("Reading IMDb ratings data...")
    ratings_df = pd.read_csv(RATINGS_FILE, sep='\t', usecols=['tconst', 'averageRating'])
    ratings_df.set_index('tconst', inplace=True)
    print(f"Loaded {len(ratings_df)} rating entries.")

    print("Reading IMDb AKAS data for US content ratings...")
    akas_df = pd.read_csv(AKAS_FILE, sep='\t', usecols=['titleId', 'region', 'attributes'], low_memory=False)
    us_ratings = akas_df[(akas_df['region'] == 'US') & (akas_df['attributes'].notna())].copy()
    us_ratings = us_ratings[us_ratings['attributes'].isin(['G', 'PG', 'PG-13', 'R', 'NC-17'])]
    us_ratings = us_ratings.drop_duplicates(subset='titleId', keep='first').set_index('titleId')
    us_ratings.rename(columns={'attributes': 'contentRating'}, inplace=True)
    print(f"Found {len(us_ratings)} unique US content ratings.")

    print("Merging all data sources into a master database...")
    master_df = movies_df.join(ratings_df, how='left')
    master_df = master_df.join(us_ratings['contentRating'], how='left')

    master_df.reset_index(inplace=True)
    master_df['cleaned_title'] = master_df['primaryTitle'].str.lower().str.strip()
    
    master_df['averageRating'].fillna(0.0, inplace=True)
    master_df['contentRating'].fillna('NR', inplace=True)
    
    final_df = master_df[['tconst', 'cleaned_title', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'contentRating']]

    print(f"Connecting to SQLite database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    
    print(f"Writing {len(final_df)} movies to the master database...")
    final_df.to_sql('movies', conn, if_exists='replace', index=False)

    print("Creating index for fast lookups...")
    conn.execute("CREATE INDEX idx_cleaned_title ON movies (cleaned_title);")
    
    conn.close()
    print("IMDb master lookup database built successfully.")

if __name__ == "__main__":
    build_db()
