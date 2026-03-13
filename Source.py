import pandas as pd
import numpy as np
import ast
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

nltk.download('punkt')

print('✅ All libraries imported successfully')

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print(f'Movies shape: {movies.shape}')
print(f'Credits shape: {credits.shape}')
movies.head(3)

# Merge on title
movies = movies.merge(credits, on='title')

# Keep only useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

print(f'Merged shape: {movies.shape}')
movies.head(3)

# Drop rows with missing values
movies.dropna(inplace=True)
print(f'After dropna: {movies.shape}')

# Extract genre/keyword names from JSON strings
def extract_names(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)

# Extract top 3 cast members
def extract_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

movies['cast'] = movies['cast'].apply(extract_cast)

# Extract director from crew
def extract_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(extract_director)

movies[['title', 'genres', 'keywords', 'cast', 'crew']].head(3)

# Remove spaces within names so they're treated as single tokens
# e.g. "Sam Worthington" → "SamWorthington"
def collapse_spaces(lst):
    return [i.replace(" ", "") for i in lst]

movies['genres'] = movies['genres'].apply(collapse_spaces)
movies['keywords'] = movies['keywords'].apply(collapse_spaces)
movies['cast'] = movies['cast'].apply(collapse_spaces)
movies['crew'] = movies['crew'].apply(collapse_spaces)

# Combine all into tags
movies['tags'] = (
    movies['overview'].apply(lambda x: x.split()) +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# Join list into a single string and lowercase
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# Final dataframe
new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df.head(3)

ps = PorterStemmer()

def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem_text)

# Example
print('Sample tags for Avatar:')
print(new_df[new_df['title'] == 'Avatar']['tags'].values[0][:300], '...')

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray().astype('float32')

print(f'Vector shape: {vectors.shape}')
print(f'Each movie → vector of {vectors.shape[1]} dimensions')

# Peek at top words
print('Sample vocabulary words:')
print(cv.get_feature_names_out()[:20])

similarity = cosine_similarity(vectors)

print(f'Similarity matrix shape: {similarity.shape}')
print(f'\nSample (Avatar vs first 5 movies):')
print(similarity[0][:5])

def recommend(movie, n=5):
    """
    Recommend n similar movies based on cosine similarity.
    
    Args:
        movie (str): Movie title
        n (int): Number of recommendations (default 5)
    """
    # Check if movie exists
    matches = new_df[new_df['title'] == movie]
    if matches.empty:
        print(f'❌ Movie "{movie}" not found in dataset.')
        return
    
    # Get index
    idx = matches.index[0]
    
    # Sort by similarity score (descending), skip index 0 (itself)
    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )
    
    print(f'🎬 Movies similar to "{movie}":\n')
    for rank, (i, score) in enumerate(distances[1:n+1], 1):
        print(f'  {rank}. {new_df.iloc[i].title:<40} (score: {score:.4f})')

recommend('Avatar')
recommend('The Dark Knight')
recommend('Inception')
recommend('Interstellar')

import matplotlib.pyplot as plt
import seaborn as sns

# Show similarity heatmap for first 10 movies
sample = similarity[:10, :10]
labels = new_df['title'][:10].tolist()

plt.figure(figsize=(12, 8))
sns.heatmap(
    sample,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt='.2f',
    cmap='Blues'
)
plt.title('Cosine Similarity Heatmap (First 10 Movies)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import pickle

pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print('✅ Model saved as movies.pkl and similarity.pkl')