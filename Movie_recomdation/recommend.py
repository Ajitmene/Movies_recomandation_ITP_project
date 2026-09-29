import numpy as np
import pandas as pd
import nltk
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/movies_metadata.csv')

df.head()

df.columns

df.info()
df.shape
df.isnull().sum()

# Remove duplicates and keep useful columns
df = df.drop_duplicates().reset_index(drop = True)

# keep poster_path and imdb_id, we need them for the images
'''reset_index(drop=True) renumbers the rows 0, 1, 2… again after rows were removed.
Then you keep only the 8 columns you need:
title, overview, genres, tagline are used to find similar movies.
poster_path and imdb_id are used to get poster images.'''

df = df[['title', 'overview', 'genres','tagline','vote_average','popularity','poster_path','imdb_id']]
df


# Handle missing values
df.isnull().sum()
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')


# Clean the genres column
df.iloc[0]['genres']

'''
ast.literal_eval(x) turns the text into a real Python list of dictionaries.
[i['name'] for i in ...] takes only the names: ['Animation', 'Comedy', 'Family'].
" ".join(...) joins them into one string: "Animation Comedy Family".
.apply(...) does this for every row.
'''
import ast
df['genres'] = df['genres'].apply(lambda x :" ".join([i['name'] for i in ast.literal_eval(x)]))
df.head()

# Build poster URLs
# build full image url from poster_path (TMDB image server, no API key needed)

'''
If poster_path is valid text like /abc.jpg, the TMDB website address is added in front of it to make a full link.
Otherwise, a "No Poster" placeholder image is used

TMDB is blocked in India and your api.py now gets posters from OMDb using imdb_id, this poster_url
'''
BASE_URL = 'https://image.tmdb.org/t/p/w500'
NO_POSTER = 'https://placehold.co/500x750?text=No+Poster'

def make_poster_url(path):
  if isinstance(path, str) and path.startswith('/'):
    return BASE_URL + path
  return NO_POSTER

df['poster_url'] = df['poster_path'].apply(make_poster_url)
df[['title','poster_url']].head()
df.isnull().sum()

# Create the "tags" column
'''
This joins the overview, genres and tagline into one big text per movie. That text describes the movie, and it's what gets compared to find similar movies.
'''
df['tags'] = df['overview']+ " " + df['genres'] +" "+ df['tagline']
df.head()
df['tags'][1]

# Clean the text (NLP preprocessing)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
     
def preprocess_text(text):
  #lowercase
  text = str(text).lower()  # 1. make lowercase
  text = re.sub(r'[^a-zA-Z\s]', '', text) # 2. remove numbers & punctuation
  words = text.split() # 3. split into words
  words = [word for word in words if word not in stop_words] # 4. remove stopwords
  #lemmatize
  words = [lemmatizer.lemmatize(word) for word in words] # 5. base form

  return " ".join(words) #  6. join back into text

df['tags'] = df['tags'].apply(preprocess_text)

# Create a title-to-row-number lookup
'''
indices works like a dictionary: indices['Avatar'] gives the row number of Avatar.
Some titles appear more than once (like "Betrayal"), so only the first one is kept. This way, one title always gives one row number.
'''

df = df.reset_index(drop = True)
indices = pd.Series(df.index,index = df['title'])
# some titles appear more than once, keep only the first one
indices = indices[~indices.index.duplicated(keep='first')]
indices

'''
Convert text into numbers (TF-IDF)

max_features=50000 keeps only the 50,000 most useful words.
ngram_range=(1,2) uses single words ("space") and pairs of words ("space ship").

The result is a table with 45,447 rows (one per movie) and 50,000 columns (one per word).
'''

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=50000,ngram_range=(1,2),stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])
tfidf_matrix

'''
If the title isn't in the dataset, it returns "Movie not found".
idx is the row number of the chosen movie.
cosine_similarity compares this movie's numbers with every other movie. The score goes from 0 (completely different) to 1 (the same).
argsort()[::-1] sorts the movies from most similar to least similar.
[1:n+1] skips position 0, because the most similar movie is the movie itself, and then takes the next n movies.
It returns their titles and poster URLs.
'''
# The recommend function (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
def recommend(title, n = 10):
  if title not in indices:
    return ['Movie not found']

  idx = indices[title]
  sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
  similar_idx = sim_score.argsort()[::-1][1:n+1]
  return df[['title','poster_url']].iloc[similar_idx]


recommend('Avatar',5)

# Save everything for FastAPI
     
import pickle

pickle.dump(tfidf_matrix,open('tfidf_matrix.pkl','wb'))

pickle.dump(indices,open('indices.pkl','wb'))

df.to_pickle('df.pkl')

pickle.dump(tfidf,open('tfidf.pkl','wb'))