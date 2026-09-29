import pickle
import requests
import pandas as pd
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

# get your free key from https://www.omdbapi.com/apikey.aspx
OMDB_API_KEY = "e4aec671"
NO_POSTER = "https://placehold.co/500x750?text=No+Poster"

# load the files saved from the notebook
df = pd.read_pickle("df.pkl")
indices = pickle.load(open("indices.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# Create the app
'''
This creates your API application. The command uvicorn api:app means: in the file api.py, run the object called app.
'''
app = FastAPI()

# The poster cache
'''
If the same movie is recommended again, the poster comes from here instead of calling OMDb again. That makes it faster and saves your 1,000 daily requests.
'''
poster_cache = {}  # remember posters so we don't call OMDb again for the same movie


# The get_poster() function
# a valid IMDb id is text that starts with "tt", like tt0114709. If it's missing or broken, the placeholder is returned straight away.
# if this poster was fetched before, it's returned from memory.

def get_poster(imdb_id):
    if not isinstance(imdb_id, str) or not imdb_id.startswith("tt"):
        return NO_POSTER

    if imdb_id in poster_cache:
        return poster_cache[imdb_id]
    '''
The URL is built, for example: http://www.omdbapi.com/?i=tt0114709&apikey=2fdf6f0a
requests.get(url, timeout=5) calls OMDb and waits at most 5 seconds.
.json() converts OMDb's reply into a Python dictionary, like {"Title": "Toy Story", "Poster": "https://...jpg", ...}.
data.get("Poster", "N/A") takes the poster link, or gives "N/A" if the key is missing.
OMDb itself sends "N/A" when a movie has no poster, so that case gets the placeholder too.
try / except means that if anything goes wrong (no internet, OMDb is down, a bad reply), the placeholder is used instead of crashing the whole API.

Finally, the poster is saved in the cache and returned.
    '''

    try:
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        poster = data.get("Poster", "N/A")
        if poster == "N/A":
            poster = NO_POSTER
    except Exception:
        poster = NO_POSTER

    poster_cache[imdb_id] = poster
    return poster


# The home endpoint
'''
@app.get("/") means: when someone opens http://127.0.0.1:8000/, run this function.
FastAPI automatically converts the returned dictionary into JSON.
It's just a simple check that the server is working.
'''
@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}



#The movies endpoint
'''
This runs when someone opens http://127.0.0.1:8000/movies.
indices.index holds all the movie titles, and list() turns them into a normal Python list.
Streamlit uses this list to fill the dropdown menu.
'''
@app.get("/movies")
def get_movies():
    # list of all movie titles (for the dropdown in streamlit)
    return {"movies": list(indices.index)}


# The recommend endpoint

'''
his runs when someone opens http://127.0.0.1:8000/recommend?title=Avatar&n=5.
FastAPI reads title and n from the URL automatically. These are called query parameters.
title: str means the title must be text.
n: int = 5 means n must be a number, and it's 5 if you don't give one.
'''
@app.get("/recommend")
def recommend(title: str, n: int = 5):
    if title not in indices:
        return {"error": "Movie not found"} #If the title doesn't exist, it returns an error message.

    '''
    This is the same logic as in the notebook:

    Get the movie's row number.
    Compare it with all movies to get a similarity score for each.
    Sort from most to least similar, skip the movie itself, and take the top n.
     '''
    idx = indices[title]
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n + 1]

    '''
    df["title"].iloc[i] gets the title at row i.
    get_poster(df["imdb_id"].iloc[i]) gets its poster from OMDb.
    Both are added to the results list.
    '''

    results = []
    for i in similar_idx:
        results.append({
            "title": df["title"].iloc[i],
            "poster_url": get_poster(df["imdb_id"].iloc[i]),
        })

    #The final JSON sent to Streamlit looks like this:
    '''
     The final JSON sent to Streamlit looks like this:

json
{
  "recommendations": [
    {"title": "Avatar 2", "poster_url": "https://m.media-amazon.com/...jpg"},
    {"title": "Thor: Ragnarok", "poster_url": "https://m.media-amazon.com/...jpg"}
  ]
}
    '''

    return {"recommendations": results}
