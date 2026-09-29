import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")


# get all movie titles from the API (cached so it loads only once)
@st.cache_data
def get_movies():
    response = requests.get(f"{API_URL}/movies")
    return response.json()["movies"]


movies = get_movies()

selected_movie = st.selectbox("Choose a movie", movies)
n = st.slider("How many recommendations?", 1, 10, 5)

if st.button("Recommend"):
    response = requests.get(
        f"{API_URL}/recommend", params={"title": selected_movie, "n": n}
    )
    if response.status_code != 200:
         st.error("Backend error! Check the uvicorn terminal.")
         st.stop()


    data = response.json()

    if "error" in data:
        st.error(data["error"])
    else:
        recs = data["recommendations"]
        cols = st.columns(5)  # 5 posters per row
        for i, movie in enumerate(recs):
            with cols[i % 5]:
                st.image(movie["poster_url"])
                st.caption(movie["title"])
