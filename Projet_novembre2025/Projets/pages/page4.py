import streamlit as st
import pandas as pd
from requests import post, get,exceptions
import base64
import json
from datetime import datetime as time

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Spotify Songs Recommender",
    page_icon="🎵",
    layout="wide",
)
st.title("**Spotify Songs Recommender**")
st.markdown("---")
st.markdown("""
Bienvenue sur la plateforme **Spotify Songs Recommender** ! Cette application vous permet de découvrir des chansons similaires à vos morceaux préférés en utilisant des techniques avancées de traitement du signal audio et de machine learning.
 Téléchargez simplement une chanson au format MP3, et notre système analysera ses caractéristiques acoustiques pour vous recommander des titres similaires issus de la vaste bibliothèque de Spotify.
 Suivez les étapes dans la barre de navigation ci-dessus pour commencer votre exploration musicale.""")

# Page configuration
st.set_page_config(
    page_title="Spotify Track Explorer",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #191414;
    }
    .stApp {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
    }
    h1 {
        color: #1DB954;
        text-align: center;
    }
    .track-card {
        background-color: #282828;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎵 Spotify Time Machine")
st.markdown("### Explore tracks from 2013 to 2020")
client_id = st.secrets.get("SPOTIFY_CLIENT_ID", "")
client_secret = st.secrets.get("SPOTIFY_SECRET_CODE", "")

# Fonctions to Use Spotify API    
@st.cache_data
def get_token():
    auth_string = client_id + ':' + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + auth_base64,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token=json_result['access_token']
    return token

def get_auth_header(token):
    return {'Authorization': 'Bearer ' + token}

# Search of album function
def search_albums(token, query, limit=50, offset=0):
        """Basic search with pagination"""
        url = 'https://api.spotify.com/v1/search'
        headers = get_auth_header(token)
        params = {
            "q": query,
            "type": "album",
            "limit": limit,
            "offset": offset
        }
        response = get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
# Search albums year by year
@st.cache_data
def strategy_1_year_by_year(start_year, end_year, genre=None):
        """Strategy 1: Query year by year"""
        all_albums = []
        token = get_token()
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching albums from {year}...")
            
            # Build query with year filter
            query = f"year:{year}"
            
            offset = 0
            while True:
                try:
                    results =search_albums(token,query, limit=50, offset=offset)
                    albums = results['albums']['items']
                    
                    if not albums:
                        break
                    
                    all_albums.extend(albums)
                    offset += 50
                    
                    # Stop if we've reached the total (max ~1000)
                    if offset >= results['albums']['total'] or offset >= 1000:
                        break
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except exceptions.RequestException as e:
                    print(f"Error: {e}")
                    break
        
        return all_albums


# Year slider
year = st.slider(
    "Select Year",
    min_value=1970,
    max_value=time.now().year,
    value=(2015,2010),
    step=1
)
st.write(f"You started from the year: {year[0]} to {year[1]}")



st.markdown(f"### 📅 Searching from the year: **{year[0]}** to **{year[1]}**")

# Search button
if st.button("🔍 Search Tracks", type="primary", use_container_width=True):
    if not client_id or not client_secret:
        st.error("⚠️ Please enter your Spotify API credentials in the sidebar!")
    else:
        albums = strategy_1_year_by_year(year[0], year[1])
        st.write(f"Found {len(albums)} albums using year-by-year strategy")
        st.write(albums)
              

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #B3B3B3;'>
        <p>Made with ❤️ using Streamlit and Spotify API</p>
        <p>🎵 Discover music through time</p>
    </div>
""", unsafe_allow_html=True)
