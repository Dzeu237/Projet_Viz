import streamlit as st
import plotly.express as px
import pandas as pd
from requests import post, get,exceptions
import base64
import json
from datetime import datetime
import time
import pycountry
from math import ceil

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

#Fonctions to Use Spotify API    
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

def convert_country(country):
    try:
        country = pycountry.countries.get(alpha_2=country.upper())
        return country.alpha_3 if country else None
    except (AttributeError, KeyError):
        return None

#Search of album function
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

@st.cache_data
def search_artist(token, artist_name,start, end):
    url = 'https://api.spotify.com/v1/search'
    headers = get_auth_header(token)
    query = f'?q={artist_name}&type=artist&limit=1&year={start}-{end}'
    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)['artists']['items']
    if len(json_result) == 0:
        return None
    return json_result[0]

@st.cache_data
def get__songs_by_artist(token, artist_id):
    url = f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks'
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

# Year slider
year = st.slider(
    "Select Year",
    min_value=1970,
    max_value=datetime.now().year,
    value=(2015,2010),
    step=1
)
st.write(f"You started from the year: {year[0]} to {year[1]}")

st.markdown(f"### 📅 Searching from the year: **{year[0]}** to **{year[1]}**")

# Search button
if st.button("🔍 Search Albums", type="primary", width='content'):
    if not client_id or not client_secret:
        st.error("⚠️ Please enter your Spotify API credentials in the sidebar!")
    else:
        albums = strategy_1_year_by_year(year[0], year[1])
        st.write(f"Found {len(albums)} albums using year-by-year strategy")

    #Create a dataframe to display albums
        album_data = []
        for album in albums:
            album_info = {
                 "Album_ID": album['id'],
                "Album_Name": album['name'],
                "Album_type": album['album_type'],
                "Total_Tracks": album['total_tracks'],
                "market": album['available_markets'],
                "image_url": album['images'][0]['url'] if album['images'] else None,
                "release_date": album['release_date'],
                "Artist_id":", ".join([artist["id"] for artist in album["artists"]]),
                "Artist": ", ".join([artist['name'] for artist in album['artists']]),
            }
            album_data.append(album_info)
        
        df_albums = pd.DataFrame(album_data)
        st.session_state.df=df_albums
# patch='Data/Album.csv'
# df_albums=pd.read_csv(patch)
# df_albums['release_date']=df_albums['release_date'].apply(lambda x: x.split('-')[0].strip())
if 'df' in st.session_state:
    df_albums=st.session_state.df
    
    #df_albums["market"]=df_albums["market"].apply(lambda x:x.split(','))
    Serie_country=df_albums['market'].explode(ignore_index=True)
    list_country=set(Serie_country)
    iso_map={country:convert_country(country) for country in list_country}
    df_albums['release_date']=df_albums['release_date'].apply(lambda x: x.split('-')[0].strip())

    tab1,tab2=st.tabs('Visualize Album,Visualise Artist'.split(','),width= "stretch")

    with tab1:
        with st.sidebar:
            art=df_albums["Artist"].sort_values().unique().tolist()
            artist=st.selectbox(label="Artist",options=["All"]+art,width="stretch",index=0)
            if artist != "All":
                df_artist=df_albums[df_albums['Artist'] == artist]
            else:
                df_artist=df_albums.copy()
            st.markdown("**Metrique**")
            st.metric(label="Num.Album",value=df_albums.shape[0])
            st.metric(label="Num.Artist",value=len(df_albums['Artist'].unique().tolist()))
            st.metric(label="Num.Track",value=df_albums["Total_Tracks"].sum())

    


        # Configuration de la pagination
        CARDS_PER_PAGE = 9  # Nombre de cartes par page

        # Initialiser la page actuelle dans session_state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # Calculer le nombre total de pages
        total_pages = ceil(len(df_artist) / CARDS_PER_PAGE)

        # Fonction pour changer de page
        def go_to_page(page_num):
            st.session_state.current_page = page_num


        # Calculer les indices pour la pagination
        start_idx = (st.session_state.current_page - 1) * CARDS_PER_PAGE
        end_idx = min(start_idx + CARDS_PER_PAGE, len(df_artist))

        # Extraire la portion du DataFrame pour la page actuelle
        df_current_page = df_artist.iloc[start_idx:end_idx]

        # Afficher les cartes en grille
        cols_per_row = 3
        rows = ceil(len(df_current_page) / cols_per_row)

        for row in range(rows):
            cols = st.columns(cols_per_row)
            st.divider()
            for col_idx in range(cols_per_row):
                album_idx = row * cols_per_row + col_idx
                if album_idx < len(df_current_page):
                    # Récupérer la ligne du DataFrame
                    album = df_current_page.iloc[album_idx]
                    with cols[col_idx]:
                        col1,col2=st.columns([1,2])
                        # Carte d'album
                        with col1:
                            st.image(album["image_url"], width = 500)
                        with col2:
                            st.markdown(f"### {album['Album_Name']}")
                            st.markdown(f"**🎤 Artiste:** {album['Artist']}")
                            st.markdown(f"**🎵 Tracks:** {album['Total_Tracks']}")
                            st.markdown(f"**💰 Marché:** {len(album['market'])} Pays")
                            st.markdown(f"**📅 Année:** {album['release_date']}")
                        
        # Système de pagination numéroté

        st.markdown(f"### Page {st.session_state.current_page} sur {total_pages}")

        # Créer les boutons de pagination
        col_pagination = st.columns([1, 3, 1])

        with col_pagination[0]:
            if st.session_state.current_page > 1:
                if st.button("⬅️ Précédent"):
                    go_to_page(st.session_state.current_page - 1)
                    st.rerun()

        with col_pagination[1]:
            # Afficher les numéros de page
            page_buttons = st.columns(min(total_pages, 10))
            
            # Logique pour afficher les pages (avec ellipses si trop de pages)
            if total_pages <= 10:
                pages_to_show = list(range(1, total_pages + 1))
            else:
                # Afficher: 1 ... pages autour de current ... dernière
                current = st.session_state.current_page
                if current <= 4:
                    pages_to_show = list(range(1, 8)) + ['...', total_pages]
                elif current >= total_pages - 3:
                    pages_to_show = [1, '...'] + list(range(total_pages - 6, total_pages + 1))
                else:
                    pages_to_show = [1, '...'] + list(range(current - 2, current + 3)) + ['...', total_pages]
            
            for idx, page in enumerate(pages_to_show):
                if idx < len(page_buttons):
                    with page_buttons[idx]:
                        if page == '...':
                            st.markdown("...")
                        else:
                            # Bouton actif ou inactif
                            if page == st.session_state.current_page:
                                st.markdown(f"**[{page}]**")
                            else:
                                if st.button(str(page), key=f"page_{page}"):
                                    go_to_page(page)
                                    st.rerun()

        with col_pagination[2]:
            if st.session_state.current_page < total_pages:
                if st.button("Suivant ➡️"):
                    go_to_page(st.session_state.current_page + 1)
                    st.rerun()

        # Sélecteur de page direct
        st.markdown("---")
        col_select = st.columns([2, 1, 2])
        with col_select[1]:
            selected_page = st.selectbox(
                "Aller à la page:",
                options=list(range(1, total_pages + 1)),
                index=st.session_state.current_page - 1,
                key="page_selector"
            )
            if selected_page != st.session_state.current_page:
                go_to_page(selected_page)
                st.rerun()
        
    with tab2:
        if artist == "All":
            st.warning("Choisir d'abord un artist")
        else:
            col1,col2=st.columns([1,3])
            with col1:
                st.write(df_artist.head(5))
                id=df_artist['Artist_id'].unique()
                st.write(id[0])
                token=get_token()
                songs=get__songs_by_artist(token,id)
                st.write(songs)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #B3B3B3;'>
        <p>Made with ❤️ using Streamlit and Spotify API</p>
        <p>🎵 Discover music through time</p>
    </div>
""", unsafe_allow_html=True)
