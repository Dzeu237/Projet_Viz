"""
Spotify Songs Recommender
Application Streamlit pour découvrir et recommander des chansons similaires
"""

import streamlit as st
import pandas as pd
from requests import post, get, exceptions
import base64
from datetime import datetime
import time
import pycountry
from math import ceil
import humanize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Spotify Songs Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLES CSS PERSONNALISÉS
# ============================================================================

st.markdown("""
    <style>
    /* En-têtes */
    h1 {
        color: #1DB954;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    h2 {
        color: #1DB954;
        font-weight: 600;
    }
    
    /* Cartes d'album */
    .track-card {
        background-color: #282828;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .track-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.3);
    }
    
    /* Boutons */
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Divider personnalisé */
    hr {
        margin: 30px 0;
        border-color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES
# ============================================================================

CARDS_PER_PAGE = 9
COLS_PER_ROW = 3
MUSIC_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
DATASET_URL = 'https://github.com/Dzeu237/Projet_Viz/blob/main/Projet_novembre2025/Projets/Data/Song/Scrobble_Features.csv?raw=true'

# ============================================================================
# FONCTIONS API SPOTIFY
# ============================================================================

@st.cache_data
def get_token():
    """
    Obtient un token d'accès Spotify via le flux Client Credentials.
    Le token est stocké dans st.session_state pour réutilisation.
    """
    if "spotify_token" in st.session_state:
        return st.session_state.spotify_token

    client_id = st.secrets.get("SPOTIFY_CLIENT_ID", "")
    client_secret = st.secrets.get("SPOTIFY_SECRET_CODE", "")
    
    if not client_id or not client_secret:
        st.error("⚠️ Veuillez fournir les identifiants API Spotify dans les secrets Streamlit.")
        return None

    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    try:
        response = post(url, headers=headers, data=data)
        response.raise_for_status()
        token = response.json()["access_token"]
        st.session_state.spotify_token = token
        return token
    except exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention du token Spotify: {e}")
        return None


def get_auth_header(token):
    """Retourne les en-têtes d'autorisation pour les requêtes API"""
    return {'Authorization': f'Bearer {token}'}


def search_albums(token, query, limit=50, offset=0):
    """Recherche basique d'albums avec pagination"""
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


@st.cache_data
def strategy_1_year_by_year(start_year, end_year, genre=None):
    """
    Recherche d'albums année par année
    Retourne une liste de tous les albums trouvés
    """
    all_albums = []
    token = get_token()
    
    for year in range(start_year, end_year + 1):
        query = f"year:{year}"
        offset = 0
        
        while True:
            try:
                results = search_albums(token, query, limit=50, offset=offset)
                albums = results['albums']['items']
                
                if not albums:
                    break
                
                all_albums.extend(albums)
                offset += 50
                
                # Limite maximale de l'API Spotify
                if offset >= results['albums']['total'] or offset >= 1000:
                    break
                
                time.sleep(0.1)  # Rate limiting
                
            except exceptions.RequestException as e:
                st.warning(f"Erreur lors de la recherche pour l'année {year}: {e}")
                break
    
    return all_albums


def search_artist(artist_name):
    """
    Recherche un artiste par son nom
    Retourne un dictionnaire ou None
    """
    token = get_token()
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    params = {"q": artist_name, "type": "artist", "limit": 1}

    try:
        response = get(url, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("artists", {}).get("items", [])
        return items[0] if items else None
    except exceptions.RequestException as e:
        st.error(f"Erreur API Spotify: {e}")
        return None


def get__songs_by_artist(artist_id, market='US'):
    """
    Récupère les 5 meilleures chansons d'un artiste
    """
    token = get_token()
    if not token:
        return []

    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks"
    headers = get_auth_header(token)
    params = {"market": market}

    try:
        response = get(url, headers=headers, params=params)
        response.raise_for_status()
        tracks = response.json().get("tracks", [])[:5]
        return tracks
    except exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des chansons: {e}")
        return []

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def convert_country(country):
    """Convertit un code pays alpha-2 en alpha-3"""
    try:
        country = pycountry.countries.get(alpha_2=country.upper())
        return country.alpha_3 if country else None
    except (AttributeError, KeyError):
        return None


def process_albums_data(albums):
    """
    Traite les données brutes d'albums et retourne un DataFrame
    """
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
            "Artist_id": ", ".join([artist["id"] for artist in album["artists"]]),
            "Artist": ", ".join([artist['name'] for artist in album['artists']]),
        }
        album_data.append(album_info)
    
    df_albums = pd.DataFrame(album_data)
    df_albums['Artist_id'] = df_albums['Artist_id'].apply(lambda x: x.split(","))
    df_albums["Artist"] = df_albums['Artist'].apply(lambda x: x.split(","))
    df_albums = df_albums.explode(["Artist_id"])
    df_albums = df_albums.explode(["Artist"])
    df_albums['release_date'] = df_albums['release_date'].apply(lambda x: x.split('-')[0].strip())
    
    return df_albums


def go_to_page(page_num):
    """Change la page actuelle dans la pagination"""
    st.session_state.current_page = page_num

# ============================================================================
# COMPOSANTS D'AFFICHAGE - SIDEBAR
# ============================================================================

def render_sidebar_metrics(df_albums):
    """Affiche les métriques dans la sidebar"""
    with st.sidebar:
        st.markdown("### 📊 Statistiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Albums",
                value=len(df_albums["Album_ID"].unique())
            )
        
        with col2:
            st.metric(
                label="Artistes",
                value=len(df_albums['Artist'].unique())
            )
        
        with col3:
            st.metric(
                label="Tracks",
                value=df_albums.drop_duplicates(subset=["Album_ID"])["Total_Tracks"].sum()
            )

# ============================================================================
# COMPOSANTS D'AFFICHAGE - PAGINATION
# ============================================================================

def render_pagination(total_pages):
    """Affiche le système de pagination complet"""
    
    st.markdown(f"### 📄 Page {st.session_state.current_page} sur {total_pages}")
    
    # Boutons de navigation
    col_pagination = st.columns([1, 3, 1])
    
    # Bouton Précédent
    with col_pagination[0]:
        if st.session_state.current_page > 1:
            if st.button("⬅️ Précédent", width='content'):
                go_to_page(st.session_state.current_page - 1)
                st.rerun()
    
    # Numéros de page
    with col_pagination[1]:
        page_buttons = st.columns(min(total_pages, 10))
        
        if total_pages <= 10:
            pages_to_show = list(range(1, total_pages + 1))
        else:
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
                    elif page == st.session_state.current_page:
                        st.markdown(f"**[{page}]**")
                    else:
                        if st.button(str(page), key=f"page_{page}"):
                            go_to_page(page)
                            st.rerun()
    
    # Bouton Suivant
    with col_pagination[2]:
        if st.session_state.current_page < total_pages:
            if st.button("Suivant ➡️", width='content'):
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

# ============================================================================
# COMPOSANTS D'AFFICHAGE - ALBUMS
# ============================================================================

def render_album_card(album):
    """Affiche une carte d'album individuelle"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if album["image_url"]:
            st.image(album["image_url"], width='content')
    
    with col2:
        st.markdown(f"### {album['Album_Name']}")
        st.markdown(f"**🎤 Artiste:** {album['Artist']}")
        st.markdown(f"**🎵 Tracks:** {album['Total_Tracks']}")
        st.markdown(f"**💰 Marché:** {len(album['market'])} Pays")
        st.markdown(f"**📅 Année:** {album['release_date']}")


def render_albums_grid(df_artist):
    """Affiche la grille d'albums avec pagination"""
    
    # Initialiser la pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    total_pages = ceil(len(df_artist) / CARDS_PER_PAGE)
    
    # Calculer les indices
    start_idx = (st.session_state.current_page - 1) * CARDS_PER_PAGE
    end_idx = min(start_idx + CARDS_PER_PAGE, len(df_artist))
    df_current_page = df_artist.iloc[start_idx:end_idx]
    
    # Afficher les cartes
    rows = ceil(len(df_current_page) / COLS_PER_ROW)
    
    for row in range(rows):
        cols = st.columns(COLS_PER_ROW)
        st.divider()
        
        for col_idx in range(COLS_PER_ROW):
            album_idx = row * COLS_PER_ROW + col_idx
            
            if album_idx < len(df_current_page):
                album = df_current_page.iloc[album_idx]
                with cols[col_idx]:
                    render_album_card(album)
    
    # Afficher la pagination
    st.markdown("---")
    render_pagination(total_pages)

# ============================================================================
# COMPOSANTS D'AFFICHAGE - ARTISTE
# ============================================================================

def render_artist_info(artist_name, df_artist):
    """Affiche les informations détaillées d'un artiste"""
    
    st.subheader("🎤 Description de l'artiste", divider=True)
    
    col1, col2 = st.columns([1, 2])
    artist_id = df_artist['Artist_id'].iloc[0]
    
    with col1:
        artist_data = search_artist(artist_name)
        
        if artist_data:
            # Afficher l'image et les informations
            image_url = artist_data.get("images", [{}])[0].get("url") if artist_data.get("images") else None
            
            if image_url:
                st.image(image_url, caption=artist_data.get("name"), width='content')
            
            st.markdown(f"**Nom:** {artist_data.get('name')}")
            st.markdown(f"**Genres:** {', '.join(artist_data.get('genres', [])) or 'Non spécifié'}")
            st.markdown(f"**Followers:** {humanize.metric(artist_data.get('followers', {}).get('total', 0))}")
            st.markdown(f"**Popularité:** {artist_data.get('popularity', 0)}/100")
    
    with col2:
        st.markdown("### 📊 Statistiques de l'artiste")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                label="Albums",
                value=len(df_artist['Album_ID'].unique())
            )
        
        with metrics_col2:
            st.metric(
                label="Total Tracks",
                value=df_artist['Total_Tracks'].sum()
            )
        
        with metrics_col3:
            st.metric(
                label="Marchés",
                value=len(set([m for markets in df_artist['market'] for m in markets]))
            )


def render_top_songs(artist_name, artist_id):
    """Affiche les meilleures chansons d'un artiste"""
    
    st.subheader(f"🎵 Top Songs - {artist_name}", divider=True)
    
    songs = get__songs_by_artist(artist_id)
    
    if not songs:
        st.warning("Aucune chanson trouvée pour cet artiste.")
        return
    
    # Créer un DataFrame des chansons
    songs_data = []
    for song in songs:
        songs_data.append({
            "song_name": song.get("name"),
            "album_name": song.get("album", {}).get("name"),
            "image": song.get("album", {}).get("images", [{}])[0].get("url"),
            "duration": song.get("duration_ms"),
            "popularity": song.get("popularity"),
            "track_number": song.get("track_number")
        })
    
    df_songs = pd.DataFrame(songs_data)
    
    # Afficher en grille
    rows = ceil(len(df_songs) / 3)
    
    for row in range(rows):
        cols = st.columns(3)
        st.divider()
        
        for col_idx in range(3):
            song_idx = row * 3 + col_idx
            
            if song_idx < len(df_songs):
                song = df_songs.iloc[song_idx]
                
                with cols[col_idx]:
                    if song["image"]:
                        st.image(song["image"], width='content')
                    
                    st.markdown(f"### {song['song_name']}")
                    st.markdown(f"**Album:** {song['album_name']}")
                    st.markdown(f"**Durée:** {humanize.precisedelta(pd.to_timedelta(song['duration'], unit='ms'))}")
                    st.markdown(f"**Popularité:** {song['popularity']}/100")
                    st.markdown(f"**Track #:** {song['track_number']}")

# ============================================================================
# COMPOSANTS D'AFFICHAGE - RECOMMANDATIONS
# ============================================================================

@st.cache_data
def load_recommendation_data():
    """Charge et prépare les données pour les recommandations"""
    df = pd.read_csv(DATASET_URL, index_col=0)
    df_clean = df.dropna(subset=MUSIC_FEATURES).copy()
    
    # Normalisation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean[MUSIC_FEATURES])
    df_scaled = pd.DataFrame(features_scaled, columns=MUSIC_FEATURES, index=df_clean.index)
    
    # Calcul de similarité
    similarity_matrix = cosine_similarity(df_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=df_clean.index, columns=df_clean.index)
    
    return df, df_clean, similarity_df


def recommend_songs(song_id, similarity_df, df_clean, num_recommendations=5):
    """Recommande des chansons similaires"""
    if song_id not in similarity_df.index:
        return None
    
    similar_songs = similarity_df[song_id].sort_values(ascending=False)
    top_songs = similar_songs.drop(song_id).head(num_recommendations)
    
    return df_clean.loc[top_songs.index]


def render_recommendations_tab():
    """Affiche l'onglet des recommandations"""
    
    st.subheader("🎵 Système de Recommandation", divider=True)
    
    # Chargement des données
    with st.spinner("Chargement des données de recommandation..."):
        df, df_clean, similarity_df = load_recommendation_data()
    
    # Sélection de l'artiste et de la chanson
    col1, col2 = st.columns(2)
    
    with col1:
        selected_artist = st.selectbox(
            "🎤 Sélectionner un artiste",
            options=sorted(df['Performer'].unique().tolist()),
            key="rec_artist"
        )
    
    with col2:
        artist_songs = df[df['Performer'] == selected_artist]['Song'].sort_values().unique().tolist()
        selected_song = st.selectbox(
            "🎵 Sélectionner une chanson",
            options=artist_songs,
            key="rec_song"
        )
    
    # Bouton de recherche
    if st.button("🔍 Trouver des chansons similaires", type="primary", width='content'):
        
        # Trouver l'ID de la chanson
        song_id = df[(df['Song'] == selected_song) & (df['Performer'] == selected_artist)].index[0]
        
        # Obtenir les recommandations
        with st.spinner("Recherche de chansons similaires..."):
            recommendations = recommend_songs(song_id, similarity_df, df_clean, num_recommendations=5)
        
        if recommendations is not None and len(recommendations) > 0:
            st.success(f"✅ {len(recommendations)} recommandations trouvées!")
            
            st.markdown("---")
            st.subheader(f"🎯 Recommandations pour '{selected_song}' par {selected_artist}")
            
            # Afficher les recommandations
            for idx, (index, row) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"### {idx}. {row['Song']}")
                        st.markdown(f"**Artiste:** {row['Performer']}")
                        st.markdown(f"**Durée:** {humanize.precisedelta(pd.to_timedelta(row['spotify_track_duration_ms'], unit='ms'))}")
                    
                    with col_b:
                        st.metric(
                            label="Popularité",
                            value=f"{row['spotify_track_popularity']}/100"
                        )
                    
                    st.divider()
        else:
            st.error("❌ Aucune recommandation trouvée pour cette chanson.")

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # En-tête
    st.title("**🎵 Spotify Songs Recommender**")
    st.markdown("---")
    
    st.markdown("""
    ### Bienvenue sur Spotify Songs Recommender!
    
    Cette application vous permet de découvrir des chansons similaires à vos morceaux préférés 
    en utilisant des techniques avancées de **machine learning** et d'analyse audio.
    
    📱 Explorez la vaste bibliothèque de Spotify et découvrez de nouveaux artistes !
    """)
    
    st.markdown("---")
    
    # Section de recherche d'années
    st.markdown("## 🕰️ Spotify Time Machine")
    st.markdown("### Explorez la musique à travers les décennies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.slider(
            "📅 Année de début",
            min_value=1970,
            max_value=datetime.now().year,
            value=2013,
            step=1
        )
    
    with col2:
        end_year = st.slider(
            "📅 Année de fin",
            min_value=start_year,
            max_value=datetime.now().year,
            value=2020,
            step=1
        )
    
    st.markdown(f"### 🔍 Recherche de **{start_year}** à **{end_year}**")
    
    # Bouton de recherche
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        search_clicked = st.button(
            "🔍 Rechercher les Albums",
            type="primary",
            width='content'
        )
    
    if search_clicked:
        with st.spinner(f"🔍 Recherche des albums de {start_year} à {end_year}..."):
            albums = strategy_1_year_by_year(start_year, end_year)
            
            if albums:
                st.success(f"✅ {len(albums)} albums trouvés!")
                df_albums = process_albums_data(albums)
                st.session_state.df = df_albums
            else:
                st.warning("⚠️ Aucun album trouvé pour cette période.")
    
    # Affichage des onglets si des données sont disponibles
    if 'df' in st.session_state:
        st.markdown("---")
        
        df_albums = st.session_state.df
        df_albums = df_albums.drop_duplicates(subset=['Album_Name', 'Album_ID'])
        
        # Créer les onglets
        tab1, tab2, tab3 = st.tabs([
            "📀 Albums",
            "🎤 Artistes",
            "🎵 Recommandations"
        ])
        
        # ONGLET 1: Albums
        with tab1:
            with st.sidebar:
                st.markdown("### 🎨 Filtres")
                
                artists_list = sorted(df_albums["Artist"].unique().tolist())
                artist_name = st.selectbox(
                    label="Filtrer par artiste",
                    options=["Tous les artistes"] + artists_list,
                    index=0
                )
                
                if artist_name != "Tous les artistes":
                    df_artist = df_albums[df_albums['Artist'] == artist_name]
                else:
                    df_artist = df_albums.copy()
                
                # Métriques
                render_sidebar_metrics(df_albums)
            
            # Afficher la grille d'albums
            render_albums_grid(df_artist)
        
        # ONGLET 2: Artistes
        with tab2:
            if artist_name == "Tous les artistes":
                st.info("👈 Veuillez sélectionner un artiste spécifique dans la barre latérale.")
            else:
                render_artist_info(artist_name, df_artist)
                st.markdown("---")
                render_top_songs(artist_name, df_artist['Artist_id'].iloc[0])
        
        # ONGLET 3: Recommandations
        with tab3:
            render_recommendations_tab()
    
    else:
        st.info("👆 Veuillez d'abord rechercher des albums en utilisant le bouton ci-dessus.")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; color: #B3B3B3; margin-top: 50px;'>
            <p>Créé avec ❤️ en utilisant Streamlit et l'API Spotify</p>
            <p>🎵 Découvrez la musique à travers le temps</p>
        </div>
    """, unsafe_allow_html=True)

# Point d'entrée
if __name__ == "__main__":
    main()

