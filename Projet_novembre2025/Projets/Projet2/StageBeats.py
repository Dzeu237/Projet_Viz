import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Configuration de la page
st.set_page_config(
    page_title="Spotify Match - Connexion Étudiants",
    page_icon="🎵",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">🎵 Spotify Match</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Connecte-toi avec des étudiants partageant tes goûts musicaux</div>', unsafe_allow_html=True)

# Sidebar pour la configuration
with st.sidebar:
    st.header("🔐 Configuration API Spotify")
    st.markdown("""
    Pour utiliser ce dashboard, vous avez besoin de:
    1. Créer une app sur [Spotify Developer](https://developer.spotify.com/dashboard)
    2. Obtenir vos identifiants Client ID et Client Secret
    """)
    
    client_id = st.text_input("Client ID", type="password")
    client_secret = st.text_input("Client Secret", type="password")
    
    st.markdown("---")
    st.header("👤 Ton Profil")
    user_name = st.text_input("Nom/Pseudo", placeholder="Ex: Marie Dupont")
    user_domain = st.selectbox("Domaine d'études", 
                               ["Informatique", "Marketing", "Design", "Business", 
                                "Ingénierie", "Sciences", "Arts", "Autre"])
    user_stage = st.selectbox("Type de stage recherché",
                             ["Développement", "Data Science", "Marketing Digital",
                              "Design UX/UI", "Gestion de projet", "Communication",
                              "Consulting", "Autre"])

# Initialisation de l'API Spotify
@st.cache_resource
def init_spotify(cid, secret):
    try:
        credentials = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(client_credentials_manager=credentials)
    except:
        return None

if client_id and client_secret:
    sp = init_spotify(client_id, client_secret)
    
    if sp:
        st.success("✅ Connexion à l'API Spotify réussie!")
        
        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyse Musicale", "🤝 Matching", "📊 Statistiques", "💡 Recommandations"])
        
        with tab1:
            st.header("Analyse de tes goûts musicaux")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_query = st.text_input("🎤 Recherche ton artiste préféré", 
                                            placeholder="Ex: Daft Punk, Stromae, Jul...")
                
            with col2:
                search_type = st.selectbox("Type", ["Artiste", "Track", "Album"])
            
            if search_query:
                type_map = {"Artiste": "artist", "Track": "track", "Album": "album"}
                results = sp.search(q=search_query, type=type_map[search_type], limit=10)
                
                if search_type == "Artiste" and results['artists']['items']:
                    st.subheader("📋 Résultats de recherche")
                    
                    artists_data = []
                    for artist in results['artists']['items']:
                        artists_data.append({
                            'Nom': artist['name'],
                            'Popularité': artist['popularity'],
                            'Followers': f"{artist['followers']['total']:,}",
                            'Genres': ', '.join(artist['genres'][:3]) if artist['genres'] else 'N/A',
                            'ID': artist['id']
                        })
                    
                    df_artists = pd.DataFrame(artists_data)
                    
                    selected_artist = st.selectbox(
                        "Sélectionne un artiste pour voir les détails",
                        df_artists['Nom'].tolist()
                    )
                    
                    if selected_artist:
                        artist_id = df_artists[df_artists['Nom'] == selected_artist]['ID'].values[0]
                        artist_info = sp.artist(artist_id)
                        top_tracks = sp.artist_top_tracks(artist_id, country='FR')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{artist_info['popularity']}</h3>
                                <p>Popularité</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            followers = artist_info['followers']['total']
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{followers:,}</h3>
                                <p>Followers</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            genres_count = len(artist_info['genres'])
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{genres_count}</h3>
                                <p>Genres</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.subheader("🎵 Top Tracks")
                        tracks_df = pd.DataFrame([
                            {
                                'Titre': t['name'],
                                'Album': t['album']['name'],
                                'Popularité': t['popularity'],
                                'Durée': f"{t['duration_ms']//60000}:{(t['duration_ms']//1000)%60:02d}"
                            }
                            for t in top_tracks['tracks'][:5]
                        ])
                        
                        fig = px.bar(tracks_df, x='Popularité', y='Titre', 
                                    orientation='h',
                                    title=f"Popularité des top tracks de {selected_artist}",
                                    color='Popularité',
                                    color_continuous_scale='greens')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(tracks_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.header("🤝 Trouve ton match musical")
            
            st.markdown("""
            <div class="recommendation-box">
            <h4>💡 Comment ça marche ?</h4>
            <p>Entre tes artistes préférés et découvre des étudiants qui partagent tes goûts musicaux. 
            C'est un excellent moyen de briser la glace et de créer des connexions professionnelles authentiques!</p>
            </div>
            """, unsafe_allow_html=True)
            
            if user_name:
                st.subheader(f"Profil de {user_name}")
                st.write(f"**Domaine:** {user_domain} | **Stage recherché:** {user_stage}")
                
                fav_artists = st.text_area(
                    "Liste tes 3-5 artistes préférés (un par ligne)",
                    placeholder="Daft Punk\nStromae\nChristine and the Queens\n..."
                )
                
                if fav_artists:
                    artists_list = [a.strip() for a in fav_artists.split('\n') if a.strip()]
                    
                    st.subheader("🎯 Analyse de compatibilité")
                    
                    # Simulation de profils d'étudiants (en production, cela viendrait d'une base de données)
                    sample_profiles = [
                        {"name": "Thomas L.", "domain": "Informatique", "stage": "Développement", 
                         "artists": ["Daft Punk", "Justice", "The Chemical Brothers"], "match": 85},
                        {"name": "Sophie M.", "domain": "Marketing", "stage": "Marketing Digital", 
                         "artists": ["Stromae", "Angèle", "Lomepal"], "match": 70},
                        {"name": "Lucas B.", "domain": "Design", "stage": "Design UX/UI", 
                         "artists": ["Christine and the Queens", "Phoenix", "Air"], "match": 65},
                        {"name": "Emma D.", "domain": "Business", "stage": "Consulting", 
                         "artists": ["Orelsan", "PNL", "Nekfeu"], "match": 45},
                    ]
                    
                    for profile in sample_profiles:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **{profile['name']}** - {profile['domain']} 
                            
                            Stage: {profile['stage']}
                            
                            Artistes: {', '.join(profile['artists'])}
                            """)
                        
                        with col2:
                            color = "green" if profile['match'] > 70 else "orange" if profile['match'] > 50 else "red"
                            st.markdown(f"<h2 style='color:{color};text-align:center'>{profile['match']}%</h2>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align:center'>Match</p>", unsafe_allow_html=True)
                        
                        st.markdown("---")
            else:
                st.warning("⚠️ Remplis ton profil dans la barre latérale pour commencer!")
        
        with tab3:
            st.header("📊 Statistiques et insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎼 Genres musicaux populaires")
                genres_data = {
                    'Genre': ['Pop', 'Hip-Hop/Rap', 'Électro', 'Rock', 'R&B', 'Indie'],
                    'Étudiants': [145, 132, 98, 87, 76, 54]
                }
                df_genres = pd.DataFrame(genres_data)
                
                fig = px.pie(df_genres, values='Étudiants', names='Genre',
                            title="Distribution des genres préférés",
                            color_discrete_sequence=px.colors.sequential.Greens_r)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Domaines d'études")
                domains_data = {
                    'Domaine': ['Informatique', 'Business', 'Marketing', 'Design', 'Ingénierie'],
                    'Nombre': [156, 134, 112, 98, 92]
                }
                df_domains = pd.DataFrame(domains_data)
                
                fig = px.bar(df_domains, x='Domaine', y='Nombre',
                           title="Étudiants par domaine",
                           color='Nombre',
                           color_continuous_scale='greens')
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📈 Tendances de connexion")
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
            connections = [45, 67, 89, 112, 134, 156]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=connections, mode='lines+markers',
                                    name='Connexions',
                                    line=dict(color='#1DB954', width=3),
                                    marker=dict(size=10)))
            fig.update_layout(title="Évolution des connexions étudiants",
                            xaxis_title="Mois",
                            yaxis_title="Nombre de connexions")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("💡 Recommandations personnalisées")
            
            st.markdown("""
            <div class="recommendation-box">
            <h4>🎧 Pourquoi utiliser la musique pour networker ?</h4>
            <ul>
                <li><strong>Briser la glace:</strong> La musique est un sujet universel et authentique</li>
                <li><strong>Découvrir des affinités:</strong> Les goûts musicaux révèlent souvent des valeurs communes</li>
                <li><strong>Créer du lien:</strong> Partager des playlists ou des recommandations renforce les connexions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📻 Suggestions de podcasts pour stages")
            
            podcasts = [
                {"title": "Le Gratin", "desc": "Interviews d'entrepreneurs et créatifs", "relevance": "Networking"},
                {"title": "Génération Do It Yourself", "desc": "Parcours de jeunes entrepreneurs", "relevance": "Inspiration"},
                {"title": "VLAN!", "desc": "Nouveau monde du travail", "relevance": "Carrière"},
            ]
            
            for podcast in podcasts:
                st.markdown(f"""
                **🎙️ {podcast['title']}**
                
                {podcast['desc']}
                
                *Pertinence: {podcast['relevance']}*
                """)
                st.markdown("---")
            
            st.subheader("🎵 Playlists recommandées pour travailler")
            
            playlist_query = st.text_input("Cherche une playlist", placeholder="focus, study, coding...")
            
            if playlist_query:
                playlists = sp.search(q=playlist_query, type='playlist', limit=5)
                
                for playlist in playlists['playlists']['items']:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{playlist['name']}**
                        
                        Par {playlist['owner']['display_name']} • {playlist['tracks']['total']} titres
                        """)
                    
                    with col2:
                        if playlist['external_urls']['spotify']:
                            st.markdown(f"[Écouter]({playlist['external_urls']['spotify']})")
                    
                    st.markdown("---")
    else:
        st.error("❌ Erreur de connexion à l'API Spotify. Vérifie tes identifiants.")
else:
    st.info("👈 Entre tes identifiants Spotify API dans la barre latérale pour commencer!")
    
    st.markdown("""
    ### 🚀 Comment démarrer ?
    
    1. **Crée une application Spotify:**
       - Va sur [Spotify for Developers](https://developer.spotify.com/dashboard)
       - Clique sur "Create app"
       - Note ton Client ID et Client Secret
    
    2. **Configure ton profil:**
       - Entre tes informations dans la barre latérale
       - Ajoute tes artistes préférés
    
    3. **Explore et connecte:**
       - Découvre des étudiants avec des goûts similaires
       - Partage des playlists et podcasts
       - Développe ton réseau professionnel!
    
    ### 🎯 Fonctionnalités
    
    - ✅ Analyse de goûts musicaux
    - ✅ Matching basé sur les artistes préférés
    - ✅ Statistiques de la communauté
    - ✅ Recommandations de podcasts et playlists
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎵 Spotify Match - Connecte-toi autrement | Propulsé par Spotify API</p>
    </div>
""", unsafe_allow_html=True)