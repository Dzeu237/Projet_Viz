import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

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

# Sidebar for Spotify credentials
with st.sidebar:
    st.header("🔑 Spotify API Credentials")
    st.markdown("Get your credentials from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)")
    
    client_id = st.text_input("Client ID", type="password")
    client_secret = st.text_input("Client Secret", type="password")
    
    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. Enter your Spotify API credentials
    2. Select a year with the slider
    3. Click 'Search Tracks'
    4. Explore the results!
    """)

# Year slider
year = st.slider(
    "Select Year",
    min_value=2013,
    max_value=2020,
    value=2016,
    step=1
)

st.markdown(f"### 📅 Searching for tracks from **{year}**")

# Search button
if st.button("🔍 Search Tracks", type="primary", use_container_width=True):
    if not client_id or not client_secret:
        st.error("⚠️ Please enter your Spotify API credentials in the sidebar!")
    else:
        try:
            # Authenticate with Spotify
            with st.spinner("Connecting to Spotify..."):
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Search for tracks
            with st.spinner(f"Searching for tracks from {year}..."):
                results = sp.search(q=f"year:{year}", type="track", limit=20)
                tracks = results['tracks']['items']
            
            if tracks:
                st.success(f"✅ Found {len(tracks)} tracks from {year}!")
                
                # Display tracks
                for idx, track in enumerate(tracks, 1):
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Album artwork
                        if track['album']['images']:
                            st.image(track['album']['images'][0]['url'], width=150)
                        else:
                            st.markdown("🎵")
                    
                    with col2:
                        st.markdown(f"### {idx}. {track['name']}")
                        
                        # Artist(s)
                        artists = ", ".join([artist['name'] for artist in track['artists']])
                        st.markdown(f"**👤 Artist(s):** {artists}")
                        
                        # Album
                        st.markdown(f"**💿 Album:** {track['album']['name']}")
                        
                        # Release date
                        if 'release_date' in track['album']:
                            st.markdown(f"**📅 Release:** {track['album']['release_date']}")
                        
                        # Preview URL
                        if track['preview_url']:
                            st.audio(track['preview_url'])
                        
                        # Spotify link
                        st.markdown(f"[🎧 Open in Spotify]({track['external_urls']['spotify']})")
                    
                    st.markdown("---")
                
                # Create a DataFrame for download
                df_data = []
                for track in tracks:
                    df_data.append({
                        'Track Name': track['name'],
                        'Artist(s)': ", ".join([artist['name'] for artist in track['artists']]),
                        'Album': track['album']['name'],
                        'Release Date': track['album'].get('release_date', 'N/A'),
                        'Spotify URL': track['external_urls']['spotify']
                    })
                
                df = pd.DataFrame(df_data)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"spotify_tracks_{year}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning(f"No tracks found for {year}")
                
        except spotipy.exceptions.SpotifyException as e:
            st.error(f"❌ Spotify API Error: {str(e)}")
            st.info("Make sure your Client ID and Client Secret are correct.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #B3B3B3;'>
        <p>Made with ❤️ using Streamlit and Spotify API</p>
        <p>🎵 Discover music through time</p>
    </div>
""", unsafe_allow_html=True)
