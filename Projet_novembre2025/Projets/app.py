import streamlit as st

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Plateforme d'Expérimentation NLP",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design épuré et professionnel
st.markdown("""
<style>
    /* Cacher la sidebar par défaut */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Navigation horizontale */
    .navbar {
        background: linear-gradient(90deg, #2E3192 0%, #1BFFFF 100%);
        padding: 5rem 10rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 1rem 2rem -1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .navbar-brand {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    
    .navbar-links {
        display: flex;
        gap: 1rem;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E3192 0%, #1BFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .project-card {
        background: white;
    
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #2E3192;
        transition: transform 0.3s ease;
    }
    .skill-badge {
        display: inline-block;
        background: #f0f2f6;
   
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        color: #2E3192;
        font-weight: 500;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .contact-link {
        color: #2E3192;
        text-decoration: none;
        font-weight: 500;
    }
    .section-title {
        color: #2E3192;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1BFFFF;
        display: inline-block;
    
    }
    .stButton>button {
        width: 100%;
    }
    
    /* Réduire l'espace en haut */
    .block-container {
        padding-top: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- ENTETE ---
st.markdown("---")
st.title("**Description**")
st.markdown("---")
st.write(
    """
    Cette plateforme d'expérimentation en traitement du langage naturel (NLP) est conçue pour permettre aux utilisateurs de tester et d'évaluer divers modèles et techniques NLP. 
    base sur des problematique d'entreprise telle que:
    * Analyse des tickets de support client pour identifier les tendances et améliorer le service.
    * Prédiction de l'adéquation des candidats aux postes à pourvoir en utilisant des modèles de matching avancés.
    * Analyse de sentiment des articles de presse et des publications sur les réseaux sociaux pour comprendre l'opinion publique.
    """
)

# --- Barre latérale de navigation ---
col_nav1, col_nav2, col_nav3 = st.columns([ 1, 1, 1])

with col_nav3:
    st.page_link(label="Financial Health Analysis", page="Pages\page1.py",icon="📊" ,use_container_width=True)

with col_nav1:
    st.page_link(label="Easy Model Prediction", page="Pages\page2.py",icon="🏠" ,use_container_width=True)

with col_nav2:
    st.page_link(label="Student AI Usage Trend", page="Pages\page3.py",icon="🤖" ,use_container_width=True)

