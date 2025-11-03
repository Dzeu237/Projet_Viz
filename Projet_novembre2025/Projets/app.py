import streamlit as st

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Plateforme d'Expérimentation NLP",
    page_icon="💼",
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
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .challenge-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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


# Header principal
st.markdown("""
    <div class="main-header">
        <h1>📊 Avant-Propos</h1>
        <h3>Transformer les données en décisions stratégiques</h3>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("## 💡 Pourquoi ce portfolio ?")

st.markdown("""
Dans un monde professionnel où **80% des entreprises collectent des données** mais 
seulement **30% les exploitent efficacement**, le rôle du Data Analyst devient crucial. 
Ce portfolio présente des solutions concrètes aux défis quotidiens rencontrés par les organisations 
dans leur quête de transformation data-driven.

Chaque projet répond à une **problématique réelle du monde du travail**, démontrant comment l'analyse 
de données peut générer de la valeur mesurable et des décisions éclairées.
""")

st.divider()

# Les défis métier
st.markdown("## 🎯 Les défis métier adressés")

st.markdown("""
        <div class="challenge-box" style="border-left-color: #8b5cf6;">
            <h4>👥 Compréhension des comportements</h4>
            <p><strong>Problème :</strong> Les organisations éducatives et RH peinent à personnaliser 
            leurs approches par manque de segmentation.</p>
            <p><strong>Solution :</strong> Analyse comportementale révélant des patterns cachés pour 
            adapter stratégies pédagogiques et formations.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
        <div class="challenge-box" style="border-left-color: #10b981;">
            <h4>📈 Pilotage stratégique</h4>
            <p><strong>Problème :</strong> 67% des dirigeants manquent de visibilité sur la santé 
            réelle de leur entreprise.</p>
            <p><strong>Solution :</strong> Tableau de bord holistique fournissant une vision 360° 
            avec indicateurs actionnables pour anticiper les risques.</p>
        </div>
    """, unsafe_allow_html=True)
    
st.markdown("""
            <div class="challenge-box" style="border-left-color: #f59e0b;">
                <h4>💰 ROI mesurable</h4>
                <p><strong>Problème :</strong> Difficulté à prouver la valeur des investissements data.</p>
                <p><strong>Solution :</strong> Métriques concrètes (gain de temps, précision, réduction 
                coûts) pour justifier chaque initiative analytique.</p>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# Valeur ajoutée
st.markdown("## 💼 La valeur d'un Data Analyst dans l'entreprise moderne")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-box">
            <h2>25-30%</h2>
            <p>Amélioration moyenne de la productivité grâce à l'automatisation des analyses</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-box">
            <h2>15-20%</h2>
            <p>Réduction des coûts opérationnels via l'optimisation data-driven</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-box">
            <h2>3-5x</h2>
            <p>ROI typique des investissements en analytics selon McKinsey</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# Mon approche
st.markdown("## 🚀 Mon approche")

st.markdown("""
<div class="section-box">
    <h4>1️⃣ Orientation business</h4>
    <p>Chaque analyse commence par la compréhension de l'enjeu métier, pas par la technologie</p>
</div>

<div class="section-box">
    <h4>2️⃣ Accessibilité</h4>
    <p>Des outils pensés pour les utilisateurs métier, avec interfaces intuitives et automatisation</p>
</div>

<div class="section-box">
    <h4>3️⃣ Impact mesurable</h4>
    <p>Des métriques claires pour quantifier la valeur créée et justifier les investissements</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Call to action
st.markdown("### 📌 Découvrez comment ces projets peuvent s'appliquer à vos défis organisationnels")

# --- Barre latérale de navigation ---
col_nav1, col_nav2, col_nav3 = st.columns([ 1, 1, 1])

with col_nav1:
    if st.button(label="Financial Health Analysis" ,icon="💄" ,use_container_width=True):
        st.switch_page("pages/page1.py")

with col_nav2:
    if st.button(label="Easy Model Prediction",icon="🏠" ,use_container_width=True):
        st.switch_page("pages/page2.py")

with col_nav3:
    if st.button(label="Student AI Usage Trend",icon="🤖" ,use_container_width=True):
        st.switch_page("pages/page3.py")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💡 <em>Ce portfolio démontre des compétences concrètes en analyse de données, 
        machine learning et visualisation, applicables immédiatement en entreprise.</em></p>
    </div>
""", unsafe_allow_html=True)

