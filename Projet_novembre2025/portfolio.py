import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Claude Dzeugueu | Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state.current_page='home'

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

# Navigation horizontale
col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_nav6 = st.columns([2, 1, 1, 1, 1, 1])

with col_nav1:
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #2E3192; padding: 0.5rem 0;">📊 Portfolio</div>', unsafe_allow_html=True)

with col_nav2:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.current_page = 'home'


with col_nav3:
    if st.button("👤 About", key="nav_about", use_container_width=True):
        st.session_state.current_page = 'about'


with col_nav4:
    if st.button("💼 Projects", key="nav_projects", use_container_width=True):
        st.session_state.current_page = 'projects'


with col_nav5:
    if st.button("📧 Contact", key="nav_contact", use_container_width=True):
        st.session_state.current_page = 'contact'


with col_nav6:
    st.markdown(f'<div style="text-align: center; padding: 0.5rem 0; color: #666; font-size: 0.9rem;">Stage Jan 2026</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== PAGE HOME ====================
if st.session_state.current_page == 'home':
    # Hero Section

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <img src="https://github.com/Dzeu237/Projet_Viz/blob/main/Projet_novembre2025/Assets/PP.JPG?raw=true" style="
            width: 250px;
            height: 250px;
            border-radius: 50%;
            object-fit: cover;
            border: 5px solid #2E3192;
            box-shadow: 0 10px 20px rgba(46, 49, 146, 0.3);
        ">
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h1 class="main-header">Claude Dzeugueu</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Data Analyst | Machine Learning Engineer</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Analyste passionné par la manipulation des données, je recherche des opportunités dans
        le domaine de la data pour participer à des projets porteurs de valeurs, en contribuant
        à la prise de décision basée sur la data.
        """)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("📍 **Paris et Périphéries**")
        with col_b:
            st.markdown("🎓 **EFREI Paris - M1**")
        with col_c:
            st.markdown("📅 **Stage dès Janvier 2026**")
    
    with col2:
        st.markdown("### 🎯 Recherche Active")
        st.info("**Stage Césure M1**\n\nData Analyst / BI Analyst\n\nJanvier 2026")
    
    st.markdown("---")
    
    # Technologies
    st.markdown("### 💻 Stack Technique")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🐍 Languages**")
        st.markdown('<span class="skill-badge">Python</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">R</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">JavaScript</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">SQL</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📊 Data & BI**")
        st.markdown('<span class="skill-badge">Power BI</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Streamlit</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">SAP</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Excel</span>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**🤖 ML & AI**")
        st.markdown('<span class="skill-badge">Scikit-learn</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">TensorFlow</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Keras</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Pandas</span>', unsafe_allow_html=True)

# ==================== PAGE ABOUT ====================
elif st.session_state.current_page == 'about':
    st.markdown('<h1 class="section-title">À Propos de Moi</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Formation
        **EFREI Paris** - Ingénieur Data & IA  
        *Septembre 2023 - Juillet 2025 | Villejuif, France*
        
        - Deuxième année de Cycle Ingénieur
        - Spécialité: Informatique
        - Majeure: **Business Intelligence & Analytics**
        
        **Matières clés:**
        - Data Visualisation
        - Machine Learning
        - Advanced Database
        - Optimisation
        - Gestion Financière
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Objectifs Professionnels")
        st.markdown("""
        Je suis à la recherche d'un **stage de césure M1** en tant que **Data Analyst** 
        pour mettre en pratique mes compétences en analyse de données, visualisation 
        et machine learning dans un environnement professionnel stimulant.
        
        Mon objectif est de contribuer à des projets data-driven qui génèrent un impact 
        mesurable sur la performance business.
        """)
    
    with col2:
        st.markdown("### 🌟 Soft Skills")
        
        
        st.markdown("### 🌍 Langues")
        st.markdown("🇫🇷 **Français** - Natif")
        st.markdown("🇬🇧 **Anglais** - TOEIC 825/990")

# ==================== PAGE PROJECTS LIST ====================
elif st.session_state.current_page == 'projects':
    st.markdown('<h1 class="section-title">Mes Projets</h1>', unsafe_allow_html=True)
    st.markdown("Découvrez une sélection de projets réalisés en data analysis, machine learning et business intelligence.")
    
    st.markdown("---")
    
    # Projet 1: Student Performance Dashboard
    st.markdown("""
    <div class="project-card">
        <h2>Easy Model Prediction</h2>
        <p style="color: #666; font-size: 0.9rem;">Septembre 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Description:**  
        Plateforme AutoML de Prédiction et Analyse de Données.
        
        **Réalisations clés:**
        - ✅ Exploration automatisée de données CSV
        - ✅ Implémentation de pipelines de preprocessing
        - ✅ Intégration de multiples algorithmes ML
        - ✅ Création d'un système de comparaison des modèles avec métriques de performance
        - ✅ Déploiement d'un module de prédiction sur échantillon de test
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">Streamlit</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Pandas</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Scikit-learn</span>', unsafe_allow_html=True)
        
        st.markdown("**📈 Métriques**")
        st.metric("Gain de temps", "75%")
        st.metric("Précision moyenne", "85-92%")
        

        
    
    st.markdown("---")
    
    # Projet 2: Retail Analytics
    st.markdown("""
    <div class="project-card">
        <h2>Tableau de Bord Décisionnel pour Diagnostic d'Entreprise</h2>
        <p style="color: #666; font-size: 0.9rem;">Janvier 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Description:**  
        Pipeline ETL complet pour l'analyse des performances commerciales d'une chaîne 
        de magasins.
        
        **Réalisations clés:**
        - ✅ Segmentation des metriques en 4 dimensions (Financière, Opérationnelle, Qualité, Commerciale)
        - ✅ Création de visualisations dynamiques avec des recommandations actionnables
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">Streamlit</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Ploty</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Pandas</span>', unsafe_allow_html=True)
        
        st.markdown("**📈 Métriques**")
        st.metric("Réduction délai décisionnel", "60%")
        st.metric("Couverture", "4 axes stratégiques")


    
    st.markdown("---")
    
    # Projet 3: Architecture Database
    st.markdown("""
    <div class="project-card">
        <h2>Analyse Comportementale : Utilisation de l'IA par les Étudiants</h2>
        <p style="color: #666; font-size: 0.9rem;">Novembre 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Description:**  
        Conception d'une architecture de base de données robuste pour une plateforme 
        de réseau social type Facebook, respectant les propriétés ACID.
        
        **Réalisations clés:**
        - ✅ Collecte et nettoyage de données d'utilisation de l'IA académique
        - ✅ Application d'algorithmes de clustering pour segmentation
        - ✅ Création de visualisations pour représentation multi-dimensionnelle
        - ✅ Production de recommandations pédagogiques par segment identifié
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">PostgreSQL</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Python</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">SQL</span>', unsafe_allow_html=True)
        
        st.markdown("**🎯 Concepts**")
        st.metric("Segmentation", "✓")
        st.metric("Variance expliquée", "78%")

    
    st.markdown("---")
    
    # Projets à venir
    st.markdown('<h2 class="section-title">🚀 Projets à Venir</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card" style="opacity: 0.7;">
            <h3>🔮 Projet 4 - Coming Soon</h3>
            <p>Espace réservé pour un futur projet d'analyse de données ou de machine learning.</p>
            <p style="color: #999;">📅 En développement...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="project-card" style="opacity: 0.7;">
            <h3>🔮 Projet 5 - Coming Soon</h3>
            <p>Espace réservé pour un futur projet d'analyse de données ou de machine learning.</p>
            <p style="color: #999;">📅 En développement...</p>
        </div>
        """, unsafe_allow_html=True)
    
# Call to action
st.markdown("### 📌 Découvrez comment ces projets peuvent s'appliquer à vos défis organisationnels")

if st.button("🔍 Explorer les projets", use_container_width=True, type="primary"):
    st.page_link(page="https://github.com/Dzeu237/Projet_Viz/blob/main/Projet_novembre2025/Projets/app.py",label="Voir le code source sur GitHub",width = "content")


# ==================== PAGE CONTACT ====================
elif st.session_state.current_page == 'contact':
    st.markdown('<h1 class="section-title">Me Contacter</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📬 Informations de Contact")
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin-top: 1rem;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                <strong>📧 Email:</strong><br>
                <a href="mailto:claude-bernard.dzeugueu@efrei.net" class="contact-link">
                    claude-bernard.dzeugueu@efrei.net
                </a>
            </p>
            
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                <strong>📱 Téléphone:</strong><br>
                <a href="tel:0744516792" class="contact-link">07 44 51 67 92</a>
            </p>
            
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                <strong>📍 Localisation:</strong><br>
                Paris et Périphéries, France
            </p>
            
            <p style="font-size: 1.1rem; margin-bottom: 0;">
                <strong>💼 LinkedIn:</strong><br>
                <a href="https://linkedin.com/in/claude-dzeugueu" class="contact-link" target="_blank">
                    linkedin.com/in/claude-dzeugueu
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Disponibilité")
        st.success("✅ Disponible pour un stage dès **Janvier 2026**")
        st.info("📅 Stage Césure M1 - 6 mois minimum")
    
    with col2:
        st.markdown("### ✉️ Envoyez-moi un message")
        
        with st.form("contact_form"):
            name = st.text_input("Nom *")
            email = st.text_input("Email *")
            company = st.text_input("Entreprise")
            subject = st.selectbox(
                "Sujet *",
                ["Stage/Alternance", "Projet collaboratif", "Question technique", "Autre"]
            )
            message = st.text_area("Message *", height=150)
            
            submitted = st.form_submit_button("📤 Envoyer le message")
            
            if submitted:
                if name and email and message:
                    st.success("✅ Message envoyé avec succès ! Je vous répondrai dans les plus brefs délais.")
                else:
                    st.error("⚠️ Veuillez remplir tous les champs obligatoires (*)")
        
        st.markdown("---")
        st.markdown("### 🔗 Réseaux Professionnels")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com)")
        with col_b:
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com)")
        with col_c:
            st.markdown("[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:claude-bernard.dzeugueu@efrei.net)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>© 2025 Claude Dzeugueu | Data Analyst Portfolio</p>
    <p style="font-size: 0.9rem;">Conçu avec Streamlit 🎈 | Propulsé par Python 🐍</p>
</div>
""", unsafe_allow_html=True)