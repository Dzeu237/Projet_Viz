import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Claude Dzeugueu | Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS personnalisé pour un design épuré et professionnel
st.markdown("""
<style>
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
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #2E3192;
        transition: transform 0.3s ease;
    }
    .skill-badge {
        display: inline-block;
        background: #f0f2f6;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        color: #2E3192;
        font-weight: 500;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
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
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("### 🎯 Navigation")
    
    # Gestion de la navigation
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    
    if st.button("👤 About", use_container_width=True):
        st.session_state.current_page = 'about'
        st.rerun()
    
    if st.button("💼 Projects", use_container_width=True):
        st.session_state.current_page = 'projects'
        st.rerun()
    
    if st.button("📧 Contact", use_container_width=True):
        st.session_state.current_page = 'contact'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Projets Réalisés", "3+")
    st.metric("Technologies", "15+")
    st.metric("TOEIC Score", "750/990")

# ==================== PAGE HOME ====================
if st.session_state.current_page == 'home':
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">Claude Dzeugueu</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Data Analyst | Machine Learning Engineer</p>', unsafe_allow_html=True)
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
        
        # Radar Chart des soft skills
        soft_skills = pd.DataFrame({
            'Compétence': ['Rigueur', 'Esprit Critique', 'Dynamisme', 
                          'Goût du Défi', 'Curiosité'],
            'Niveau': [90, 85, 88, 82, 95]
        })
        
        fig = go.Figure(data=go.Scatterpolar(
            r=soft_skills['Niveau'],
            theta=soft_skills['Compétence'],
            fill='toself',
            line_color='#2E3192',
            fillcolor='rgba(46, 49, 146, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🌍 Langues")
        st.markdown("🇫🇷 **Français** - Natif")
        st.markdown("🇬🇧 **Anglais** - TOEIC 750/990")

# ==================== PAGE PROJECTS LIST ====================
elif st.session_state.current_page == 'projects':
    st.markdown('<h1 class="section-title">Mes Projets</h1>', unsafe_allow_html=True)
    st.markdown("Découvrez une sélection de projets réalisés en data analysis, machine learning et business intelligence.")
    
    st.markdown("---")
    
    # Projet 1: Student Performance Dashboard
    st.markdown("""
    <div class="project-card">
        <h2>📊 Dashboard Student Performance Insights</h2>
        <p style="color: #666; font-size: 0.9rem;">Septembre 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Description:**  
        Plateforme d'analyse prédictive des performances étudiantes avec collecte 
        de données en temps réel et modèle de machine learning.
        
        **Réalisations clés:**
        - ✅ Collecte et préparation de données en temps réel depuis une source en ligne
        - ✅ Analyse et calcul d'indicateurs de performance pertinents
        - ✅ Identification des patterns et modèle prédictif avec **75% de précision**
        - ✅ Interface interactive pour visualiser les tendances
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">Streamlit</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Pandas</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Folium</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Scikit-learn</span>', unsafe_allow_html=True)
        
        st.markdown("**📈 Métriques**")
        st.metric("Précision ML", "75%")
        st.metric("Sources de données", "Temps réel")
        

        
    
    st.markdown("---")
    
    # Projet 2: Retail Analytics
    st.markdown("""
    <div class="project-card">
        <h2>🛒 Analyse Retail - Performances Multi-Magasins</h2>
        <p style="color: #666; font-size: 0.9rem;">Janvier 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Description:**  
        Pipeline ETL complet pour l'analyse des performances commerciales d'une chaîne 
        de magasins avec dashboard dynamique Power BI.
        
        **Réalisations clés:**
        - ✅ Pipeline d'extraction et traitement de données de ventes multi-magasins
        - ✅ Modélisation des données et analyse des KPIs commerciaux
        - ✅ Identification des tendances de vente par magasin et catégorie
        - ✅ Dashboard dynamique pour le suivi quotidien des performances
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">MySQL</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Python</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Power BI</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">OpenAI</span>', unsafe_allow_html=True)
        
        st.markdown("**📊 KPIs suivis**")
        st.metric("Magasins analysés", "Multiple")
        st.metric("Fréquence update", "Quotidien")

    
    st.markdown("---")
    
    # Projet 3: Architecture Database
    st.markdown("""
    <div class="project-card">
        <h2>🗄️ Architecture BD - Plateforme Réseau Social</h2>
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
        - ✅ Analyse approfondie des relations entre entités
        - ✅ Conception de tables et relations cohérentes
        - ✅ Optimisation des requêtes et indexation
        - ✅ Documentation détaillée des choix d'architecture ACID
        """)
    
    with col2:
        st.markdown("**🛠️ Technologies**")
        st.markdown('<span class="skill-badge">PostgreSQL</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">Python</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">SQL</span>', unsafe_allow_html=True)
        
        st.markdown("**🎯 Concepts**")
        st.metric("Propriétés ACID", "✓")
        st.metric("Scalabilité", "High")

    
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