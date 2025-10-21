import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime
import base64
from pathlib import Path
import re

# Configuration de la page
st.set_page_config(
    page_title="Spotify Match - Inscription/Connexion",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .profile-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .info-label {
        font-weight: bold;
        color: #1DB954;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        width: 100%;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
    </style>
""", unsafe_allow_html=True)

# Initialiser le dossier pour les uploads
UPLOAD_DIR = "user_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# === FONCTIONS BASE DE DONNÉES ===

def init_database():
    """Initialise la base de données avec toutes les tables nécessaires"""
    conn = sqlite3.connect('spotify_match.db')
    c = conn.cursor()
    
    # Table utilisateurs
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pseudo TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            telephone TEXT,
            niveau_etude TEXT,
            contrat_recherche TEXT,
            domaine TEXT,
            photo_path TEXT,
            cv_path TEXT,
            autres_fichiers TEXT,
            date_inscription TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            derniere_connexion TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash le mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Valide le format de l'email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Valide le format du téléphone"""
    pattern = r'^(\+33|0)[1-9](\d{2}){4}$'
    return re.match(pattern, phone.replace(" ", "")) is not None

def save_uploaded_file(uploaded_file, user_id, file_type):
    """Sauvegarde un fichier uploadé"""
    if uploaded_file is not None:
        user_folder = os.path.join(UPLOAD_DIR, f"user_{user_id}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        file_name = f"{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
        file_path = os.path.join(user_folder, file_name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None

def create_user(pseudo, email, password, telephone, niveau_etude, contrat_recherche, domaine, photo, cv, autres_fichiers):
    """Crée un nouvel utilisateur dans la base de données"""
    conn = sqlite3.connect('spotify_match.db')
    c = conn.cursor()
    
    try:
        # Insérer l'utilisateur d'abord pour obtenir l'ID
        c.execute('''
            INSERT INTO users (pseudo, email, password, telephone, niveau_etude, 
                             contrat_recherche, domaine)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (pseudo, email, hash_password(password), telephone, niveau_etude, 
              contrat_recherche, domaine))
        
        user_id = c.lastrowid
        
        # Sauvegarder les fichiers
        photo_path = save_uploaded_file(photo, user_id, "photo")
        cv_path = save_uploaded_file(cv, user_id, "cv")
        
        autres_paths = []
        if autres_fichiers:
            for idx, fichier in enumerate(autres_fichiers):
                path = save_uploaded_file(fichier, user_id, f"doc_{idx}")
                if path:
                    autres_paths.append(path)
        
        # Mettre à jour avec les chemins des fichiers
        c.execute('''
            UPDATE users 
            SET photo_path = ?, cv_path = ?, autres_fichiers = ?
            WHERE id = ?
        ''', (photo_path, cv_path, "|".join(autres_paths), user_id))
        
        conn.commit()
        return True, "Inscription réussie !"
    except sqlite3.IntegrityError as e:
        if "pseudo" in str(e):
            return False, "Ce pseudo est déjà utilisé"
        elif "email" in str(e):
            return False, "Cet email est déjà utilisé"
        else:
            return False, "Erreur lors de l'inscription"
    except Exception as e:
        return False, f"Erreur: {str(e)}"
    finally:
        conn.close()

def verify_user(email, password):
    """Vérifie les identifiants de connexion"""
    conn = sqlite3.connect('spotify_match.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM users 
        WHERE email = ? AND password = ?
    ''', (email, hash_password(password)))
    
    user = c.fetchone()
    
    if user:
        # Mettre à jour la dernière connexion
        c.execute('''
            UPDATE users 
            SET derniere_connexion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (user[0],))
        conn.commit()
    
    conn.close()
    return user

def get_user_by_id(user_id):
    """Récupère les informations d'un utilisateur par son ID"""
    conn = sqlite3.connect('spotify_match.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    
    conn.close()
    return user

# === INITIALISATION ===
init_database()

# Initialiser les états de session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'show_page' not in st.session_state:
    st.session_state.show_page = 'login'

# === INTERFACE ===

st.markdown('<div class="main-header">🎵 Spotify Match</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Connecte-toi avec des étudiants partageant tes goûts musicaux</div>', unsafe_allow_html=True)

# === PAGE DE PROFIL (si connecté) ===
if st.session_state.logged_in:
    st.markdown("---")
    
    # Récupérer les infos utilisateur
    user = get_user_by_id(st.session_state.user_id)
    
    if user:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="profile-card">
                <h2>👋 Bienvenue, {user[1]} !</h2>
                <p>Connecté en tant que: {user[2]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🚪 Déconnexion", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.rerun()
        
        # Afficher les informations du profil
        st.header("📋 Mon Profil")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="info-label">📧 Email</p>', unsafe_allow_html=True)
            st.write(user[2])
            
            st.markdown('<p class="info-label">📱 Téléphone</p>', unsafe_allow_html=True)
            st.write(user[4] if user[4] else "Non renseigné")
            
            st.markdown('<p class="info-label">🎓 Niveau d\'étude</p>', unsafe_allow_html=True)
            st.write(user[5] if user[5] else "Non renseigné")
            
            st.markdown('<p class="info-label">💼 Contrat recherché</p>', unsafe_allow_html=True)
            st.write(user[6] if user[6] else "Non renseigné")
        
        with col2:
            st.markdown('<p class="info-label">🏢 Domaine</p>', unsafe_allow_html=True)
            st.write(user[7] if user[7] else "Non renseigné")
            
            st.markdown('<p class="info-label">📅 Date d\'inscription</p>', unsafe_allow_html=True)
            st.write(user[11])
            
            st.markdown('<p class="info-label">🕐 Dernière connexion</p>', unsafe_allow_html=True)
            st.write(user[12] if user[12] else "Première connexion")
        
        # Section fichiers
        st.header("📎 Mes Documents")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if user[8]:  # photo_path
                st.success("✅ Photo de profil")
                if os.path.exists(user[8]):
                    with open(user[8], "rb") as f:
                        st.download_button("📥 Télécharger photo", f, file_name="photo.jpg")
            else:
                st.info("📷 Pas de photo")
        
        with col2:
            if user[9]:  # cv_path
                st.success("✅ CV")
                if os.path.exists(user[9]):
                    with open(user[9], "rb") as f:
                        st.download_button("📥 Télécharger CV", f, file_name="cv.pdf")
            else:
                st.info("📄 Pas de CV")
        
        with col3:
            if user[10]:  # autres_fichiers
                fichiers = user[10].split("|")
                st.success(f"✅ {len(fichiers)} document(s)")
                for idx, fichier in enumerate(fichiers):
                    if os.path.exists(fichier):
                        with open(fichier, "rb") as f:
                            st.download_button(f"📥 Doc {idx+1}", f, 
                                             file_name=os.path.basename(fichier),
                                             key=f"doc_{idx}")
            else:
                st.info("📁 Pas de documents")

# === PAGE DE CONNEXION/INSCRIPTION ===
else:
    # Boutons pour changer de page
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Connexion", use_container_width=True, 
                    type="primary" if st.session_state.show_page == 'login' else "secondary"):
            st.session_state.show_page = 'login'
            st.rerun()
    with col2:
        if st.button("📝 Inscription", use_container_width=True,
                    type="primary" if st.session_state.show_page == 'register' else "secondary"):
            st.session_state.show_page = 'register'
            st.rerun()
    
    st.markdown("---")
    
    # === FORMULAIRE DE CONNEXION ===
    if st.session_state.show_page == 'login':
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.header("🔐 Connexion")
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="exemple@email.com")
            password = st.text_input("🔒 Mot de passe", type="password")
            
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("⚠️ Veuillez remplir tous les champs")
                elif not validate_email(email):
                    st.error("⚠️ Format d'email invalide")
                else:
                    user = verify_user(email, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.success("✅ Connexion réussie !")
                        st.rerun()
                    else:
                        st.error("❌ Email ou mot de passe incorrect")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # === FORMULAIRE D'INSCRIPTION ===
    else:
        st.header("📝 Créer un compte")
        
        with st.form("register_form"):
            st.subheader("Informations personnelles")
            
            col1, col2 = st.columns(2)
            with col1:
                pseudo = st.text_input("👤 Pseudo *", placeholder="john_doe")
                email = st.text_input("📧 Email *", placeholder="exemple@email.com")
            
            with col2:
                password = st.text_input("🔒 Mot de passe *", type="password")
                confirm_password = st.text_input("🔒 Confirmer mot de passe *", type="password")
            
            telephone = st.text_input("📱 Téléphone", placeholder="+33 6 12 34 56 78")
            
            st.markdown("---")
            st.subheader("Formation et recherche")
            
            col1, col2 = st.columns(2)
            with col1:
                niveau_etude = st.selectbox("🎓 Niveau d'étude *", [
                    "Sélectionner...",
                    "Bac",
                    "Bac+1",
                    "Bac+2 (BTS/DUT)",
                    "Bac+3 (Licence)",
                    "Bac+4",
                    "Bac+5 (Master)",
                    "Bac+8 (Doctorat)",
                    "Autre"
                ])
                
                domaine = st.selectbox("🏢 Domaine *", [
                    "Sélectionner...",
                    "Informatique / Tech",
                    "Marketing / Communication",
                    "Design / Créatif",
                    "Business / Management",
                    "Ingénierie",
                    "Sciences",
                    "Droit",
                    "Finance",
                    "RH",
                    "Santé",
                    "Autre"
                ])
            
            with col2:
                contrat_recherche = st.selectbox("💼 Type de contrat recherché *", [
                    "Sélectionner...",
                    "Stage",
                    "Alternance",
                    "CDI",
                    "CDD",
                    "Freelance",
                    "VIE",
                    "Autre"
                ])
            
            st.markdown("---")
            st.subheader("Documents")
            
            col1, col2 = st.columns(2)
            with col1:
                photo = st.file_uploader("📷 Photo de profil", 
                                        type=['jpg', 'jpeg', 'png'],
                                        help="Format: JPG, PNG (max 5MB)")
            
            with col2:
                cv = st.file_uploader("📄 CV", 
                                     type=['pdf', 'doc', 'docx'],
                                     help="Format: PDF, DOC, DOCX (max 10MB)")
            
            autres_fichiers = st.file_uploader("📁 Autres documents (lettres de motivation, portfolio...)", 
                                              type=['pdf', 'doc', 'docx', 'jpg', 'png'],
                                              accept_multiple_files=True,
                                              help="Formats: PDF, DOC, DOCX, JPG, PNG (max 10MB chacun)")
            
            st.markdown("---")
            st.markdown("*Champs obligatoires")
            
            submit = st.form_submit_button("✨ Créer mon compte", use_container_width=True)
            
            if submit:
                # Validation
                errors = []
                
                if not all([pseudo, email, password, confirm_password]):
                    errors.append("⚠️ Veuillez remplir tous les champs obligatoires")
                
                if not validate_email(email):
                    errors.append("⚠️ Format d'email invalide")
                
                if telephone and not validate_phone(telephone):
                    errors.append("⚠️ Format de téléphone invalide (ex: 06 12 34 56 78)")
                
                if password != confirm_password:
                    errors.append("⚠️ Les mots de passe ne correspondent pas")
                
                if len(password) < 6:
                    errors.append("⚠️ Le mot de passe doit contenir au moins 6 caractères")
                
                if niveau_etude == "Sélectionner...":
                    errors.append("⚠️ Veuillez sélectionner un niveau d'étude")
                
                if contrat_recherche == "Sélectionner...":
                    errors.append("⚠️ Veuillez sélectionner un type de contrat")
                
                if domaine == "Sélectionner...":
                    errors.append("⚠️ Veuillez sélectionner un domaine")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Créer l'utilisateur
                    success, message = create_user(
                        pseudo, email, password, telephone,
                        niveau_etude, contrat_recherche, domaine,
                        photo, cv, autres_fichiers
                    )
                    
                    if success:
                        st.success("✅ " + message)
                        st.info("👉 Vous pouvez maintenant vous connecter !")
                        st.balloons()
                    else:
                        st.error("❌ " + message)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎵 Spotify Match - Connecte-toi autrement | Tous droits réservés © 2025</p>
        <p style='font-size: 0.8rem;'>🔒 Vos données sont sécurisées et ne seront jamais partagées</p>
    </div>
""", unsafe_allow_html=True)