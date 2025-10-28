import streamlit as st

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Plateforme Job Prediction",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENTETE ---
st.markdown("---")
st.title("**Analyse de Prédiction d'Emploi**")
st.markdown("---")
st.write(
    """
Le système d'apprentissage continu s'améliore au fil des recrutements réussis, affinant 
constamment ses prédictions pour maximiser le taux de rétention et la satisfaction des 
nouveaux employés.\n
Plateforme de recrutement intelligente propulsée par l'IA qui transforme le matching 
candidat-poste. Notre technologie NLP analyse instantanément descriptions de postes et 
profils candidats pour révéler les meilleures correspondances.
* recrutements plus rapides, plus pertinents et débarrassés des biais humains.
* Gagnez jusqu'à 70% de temps sur la présélection tout en augmentant la qualité de votre candidature.
"""
)