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
 """Cette plateforme d'analyse prédictive combine des modèles de traitement du langage naturel (NLP) 
pour évaluer le sentiment des articles de presse et des publications sur les réseaux sociaux. 
Elle mesure l'opinion publique sur divers sujets et offre trois fonctionnalités clés : 
* la génération de nuages de mots thématiques,
* l'identification des tendances dominantes dans les textes,
* l'attribution de scores de sentiment (positif, négatif, neutre).\n

Plateforme d'analyse de sentiment alimentée par l'IA qui scrute articles de presse et réseaux 
sociaux pour décrypter l'opinion publique. Visualisez les thèmes émergents via des nuages de 
mots, identifiez les tendances clés et obtenez des scores de sentiment instantanés./n

Cette solution analytique s'appuie sur des algorithmes de NLP pour extraire et quantifier 
le sentiment exprimé dans les médias traditionnels et sociaux. Le système propose une analyse 
tridimensionnelle : cartographie thématique par nuage de mots, détection de tendances textuelles, 
et classification sentimentale tripartite (positif/négatif/neutre).\n

Plateforme qui analyse automatiquement les textes (articles, réseaux sociaux) pour comprendre 
l'opinion publique. 
* Elle crée des nuages de mots pour visualiser les thèmes principaux, 
* Détecte les sujets tendance et attribue un score de sentiment à chaque contenu."""
)