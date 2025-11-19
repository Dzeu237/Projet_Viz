import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Analyse Bancaire", layout="wide", page_icon="🏦")

# Titre principal
st.title("🏦 Plateforme d'Analyse Bancaire")
st.markdown("---")

# Chargement des données
@st.cache_data
def load_data(patch:str):
    # Vous devrez remplacer ceci par le chemin vers votre fichier CSV
    # df = pd.read_csv('votre_fichier.csv')

    df = pd.read_csv(patch)
    return df

df = load_data("D:\Document\EFREI\Semestre 7 et 8\Projet_Viz\Projet_novembre2025\Projets\Data\Customer-Churn-Records.csv")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", 
                        ["📊 Aperçu des Données", 
                         "🔧 Transformation", 
                         "📈 Visualisations", 
                         "🤖 Modèles Prédictifs"])

# PAGE 1: APERÇU DES DONNÉES
if page == "📊 Aperçu des Données":
    st.header("📊 Aperçu des Données")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de clients", len(df))
    with col2:
        st.metric("Taux de sortie", f"{df['Exited'].mean()*100:.1f}%")
    with col3:
        st.metric("Âge moyen", f"{df['Age'].mean():.1f} ans")
    with col4:
        st.metric("Score crédit moyen", f"{df['CreditScore'].mean():.0f}")
    
    st.subheader("Échantillon des données")
    st.dataframe(df.head(20), width="stretch")
    
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(), width="stretch")
    
    st.subheader("Informations sur les colonnes")
    info_df = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.values,
        'Valeurs non-nulles': df.count().values,
        'Valeurs nulles': df.isnull().sum().values
    })
    st.dataframe(info_df, width="stretch")

# PAGE 2: TRANSFORMATION
elif page == "🔧 Transformation":
    st.header("🔧 Transformation des Données")
    
    st.subheader("Colonnes actuelles")
    st.write(list(df.columns))
    
    # Sélection des colonnes à supprimer
    st.subheader("Supprimer des colonnes")
    cols_to_drop = st.multiselect(
        "Sélectionner les colonnes à supprimer",
        df.columns.tolist(),
        default=['RowNumber', 'CustomerId', 'Surname']
    )
    
    # Création de nouvelles colonnes
    st.subheader("Créer de nouvelles colonnes")
    
    add_features = st.checkbox("Ajouter des colonnes dérivées")
    
    if add_features:
        st.write("**Nouvelles colonnes à créer:**")
        st.write("- **BalanceCategory**: Catégorie du solde (Bas/Moyen/Élevé)")
        st.code("""
            df_transformed['BalanceCategory'] = pd.cut(
                df_transformed['Balance'], 
                bins=[0, 50000, 100000, float('inf')],
                labels=['Bas', 'Moyen', 'Élevé']
            )
        """,language="python")

        st.write("- **AgeGroup**: Groupe d'âge (Jeune/Adulte/Senior)")
        st.code(""" 
            df_transformed['AgeGroup'] = pd.cut(
                df_transformed['Age'],
                bins=[0, 30, 45, 100],
                labels=['Jeune', 'Adulte', 'Senior']
            )
""",language="python")
        
        st.write("- **CreditScoreCategory**: Catégorie du score de crédit")
        st.code("""
            df_transformed['CreditScoreCategory'] = pd.cut(
                df_transformed['CreditScore'],
                bins=[0, 600, 700, 850],
                labels=['Faible', 'Moyen', 'Élevé']
            )
""",language="python")
        st.write("- **BalancePerProduct**: Solde moyen par produit")
        st.code("""
            df_transformed['BalancePerProduct'] = df_transformed['Balance'] / df_transformed['NumOfProducts'].replace(0, 1)
""",language="python")
        
        st.write("- **Randomize result of loan authorization**: Generer aleatoirement le resultat de la demande de pret")
        st.code("""probs = df["CreditScore"] / df["CreditScore"].sum()
            df["category"] = np.random.choice(
                [0, 1],
                size=len(df),
            p=[1 - probs.mean(), probs.mean()]).astype(str)""",language='python')

    
    if st.button("Appliquer les transformations"):
        df_transformed = df.copy()
        
        # Supprimer les colonnes
        df_transformed = df_transformed.drop(columns=cols_to_drop, errors='ignore')
        
        
        # Ajouter les nouvelles colonnes
        if add_features:
            # Balance Category
            df_transformed['BalanceCategory'] = pd.cut(
                df_transformed['Balance'], 
                bins=[0, 50000, 100000, float('inf')],
                labels=['Bas', 'Moyen', 'Élevé']
            )
            
            # Age Group
            df_transformed['AgeGroup'] = pd.cut(
                df_transformed['Age'],
                bins=[0, 30, 45, 100],
                labels=['Jeune', 'Adulte', 'Senior']
            )
            
            # Credit Score Category
            df_transformed['CreditScoreCategory'] = pd.cut(
                df_transformed['CreditScore'],
                bins=[0, 600, 700, 850],
                labels=['Faible', 'Moyen', 'Élevé']
            )
            
            # Balance per Product
            df_transformed['BalancePerProduct'] = df_transformed['Balance'] / df_transformed['NumOfProducts'].replace(0, 1)

            #Randomize loan demand
            probs = df["CreditScore"] / df["CreditScore"].sum()
            df["category"] = np.random.choice(
                [0, 1],
                size=len(df),
            p=[1 - probs.mean(), probs.mean()]).astype(str)
        
        st.success("Transformations appliquées avec succès!")
        st.subheader("Données transformées")
        st.dataframe(df_transformed.head(20), width="stretch")
        
        # Sauvegarder dans session_state
        st.session_state['df_transformed'] = df_transformed

# PAGE 3: VISUALISATIONS
elif page == "📈 Visualisations":
    st.header("📈 Visualisations")
    
    # Utiliser les données transformées si disponibles
    df_viz = st.session_state.get('df_transformed', df)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Relations", "Comparaisons", "Carte de chaleur"])
    
    with tab1:
        st.subheader("Distribution des variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution de l'âge
            fig_age = px.histogram(
                df_viz, 
                x='Age', 
                color='Exited',
                title='Distribution de l\'âge par statut de sortie',
                labels={'Exited': 'A quitté la banque'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_age, width="stretch")
            
            # Distribution du score de crédit
            fig_credit = px.histogram(
                df_viz,
                x='CreditScore',
                color='Geography',
                title='Distribution du Score de Crédit par Géographie',
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_credit, width="stretch")
        
        with col2:
            # Distribution du solde
            fig_balance = px.box(
                df_viz,
                x='Geography',
                y='Balance',
                color='Exited',
                title='Distribution du Solde par Géographie et Sortie'
            )
            st.plotly_chart(fig_balance, width="stretch")
            
            # Répartition par type de carte
            card_counts = df_viz['Card Type'].value_counts()
            fig_card = px.pie(
                values=card_counts.values,
                names=card_counts.index,
                title='Répartition par Type de Carte'
            )
            st.plotly_chart(fig_card, width="stretch")
    
    with tab2:
        st.subheader("Relations entre variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Relation Âge vs Solde
            fig_scatter1 = px.scatter(
                df_viz,
                x='Age',
                y='Balance',
                color='Exited',
                size='EstimatedSalary',
                title='Relation Âge vs Solde (taille = Salaire estimé)',
                labels={'Exited': 'A quitté'},
                opacity=0.6
            )
            st.plotly_chart(fig_scatter1, width="stretch")
        
        with col2:
            # Relation Score de Crédit vs Salaire
            fig_scatter2 = px.scatter(
                df_viz,
                x='CreditScore',
                y='EstimatedSalary',
                color='Geography',
                title='Score de Crédit vs Salaire Estimé',
                opacity=0.6
            )
            st.plotly_chart(fig_scatter2, width="stretch")
    
    with tab3:
        st.subheader("Comparaisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Taux de sortie par géographie
            exit_by_geo = df_viz.groupby('Geography')['Exited'].mean() * 100
            fig_geo = px.bar(
                x=exit_by_geo.index,
                y=exit_by_geo.values,
                title='Taux de Sortie par Géographie (%)',
                labels={'x': 'Géographie', 'y': 'Taux de sortie (%)'},
                color=exit_by_geo.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_geo, width="stretch")
            
            # Satisfaction par type de carte
            satisfaction_card = df_viz.groupby('Card Type')['Satisfaction Score'].mean()
            fig_sat = px.bar(
                x=satisfaction_card.index,
                y=satisfaction_card.values,
                title='Score de Satisfaction Moyen par Type de Carte',
                labels={'x': 'Type de Carte', 'y': 'Score moyen'},
                color=satisfaction_card.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_sat, width="stretch")
        
        with col2:
            # Salaire moyen par genre et géographie
            salary_by_gender_geo = df_viz.groupby(['Geography', 'Gender'])['EstimatedSalary'].mean().reset_index()
            fig_salary = px.bar(
                salary_by_gender_geo,
                x='Geography',
                y='EstimatedSalary',
                color='Gender',
                title='Salaire Moyen par Genre et Géographie',
                barmode='group'
            )
            st.plotly_chart(fig_salary, width="stretch")
            
            # Points gagnés par type de carte
            points_card = df_viz.groupby('Card Type')['Point Earned'].mean()
            fig_points = px.bar(
                x=points_card.index,
                y=points_card.values,
                title='Points Moyens Gagnés par Type de Carte',
                labels={'x': 'Type de Carte', 'y': 'Points moyens'},
                color=points_card.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_points, width="stretch")
    
    with tab4:
        st.subheader("Carte de chaleur des corrélations")
        
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = df_viz[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            title='Matrice de Corrélation',
            labels=dict(color="Corrélation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_heatmap, width="stretch")

# PAGE 4: MODÈLES PRÉDICTIFS
elif page == "🤖 Modèles Prédictifs":
    st.header("🤖 Modèles Prédictifs - Prédiction de Sortie Client")
    
    st.info("Objectif: Prédire si un client quittera la banque (Exited = 1) ou non (Exited = 0)")
    
    # Préparation des données
    df_model = df.copy()
    
    # Encodage des variables catégorielles
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    le_card = LabelEncoder()
    
    df_model['Geography_encoded'] = le_geo.fit_transform(df_model['Geography'])
    df_model['Gender_encoded'] = le_gender.fit_transform(df_model['Gender'])
    df_model['CardType_encoded'] = le_card.fit_transform(df_model['Card Type'])
    
    # Sélection des features
    feature_cols = ['CreditScore', 'Geography_encoded', 'Gender_encoded', 'Age', 
                   'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
                   'IsActiveMember', 'EstimatedSalary', 'CardType_encoded',
                   'Satisfaction Score', 'Point Earned']
    
    X = df_model[feature_cols]
    y = df_model['Exited']
    
    # Split des données
    test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.subheader("Configuration des données")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Données d'entraînement", len(X_train))
    with col2:
        st.metric("Données de test", len(X_test))
    with col3:
        st.metric("Features utilisées", len(feature_cols))
    
    # Choix du modèle
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Régression Logistique", "Arbre de Décision", "Comparer les deux"]
    )
    
    if st.button("Entraîner le(s) modèle(s)"):
        with st.spinner("Entraînement en cours..."):
            
            if model_choice in ["Régression Logistique", "Comparer les deux"]:
                st.subheader("📊 Régression Logistique")
                
                # Entraînement
                lr_model = LogisticRegression(max_iter=1000, random_state=42)
                lr_model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_lr = lr_model.predict(X_test)
                accuracy_lr = accuracy_score(y_test, y_pred_lr)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Précision", f"{accuracy_lr*100:.2f}%")
                    
                    # Matrice de confusion
                    cm_lr = confusion_matrix(y_test, y_pred_lr)
                    fig_cm_lr = px.imshow(
                        cm_lr,
                        labels=dict(x="Prédit", y="Réel", color="Nombre"),
                        x=['Reste', 'Sort'],
                        y=['Reste', 'Sort'],
                        title='Matrice de Confusion - Régression Logistique',
                        text_auto=True,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm_lr, width="stretch")
                
                with col2:
                    # Rapport de classification
                    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
                    st.write("**Rapport de classification:**")
                    st.dataframe(pd.DataFrame(report_lr).transpose())
                    
                    # Importance des features (coefficients)
                    feature_importance_lr = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': np.abs(lr_model.coef_[0])
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp_lr = px.bar(
                        feature_importance_lr.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features (Régression Logistique)'
                    )
                    st.plotly_chart(fig_imp_lr, width="stretch")
            
            if model_choice in ["Arbre de Décision", "Comparer les deux"]:
                st.subheader("🌳 Arbre de Décision")
                
                # Paramètres de l'arbre
                max_depth = st.sidebar.slider("Profondeur maximale de l'arbre", 2, 20, 5)
                
                # Entraînement
                dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                dt_model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_dt = dt_model.predict(X_test)
                accuracy_dt = accuracy_score(y_test, y_pred_dt)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Précision", f"{accuracy_dt*100:.2f}%")
                    
                    # Matrice de confusion
                    cm_dt = confusion_matrix(y_test, y_pred_dt)
                    fig_cm_dt = px.imshow(
                        cm_dt,
                        labels=dict(x="Prédit", y="Réel", color="Nombre"),
                        x=['Reste', 'Sort'],
                        y=['Reste', 'Sort'],
                        title='Matrice de Confusion - Arbre de Décision',
                        text_auto=True,
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_cm_dt, width="stretch")
                
                with col2:
                    # Rapport de classification
                    report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
                    st.write("**Rapport de classification:**")
                    st.dataframe(pd.DataFrame(report_dt).transpose())
                    
                    # Importance des features
                    feature_importance_dt = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': dt_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp_dt = px.bar(
                        feature_importance_dt.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features (Arbre de Décision)'
                    )
                    st.plotly_chart(fig_imp_dt, width="stretch")
            
            if model_choice == "Comparer les deux":
                st.subheader("📊 Comparaison des Modèles")
                
                comparison_df = pd.DataFrame({
                    'Modèle': ['Régression Logistique', 'Arbre de Décision'],
                    'Précision': [accuracy_lr*100, accuracy_dt*100]
                })
                
                fig_comparison = px.bar(
                    comparison_df,
                    x='Modèle',
                    y='Précision',
                    title='Comparaison des Précisions',
                    text='Précision',
                    color='Précision',
                    color_continuous_scale='Viridis'
                )
                fig_comparison.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig_comparison, width="stretch")
                
                if accuracy_lr > accuracy_dt:
                    st.success(f"🏆 La Régression Logistique performe mieux avec {accuracy_lr*100:.2f}% de précision")
                else:
                    st.success(f"🏆 L'Arbre de Décision performe mieux avec {accuracy_dt*100:.2f}% de précision")

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ en utilisant Streamlit, Plotly et Scikit-learn")