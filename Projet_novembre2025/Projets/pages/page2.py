import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score

# --- Import des modèles ---
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Plateforme d'Expérimentation ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENTETE ---
st.markdown("---")
st.title("**Test Model**")
st.markdown("---")

# Fonction pour entraîner et évaluer les modèles de classification
def train_classification_models(X_train, y_train, X_test, y_test, models_to_run):
    results = {}
    for model_name, model in models_to_run.items():
        st.write(f"--- Entraînement de : **{model_name}** ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'metrics': {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1},
            'confusion_matrix': cm
        }
    return results

# Fonction pour entraîner et évaluer les modèles de régression
def train_regression_models(X_train, y_train, X_test, y_test, models_to_run):
    results = {}
    for model_name, model in models_to_run.items():
        st.write(f"--- Entraînement de : **{model_name}** ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'metrics': {'Mean Squared Error': mse, 'R2 Score': r2}
        }
    return results

# --- Barre latérale de navigation ---
col_nav2, col_nav3, col_nav4, col_nav5, col_nav6 = st.columns([ 1, 1, 1, 1, 1])

with col_nav2:
    if st.button("📥 Chargement des Donnees", width='content'):
        st.session_state.current_page = 'Load'

with col_nav3:
    if st.button("🔎 Exploration", width='content'):
        st.session_state.current_page = 'Explore'

with col_nav4:
    if st.button("🔄 Transformation", width='content'):
        st.session_state.current_page = 'Transform'

with col_nav5:
    if st.button("⚙️ Test des Modeles", width='content'):
        st.session_state.current_page = 'Model'

with col_nav6:
    if st.button("🧩 Test avec un echantillon", width='content'):
        st.session_state.current_page = 'Test'


if 'current_page' in st.session_state:
    # Initialisation du session_state pour stocker les données
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None

    # --- Section 1: Chargement des Données ---
    if st.session_state.current_page == 'Load':
        st.header("1. Chargement de votre jeu de données")
        st.write("Veuillez charger un fichier CSV pour commencer l'analyse.")
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("Fichier CSV chargé avec succès !")
                st.subheader("Aperçu des données")
                st.dataframe(df.head())
                st.subheader("Informations générales")
                st.write(f"Nombre de lignes : **{df.shape[0]}**")
                st.write(f"Nombre de colonnes : **{df.shape[1]}**")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

    # Vérifie si un DataFrame est chargé avant de passer aux autres sections
    if st.session_state.df is None and st.session_state.current_page != "Load":
        st.warning("⚠️ Veuillez d'abord charger un jeu de données dans la section 'Chargement des Données'.")
    else:
        df = st.session_state.df
        
        # --- Section 2: Exploration des Données ---
        if st.session_state.current_page == 'Explore':
            st.header("2. Exploration des Données (EDA)")
            numeric= df.select_dtypes(include=['number'])

            
            st.subheader("Statistiques descriptives")
            st.dataframe(df.describe())
            st.write(f"Nombre de colonne numerique:{numeric.shape[1]} ")
            st.write (f"Nombre de donnees categorielles:{df.shape[1]-numeric.shape[1]}")
            
            st.subheader("Visualisation des distributions")
            col_to_plot = st.selectbox("Sélectionnez une colonne à visualiser", df.columns)
            if col_to_plot:
                fig = px.histogram(df, x=col_to_plot, title=f'Distribution de {col_to_plot}')
                st.plotly_chart(fig)
                
            st.subheader("Matrice de corrélation")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matrice de corrélation des variables numériques",height=800)
                st.plotly_chart(fig_corr,width='content')
            else:
                st.info("Pas assez de colonnes numériques pour afficher une matrice de corrélation.")

        # --- Section 3: Transformation et Préparation ---
        elif st.session_state.current_page == "Transform":
            st.header("3. Transformation et Préparation des Données")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sélection de la Cible et des Caractéristiques")
                target_col = st.selectbox("1. Sélectionnez la variable cible (Y)", df.columns)
                
                features_cols = st.multiselect(
                    "2. Sélectionnez les caractéristiques (X)",
                    [col for col in df.columns if col != target_col],
                    default=[col for col in df.columns if col != target_col]
                )

            if target_col and features_cols:
                X = df[features_cols]
                y = df[target_col]

                # Identifier les types de colonnes
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

                with col2:
                    st.subheader("Configuration du Preprocessing")
                    imputation_method = st.radio("Méthode d'imputation pour les valeurs manquantes numériques :", ('mean', 'median', 'most_frequent'))
                    scaling_method = st.radio("Méthode de mise à l'échelle pour les variables numériques :", ('StandardScaler', 'None'))

                # Création du pipeline de preprocessing
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=imputation_method)),
                    ('scaler', StandardScaler() if scaling_method == 'StandardScaler' else 'passthrough')
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])

                st.session_state.pipeline = preprocessor
                st.session_state.X = X
                st.session_state.y = y
                st.success("Pipeline de transformation configuré !")
                st.write("Le pipeline est prêt. Allez à la section 'Test des Modèles' pour l'entraînement.")


        # --- Section 4: Test des Modèles ---
        elif st.session_state.current_page == "Model":
            st.header("4. Entraînement et Évaluation des Modèles")
            
            if 'pipeline' not in st.session_state or st.session_state.pipeline is None:
                st.warning("⚠️ Veuillez d'abord configurer le pipeline de transformation dans la section 'Transformation'.")
            else:
                problem_type = st.selectbox("Quel type de problème traitez-vous ?", ("Classification", "Régression"))

                models_to_run = {}
                if problem_type == "Classification":
                    available_models = {
                        "Régression Logistique": LogisticRegression(max_iter=1000),
                        "Random Forest Classifier": RandomForestClassifier(),
                        "Decision Tree Classifier": DecisionTreeClassifier(),
                        "K-Nearest Neighbors": KNeighborsClassifier()
                    }
                    selected_models = st.multiselect("Choisissez les modèles à tester", list(available_models.keys()), default=list(available_models.keys()))
                    models_to_run = {name: model for name, model in available_models.items() if name in selected_models}
                else: # Régression
                    available_models = {
                        "Régression Linéaire": LinearRegression(),
                        "Ridge": Ridge(),
                        "Random Forest Regressor": RandomForestRegressor()
                    }
                    selected_models = st.multiselect("Choisissez les modèles à tester", list(available_models.keys()), default=list(available_models.keys()))
                    models_to_run = {name: model for name, model in available_models.items() if name in selected_models}

                if st.button("🚀 Lancer l'entraînement et l'évaluation"):
                    with st.spinner("Entraînement en cours..."):
                        pipeline = st.session_state.pipeline
                        X = st.session_state.X
                        y = st.session_state.y

                        # Encodage de la variable cible si c'est de la classification et qu'elle est catégorielle
                        if problem_type == "Classification" and y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            st.session_state.label_encoder = le # Sauvegarder pour plus tard
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Application du pipeline de preprocessing
                        X_train_processed = pipeline.fit_transform(X_train)
                        X_test_processed = pipeline.transform(X_test)

                        if problem_type == "Classification":
                            results = train_classification_models(X_train_processed, y_train, X_test_processed, y_test, models_to_run)
                        else:
                            results = train_regression_models(X_train_processed, y_train, X_test_processed, y_test, models_to_run)
                        
                        st.session_state.results = results
                        st.success("Entraînement terminé !")

            if st.session_state.results:
                st.subheader("📈 Résultats de l'évaluation")
                
                # Afficher les métriques dans un tableau
                metrics_data = {model: res['metrics'] for model, res in st.session_state.results.items()}
                metrics_df = pd.DataFrame(metrics_data).T
                st.dataframe(metrics_df)

                # Trouver le meilleur modèle
                if problem_type == "Classification":
                    best_model_name = metrics_df['F1-Score'].idxmax()
                else: # Régression
                    best_model_name = metrics_df['R2 Score'].idxmax()
                
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model = st.session_state.results[best_model_name]['model']

                st.success(f"🏆 Meilleur modèle : **{best_model_name}**")
                
                # Afficher les matrices de confusion pour la classification
                if problem_type == "Classification":
                    for model_name, res in st.session_state.results.items():
                        st.write(f"**Matrice de confusion pour {model_name}**")
                        fig = px.imshow(res['confusion_matrix'], text_auto=True, title=f"Matrice de confusion - {model_name}")
                        st.plotly_chart(fig)


        # --- Section 5: Test sur un échantillon ---
        elif st.session_state.current_page == "Test":
            st.header("5. Tester le meilleur modèle sur de nouvelles données")
            
            if 'best_model' not in st.session_state or st.session_state.best_model is None:
                st.warning("⚠️ Veuillez d'abord entraîner des modèles dans la section 'Test des Modèles'.")
            else:
                st.info(f"Le meilleur modèle, **{st.session_state.best_model_name}**, est prêt à être utilisé.")
                
                X_cols = st.session_state.X.columns
                input_data = {}
                
                st.subheader("Entrez les valeurs de l'échantillon :")
                for col in X_cols:
                    # Gérer les colonnes numériques
                    if st.session_state.df[col].dtype in ['int64', 'float64']:
                        input_data[col] = st.number_input(f"Valeur pour {col}", value=float(st.session_state.df[col].mean()))
                    # Gérer les colonnes catégorielles avec une liste déroulante
                    else:
                        unique_vals = st.session_state.df[col].unique()
                        input_data[col] = st.selectbox(f"Valeur pour {col}", options=unique_vals)

                if st.button("🔮 Faire une prédiction"):
                    input_df = pd.DataFrame([input_data])
                    
                    # Appliquer le même pipeline de transformation
                    pipeline = st.session_state.pipeline
                    input_processed = pipeline.transform(input_df)
                    
                    # Faire la prédiction
                    prediction = st.session_state.best_model.predict(input_processed)
                    prediction_proba = None
                    if hasattr(st.session_state.best_model, "predict_proba"):
                        prediction_proba = st.session_state.best_model.predict_proba(input_processed)
                    
                    # Afficher le résultat
                    st.subheader("Résultat de la prédiction")
                    
                    # Si la cible a été encodée, décoder le résultat
                    if 'label_encoder' in st.session_state and hasattr(st.session_state, 'label_encoder'):
                        final_prediction = st.session_state.label_encoder.inverse_transform(prediction)
                        st.success(f"La prédiction est : **{final_prediction[0]}**")
                    else:
                        st.success(f"La prédiction est : **{prediction[0]}**")
                    
                    if prediction_proba is not None:
                        st.write("Probabilités de prédiction :")
                        # Afficher les probabilités pour chaque classe
                        if 'label_encoder' in st.session_state:
                            st.dataframe(pd.DataFrame(prediction_proba, columns=st.session_state.label_encoder.classes_))
                        else:
                            st.write(prediction_proba)