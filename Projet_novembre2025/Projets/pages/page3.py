import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import prince

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Plateforme IA Stutent Usage Analytics",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENTETE ---
st.markdown("---")
st.title("**Analyse de l'usage de IA chez les etudiants**")
st.markdown("---")


#Chargement des données
@st.cache_data
def load_data():
    import pandas as pd
    df = pd.read_csv('https://github.com/Dzeu237/Projet_Viz/blob/main/Projet_novembre2025/Projets/Data/ai_assistant_usage_student_life.csv?raw=true')
    return df
df=load_data()
df_cleaned=df.drop(columns=['SessionID','SessionDate'])
df_cleaned['Satisfaction']=pd.cut(df_cleaned['SatisfactionRating'], bins=[1,2,3,5], labels=['Low','Medium','High'])
#Creation des colonnes annee et mois(en lettres)
def extract_date_features(df):
    df['Date_of_Purchase'] = pd.to_datetime(df['Date_of_Purchase'], errors='coerce')
    df['Year'] = df['Date_of_Purchase'].dt.year
    df['Month'] = df['Date_of_Purchase'].dt.month_name()
    return df

col_nav1, col_nav2= st.columns([ 1, 1],gap="Small")
with col_nav1:
        if st.button("🏠 Analytics Metrics", key="metrics", width='content'):
            st.session_state.page1 = 'Metrics'


with col_nav2:
        if st.button("👤 Segmentation Method", key="Segment", width='content'):
            st.session_state.page1 = 'Segmentation'




if 'page1'  in st.session_state:
    if st.session_state.page1 == 'Metrics':
        col1,col2,col3=st.columns([1,2,2],gap="Small",vertical_alignment="top")
        with col1:
            level=st.multiselect("IA Level",options=df_cleaned['AI_AssistanceLevel'].unique(),default=df_cleaned['AI_AssistanceLevel'].unique())
            discipline=st.multiselect("Discipline",options=df_cleaned['Discipline'].unique(),default=df_cleaned['Discipline'].unique())
            df_filtred=df_cleaned[df_cleaned['AI_AssistanceLevel'].isin(level) & df_cleaned['Discipline'].isin(discipline)]
            total_Student=df_filtred.shape[0]
            st.metric("Total Student",total_Student)
            st.metric("Average Satisfaction",f"{df_filtred['SatisfactionRating'].mean():.2f}/5")
        with col2:
            fig1=px.funnel_area(
                        names=df_filtred['StudentLevel'].value_counts().index,
                        values=df_filtred['StudentLevel'].value_counts().values
                        )
            st.plotly_chart(fig1,width='content')
            data=df_filtred.groupby('TaskType').agg({'SessionLengthMin':'mean','TotalPrompts':'mean'}).reset_index()
            fig4=go.Figure()

            fig4.add_trace(
                 go.Bar(name='Session Lenght',x=data['TaskType'], y=df_filtred['SessionLengthMin'])
                 )
                 
            fig4.add_trace(
                 go.Bar(name='Total Prompts',x=data['TaskType'], y=df_filtred['TotalPrompts'])
                 )
            fig4.update_layout(barmode='group', xaxis_tickangle=-45)
            
            # # Personnaliser les axes
            # fig4.update_xaxis(title_text="Votre axe X")
            # fig4.update_yaxis(title_text="Minutes", secondary_y=False)
            # fig4.update_yaxis(title_text="Nombre de prompts", secondary_y=True)

            # Titre et légende
            st.plotly_chart(fig4,width='content')
            
        with col3:
            fig2=px.pie(
                  names=df_filtred['FinalOutcome'].value_counts().index,
                  values=df_filtred['FinalOutcome'].value_counts().values,
                  hole=0.4
                  )
            st.plotly_chart(fig2,width='content')
            fig3=px.pie(
                  names=df_filtred['Satisfaction'].value_counts().index,
                  values=df_filtred['Satisfaction'].value_counts().values,
                  hole=0.4,
                  color_discrete_map={'Low':'red','Medium':'orange','High':'green'}
                  )
            st.plotly_chart(fig3,width='content')
    
    
    elif st.session_state.page1 == 'Segmentation':
        st.write(df_cleaned.head(10))
            # ===== EN-TÊTE AVEC MÉTRIQUES =====
        st.title("🎯 Segmentation des Sessions d'Apprentissage avec IA")
        st.markdown("---")

        # ===== PARTIE 1 : ACM =====
        st.header("1️⃣ Analyse des Correspondances Multiples (ACM)")

        with st.expander("ℹ️ Qu'est-ce que l'ACM ?", expanded=False):
            st.markdown("""
            L'ACM est une technique de réduction dimensionnelle pour les variables **catégorielles**. 
            Elle permet de visualiser les relations entre les différentes modalités et d'identifier des profils similaires.
            """)

        # Sélection des variables pour ACM
        categorical_vars = ['StudentLevel', 'Discipline', 'TaskType', 'AI_AssistanceLevel', 
                            'FinalOutcome', 'Satisfaction', 'UsedAgain']

        # Préparer les données pour ACM
        df_acm = df_cleaned[categorical_vars].copy()

        # Convertir AI_AssistanceLevel en catégorie
        df_acm['AI_AssistanceLevel'] = df_acm['AI_AssistanceLevel'].astype(str)

        # Remplacer les valeurs booléennes
        df_acm['UsedAgain'] = df_acm['UsedAgain'].map({True: 'Yes', False: 'No'})

        # Effectuer l'ACM
        @st.cache_data
        def perform_mca(df_acm, n_components=2):
            mca = prince.MCA(n_components=n_components, random_state=42)
            mca_result = mca.fit(df_acm)
            return mca, mca_result

        mca, mca_result = perform_mca(df_acm)

        # Obtenir les coordonnées
        row_coords = mca_result.transform(df_acm)

        # Variance expliquée
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variance Axe 1", f"{mca.eigenvalues_[0]:.1%}")
        with col2:
            st.metric("Variance Axe 2", f"{mca.eigenvalues_[1]:.1%}")
        with col3:
            st.metric("Variance Totale (2 axes)", f"{sum(mca.eigenvalues_[:2]):.1%}")

        # Graphiques ACM
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Projection des individus")
            
            fig_ind = px.scatter(
                x=row_coords[0],
                y=row_coords[1],
                opacity=0.5,
                labels={'x': f'Dimension 1 ({mca.eigenvalues_[0]:.1%})', 
                        'y': f'Dimension 2 ({mca.eigenvalues_[1]:.1%})'},
                title="Individus dans l'espace ACM"
            )
            fig_ind.update_traces(marker=dict(size=6, color='steelblue'))
            fig_ind.update_layout(height=500)
            st.plotly_chart(fig_ind, width='content')

        with col2:
            st.subheader("Projection des modalités")
            
            # Obtenir les coordonnées des modalités
            column_coords = mca.column_coordinates(df_acm)
            
            # Créer le dataframe pour les modalités
            modalities_df = pd.DataFrame({
                'Dim1': column_coords[0],
                'Dim2': column_coords[1],
                'Modalite': column_coords.index
            })
            
            # Extraire la variable d'origine
            modalities_df['Variable'] = modalities_df['Modalite'].str.split('_').str[0]
            
            fig_mod = px.scatter(
                modalities_df,
                x='Dim1',
                y='Dim2',
                color='Variable',
                text='Modalite',
                labels={'Dim1': f'Dimension 1 ({mca.eigenvalues_[0]:.1%})', 
                        'Dim2': f'Dimension 2 ({mca.eigenvalues_[1]:.1%})'},
                title="Modalités dans l'espace ACM"
            )
            fig_mod.update_traces(textposition='top center', marker=dict(size=10))
            fig_mod.update_layout(height=500)
            st.plotly_chart(fig_mod, width='content')

        st.markdown("---")

        # ===== PARTIE 2 : K-MEANS =====
        st.header("2️⃣ Clustering K-Means")

        with st.expander("ℹ️ Qu'est-ce que le K-Means ?", expanded=False):
            st.markdown("""
            Le K-Means est un algorithme de clustering qui regroupe les observations similaires en **K clusters**. 
            Nous l'appliquons sur les coordonnées ACM pour identifier des segments d'utilisateurs.
            """)

        # Utiliser les 2 premières dimensions ACM pour le clustering
        n_dims_clustering = 2
        X_clustering = row_coords[[i for i in range(n_dims_clustering)]]

        # Méthode du coude
        @st.cache_data
        def compute_elbow(X, max_k):
            inertias = []
            silhouettes = []
            K_range = range(2, max_k + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, kmeans.labels_))
            
            return list(K_range), inertias, silhouettes

        K_range, inertias, silhouettes = compute_elbow(X_clustering, max_k=8)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Méthode du coude")
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=K_range,
                y=inertias,
                mode='lines+markers',
                name='Inertie',
                line=dict(color='steelblue', width=3),
                marker=dict(size=10)
            ))
            fig_elbow.update_layout(
                xaxis_title="Nombre de clusters",
                yaxis_title="Inertie",
                height=400
            )
            st.plotly_chart(fig_elbow, width='content')

        with col2:
            st.subheader("Score de Silhouette")
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=K_range,
                y=silhouettes,
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='coral', width=3),
                marker=dict(size=10)
            ))
            fig_sil.update_layout(
                xaxis_title="Nombre de clusters",
                yaxis_title="Score de Silhouette",
                height=400
            )
            st.plotly_chart(fig_sil, width='content')

        # Clustering final avec K optimal
        #optimal_k = 3  # Vous pouvez ajuster selon les graphiques ci-dessus
        optimal_k = st.selectbox("Sélectionner le nombre de clusters (K)", options=range(2,9),index=1)
        st.info(f"""🎯 Nombre de clusters sélectionné : **{optimal_k}**\n
                # L'interpretation qui suit est uniquement pour 3 clusters libre adapter selon le K choisi et de definir une nouvelle interpretation """)

        # Effectuer le clustering
        @st.cache_data
        def perform_clustering(X, k):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            return labels, kmeans, score

        labels, kmeans, silhouette = perform_clustering(X_clustering, optimal_k)

        # Ajouter les labels au dataframe
        df_cleaned['Cluster'] = labels
    

        # Métriques du clustering
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col2:
            st.metric("Inertie", f"{kmeans.inertia_:.0f}")
        with col3:
            st.metric("Itérations", kmeans.n_iter_)
        st.write(df_cleaned.head(10))

        # Visualisation des clusters
        st.subheader("Visualisation des clusters")

        col1, col2 = st.columns(2)

        with col1:
            # Clusters sur axes ACM
            fig_clusters = px.scatter(
                x=row_coords[0],
                y=row_coords[1],
                color=labels.astype(str),
                labels={'x': f'Dimension 1', 'y': f'Dimension 2', 'color': 'Cluster'},
                title="Clusters projetés sur les axes ACM",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_clusters.update_traces(marker=dict(size=8))
            fig_clusters.update_layout(height=500)
            st.plotly_chart(fig_clusters, width='content')

        with col2:
            # Taille des clusters
            cluster_sizes = df_cleaned['Cluster'].value_counts().sort_index()
            fig_sizes = px.bar(
                x=cluster_sizes.index.astype(str),
                y=cluster_sizes.values,
                labels={'x': 'Cluster', 'y': 'Nombre de sessions'},
                title="Taille des clusters",
                color=cluster_sizes.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_sizes.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_sizes, width='content')

        st.markdown("---")

        # ===== PARTIE 3 : INTERPRÉTATION =====
        st.header("3️⃣ Interprétation des Résultats")

        # A. PROFILS DES SEGMENTS
        st.subheader("A. Profils des segments")

        # Créer des tabs pour chaque segment
        tabs = st.tabs([f"📊 Cluster {i+1}" for i in range(optimal_k)])

        for idx, tab in enumerate(tabs):
            with tab:
                cluster_data = df_cleaned[df_cleaned['Cluster'] == idx]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Sessions", 
                        f"{len(cluster_data)} ({len(cluster_data)/len(df_cleaned)*100:.1f}%)"
                    )
                with col2:
                    st.metric(
                        "Minutes moy.", 
                        f"{cluster_data['SessionLengthMin'].mean():.1f}"
                    )
                with col3:
                    st.metric(
                        "Prompts moy.", 
                        f"{cluster_data['TotalPrompts'].mean():.1f}"
                    )
                with col4:
                    st.metric(
                        "Niveau IA moy.", 
                        f"{cluster_data['AI_AssistanceLevel'].mean():.1f}"
                    )
                
                # Caractéristiques principales
                st.markdown("**Caractéristiques principales :**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Discipline la plus fréquente
                    top_discipline = cluster_data['Discipline'].mode()[0]
                    st.write(f"🎓 **Discipline dominante** : {top_discipline}")
                    
                    # TaskType le plus fréquent
                    top_task = cluster_data['TaskType'].mode()[0]
                    st.write(f"📝 **Type de tâche dominant** : {top_task}")
                    
                    # StudentLevel le plus fréquent
                    top_level = cluster_data['StudentLevel'].mode()[0]
                    st.write(f"👤 **Niveau étudiant dominant** : {top_level}")
                
                with col2:
                    # Satisfaction moyenne
                    satisfaction_dist = cluster_data['Satisfaction'].value_counts(normalize=True)
                    st.write(f"😊 **Satisfaction** :")
                    for sat, pct in satisfaction_dist.items():
                        st.write(f"   - {sat}: {pct*100:.1f}%")
                    
                    # UsedAgain
                    used_again_pct = cluster_data['UsedAgain'].mean() * 100
                    st.write(f"🔄 **Réutilisation** : {used_again_pct:.1f}%")

        # B. COMPARAISON VISUELLE
        st.subheader("B. Comparaison entre clusters")

        col1, col2 = st.columns(2)

        with col1:
            # Box plot pour SessionLengthMin
            fig_box1 = px.box(
                df_cleaned,
                x='Cluster',
                y='SessionLengthMin',
                color='Cluster',
                title="Durée des sessions par cluster",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box1.update_layout(showlegend=False)
            st.plotly_chart(fig_box1, width='content')

        with col2:
            # Box plot pour TotalPrompts
            fig_box2 = px.box(
                df_cleaned,
                x='Cluster',
                y='TotalPrompts',
                color='Cluster',
                title="Nombre de prompts par cluster",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box2.update_layout(showlegend=False)
            st.plotly_chart(fig_box2, width='content')

        # Distribution des variables catégorielles
        col1, col2 = st.columns(2)

        with col1:
            # Satisfaction par cluster
            satisfaction_cluster = pd.crosstab(
                df_cleaned['Cluster'], 
                df_cleaned['Satisfaction'], 
                normalize='index'
            ) * 100
            
            fig_sat = go.Figure()
            for col in satisfaction_cluster.columns:
                fig_sat.add_trace(go.Bar(
                    name=col,
                    x=satisfaction_cluster.index +1,
                    y=satisfaction_cluster[col]
                ))
            
            fig_sat.update_layout(
                title="Distribution de la Satisfaction par cluster",
                xaxis_title="Cluster",
                yaxis_title="Pourcentage (%)",
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig_sat, width='content')

        with col2:
            # TaskType par cluster
            task_cluster = pd.crosstab(
                df_cleaned['Cluster'], 
                df_cleaned['TaskType'], 
                normalize='index'
            ) * 100
            
            fig_task = go.Figure()
            for col in task_cluster.columns:
                fig_task.add_trace(go.Bar(
                    name=col,
                    x=task_cluster.index +1,
                    y=task_cluster[col]
                ))
            
            fig_task.update_layout(
                title="Distribution des Types de tâches par cluster",
                xaxis_title="Cluster",
                yaxis_title="Pourcentage (%)",
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig_task, width='content')

        # Tableau récapitulatif
        st.subheader("Tableau récapitulatif des segments")

        summary_data = []
        for i in range(optimal_k):
            cluster_data = df_cleaned[df_cleaned['Cluster'] == i]
            summary_data.append({
                'Segment': f'Cluster {i+1}',
                'Taille': f"{len(cluster_data)} ({len(cluster_data)/len(df_cleaned)*100:.1f}%)",
                'Minutes moy.': f"{cluster_data['SessionLengthMin'].mean():.1f}",
                'Prompts moy.': f"{cluster_data['TotalPrompts'].mean():.1f}",
                'Niveau IA moy.': f"{cluster_data['AI_AssistanceLevel'].mean():.1f}",
                'Satisfaction dominante': cluster_data['Satisfaction'].mode()[0],
                'Réutilisation': f"{cluster_data['UsedAgain'].mean()*100:.1f}%"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='content')
        st.session_state.fig=fig_clusters
        st.session_state.kmeans=kmeans
        st.session_state.categorical_vars=categorical_vars

        # C. INSIGHTS STRATÉGIQUES
        st.subheader("C. Insights & Recommandations")

        insights = {
            1: """
                ***Les Rédacteurs Efficaces*** (28% des étudiants)\n
    **Qui sont-ils ?**\n

    - Étudiants en informatique, niveau licence
    - Utilisent l'IA principalement pour rédiger (rapports, devoirs écrits)

    **Comment les reconnaître ?**\n

    - Sessions les plus longues (20 minutes en moyenne)
    - Utilisent peu l'IA (niveau 2.4/5) - préfèrent travailler seuls
    - Font peu de demandes à l'IA (5.7 prompts par session)
    - Satisfaction mitigée : la moitié est moyennement satisfaite, 38% peu satisfaits
    - Reviennent modérément sur l'outil (82% de réutilisation)

    **En résumé** \n
    - Des étudiants techniques qui utilisent l'IA en complément, pas comme outil principal."""
            ,

            2: """ ***Les Studieux Satisfaits*** (25% des étudiants)\n
    **Qui sont-ils ?**\n

    -Étudiants en mathématiques, niveau licence
    -Utilisent l'IA pour réviser et étudier

    **Comment les reconnaître ?**

    - Sessions moyennes (19.8 minutes)
    - Utilisation intermédiaire de l'IA (niveau 3.7/5)
    - Nombre de prompts moyen (5.6 par session)
    - Très satisfaits : 75% donnent une note élevée
    - Taux de retour le plus faible (22.6%)

    **En résumé**\n
    Des étudiants sérieux qui utilisent l'IA comme assistant d'étude et apprécient beaucoup, mais n'en deviennent pas dépendants.

    """ ,
            3: 
                """ ***Les Créatifs Engagés*** (47% des étudiants)\n
    **Qui sont-ils ?**

    - Étudiants en biologie, niveau licence
    - Utilisent l'IA pour rédiger des travaux créatifs

    **Comment les reconnaître ?**

    - Sessions les plus courtes mais les plus nombreuses (19.7 minutes)
    - Utilisation maximale de l'IA (niveau 4.0/5)
    - Nombre de prompts similaire (5.6 par session)
    - Satisfaction exceptionnelle : 94% très satisfaits, 0% insatisfaits
    - Reviennent systématiquement (90% de réutilisation)

    **En résumé**\n 
    Le groupe le plus important et le plus fidèle. Des étudiants qui ont pleinement adopté l'IA dans leur routine de travail et en sont ravis. """
            
        }

        for i,k in (insights.items()):
            with st.expander(f"💡 Cluster {i}"):
                st.write(k)

        # Bouton de téléchargement
        st.markdown("---")
        st.subheader("📥 Télécharger les résultats")

        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données avec clusters",
            data=csv,
            file_name="sessions_avec_clusters.csv",
            mime="text/csv",
        )

else:
    st.write(
        """
    Transformez votre support client en centre d'intelligence stratégique. Notre plateforme 
    analyse automatiquement vos tickets pour:
    * Révéler tendances cachées, problèmes systémiques et opportunités d'amélioration,
    *  Dashboards visuels, détection d'incidents récurrents, alertes proactives 
    """
    )
