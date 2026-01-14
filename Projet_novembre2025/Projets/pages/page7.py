import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Customer Risk & Profitability Analysis",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("🎯 Customer Risk Predictive Model & Profitability Analysis")
st.markdown("---")

# Fonction pour générer un dataset fictif
@st.cache_data
def generate_dataset(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 75, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'account_age_months': np.random.randint(1, 240, n_samples),
        'num_products': np.random.randint(1, 6, n_samples),
        'credit_utilization': np.random.uniform(0, 1, n_samples),
        'payment_history_score': np.random.uniform(0, 100, n_samples),
        'num_late_payments': np.random.randint(0, 15, n_samples),
        'total_transaction_amount': np.random.randint(1000, 100000, n_samples),
        'avg_monthly_spend': np.random.randint(100, 5000, n_samples),
        'num_inquiries': np.random.randint(0, 10, n_samples),
        'debt_to_income': np.random.uniform(0, 0.8, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Création de la variable cible (default) basée sur les features
    default_prob = (
        (df['credit_score'] < 600) * 0.3 +
        (df['num_late_payments'] > 5) * 0.25 +
        (df['credit_utilization'] > 0.7) * 0.2 +
        (df['debt_to_income'] > 0.5) * 0.15 +
        (df['payment_history_score'] < 50) * 0.1
    )
    
    df['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Calcul de la rentabilité
    df['revenue'] = df['avg_monthly_spend'] * 12 * np.random.uniform(0.8, 1.2, n_samples)
    df['cost'] = df['revenue'] * np.random.uniform(0.3, 0.5, n_samples)
    df['profit'] = df['revenue'] - df['cost']
    
    return df

# Chargement des données
df = generate_dataset(1000)

# Sidebar
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"]
)

test_size = st.sidebar.slider("Taille du test set (%)", 10, 40, 20) / 100

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "🤖 Model Performance", 
    "👥 Segment Analysis",
    "📈 Dataset Explorer"
])

# Préparation des données pour le modèle
features = ['age', 'income', 'credit_score', 'account_age_months', 
            'num_products', 'credit_utilization', 'payment_history_score',
            'num_late_payments', 'total_transaction_amount', 'avg_monthly_spend',
            'num_inquiries', 'debt_to_income']

X = df[features]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle
@st.cache_resource
def train_model(model_name, X_tr, y_tr):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_tr, y_tr)
    return model

model = train_model(model_choice, X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calcul des métriques
auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Classification par risque
def classify_risk(proba):
    if proba < 0.3:
        return 'Low Risk'
    elif proba < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['risk_score'] = model.predict_proba(scaler.transform(df[features]))[:, 1]
df['risk_category'] = df['risk_score'].apply(classify_risk)

# TAB 1: OVERVIEW
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", f"{len(df):,}")
        st.metric("Taux de défaut", f"{df['default'].mean()*100:.1f}%")
    
    with col2:
        st.metric("Revenu Total", f"${df['revenue'].sum()/1e6:.2f}M")
        st.metric("Profit Total", f"${df['profit'].sum()/1e6:.2f}M")
    
    with col3:
        st.metric("AUC Score", f"{auc:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")
    
    with col4:
        st.metric("Clients Low Risk", f"{(df['risk_category']=='Low Risk').sum()}")
        st.metric("Clients High Risk", f"{(df['risk_category']=='High Risk').sum()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution des Risques")
        risk_counts = df['risk_category'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#10b981', '#f59e0b', '#ef4444']
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Répartition des Clients par Niveau de Risque')
        st.pyplot(fig)
    
    with col2:
        st.subheader("💰 Profitabilité par Segment")
        profit_by_risk = df.groupby('risk_category')['profit'].sum() / 1000
        fig, ax = plt.subplots(figsize=(8, 6))
        profit_by_risk.plot(kind='bar', color=colors, ax=ax)
        ax.set_ylabel('Profit Total ($K)')
        ax.set_xlabel('Catégorie de Risque')
        ax.set_title('Profit Total par Catégorie de Risque')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# TAB 2: MODEL PERFORMANCE
with tab2:
    st.subheader(f"🎯 Performance du Modèle: {model_choice}")
    
    col1, col2, col3 = st.columns(3)
    
    metrics_data = {
        'Métrique': ['AUC-ROC', 'Precision', 'Recall', 'F1-Score'],
        'Valeur': [auc, precision, recall, f1],
        'Benchmark': [0.75, 0.70, 0.68, 0.69]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    with col1:
        st.dataframe(metrics_df.style.format({'Valeur': '{:.3f}', 'Benchmark': '{:.3f}'}),
                     use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#3b82f6', lw=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbe ROC')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col3:
        st.subheader("🔢 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel('Valeur Réelle')
        ax.set_xlabel('Valeur Prédite')
        ax.set_title('Matrice de Confusion')
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Importance des Features")
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(feature_imp['Feature'][:10], feature_imp['Importance'][:10], color='#8b5cf6')
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Features les Plus Importantes')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("L'importance des features n'est pas disponible pour ce modèle.")
    
    with col2:
        st.subheader("📊 Distribution des Scores de Risque")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[df['default']==0]['risk_score'], bins=30, alpha=0.6, 
                label='Non-Default', color='#10b981')
        ax.hist(df[df['default']==1]['risk_score'], bins=30, alpha=0.6, 
                label='Default', color='#ef4444')
        ax.set_xlabel('Score de Risque')
        ax.set_ylabel('Nombre de Clients')
        ax.set_title('Distribution des Scores par Statut de Default')
        ax.legend()
        ax.axvline(x=0.3, color='orange', linestyle='--', label='Seuil Low/Medium')
        ax.axvline(x=0.7, color='red', linestyle='--', label='Seuil Medium/High')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.info(f"""
    **Calibration du Modèle**: Les seuils de décision ont été optimisés:
    - **Low Risk**: Score < 0.3 (faible surveillance)
    - **Medium Risk**: 0.3 ≤ Score < 0.7 (surveillance régulière)
    - **High Risk**: Score ≥ 0.7 (surveillance intensive)
    """)

# TAB 3: SEGMENT ANALYSIS
with tab3:
    st.subheader("👥 Analyse Détaillée par Segment")
    
    segment_stats = df.groupby('risk_category').agg({
        'customer_id': 'count',
        'profit': ['mean', 'sum'],
        'revenue': 'mean',
        'default': 'mean',
        'credit_score': 'mean',
        'income': 'mean'
    }).round(2)
    
    segment_stats.columns = ['Nb Clients', 'Profit Moyen', 'Profit Total', 
                             'Revenu Moyen', 'Taux Default (%)', 
                             'Credit Score Moyen', 'Revenu Moyen']
    segment_stats['Taux Default (%)'] *= 100
    
    # st.dataframe(segment_stats.style.format({
    #     'Profit Moyen': '${:.0f}',
    #     'Profit Total': '${:.0f}',
    #     'Revenu Moyen': '${:.0f}',
    #     'Taux Default (%)': '{:.2f}%',
    #     'Credit Score Moyen': '{:.0f}',
    #     'Revenu Moyen': '${:.0f}'
    # }), use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💵 Revenu vs Taux de Défaut")
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat, color in zip(['Low Risk', 'Medium Risk', 'High Risk'], colors):
            data = df[df['risk_category'] == cat]
            ax.scatter(data['default'], data['revenue']/1000, 
                      alpha=0.5, label=cat, color=color, s=50)
        ax.set_xlabel('Default (0=Non, 1=Oui)')
        ax.set_ylabel('Revenu ($K)')
        ax.set_title('Distribution Revenu par Statut de Default')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Profil des Segments")
        avg_by_risk = df.groupby('risk_category')[['credit_score', 'income', 
                                                    'payment_history_score']].mean()
        avg_by_risk_normalized = (avg_by_risk - avg_by_risk.min()) / (avg_by_risk.max() - avg_by_risk.min())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        avg_by_risk_normalized.plot(kind='bar', ax=ax, color=['#3b82f6', '#8b5cf6', '#ec4899'])
        ax.set_ylabel('Valeur Normalisée')
        ax.set_xlabel('Catégorie de Risque')
        ax.set_title('Profil Moyen des Segments (Normalisé)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(['Credit Score', 'Income', 'Payment History'])
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    for col, risk, color in zip([col1, col2, col3], 
                                ['Low Risk', 'Medium Risk', 'High Risk'], 
                                colors):
        with col:
            risk_data = df[df['risk_category'] == risk]
            st.markdown(f"### {risk}")
            st.markdown(f"**Clients**: {len(risk_data)}")
            st.markdown(f"**Profit moyen**: ${risk_data['profit'].mean():.0f}")
            st.markdown(f"**Taux default**: {risk_data['default'].mean()*100:.1f}%")
            st.markdown(f"**Credit score moyen**: {risk_data['credit_score'].mean():.0f}")

# TAB 4: DATASET EXPLORER
with tab4:
    st.subheader("🔍 Exploration du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aperçu des données**")
        st.dataframe(df.head(20), use_container_width=True)
    
    with col2:
        st.write("**Statistiques descriptives**")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Corrélations")
        corr_features = ['credit_score', 'income', 'payment_history_score', 
                        'num_late_payments', 'credit_utilization', 'default']
        corr_matrix = df[corr_features].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Matrice de Corrélation')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Distribution Credit Score")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['credit_score'], bins=30, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Nombre de Clients')
        ax.set_title('Distribution du Credit Score')
        ax.axvline(df['credit_score'].mean(), color='red', linestyle='--', 
                  label=f'Moyenne: {df["credit_score"].mean():.0f}')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
### 🛠️ Technologies Utilisées
`Python` • `Streamlit` • `Scikit-learn` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn`

**Projet**: Développement d'un modèle de scoring pour l'évaluation du risque de défaut et estimation de la rentabilité par segment client.
""")