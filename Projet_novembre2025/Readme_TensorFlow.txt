🧠 Concept de l'Agent IA
Type d'agent : Assistant de Classification de Tickets Support
Fonctionnalités :

Classifier automatiquement les demandes clients (Urgence, Département, Sentiment)
Générer des réponses appropriées
Apprendre de nouvelles données via réentraînement
Fournir un score de confiance pour ses décisions


📊 Sources de datasets
Dataset principal recommandé :
1. Customer Support Tickets (Kaggle)

URL : https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
Contenu : ~50,000 tickets avec catégories, priorités, sentiments

2. Twitter US Airline Sentiment

URL : https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
Contenu : 14,640 tweets classés (positif, négatif, neutre)

3. Complaints Dataset (Consumer Financial Protection Bureau)

URL : https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints
Contenu : Vraies plaintes consommateurs avec descriptions textuelles

4. Alternative : Générer données synthétiques

Utiliser ChatGPT/Claude pour créer 1000+ exemples de tickets


🏗️ Architecture du projet
agent-ia-support/
│
├── data/
│   ├── raw/
│   │   └── support_tickets.csv
│   ├── processed/
│   │   ├── tickets_clean.csv
│   │   └── embeddings.npy
│   └── models/
│       ├── tokenizer.pkl
│       └── label_encoder.pkl
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_agent_creation.ipynb
│   └── 05_demo_interactive.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── models/
│   │   ├── classifier.py          # Modèle Keras
│   │   ├── sentiment_analyzer.py  # Modèle sentiment
│   │   └── priority_detector.py   # Modèle priorité
│   ├── agent/
│   │   ├── agent_core.py          # Logique agent
│   │   ├── decision_maker.py      # Règles décision
│   │   └── response_generator.py  # Génération réponses
│   └── utils/
│       ├── text_processing.py
│       └── visualization.py
│
├── models/
│   ├── category_classifier.h5
│   ├── sentiment_model.h5
│   └── priority_model.h5
│
├── app/
│   └── streamlit_app.py           # Interface web
│
├── requirements.txt
└── README.md

📋 Structure des données
Format du dataset (support_tickets.csv) :
ColonneTypeDescriptionticket_idintID uniquecustomer_messagetextMessage du clientcategorytextTechnique/Facturation/Commercial/SAVprioritytextBasse/Moyenne/Haute/CritiquesentimenttextPositif/Neutre/Négatifresolution_timefloatTemps résolution (heures)customer_satisfactionintNote 1-5