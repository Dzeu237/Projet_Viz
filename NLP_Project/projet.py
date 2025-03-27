import streamlit as st
import streamlit_option_menu
import spacy
import pandas as pd


st.title('NLP Project')
st.title('__**Resume**__')
st.write('''Pour ce portfolio, nous allons présenter les différentes utilisations du NLP (Traitement du Langage Naturel) dans les problématiques d'entreprise.
Nous mettrons en lumière plusieurs projets innovants. Pour chacun de ces projets, nous détaillerons leur valeur ajoutée spécifique par rapport aux solutions existantes, en soulignant les gains en précision, en efficacité opérationnelle et en impact financier mesurable.''')

st.title('Projet I:**Text Analysis**')
patch='video_game_reviews.csv'
data=pd.read_csv(patch,delimiter=',')
st.write(data.head(5))
nlp=spacy.load('en_core_web_sm')
