import streamlit as st
import pandas as pd
import re
from kaggle.api.kaggle_api_extended import KaggleApi

def is_valid_search_term(term):
    # Vérifie que la chaîne n'est pas vide
    if not term:
        st.write('Please enter a valid search term with no special characters or numbers.')
        return False
    # Vérifie qu'il n'y a pas de caractères spéciaux ou de chiffres
    if re.search(r'[^a-zA-Z]', term):
        st.write('Please enter a valid search term with no special characters or numbers.')
        return False
    return True

# Configuration de la page Streamlit
st.set_page_config(layout="centered")
st.title("DataSet Search Results")
st.write('''For this repository, we use different APIs from popular websites like Kaggle, GitHub, etc.,
         to download datasets and store them for future transformation and analysis.''')

st.write('**Kaggle API:**')
st.write('''Kaggle is the world’s largest data science community with powerful tools and resources to help you achieve your data science goals.
         Here we search for datasets using different terms and display the results in a DataFrame.''')

# Interface utilisateur
a, b = st.columns(2,vertical_alignment='bottom')
search = a.text_input(label='**Kaggle Search**', placeholder="Enter a Term", key='Text')
search_button = b.button('Search')

if 'load_state' not in st.session_state:
    st.session_state.load_state = False

# Configuration de l'API Kaggle
api = KaggleApi()
api.authenticate()

if search_button or st.session_state.load_state:
    st.session_state.load_state = True
    search_term = search
    if is_valid_search_term(search_term):
        datasets = api.dataset_list(search=search_term, sort_by="hottest", page=5, file_type="all")
        data = []
        for dataset in datasets:
            data.append({
                "Title": dataset.title,
                "Ref": dataset.ref,
                "Size": dataset.size,
                "Last Updated": dataset.lastUpdated,
                "Download Count": dataset.downloadCount
            })
        if len(data) == 0:
            st.write("No dataset found for the search term.")
        else:
            df = pd.DataFrame(data, columns=["Title", "Ref", "Size", "Last Updated", "Download Count"]).sort_values(by="Last Updated", ascending=False)
            st.write(df.set_index('Title'))
            st.session_state.df = df  # Store the DataFrame in session state

            option = df['Title']
            download = st.multiselect(label='**Kaggle Download**', options=option, key='Download_data')

            if download:
                dataset_refs = df['Ref'][df['Title'].isin(download)]
                if st.button("Download Selected Datasets"):
                    with st.spinner('Downloading...'):
                        for dataset_ref in dataset_refs:
                            # Download the dataset
                            api.dataset_download_files(dataset_ref, path='.\Data\Raw_Data', unzip=False,force=True)
                        st.success('Selected datasets downloaded successfully!')