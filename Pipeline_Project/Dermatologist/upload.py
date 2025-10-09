import streamlit as st

#Upload de fichiers
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
else:
    st.write("No file uploaded")