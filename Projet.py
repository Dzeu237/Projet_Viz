import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Project Template")
st.write("Welcome to your Streamlit project!")

# Add your project components here
with st.sidebar:
    st.title("Projet Files")
st.header("Repartion des Genres dans les metiers de l'Informatique en 2021")
st.write("En 2021, le secteur informatique demeure fortement masculinisé "
        "malgré une légère progression de la mixité. Les femmes représentent"
        "environ 20% des effectifs, avec une présence accrue dans certains domaines comme"
        " la gestion de projet (35%) et l'analyse de données (30%)."
        " Cette sous-représentation s'explique notamment par des facteurs socioculturels persistants"
        " et des stéréotypes de genre ancrés dès l'orientation scolaire."
        " La réduction de ces écarts constitue un enjeu majeur pour l'innovation et la compétitivité du secteur.")

#Data Load and Preparation
patch="employee_data.csv"
data=pd.read_csv(patch,delimiter=",")
data.drop('ID',axis=1)
st.write(data.head())
option=st.selectbox('Chose your position',options=data['Position'].unique())
df=data.groupby('Position').get_group(option)

# Calculate the percentage of males
total_count = len(df)
male_count = len(df[df['Gender'] == 'M'])
male_percentage = (male_count / total_count) * 100
female_percentage=  male_percentage - 1

a,b=st.columns(2)
a.metric(label="Male Percentage",value=f"{male_percentage:.2f}%")
b.metric(label="Female Percentage",value=f"{female_percentage:.2f}%")
a.metric(label="Min Salary",value=f'$ {df['Salary'].min():.0f}')
b.metric(label="Max Salary",value=f'$ {df['Salary'].max():.0f}')
st.title("Salary by Experience and Gender in "+option)

# Create a Seaborn pairplot
plot = sns.lineplot(data=df,y='Salary',x='Experience (Years)',hue='Gender',style='Gender',markers=True)
st.pyplot(plot.get_figure())
