import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.title("Repartion des Genres dans les metiers de l'Informatique en 2021")
st.write("En 2021, le secteur informatique demeure fortement masculinisé "
        "malgré une légère progression de la mixité. Les femmes représentent"
        "environ 20% des effectifs, avec une présence accrue dans certains domaines comme"
        " la gestion de projet (35%) et l'analyse de données (30%)."
        " Cette sous-représentation s'explique notamment par des facteurs socioculturels persistants"
        " et des stéréotypes de genre ancrés dès l'orientation scolaire."
        " La réduction de ces écarts constitue un enjeu majeur pour l'innovation et la compétitivité du secteur.")

#Data Load and Preparation
patch="Data/employee_data.csv"
data=pd.read_csv(patch,delimiter=",")
data.drop('ID',axis=1,inplace=True)

# General Overview
    # Calculate the percentage of males and females
total_count = len(data)
male_count = len(data[data['Gender'] == 'M'])
female_count = len(data[data['Gender'] == 'F'])
male_percentage = (male_count / total_count) * 100
female_percentage=( female_count/ total_count) * 100

st.title('General Overview:')
a,b,c=st.columns(3)
a.metric(label="Male Percentage",value=f"{male_percentage:.2f}%")
c.metric(label="Female Percentage",value=f"{female_percentage:.2f}%")
a.metric(label="Min Salary",value=f'$ {data['Salary'].min():.0f}')
c.metric(label="Max Salary",value=f'$ {data['Salary'].max():.0f}')
plot=sns.barplot(data=data,y='Salary',x='Position',errorbar=None,estimator='mean')
plt.xticks(rotation=90)
plt.xlabel(xlabel=None)
plt.figure(figsize=(8,5))

plot.axhline(y=data['Salary'].median(), color='b', linestyle='--', label=f'Median Salary: ${data['Salary'].median():.0f}')
plot.axhline(y=data['Salary'].mean(), color='r', linestyle='--', label=f'Average Salary: ${data['Salary'].mean():.0f}')
plot.legend(loc='upper left')
st.pyplot(plot.get_figure(),clear_figure=True)


# Overview by Expereince and Salary
st.title('Position Overview:')
option=st.selectbox('**Chose your position**',options=data['Position'].unique())
df=data.groupby('Position').get_group(option)
# Calculate the percentage of males and females
total_count = len(df)
male_count = len(df[df['Gender'] == 'M'])
female_count = len(df[df['Gender'] == 'F'])
male_percentage = (male_count / total_count) * 100
female_percentage=( female_count/ total_count) * 100

a,b,c=st.columns(3)
a.metric(label="Male Percentage",value=f"{male_percentage:.2f}%")
c.metric(label="Female Percentage",value=f"{female_percentage:.2f}%")
a.metric(label="Min Salary",value=f'$ {df['Salary'].min():.0f}')
c.metric(label="Max Salary",value=f'$ {df['Salary'].max():.0f}')
st.title("Salary by Experience and Gender in "+option)

a,b=st.columns([5,5])
# Create a Seaborn pairplot
plot = sns.lineplot(data=df,y='Salary',x='Experience (Years)',hue='Gender',style='Gender',markers=True)
plot.axhline(y=df['Salary'].median(), color='b', linestyle='--', label=f'Median Salary: ${df['Salary'].median():.0f}')
plot.axhline(y=df['Salary'].mean(), color='r', linestyle='--', label=f'Average Salary: ${df['Salary'].mean():.0f}')
plot.legend(loc='upper left')
a.pyplot(fig=plot.get_figure(),clear_figure=True)


#Create a Seaborn barPlot
tab=df[['Experience (Years)','Gender']].groupby('Gender').value_counts().reset_index(name='Gender_Count')
plot1=sns.barplot(data=tab,x='Experience (Years)', y='Gender_Count',hue='Gender',palette='viridis',errorbar=None)
b.pyplot(fig=plot1.get_figure(),clear_figure=True)
