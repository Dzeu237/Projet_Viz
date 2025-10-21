import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder



# Configuration de la page Streamlit
st.set_page_config(layout="wide")

st.markdown('<h1 class="section-title">📊 Dashboard Student Performance Insights</h1>', unsafe_allow_html=True)
st.markdown("*Septembre 2024*")
st.title('Student Performance')
#Preparation des Donnees
path="StudentPerformanceFactors.csv"
data=pd.read_csv(path,delimiter=",").dropna()
data.columns=data.columns.str.strip()

def categorize_time_of_day(hour):
    if 9 <= hour <= 12:
        return 'Morning'
    elif 13 <= hour <= 15:
        return 'Afternoon'
    elif 16 <= hour < 18:
        return 'Evening'


def categorize_Notes(note):
    if 55 <= note <= 70:
        return 'Poor'
    elif 71 <= note <= 85:
        return 'Average'
    else:
        return 'Good'

    
def categorize_student(hour):
    if 1 <= hour <= 11:
        return 'Disengaged'
    elif 12 <= hour <= 22:
        return 'Moderate'
    elif 23 <= hour < 33:
        return 'Assiduous'
    else:
        return 'Nerd'

st.write('La performance des étudiants est un indicateur multidimensionnel qui reflète non seulement les résultats académiques'
         ', mais aussi le développement global de l/apprenant. Cette analyse examine les principaux facteurs'
          ' influençant la réussite scolaire et propose des recommandations pour l/optimisation des résultats.')  
    # Tabs pour organiser l'information

tab1, tab2, tab3, tab4 = st.tabs([ " 🔎Exploration & Transformation","📊 Visualisation","  Analyse", "💻 Tests & Démo"])

with tab1:
    st.markdown("## Exploration & Transformation des Données")
    st.markdown("### 1️⃣ Aperçu des Données")
    st.markdown("Le dataset contient des informations sur les performances académiques des étudiants, ainsi que sur divers facteurs"
                " pouvant influencer ces performances, tels que le temps d'étude, l'assiduité, le soutien familial et les ressources disponibles.")
    st.write(data.head(10))
    col1,col2,col3=st.columns(3)
    with col1:
        st.metric("Nombre de lignes", data.shape[0])
        st.metric("Nombre de colonnes", data.shape[1])
    with col2:
        st.metric("Nombre de valeurs manquantes", data.isnull().sum().sum())
    with col3:
        st.metric("Nombre de colonnes catégorielles", data.select_dtypes(include=['object']).shape[1])
        st.metric("Nombre de colonnes numériques", data.select_dtypes(include=['number']).shape[1])


        # Load Data & Transformation
    st.markdown("### 2️⃣ Transformation des Données")
    co1,col2,col3=st.columns(3)
    with co1:
        st.markdown("**Function to categorize Time_of_Day**")
        st.code("""def categorize_time_of_day(hour):
    if 9 <= hour <= 12:
        return 'Morning'
    elif 13 <= hour <= 15:
        return 'Afternoon'
    elif 16 <= hour < 18:
        return 'Evening'""", language="python")
    with col2:
        st.markdown("**Function to categorize Notes**")
        st.code("""def categorize_Notes(note):
    if 55 <= note <= 70:
        return 'Poor'
    elif 71 <= note <= 85:
        return 'Average'
    else:
        return 'Good'""", language="python")
    with col3:
        st.markdown("**Function to categorize Student_Note**")
        st.code("""def categorize_student(hour):
    if 1 <= hour <= 11:
        return 'Disengaged'
    elif 12 <= hour <= 22:
        return 'Moderate'
    elif 23 <= hour < 33:
        return 'Assiduous'
    else:
        return 'Nerd'""", language="python")

    data['Previous_Eval']=data['Previous_Scores'].apply(categorize_Notes)
    data['Exam_Eval']=data['Exam_Score'].apply(categorize_Notes)

    col= data['Hours_Studied'].apply(categorize_student)
    data.insert(0,'Student_Category',col)

    data['School_Type']=data['School_Type'].str.strip()
    data['Access_to_Resources']=data['Access_to_Resources'].str.strip()

    st.write("### 3️⃣ Résumé des Transformations Appliquées")
    st.write(data.head(10))
    st.markdown("""
    - **Categorisation des notes** : Mauvais, Moyen, Bon
    - **Categorisation du type d'étudiant** : Désengagé, Modéré, Assidu, Nerd""")

    # Function to categorize Student_Note

with tab2:
    a,b,c=st.columns(3)
    School_Type=a.selectbox(label='School Type',options=['All','Public', 'Private'])
    Category=b.selectbox(label='Category',options=['All','Disengaged','Moderate','Assiduous','Nerd'])
    Ressources=c.selectbox(label='Resources_Access',options=['All','High','Medium','Low'])
    #Update Dataframe
    filtred_data=data.copy()

# Apply School Type filter
    if School_Type != 'All':
        filtred_data = filtred_data[filtred_data['School_Type'] == School_Type]

    if Category != 'All':
       filtred_data = filtred_data[filtred_data['Student_Category'] == Category]
    
    # Apply Disabilities filter
    if  Ressources != 'All':
        filtred_data = filtred_data[filtred_data['Access_to_Resources'] == Ressources]

    # Measures Indicator
    total_count=len(filtred_data)
    Gender=filtred_data['Gender'].value_counts()
    Percent_Male=(Gender[0]/total_count)*100
    Percent_Female=(Gender[1]/total_count)*100
    attendance_rate=round(filtred_data['Attendance'].mean(),2)
    study_hour_average=filtred_data['Hours_Studied'].mean()
    exam_score=round(filtred_data['Exam_Score'].mean(),2)
    previous_score=round(filtred_data['Previous_Scores'].mean(),2)

    a,b,c=st.columns(3)
    a.metric(label='Male Percent',value=f"{round(Percent_Male,2)} %")
    b.metric(label='',value='',label_visibility='hidden')
    b.metric(label='Avarage Score',value=f"{exam_score} Points",delta=round((exam_score-previous_score),2))
    c.metric(label='Female Percent',value=f"{round(Percent_Female,2)} %")
    a.metric(label='Attendance Rate',value=f"{attendance_rate} %")
    c.metric(label='Average Study Hours',value=f"{study_hour_average:.0f} Hours")

    school=data[['School_Type','Family_Income']].groupby('Family_Income').value_counts().reset_index(name='Count').sort_values(by='Count',ascending=False)



    motivation=filtred_data[['Parental_Involvement','Access_to_Resources','Motivation_Level','Family_Income','Teacher_Quality','School_Type']]

    Access_to_Resources=motivation['Access_to_Resources'].value_counts().reset_index(name='Count')
    Motivation_Level=motivation['Motivation_Level'].value_counts().reset_index(name='Count')
    Family_Income=motivation['Family_Income'].value_counts().reset_index(name='Count')
    Teacher_Quality=motivation['Teacher_Quality'].value_counts().reset_index(name='Count')
    Extracurricular_Activities=filtred_data['Extracurricular_Activities'].value_counts().to_dict()


    a,b,c=st.columns(3)
    fig=go.Figure(data=[
        go.Pie(labels=list(Motivation_Level['Motivation_Level']), values=list(Motivation_Level['Count']), hole=0.6)
    ])
    fig.update_traces(hoverinfo="label+percent")
    # Add annotations in the center of the donut pies.
    fig.update_layout(
        annotations=[{
            'text': 'Motivation',
            'x': 0.5,
            'y': 0.5, 
            'font_size': 15, 
            'showarrow': False
        }]
    )

    a.write(fig)

    fig=go.Figure(data=[
        go.Pie(labels=list(Teacher_Quality['Teacher_Quality']), values=list(Teacher_Quality['Count']), hole=0.6)
    ])
    fig.update_traces(hoverinfo="label+percent")
    # Add annotations in the center of the donut pies.
    fig.update_layout(
        annotations=[{
            'text': 'Teacher Quality', 
            'x': 0.5, 
            'y': 0.5, 
            'font_size': 15, 
            'showarrow': False
        }]
    )
    b.write(fig)

    fig=go.Figure(data=[
        go.Pie(labels=list(Extracurricular_Activities.keys()), values=list(Extracurricular_Activities.values()), hole=0.6)
    ])
    fig.update_traces(hoverinfo="label+percent")
    # Add annotations in the center of the donut pies.
    fig.update_layout(
        annotations=[{
            'text': 'Extra Activities', 
            'x': 0.5, 
            'y': 0.5, 
            'font_size': 15, 
            'showarrow': False
        }]
    )
    c.write(fig)

    fig=go.Figure(data=[
        go.Bar(name='Motivation_Level',x=Motivation_Level['Motivation_Level'],y=Motivation_Level['Count']),
        go.Bar(name='Access_to_Resources',x=Access_to_Resources['Access_to_Resources'],y=Access_to_Resources['Count']),
        go.Bar(name='Teacher_Quality',x=Teacher_Quality['Teacher_Quality'],y=Teacher_Quality['Count']),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')

    # Define the order for the categories
    category_order = ['Low', 'Medium', 'High']

    # Update the figure with the specified category order
    fig.update_xaxes(categoryorder='array', categoryarray=category_order)
    fig.update_yaxes(title=None)
    fig.update_traces(hoverinfo='y')
    fig.update_layout(
        title={
            'text': 'Motivation',
            'y': 0.95,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 25}
        }
    )
    a,b=st.columns(2)
    options=['Sleep_Hours','Tutoring_Sessions','Physical_Activity']
    dat=[]
    for option in options:
        df=filtred_data[[f'{option}']].value_counts().reset_index(name='Count').sort_values(by=f'{option}')
        trace=go.Scatter(y=df['Count'], x=df[f'{option}'],name=f'{option}',text=df['Count'],fill='tozeroy')
        dat.append(trace)
    # df=filtred_data[[f'{option}']].value_counts().reset_index(name='Count').sort_values(by=f'{option}')
    fig=go.Figure(data=dat)
    fig.update_layout(title={
        'text':'School_Daylife',
        'y': 0.95,
        'x': 0.45,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30}})
    # Show the updated figure
    a.write(fig)

    previous=data['Previous_Eval'].value_counts().reset_index(name='Count')
    exam=data['Exam_Eval'].value_counts().reset_index(name='Count')

    fig=go.Figure(data=[
        go.Bar(y=previous['Count'], x=previous['Previous_Eval'],name='Previous Eval',text=previous['Count']),
        go.Bar(y=exam['Count'], x=exam['Exam_Eval'],name='Exam Eval',text=exam['Count'])
    ])
    fig.update_xaxes(categoryorder='array', categoryarray=['Poor','Average','Good'])
    fig.update_layout(title={
        'text':'Exam Evolution',
        'y': 0.95,
        'x': 0.45,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30}})
    b.write(fig)

    st.title('_Environment Overview_')

    Parental_Involvement=motivation['Parental_Involvement'].value_counts().reset_index(name='Count')
    Peer_Influence=filtred_data['Peer_Influence'].value_counts().to_dict()
    Parental_Education=filtred_data['Parental_Education_Level'].value_counts().to_dict()
    a,b,c=st.columns(3)
    fig1=go.Figure(data=[
        go.Pie(labels=list(Parental_Involvement['Parental_Involvement']), values=list(Parental_Involvement['Count']), hole=0.6,title={
            'text':'Parental Involvement',
            'font': {'size': 15}})])
    fig2=go.Figure(data=[
        go.Pie(labels=list(Parental_Education.keys()), values=list(Parental_Education.values()), hole=0.6,title={
            'text':'Parental Education',
            'font': {'size': 15}})
    ])
    fig3=go.Figure(data=[
        go.Pie(labels=list(Peer_Influence.keys()), values=list(Peer_Influence.values()), hole=0.6,title={
            'text':'Peer Influence',
            'font': {'size': 15}})
    ])
    a.write(fig1)
    b.write(fig2)
    c.write(fig3)

with tab3:
    st.markdown("Chox de la Variable")
    target_option=st.selectbox(label='Features',options=list(data.columns))
    st.markdown(f"Vous avez choisi **{target_option}** comme variable cible pour la prédiction.")
    st.markdown("### Distribution de la Variable Cible")
    if data[target_option].dtype=='object':
        target=data[target_option].value_counts().reset_index(name='Count')
        fig=go.Figure(data=[
            go.Bar(y=target['Count'], x=target[target_option],text=target['Count'])
        ])
        fig.update_xaxes(categoryorder='array', categoryarray=list(target[target_option]))
        fig.update_layout(title={
            'text':f'{target_option} Distribution',
            'y': 0.95,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}})
        st.write(fig)
    else:
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=data[target_option], nbinsx=20))
        fig.update_layout(title={
            'text':f'{target_option} Distribution',
            'y': 0.95,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}})
        st.write(fig)
