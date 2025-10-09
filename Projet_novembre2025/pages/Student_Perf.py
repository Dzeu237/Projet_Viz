import streamlit as st
import pandas as pd
import plotly.graph_objects as go

if st.button("← Retour aux projets"):
        st.switch_page('portfolio.py')
        st.rerun()

def categorize_time_of_day(hour):
    if 9 <= hour <= 12:
        return 'Morning'
    elif 13 <= hour <= 15:
        return 'Afternoon'
    elif 16 <= hour < 18:
        return 'Evening'

# Configuration de la page Streamlit
st.set_page_config(layout="wide")

st.title('Student Performance')
st.write('La performance des étudiants est un indicateur multidimensionnel qui reflète non seulement les résultats académiques'
         ', mais aussi le développement global de l/apprenant. Cette analyse examine les principaux facteurs'
          ' influençant la réussite scolaire et propose des recommandations pour l/optimisation des résultats.')

#Preparation des Donnees
path="StudentPerformanceFactors.csv"
data=pd.read_csv(path,delimiter=",").dropna()
data.columns=data.columns.str.strip()
    # Function to categorize Student_Note
def categorize_Notes(note):
    if 55 <= note <= 70:
        return 'Poor'
    elif 71 <= note <= 85:
        return 'Average'
    else:
        return 'Good'
data['Previous_Eval']=data['Previous_Scores'].apply(categorize_Notes)
data['Exam_Eval']=data['Exam_Score'].apply(categorize_Notes)
    
def categorize_student(hour):
    if 1 <= hour <= 11:
        return 'Disengaged'
    elif 12 <= hour <= 22:
        return 'Moderate'
    elif 23 <= hour < 33:
        return 'Assiduous'
    else:
        return 'Nerd'

col= data['Hours_Studied'].apply(categorize_student)
data.insert(0,'Student_Category',col)


st.title('General OverView')
a,b,c=st.columns(3)
School_Type=a.selectbox(label='School Type',options=['All','Public', 'Private'])
Category=b.selectbox(label='Category',options=['All','Disengaged','Moderate','Assiduous','Nerd'])
Ressources=c.selectbox(label='Resources_Access',options=['All','High','Medium','Low'])

data['School_Type']=data['School_Type'].str.strip()
data['Access_to_Resources']=data['Access_to_Resources'].str.strip()

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
a,b,c=st.columns(3)


motivation=filtred_data[['Parental_Involvement','Access_to_Resources','Motivation_Level','Family_Income','Teacher_Quality','School_Type']]

Access_to_Resources=motivation['Access_to_Resources'].value_counts().reset_index(name='Count')
Motivation_Level=motivation['Motivation_Level'].value_counts().reset_index(name='Count')
Family_Income=motivation['Family_Income'].value_counts().reset_index(name='Count')
Teacher_Quality=motivation['Teacher_Quality'].value_counts().reset_index(name='Count')
Extracurricular_Activities=filtred_data['Extracurricular_Activities'].value_counts().to_dict()

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
a,b=st.columns(2)
fig1=go.Figure(data=[
    go.Pie(labels=list(Parental_Involvement['Parental_Involvement']), values=list(Parental_Involvement['Count']), hole=0.6,title={
        'text':'Parental Involvement',
        'font': {'size': 15}})])
fig2=go.Figure(data=[
    go.Pie(labels=list(Peer_Influence.keys()), values=list(Peer_Influence.values()), hole=0.6,title={
        'text':'Peer Influence',
        'font': {'size': 15}})
])
a.write(fig1)
b.write(fig2)
