import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
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





st.title("Call Center Employee Performance Analysis")
st.write("""Dans cette analyse approfondie de la performance des employés de notre centre d'appels,
          nous avons examiné les indicateurs clés et les tendances à travers nos opérations de service client.
          Nos conclusions révèlent des schémas significatifs dans la productivité des agents,
          la satisfaction client et l'efficacité opérationnelle, permettant d'orienter les décisions stratégiques 
         pour l'optimisation de la main-d'œuvre et les initiatives de formation""")

# Load the data
patch="Call-Center-Dataset.csv"
data=pd.read_csv(patch,delimiter=";")
df=pd.DataFrame(data)

# Clean the column Data
    #Cleaning the column names remove the Space
df.columns = df.columns.str.strip()

    #Cleaning Space from Date Colums
df['Date'] = df['Date'].str.strip()

    # Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y %H:%M')

    # Create new columns for date and time
df['Date_Only'] = df['Date'].dt.strftime('%d/%m/%Y')
df['Time_Only'] = df['Date'].dt.strftime('%H:%M')

    # Function to categorize the time of day
def categorize_time_of_day(hour):
    if 9 <= hour <= 12:
        return 'Morning'
    elif 13 <= hour <= 15:
        return 'Afternoon'
    elif 16 <= hour < 18:
        return 'Evening'
df['Daytime']=pd.to_datetime(df['Time_Only']).dt.hour.apply(categorize_time_of_day)

# Update the 'Date' column to display the day of the week
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.day_name()

# Remove or replace invalid values in 'Speed of Answer' column
df['Speed of Answer'] = df['Speed of Answer'].str.strip()  # Remove leading/trailing spaces
df['Speed of Answer'] = df['Speed of Answer'].replace('', '0')  # Replace empty strings with '0'
df['Speed of Answer'] = df['Speed of Answer'].astype(int)  # Convert to integers

    # Convert the 'Call Duration' column to numeric format
table=df['AvgTalkDuration'].str.strip()
df['AvgTalkDuration']=(pd.to_datetime(table,format='%H:%M:%S').dt.minute*60 + pd.to_datetime(table,format='%H:%M:%S').dt.second).round(0)

# Clean unnecessary columns
df.drop(columns=['Time_Only','Unnamed: 10','Call Id',''],inplace=True)

#Calculate metrics
    # Calculate the percentage of answered calls  and resolve rate
df['Answered (Y/N)']=df['Answered (Y/N)'].str.strip()
df['Resolved']=df['Resolved'].str.strip()

    # Calculate the satisfaction rate
df['Satisfaction rating']=df['Satisfaction rating'].str.strip()
df['Satisfaction rating'] = df['Satisfaction rating'].replace('', '0')
satisfaction=len(df[df['Satisfaction rating'].isin(['4','5'])])/len(df)*100

    # Calculate the percentage of answered calls and resolve rate
answered_calls = (len(df[df['Answered (Y/N)'] == 'Y'])/len(df))*100
resolved_calls = (len(df[df['Resolved'] == 'Y'])/len(df))*100
# Display the performance metrics
a,b,c=st.columns(3)
#a.metric(label="Count Calls",value=f"{len(df)}")
a.metric(label="Answered rate",value=f"{answered_calls:.2f} %")
c.metric(label="Satisfaction rate 🌟",value=f'{satisfaction:.2f} %')
b.metric(label="Resolve rate",value=f"{resolved_calls:.2f} %")
c.metric(label="Avarage time Resolve",value=f"{df['Speed of Answer'].mean():.2f} S")
a.metric(label="Avarage time discussion",value=f"{df['AvgTalkDuration'].mean():.2f} S")

#Display graphique
a,b=st.columns(2)
    # Create a Seaborn pairplot
table=df[['Daytime','Date']].groupby('Date').value_counts().reset_index(name='Daytime_Count')
plot = sns.barplot(data=table,x='Date',y='Daytime_Count',hue='Daytime',palette='viridis',errorbar=None)
plt.figure(figsize=(10,10))
for containner in plot.containers:
    plot.bar_label(containner,fontsize=8)
plot.set(xlabel=None, ylabel=None)
plot.legend(loc='upper right',ncols=3)
plot.set_ylim(0, 400)
a.pyplot(fig=plot.get_figure(),clear_figure=True)

    #Create a pie chart for Call service
fig, ax = plt.subplots()
fig=plt.figure(figsize=(8,8))
ax=plt.pie(df['Department'].value_counts(), labels=df['Department'].unique(), autopct='%1.1f%%', startangle=140)
a.pyplot(fig,clear_figure=True)

#Line chart for the satisfaction rate
tab=df['Satisfaction rating'].value_counts().reset_index(name='Count')
fig, ax1 = plt.subplots()
ax2=ax1.twinx()
tab.sort_values(by='Satisfaction rating',inplace=True)
ax2=sns.barplot(data=tab,x='Satisfaction rating',y= 'Count')
ax1=sns.barplot(data=tab,x='Satisfaction rating',y=tab['Count']/len(tab))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for containner in ax1.containers:
    ax1.bar_label(containner,fontsize=8)
b.pyplot(fig.get_figure(),clear_figure=True)

#Barchart for the satisfaction rate by Agent
tab=df.query('`Satisfaction rating` in ("4","5")')
tab=tab[['Department']].groupby('Department').value_counts().reset_index(name='Count')
tab['Total']=df['Department'].value_counts().reset_index(name='Total_count')['Total_count']
plot=sns.barplot(data=tab,x='Department',y=(tab['Count']/tab['Total'])*100,errorbar=None)
plt.xticks(rotation=45)
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.title('Satisfaction rate by Department')
plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
b.pyplot(plot.get_figure(),clear_figure=True)


# Create a Seaborn barplot
#Present the overview by Agent
st.title('Agent Overview:')
option=st.selectbox('**Chose your Agent**',options=df['Agent'].unique())
df_agent=df.groupby('Agent').get_group(option)

    # Calculate the percentage of answered calls and resolve rate
answered_calls_option = (len(df_agent[df_agent['Answered (Y/N)'] == 'Y'])/len(df_agent))*100
resolved_calls_option = (len(df_agent[df_agent['Resolved'] == 'Y'])/len(df_agent))*100
satisfaction_option=len(df_agent[df_agent['Satisfaction rating'].isin(['4','5'])])/len(df_agent)*100

# Display the performance metrics
a,b,c=st.columns(3)
#a.metric(label="Count Calls",value=f"{len(df)}")
a.metric(label="Answered rate",value=f"{answered_calls_option:.2f} %",delta=f"{answered_calls_option-answered_calls:.2f} %")
c.metric(label="Satisfaction rate 🌟",value=f'{satisfaction_option:.2f} %',delta=f"{satisfaction_option-satisfaction:.2f} %")
b.metric(label="Resolve rate",value=f"{resolved_calls_option:.2f} %",delta=f"{resolved_calls_option-resolved_calls:.2f} %")
c.metric(label="Avarage time Resolve",value=f"{df_agent['Speed of Answer'].mean():.2f} S")
a.metric(label="Avarage time discussion",value=f"{df_agent['AvgTalkDuration'].mean():.2f} S")
 
 #Display graphique
a,b=st.columns(2)
    # Create a Seaborn pairplot
table=df_agent[['Daytime','Date']].groupby('Date').value_counts().reset_index(name='Daytime_Count')
plot = sns.barplot(data=table,x='Date',y='Daytime_Count',hue='Daytime',palette='viridis',errorbar=None)
plt.figure(figsize=(8,8))
for containner in plot.containers:
    plot.bar_label(containner,fontsize=8)
plot.set(xlabel=None, ylabel=None)
plot.legend(loc='upper right')
plot.set_ylim(0,100)
a.pyplot(fig=plot.get_figure(),clear_figure=True)

    #Create a pie chart for Call service
fig, ax = plt.subplots()
fig=plt.figure(figsize=(8,8))
ax=plt.pie(df_agent['Department'].value_counts(), labels=df_agent['Department'].unique(), autopct='%1.1f%%', startangle=140)

a.pyplot(fig,clear_figure=True)

#Line chart for the satisfaction rate
tab=df_agent['Satisfaction rating'].value_counts().reset_index(name='Count')
tab.sort_values(by='Satisfaction rating',inplace=True)
plot=sns.barplot(data=tab,x='Satisfaction rating',y=tab['Count']/len(tab))
plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for containner in plot.containers:
    plot.bar_label(containner,fontsize=8)

b.pyplot(fig.get_figure(),clear_figure=True)

tab=df_agent.query('`Satisfaction rating` in ("4","5")')
tab=tab[['Department']].groupby('Department').value_counts().reset_index(name='Count')
tab['Total']=df_agent['Department'].value_counts().reset_index(name='Total_count')['Total_count']
plot=sns.barplot(data=tab,x='Department',y=(tab['Count']/tab['Total'])*100,errorbar=None)
plt.xticks(rotation=45)
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.title('Satisfaction rate by Department')
plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
b.pyplot(plot.get_figure(),clear_figure=True)


st.title('Student Performance')
st.write('La performance des étudiants est un indicateur multidimensionnel qui reflète non seulement les résultats académiques'
         ', mais aussi le développement global de l/apprenant. Cette analyse examine les principaux facteurs'
          ' influençant la réussite scolaire et propose des recommandations pour l/optimisation des résultats.')

#Preparation des Donnees
path="StudentPerformanceFactors.csv"
data=pd.read_csv(path,delimiter=",").dropna()

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
        'font_size': 20, 
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
        'font_size': 20, 
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
        'font_size': 20, 
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

st.title('Game Reviews Analysis 2010-2023')
st.write('Les jeux vidéo sont une forme de divertissement populaire qui a connu une croissance exponentielle'
         ' au cours des dernières décennies. Cette analyse examine les tendances et les performances des jeux'
         ' vidéo les plus populaires, en mettant en évidence les facteurs clés qui influencent le succès commercial.')

path='video_game_reviews.csv'
games=pd.read_csv(path,delimiter=',').dropna()
games.columns=games.columns.str.strip()


a,b,c,d=st.columns(4)
#Age Group Targeted
option1=a.selectbox(label='Genre',options=['All']+list(games['Genre'].unique()))
option2=b.selectbox(label='Age Group Targeted',options=list(games['Age Group Targeted'].unique()))
option3=c.selectbox(label='Publisher',options=['All']+list(games['Publisher'].unique()))
option4=d.selectbox(label='Developer',options=['All']+list(games['Developer'].unique()))

# Initialize with full dataset
games_filtered = games.copy()
# Apply filters only for non-'All' selections
if option1 != 'All':
    games_filtered = games_filtered[games_filtered['Genre'] == option1]
if option2 != 'All':
    games_filtered = games_filtered[games_filtered['Age Group Targeted'] == option2]
if option3 != 'All':
    games_filtered = games_filtered[games_filtered['Publisher'] == option3]
if option4 != 'All':
    games_filtered = games_filtered[games_filtered['Developer'] == option4]



# Animated Graph by using different category
def create_animated_plot(df, x_column, y_columns, hue, title):
    """
    Create an animated plot with multiple lines for each platform on the same axes
    
    Parameters:
    df: DataFrame containing the data
    x_column: Name of the column for x-axis
    y_columns: Name of the column for y-axis
    hue: List of unique values for hue (e.g., platforms)
    title: Title of the plot
    """
    # Create the base figure
    fig = go.Figure()
    
    # Add traces with initial empty data for each line
    for y_col in hue:

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[]
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title='Values',
        xaxis=dict(range=[df[x_column].min(), df[x_column].max()+2]),
        yaxis=dict(range=[df[y_columns].min()*0.86, df[y_columns].max() * 1.3])
    )
    df_filtered = df[df['Platform'] == y_col].sort_values(by=x_column)
    # Create frames for animation
    frames = []
    for k in range(1, len(df_filtered) + 1):
        frame_data = []
        for y_col in hue:
            # Filter the DataFrame for the current hue value
            df_filtered = df[df['Platform'] == y_col].sort_values(by=x_column)
            frame_data.append(
                go.Scatter(
                    x=df_filtered[x_column][:k],
                    y=df_filtered[y_columns][:k],
                    mode='lines+markers',
                    name=y_col
                )
            )
        frames.append(go.Frame(data=frame_data))
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 200, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 10}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 200, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 10}
                    }]
                }
            ]
        }]
    )
    
    # Show the plot
    st.write(fig)
    


# Create and show the plots
final=games_filtered.groupby('Platform')[['Release Year','Platform']].value_counts().reset_index(name='Count_Games')
final['Platform'] = final['Platform'].str.strip()
spe = final['Platform'].str.strip().unique()
create_animated_plot(
    final,
    x_column='Release Year',
    y_columns='Count_Games',
    hue=spe,
    title='Games Released by PlatForm Over Time'
)

# Display for each games Title related information