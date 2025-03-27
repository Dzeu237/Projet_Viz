import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.title("Call Center Employee Performance Analysis")
st.write("""Dans cette analyse approfondie de la performance des employés de notre centre d'appels,
          nous avons examiné les indicateurs clés et les tendances à travers nos opérations de service client.
          Nos conclusions révèlent des schémas significatifs dans la productivité des agents,
          la satisfaction client et l'efficacité opérationnelle, permettant d'orienter les décisions stratégiques 
         pour l'optimisation de la main-d'œuvre et les initiatives de formation""")

# Load the data
patch="Data/Call-Center-Dataset.csv"
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