import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title('Game Reviews Analysis 2010-2023')
st.write('Les jeux vidéo sont une forme de divertissement populaire qui a connu une croissance exponentielle'
         ' au cours des dernières décennies. Cette analyse examine les tendances et les performances des jeux'
         ' vidéo les plus populaires, en mettant en évidence les facteurs clés qui influencent le succès commercial.')

path='Data/video_game_reviews.csv'
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
title=st.selectbox(label='Games Description:',options=sorted(games_filtered['Game Title'].unique()))
data=games_filtered[['Platform','User Rating','Price','Release Year','Graphics Quality','Soundtrack Quality','Story Quality','User Review Text']][games_filtered['Game Title']==title].sort_values(by=['User Rating'],ascending=False).drop_duplicates(subset=['Platform'])
st.write(data.set_index('Platform'))


    #Prepare data