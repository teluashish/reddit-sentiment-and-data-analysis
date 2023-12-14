import re
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import streamlit as st
from collections import Counter
from transformers import pipeline
from model_functions import *
from reddit_scraper import reddit_scraper

def analysis_data_tabs():
    """
    Control process for selecting data when analysising subreddits.

    Args:
        None

    Returns:
        None
    """

    # Initalize buttons that need it
    st.session_state.butTokenizeDisabled = True

    with st.expander("Data Sources",expanded=True):
        #st.subheader("Data Sources")
        dataTab1, dataTab2= st.tabs(["Pre-Loaded Data", " Live Data",])
        with dataTab1:
            # Source select for preloaded data
            dataSource = pd.DataFrame()
            dataOption_selectbox = st.selectbox(
            'Select Data from Pre-Loaded sources',
            (['Reddit post and comments']),
            index=None,
            placeholder="Select data source...",)

            if dataOption_selectbox == 'Reddit post and comments':
                dataSource = pd.read_csv('reddit_posts_and_comments.csv', parse_dates = ['Creation Date'])
                st.session_state.dataSource = dataSource
            
            # Display of selected data
            st.dataframe(dataSource, use_container_width=True)
        
    with dataTab2:
        st.subheader("On the fly data")

        commentscolumn, postscolumn = st.columns(2)
        with commentscolumn:
            # button for num of comments
            num_comments = st.number_input('Number of comments', min_value=1, max_value=1000, value=3, step=1, format=None, key=None)
        with postscolumn:
            # button for number of posts
            num_posts = st.number_input('Number of posts', min_value=1, max_value=1000, value=5, step=1, format=None, key=None)

        # button for subreddit name
        subreddit_name = st.text_input('Subreddit name', value='wallstreetbets', max_chars=None, key=None, type='default')

        timeFiltercolumn, intervalcolumn = st.columns(2)
        with timeFiltercolumn:
            time_filter = st.selectbox('Time filter(Draw from the past ...)', ('day', 'week', 'month', 'year'), index=2, key=None)
        with intervalcolumn:
            # button for interval
            interval = st.selectbox('Interval', ('daily', 'weekly', 'monthly'), index=1, key=None)
        
        # You can include a button to trigger the scraping process
        if st.button('Fetch Live Data'):
            # inputs: client_id, client_secret, user_agent, num_posts, subreddit_name, interval, top_comments_count, output_file
            df = reddit_scraper('nFKOCvQQEIoW2hFeVG6kfA', 
                                '5BBB4fr-HMPtO8f4jZhle74-fYcDkQ', 
                                'Icy_Process3191', 
                                num_posts=num_posts,
                                subreddit_name=subreddit_name, 
                                time_filter=time_filter, 
                                interval=interval, 
                                top_comments_count=num_comments, 
                                output_file='reddit_posts_and_comments.csv')
            
        # Assuming the reddit_scraper function returns a dataframe
            if df is not None and not df.empty:
                st.write(f"First few rows the fetched data (out of {len(df)}):")
                st.dataframe(df.head(), use_container_width=True)
                df.to_csv('output.csv', index=True)
                interval_counts = Counter(df['Interval Number'].to_list())
                st.write(interval_counts)
                plt.bar(interval_counts.keys(), interval_counts.values())
                plt.xlabel('Interval Number')
                plt.ylabel('Number of Entries')
                plt.title('Number of Entries per Interval')
                plt.show()
                st.pyplot(plt)
            else:
                st.write("No live data fetched")
                
def analysis_model_tab(): 
    """
    Control process for analysising subreddits.

    Args:
        None

    Returns:
        None
    """
    model_directory = "content/models"
    all_items = glob.glob(os.path.join(model_directory, "*"))
    analysis_model_files = sorted([x for x in all_items if re.search('model', os.path.basename(x))])


    analysis_model_list = [f"{x}" for x in analysis_model_files]
    analysis_tokenizer_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('tokenizer', x)])
    analysis_tokenizer_list = [f"{x}" for x in analysis_tokenizer_files]
            
    # Allow for selection of model
    analysis_selected_model = st.selectbox(
        ("Select a Model"), analysis_model_list, index=None)
    st.session_state.model = analysis_selected_model
    if analysis_selected_model:
        if re.search(r'roberta', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'roberta'
        elif re.search(r'electra', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'electra'
        elif re.search(r'bert', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'bert'

    st.session_state.tokenizer = ''

    if st.button('Apply Model'):
        sentiment = pipeline('sentiment-analysis')
        cwd = os.getcwd()

        #classifier = EmotionClassifier(cwd + '/content/models/' + st.session_state.model, cwd + '/content/models/' +  st.session_state.tokenizer)
        classifier = EmotionClassifier(model_name=st.session_state.model_name, 
                                        model_path=cwd + '/' + st.session_state.model, 
                                        tokenizer_path=''#cwd + '/content/models/' +  st.session_state.tokenizer
                                        )
        print("classifer loaded")
        print(classifier)
        # Arrange Date groups by selected range

        # Predict on each piece of data and store in its date group
        df = pd.read_csv('output.csv')
        sent_scores = []

        prediction, probs = classifier.predict_emotions(df['Text'].tolist())
        df['Sentiment'] = prediction
        emotion_columns = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        df[emotion_columns] = probs
        emotion_columns.extend(['Positive','Negative'])
        # Apply sentiment analysis
        df['pos/neg'] = df['Text'].apply(lambda x: sentiment(x, max_length=512)[0]['label'])
        df['pos/neg score'] = df['Text'].apply(lambda x: sentiment(x, max_length=512)[0]['score'])

        df['Positive'] = df.apply(
            lambda x: x['pos/neg score'] if x['pos/neg'] == 'POSITIVE' else 1 - x['pos/neg score'], axis=1)
        df['Negative'] = df.apply(
            lambda x: x['pos/neg score'] if x['pos/neg'] == 'NEGATIVE' else 1 - x['pos/neg score'], axis=1)
        

        # Compute average for each emotion for each interval
        combined_averages = df.groupby('Interval Number')[emotion_columns].mean()

        combined_averages.to_csv('combined_averages.csv')

        st.write(combined_averages)
        emotion_columns = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))

        # Iterate over each emotion and plot it on the same Axes
        for emotion in emotion_columns:
            combined_averages[emotion].plot(ax=ax, marker='o', label=emotion)

        # Adding title and labels
        ax.set_title('Emotion Scores vs Interval Number')
        ax.set_ylabel('Average Emotion Score')
        ax.set_xlabel('Interval Number')

        # Invert the x-axis and adjust the x-ticks
        ax.invert_xaxis()
        ax.set_xticks(combined_averages.index)
        ax.set_xticklabels(combined_averages.index[::-1])

        # Adding legend to distinguish different emotions
        ax.legend()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
        attributes = ['Positive', 'Negative']
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))

        # Iterate over each attribute and plot it on the same Axes
        for attribute in attributes:
            combined_averages[attribute].plot(ax=ax, marker='o', label=attribute)

        # Adding title and labels
        ax.set_title('Positive and Negative Scores vs Interval Number')
        ax.set_ylabel('Average Score (considering proportion)')
        ax.set_xlabel('Interval Number')

        # Invert the x-axis and adjust the x-ticks
        ax.invert_xaxis()
        ax.set_xticks(combined_averages.index)
        ax.set_xticklabels(combined_averages.index[::-1])

        # Adding legend to distinguish between Positive and Negative scores
        ax.legend()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

