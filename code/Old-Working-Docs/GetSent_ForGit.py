#%%
import pandas as pd
import numpy as np
import os

# get data
curdir = os.getcwd() # c:\Users\schil\OneDrive\Desktop\School\6312_NLP\Project\RedditComments
df = pd.read_csv(curdir + '/2/merged_reddit_data.csv')

# create new df with only the Text column and keeps the index
df_text = df[['Text']].copy()

# get subset of data
df_text = df_text.iloc[0:10]



# ------------------------------------------------------------------
# get sentiment score for each comment with hugingface pipeline
from transformers import pipeline

def get_sentiment(text, max_length=512):
    # Convert text to string in case it's not
    text = str(text)

    # Split the text into chunks of max_length
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    # Get sentiment for each chunk and average
    sentiments = [sentiment(chunk)[0]['score'] for chunk in chunks]
    return np.mean(sentiments)

# Ensure all data in Text column are strings
df_text['Text'] = df_text['Text'].astype(str)

# Apply the get_sentiment function to each text
df_text['sentiment'] = df_text['Text'].apply(get_sentiment)

# Display the head of the dataframe
df_text.head()


#%%
# ------------------------------------------------------------------
'''
This script calculates the norm of the embedding for each token 
(excluding the special tokens [CLS] and [SEP]), and then finds 
the top k tokens based on these norms. It should resolve the "out 
of range" error you were encountering.
'''

import spacy
from transformers import AutoModel, AutoTokenizer
import torch

def extract_keywords_phrases(text, model_name="bert-base-uncased", top_k=10):
    # Load transformer model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    # Get the embeddings of the last hidden state
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Calculate the norm of embeddings for each token
    norms = torch.norm(embeddings, dim=1)

    # Adjust top_k to be no more than the number of tokens (excluding special tokens)
    token_count = embeddings.size(0) - 2 # excluding [CLS] and [SEP]
    top_k = min(top_k, token_count)

    # Get top k indices, excluding the first and last tokens ([CLS] and [SEP])
    top_k_indices = torch.topk(norms[1:-1], top_k).indices + 1

    # Convert indices to tokens
    keywords = [tokenizer.decode([inputs.input_ids[0][idx]]) for idx in top_k_indices]

    # Load spaCy model for POS tagging and noun chunks
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)

    # Get POS for keywords and key phrases
    pos_keywords = [(word, doc[i].pos_) for i, word in enumerate(keywords) if i < len(doc)]
    key_phrases = [chunk.text for chunk in doc.noun_chunks]

    return pos_keywords, key_phrases

# Example usage
text = "Hello. It is now winter, when there will always be darkness and you will always struggle to be comfortably warm."
keywords, phrases = extract_keywords_phrases(text)
print("Keywords:", keywords)
print("Key Phrases:", phrases)

# Apply the get_sentiment function to each text
#df_text[['keywords', 'key_phrases']] = df_text['Text'].apply(lambda x: extract_keywords_phrases(x), result_type='expand')



#%%

import spacy
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords_phrases(text, model_name="bert-base-uncased", top_k=10):
    # Load transformer model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    # Get the embeddings of the last hidden state
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Calculate the norm of embeddings for each token
    norms = torch.norm(embeddings, dim=1)

    # Adjust top_k to be no more than the number of tokens (excluding special tokens)
    token_count = embeddings.size(0) - 2 # excluding [CLS] and [SEP]
    top_k = min(top_k, token_count)

    # Get top k indices, excluding the first and last tokens ([CLS] and [SEP])
    top_k_indices = torch.topk(norms[1:-1], top_k).indices + 1

    # Convert indices to tokens
    keywords = [tokenizer.decode([inputs.input_ids[0][idx]]) for idx in top_k_indices]

    # TF-IDF for key phrase extraction
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_n = top_k
    top_phrases = feature_array[tfidf_sorting][:top_n]

    # Load spaCy model for POS tagging
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)

    # Get POS for keywords
    pos_keywords = [(word, doc[i].pos_) for i, word in enumerate(keywords) if i < len(doc)]

    return pos_keywords, top_phrases

# Example usage
text = "Hello. It is now winter, when there will always be darkness and you will always struggle to be comfortably warm."
keywords, phrases = extract_keywords_phrases(text)
print("Keywords:", keywords)
print("Key Phrases:", phrases)
#%%






























    
df2 = pd.read_csv(curdir + '/2/merged_reddit_data.csv')


def data_fam(df):

    # print shape of dataframe
    print('Shape:', df.shape)
    print('---' * 20)

    # print number of sample for each subreddit
    print('Samples per Subreddit:\n', df['Subreddit'].value_counts())
    print('---' * 20)

    # print number of nan values in each column
    print('Null values:\n', df.isnull().sum())
    print('---' * 20)

    # print number of rows with duplicate Text values
    print('Duplicate Text:', df.duplicated(subset=['Text']).sum())
    print('---' * 20)

    #print(df.shape[0] - df.duplicated(subset=['Text']).sum())

    return df


# function to delete specified cols, drop nan values if specified, and drop deleted/removed comments
def data_clean(df, drop_cols=None, drop_nan=False, drop_dup_txts=False, drop_del_rem=False):
    
        # drop specified columns
        if drop_cols is not None:
            df = df.drop(columns=drop_cols)
    
        # drop all rows with missing values
        if drop_nan:
            df = df.dropna()

        # drop rows with duplicate Text values
        if drop_dup_txts:
            df = df.drop_duplicates(subset=['Comments'])
    
        # drop all rows with deleted/removed comments
        #if drop_del_rem:
        #    df = df[df.Text != '[deleted]']
        #    df = df[df.Text != '[removed]']
    
        # drop all rows with empty comments
        #if drop_del_rem:
        #    df = df[df.Text != '']
    
        # drop all rows with comments that are too short/long
        #if drop_del_rem:
        #    df = df[df.Text.str.len() > 3]
        #    df = df[df.Text.str.len() < 5000]
    
        return df

# get data and print previw
df = load_data()
df = data_fam(df)
#df = data_clean(df, 
#                drop_cols=['Unnamed: 0', 'ID', 'Flair', 'is_Original',
#                            'URL', 'Title', 'Sentiment', 'num_comments',
#                            'Subreddit', 'Body', 'Upvotes'], 
#                drop_nan=True,
#                drop_dup_txts=True,
#                )

# worth considering including the following columns:
# sentiment, is_orginal, upvotes, sub, num_comments 
df = df.drop(columns=['Unnamed: 0', 'ID', 'Flair', 'is_Original',
                        'URL', 'Title', 'Sentiment', 'num_comments',
                        'Subreddit', 'Body', 'Upvotes', 'is_original',
                        'Text', 'creation_date']) 

df = df.drop_duplicates(df['Comments'], keep='first')
print(len(df))
print(df.head())

# count number of nans in each column
#print(df.isnull().sum())

#%%

df = data_fam(df)
#print('===' * 20)
#df = data_fam(df2) # this showed that combining and removing the dupes is the same as the merged file
df = data_clean(df, 
                drop_cols=['Unnamed: 0', 'ID', 'Flair'], 
                drop_nan=True, 
                #drop_dup_txts=True, 
                #drop_del_rem=False
                )
print('===' * 20)
df = data_fam(df)
print('===' * 20)
df.head()








































#%%
import pandas as pd
import numpy as np
import os

curdir = os.getcwd() # c:\Users\schil\OneDrive\Desktop\School\6312_NLP\Project\RedditComments

def load_data(): # combines data from multiple cvs files; before i realized there was a merged file
    file_names = []
    for root, dirs, files in os.walk(curdir + '/2'): # navigate to the 2 folder
        for file in files:
            if file.endswith('.csv'):
                file_names.append(file)

    # load data from the files that are in the 2 folder and have names in the file_names list
    # and store them in list
    data_list = []
    for file in file_names:
        data_list.append(pd.read_csv(curdir + '/2/' + file))

    # combine all data into a single dataframe
    data = pd.concat(data_list, ignore_index=True)

    return data 

def data_fam(df):

    # print shape of dataframe
    print('Shape:', df.shape)
    print('---' * 20)

    # print number of sample for each subreddit
    print('Samples per Subreddit:\n', df['Subreddit'].value_counts())
    print('---' * 20)

    # print number of nan values in each column
    print('Null values:\n', df.isnull().sum())
    print('---' * 20)

    # print number of rows with duplicate Text values
    print('Duplicate Text:', df.duplicated(subset=['Text']).sum())
    print('---' * 20)

    #print(df.shape[0] - df.duplicated(subset=['Text']).sum())

    return df

df = load_data()
df = df.drop_duplicates(subset=['Text'])

# worth considering including the following columns:
# sentiment, is_orginal, upvotes, sub, num_comments 
df = df.drop(columns=['Unnamed: 0', 'ID', 'Flair', 'is_Original',
                        'URL', 'Title', 'Sentiment', 'num_comments',
                        'Subreddit', 'Body', 'Upvotes']) 
#df2 = pd.read_csv(curdir + '/2/merged_reddit_data.csv')


# remove rows if they have nan values in the Text column
# count number of nans in each column
print(df.isnull().sum())
print('===' * 20)
df = df['Text'].dropna()
df.head()



#%%


