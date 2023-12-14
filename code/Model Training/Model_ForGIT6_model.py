#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import re
from nltk.corpus import stopwords
import numpy as np
'''
The first model is a BERT transformer for sequence classification. It is trained and validated with a dataset
from Hugging Face that contains almost 90,000 twitter messages, labeled with 6 different emotions.   
    '0': sadness
    '1': joy
    '2': love
    '3': anger
    '4': fear
    '5': surprise

Then, general sentiment is calculated for each message using the Hugging Face sentiment-analysis pipeline. Two 
columns are created, one for the postive/negative sentiment and then another with a score that is a numeric 
representation of the model's confidence in the sentiment that it listed.

Outputs:
    - The emotion classification model and tokenizer get saved in folders to preserve its architecture and
      weights. This allows us to run it once and then keep it for future use.
    - The classification report is saved as a csv file. It contains the precision, recall, f1-score, and support
      for each emotion.
    - The final dataframe is saved as a csv file. It contains the text, the scores for each emotion, the
      sentiment for each text, and the confidence for each message.


A subset is taken to test, debug, and develop the code (CHANGE THIS TO AT LEAST 2000 FOR TRUE RUNS!!!!).
Source: https://huggingface.co/datasets/philschmid/emotion/tree/main/data
'''

def get_data(sample_size):
  # load data
  df = pd.read_json('hug_data.jsonl', lines=True)
  df.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded

  # get subset of df for testing/debugging/development (CHANGE THIS IN THE FUTURE)
  df = df.groupby('labels').apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
  df = df.sample(frac=1).reset_index(drop=True) # shuffle df

  texts = df['text'].values
  labels = df['labels'].values

  return df, texts, labels


def tokenize_data(texts_train, texts_test):
  # load pretrained tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  # Tokenize
  input_ids_train = []
  input_ids_test = []
  attention_masks_train = []
  attention_masks_test = []

  for text in texts_train:
    # Tokenize train
    encoded_dict = tokenizer.encode_plus(text, # input texts                     
                          add_special_tokens = True, # add [CLS] at the start, [SEP] at the end
                          max_length = 128, # if input text is longer, then it gets truncated
                          padding = 'max_length', # if input text is shorter, then it gets padded to 128
                          return_attention_mask = True,   
                          return_tensors = 'tf',
                          truncation=True)

    input_ids_train.append(encoded_dict['input_ids'][0]) 
    attention_masks_train.append(encoded_dict['attention_mask'][0])

  for text in texts_test:
    # Tokenize test
    encoded_dict = tokenizer.encode_plus(text,                      
                          add_special_tokens = True, 
                          max_length = 128,           
                          padding = 'max_length',
                          return_attention_mask = True,   
                          return_tensors = 'tf',
                          truncation=True)  

    input_ids_test.append(encoded_dict['input_ids'][0])
    attention_masks_test.append(encoded_dict['attention_mask'][0])

  return input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, tokenizer

def conv_to_tensor(input_ids_train, input_ids_test, attention_masks_train, attention_masks_test,):
  # Convert to tensors
  input_ids_train = tf.stack(input_ids_train, axis=0)
  input_ids_test = tf.stack(input_ids_test, axis=0)

  attention_masks_train = tf.stack(attention_masks_train, axis=0)  
  attention_masks_test = tf.stack(attention_masks_test, axis=0)
  
  return input_ids_train, input_ids_test, attention_masks_train, attention_masks_test


def model_train (input_ids_train, attention_masks_train, labels_train, batch_size, epochs, model):
    '''
    # define model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    optimizer = Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # model compilation
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # model training
    model.fit([input_ids_train, attention_masks_train], 
              labels_train, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_split=0.3,
              shuffle=True)
    '''

    # Learning rate schedule
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1400,
        decay_rate=0.5,
        staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)

    # loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # model compilation
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # early stopping, checkpoint to get best parameters
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=.001, restore_best_weights=True)
    
    # model training
    history = model.fit([input_ids_train, attention_masks_train], 
                        labels_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.2,
                        callbacks=[early_stopping])

    return model, history


def evaluate_model(model, input_ids_test, attention_masks_test, labels_test):
  # Model evaluation for scores
  y_pred_logits = model.predict([input_ids_test, attention_masks_test]).logits
  y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
  y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()

  # Creating DataFrame
  texts_test_series = pd.Series(texts_test, name='Text')
  scores_df = pd.DataFrame(y_pred_scores, columns=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
  final_df = pd.concat([texts_test_series, scores_df], axis=1)

  # Adding overall score
  final_df['Overall_Score'] = final_df[['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']].max(axis=1)

  # get classification report
  report = classification_report(labels_test, y_pred_labels, target_names=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], output_dict=True)
  report_df = pd.DataFrame(report).transpose() # reset index needs to  be tested; intended to show labels in the final report_df cvs

  # report to csv
  #report_df.to_csv('hug_data_sample_report.csv', index=False)

  # preview report
  return final_df, report_df

# ==========================================================================================================================================================
# DATA --------------------------------------------------------------------------------------------------------------
print()
print('------------------------------------------------------------')
print('TRAIN MODEL ------------------------------------------------')
print('------------------------------------------------------------')

# define model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# define directory
cur_dir = '/home/ubuntu/NLP/Project_test/data_model'
os.chdir(cur_dir)

# get X samples of each label from the dataset
df, texts, labels = get_data(14959) # set to 14_959 for all data
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)

input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, tokenizer = tokenize_data(texts_train, texts_test)
input_ids_train, input_ids_test, attention_masks_train, attention_masks_test = conv_to_tensor(input_ids_train, 
                                                                                              input_ids_test, 
                                                                                              attention_masks_train, 
                                                                                              attention_masks_test)

# ensure shapes are same
print(input_ids_train.shape, attention_masks_train.shape, labels_train.shape)
print()

# MODEL
# train
model, history = model_train(input_ids_train, attention_masks_train, labels_train, batch_size=128, epochs=15, model=model)

# evaluate
final_df, report_df = evaluate_model(model, input_ids_test, attention_masks_test, labels_test)
final_df.to_csv('classifier_df.csv', index=False)
report_df.to_csv('classifier_report.csv', index=True)
print(final_df.head())
print(report_df)

# save model's architecture and weights; save the tokenizer
model.save_pretrained(os.path.join('bert_emotion_classifier'))
tokenizer.save_pretrained(os.path.join('bert_emotion_classifier_tokenizer'))

# ==========================================================================================================================================================
# APPLY TO REDDIT DATA --------------------------------------------------------------------------------------------------------------
print()
print('------------------------------------------------------------')
print('REDDIT DATA ------------------------------------------------')
print('------------------------------------------------------------')
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# load reddit data
# define directory
cur_dir = '/home/ubuntu/NLP/Project_test'
os.chdir(cur_dir)
reddit_data = pd.read_csv('merged_reddit_data.csv')
reddit_data = reddit_data[['Text']]
reddit_data.dropna(inplace=True)

# Load spacy model and stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text function
def preprocess_text(text):
    
    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Remove non-alphanumeric and non puncuation characters
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

    # Remove extra spaces
    #text = re.sub(' +', ' ', text)

    # Lemmatize
    doc = nlp(text)
    text = ' '.join([lemmatizer.lemmatize(token.text) for token in doc])

    return text


# apply preprocessing; takes about 5 min on cpu
reddit_data['Text'] = reddit_data['Text'].apply(preprocess_text)
texts = reddit_data['Text'].values

# Tokenize Reddit Data
input_ids_reddit, attention_masks_reddit = [], []

for text in texts:
    encoded_dict = tokenizer.encode_plus(text,                      
                        add_special_tokens = True, 
                        max_length = 128,           
                        padding = 'max_length',
                        return_attention_mask = True,   
                        return_tensors = 'tf',
                        truncation=True)  

    input_ids_reddit.append(encoded_dict['input_ids'][0])
    attention_masks_reddit.append(encoded_dict['attention_mask'][0])

# Convert tokenized data to tensors
input_ids_reddit = tf.stack(input_ids_reddit, axis=0)
attention_masks_reddit = tf.stack(attention_masks_reddit, axis=0)

# Apply model to get predictions
reddit_pred_logits = model.predict([input_ids_reddit, attention_masks_reddit]).logits
reddit_pred_scores = tf.nn.softmax(reddit_pred_logits, axis=1).numpy()

# Process predictions
reddit_texts_series = pd.Series(texts, name='Text')
reddit_scores_df = pd.DataFrame(reddit_pred_scores, columns=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
reddit_sentiment_df = pd.concat([reddit_texts_series, reddit_scores_df], axis=1)

# Save or display results
reddit_sentiment_df.to_csv('reddit_emotion_sentiments.csv', index=False)

# ==========================================================================================================================================================
# GET REDDIT DATA AT 70% SCORE --------------------------------------------------------------------------------------------------------------
print('getting new reddit data emotions...')

# define directory
cur_dir = '/home/ubuntu/NLP/Project_test'
os.chdir(cur_dir)

df_reddit = pd.read_csv('reddit_emotion_sentiments.csv')

# add a column with the highest scoring emotion converted to numeric
df_reddit['Emotion'] = df_reddit[['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']].idxmax(axis=1)
df_reddit['Emotion_Score'] = df_reddit[['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']].max(axis=1)
df_reddit['Emotion'] = df_reddit['Emotion'].map({'Sadness':0, 'Joy':1, 'Love':2, 'Anger':3, 'Fear':4, 'Surprise':5})

# keep only the rows with a score of .6 or higher
df_reddit = df_reddit[df_reddit['Emotion_Score'] >= .6]

# reformat columns so nly Emotion and text are shown
df_reddit = df_reddit[['Text', 'Emotion']]
df_reddit.rename(columns={'Text':'text'}, inplace=True)
df_reddit.rename(columns={'Emotion':'labels'}, inplace=True)
df_reddit.reset_index(drop=True, inplace=True)

df_reddit.to_csv('reddit_emotion_sentiments_format.csv')
print('Ready to train new model...')
# get texts, labels
#texts_reddit = df_reddit['text'].values
#labels_reddit = df_reddit['labels'].values

