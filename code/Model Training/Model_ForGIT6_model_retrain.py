# ==========================================================================================================================================================
# DATA - REDDIT --------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer, TFRobertaForSequenceClassification, RobertaTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import re
from nltk.corpus import stopwords
import numpy as np
import nltk
import spacy
from nltk.stem import WordNetLemmatizer

# Load spacy model and stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Plus, import or define tokenize_data and conv_to_tensor
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

def tokenize_data(texts_train, texts_test):
  # load pretrained tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  #tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
  texts_test_series = pd.Series(texts_test, name='text')
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

if __name__ == "__main__":
  # define directory and load previously trained model
  cur_dir = '/home/ubuntu/NLP/Project_test'
  os.chdir(cur_dir)

  df_reddit = pd.read_csv('reddit_emotion_sentiments_format.csv')

  cur_dir = '/home/ubuntu/NLP/Project_test/data_model'
  os.chdir(cur_dir)

  # Load the saved model
  model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
  #model = TFRobertaForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

  # get texts, labels from reddit data; apply preprocessing
  print('preprocessing reddit data')
  df_reddit['text'] = df_reddit['text'].apply(preprocess_text)

  texts_reddit = df_reddit['text'].values
  labels_reddit = df_reddit['labels'].values

  # get texts, labels from OG twitter data; concat with reddit data
  print('combining with OG twitter data')
  df, texts, labels = get_data(4000)
  texts = np.concatenate((texts, texts_reddit))
  labels = np.concatenate((labels, labels_reddit))

  # Split, tokenize, and convert to tensors
  print('splitting')
  texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)
  input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, tokenizer = tokenize_data(texts_train, texts_test)
  input_ids_train, input_ids_test, attention_masks_train, attention_masks_test = conv_to_tensor(input_ids_train, input_ids_test, attention_masks_train, attention_masks_test)

  # Check shapes
  print(input_ids_train.shape, attention_masks_train.shape, labels_train.shape)

  # Retrain the model
  print('training new model...')
  model, history = model_train(input_ids_train, 
                                              attention_masks_train, 
                                              labels_train, 
                                              batch_size=128, 
                                              epochs=15, 
                                              model = model)


  # Evaluate the retrained model
  print('evaluating...')
  final_df, report_df = evaluate_model(model, input_ids_test, attention_masks_test, labels_test)
  print('getting csv files')
  final_df.to_csv('classifier_df_reddit.csv', index=False)
  report_df.to_csv('classifier_report_reddit.csv', index=True)
  print('printing..')
  print(final_df.head())
  print(report_df)

  # Save the retrained model and tokenizer
  print('saving')
  model.save_pretrained(os.path.join('bert_emotion_classifier_reddit'))
  tokenizer.save_pretrained(os.path.join('bert_emotion_classifier_tokenizer_reddit'))
  print('done')
  