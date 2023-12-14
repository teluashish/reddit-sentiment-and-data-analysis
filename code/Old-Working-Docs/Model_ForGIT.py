#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
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

# DATA --------------------------------------------------------------------------------------------------------------
df = pd.read_json('hug_data.jsonl', lines=True)

# rename label to label_encoded
df.rename(columns={'label':'labels'}, inplace=True)

# get subset of df for testing/debugging/development (CHANGE THIS IN THE FUTURE)
df = df.groupby('labels').apply(lambda x: x.sample(n=3, random_state=42)).reset_index(drop=True)

# shuffle df
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# average length of text
max_len = df['text'].str.len().max()
min_len = df['text'].str.len().min()

# PREP --------------------------------------------------------------------------------------------------------------
texts = df['text'].values
labels = df['labels'].values

# load pretrained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Split first
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

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

# MODEL BUILD --------------------------------------------------------------------------------------------------------------
# Convert to tensors
input_ids_train = tf.stack(input_ids_train, axis=0)
input_ids_test = tf.stack(input_ids_test, axis=0)
attention_masks_train = tf.stack(attention_masks_train, axis=0)  
attention_masks_test = tf.stack(attention_masks_test, axis=0)
#print(input_ids_train.shape, attention_masks_train.shape, labels_train.shape)

# model compilation
optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # loss='binary_crossentropy' was loss

# model training
model.fit([input_ids_train, attention_masks_train], labels_train, batch_size=64, epochs=3, validation_split=0.2)


# MODEL RUN --------------------------------------------------------------------------------------------------------------
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

# Save the model's architecture and weights; save the tokenizer
model.save('bert_emotion_classifier')
tokenizer.save_pretrained('bert_emotion_classifier_tokenizer')

# Load the saved tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert_emotion_classifier_tokenizer')
#model = TFBertForSequenceClassification.from_pretrained('bert_emotion_classifier')


# MODEL OUTPUT & EVAL --------------------------------------------------------------------------------------------------------------
# get classification report
report = classification_report(labels_test, y_pred_labels, target_names=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# report to csv
report_df.to_csv('hug_data_sample_report.csv', index=False)

# preview report
print(report_df)

# GET SENTIMENT --------------------------------------------------------------------------------------------------------------
# get sentiment score for each comment with hugingface pipeline
from transformers import pipeline
import pandas as pd
import numpy as np
import os
from collections import Counter

sentiment = pipeline('sentiment-analysis')

def get_sentiment(text, max_length=512):
    # Convert text to string in case it's not
    text = str(text)

    # Split the text into chunks of max_length
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    # Get sentiment score and label for each chunk
    scores = []
    labels = []
    for chunk in chunks:
        result = sentiment(chunk)[0]
        scores.append(result['score'])
        labels.append(result['label'])
    
    # Calculate average sentiment score
    average_score = np.mean(scores)

    # Determine the most frequent label
    most_common_label = Counter(labels).most_common(1)[0][0]

    # Convert label to lowercase
    most_common_label = most_common_label.lower()

    return average_score, most_common_label

# Ensure all data in Text column are strings
final_df['Text'] = final_df['Text'].astype(str)

# Apply the get_sentiment function to each text
final_df[['Sentiment_Score', 'Sentiment_Label']] = final_df['Text'].apply(lambda x: pd.Series(get_sentiment(x)))

# find df as csv
final_df.to_csv('hug_data_sample_scores.csv', index=False)

# max and min score of "Sentiment"
#final_df['Sentiment'].max()
#final_df['Sentiment'].min()

#final_df.head()