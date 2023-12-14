import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import os
import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import TFElectraForSequenceClassification, ElectraTokenizer, ElectraConfig
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer, RobertaConfig


class EmotionClassifier:
    def __init__(self, model_path, tokenizer_path, model_name):

        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

        if model_name == 'bert':
            self.model = TFBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        elif model_name == 'electra':
            self.model = TFElectraForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = ElectraTokenizer.from_pretrained(model_path)
        elif model_name == 'roberta':
            self.model = TFRobertaForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)


    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        text = re.sub(' +', ' ', text)

        # Lemmatize
        doc = self.nlp(text)
        text = ' '.join([self.lemmatizer.lemmatize(token.text) for token in doc])

        return text

    def predict_emotions(self, texts):
        #print(texts)
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        logits = self.model.predict(
            {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}).logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()

        emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

        most_likely_emotions = [emotions[np.argmax(prob)] for prob in probabilities]
        
        return most_likely_emotions, probabilities

    def predict_emotion(self, text):
        # Tokenize the input text
        encoded_dict = self.tokenizer.encode_plus(
            text,                      
            add_special_tokens = True,
            max_length = 128,           
            padding = 'max_length',
            return_attention_mask = True,   
            return_tensors = 'tf',
            truncation=True)

        input_ids = tf.stack([encoded_dict['input_ids'][0]], axis=0)
        attention_mask = tf.stack([encoded_dict['attention_mask'][0]], axis=0)

        # Predict
        logits = self.model.predict([input_ids, attention_mask]).logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]

        # You can modify this part as per your label names
        emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        emotion_probs = dict(zip(emotions, probabilities))

        return emotion_probs

# Example usage
if __name__ == "__main__":
    # Paths to the saved model and tokenizer
    model_path = 'bert_emotion_classifier_reddit'
    tokenizer_path = 'bert_emotion_classifier_tokenizer_reddit'

    classifier = EmotionClassifier(model_path, tokenizer_path)

    # Example text
    text = "Well, his ex wife is like Batman. She’s giving all of her 60bn away"

    # Get prediction
    text = classifier.preprocess_text(text)
    prediction = classifier.predict_emotion(text)
    print("Predicted Emotion Probabilities:", prediction)