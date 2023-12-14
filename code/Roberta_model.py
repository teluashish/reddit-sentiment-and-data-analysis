import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import logging

logging.basicConfig(level=logging.INFO)

import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
from sklearn.preprocessing import LabelEncoder

class RobertaClassifier:

    def __init__(self, label_names, model_name='roberta-base'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_names = label_names

        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

    # ... [rest of the code remains the same as in ElectraClassifier until create_model method] ...

    def create_model(self, num_labels=6):
        config = RobertaConfig.from_pretrained('roberta-base', num_labels=num_labels)
        self.model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=2e-5,  # Adjusted learning rate for RoBERTa
            decay_steps=10000,          # Changed decay steps
            decay_rate=0.85,            # Adjusted decay rate
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

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

    def load_data_from_jsonl(self, filename):
        texts, labels = [], []
        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(data['label'])
                except json.JSONDecodeError as e:
                    print(f"Error in line {line_number}: {e}")
                    break

        input_ids, attention_masks = [], []
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='tf',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, 0)
        attention_masks = tf.concat(attention_masks, 0)
        labels = np.array(labels)
        return input_ids, attention_masks, labels, texts




    def train_model(self, train_data, validation_data, epochs=15, batch_size=128):
        model_path = os.path.join(os.getcwd(), 'best_model_roberta')
        logging.info(f"Model and tokenizer will be saved to: {model_path}")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3)
        ]

        try:
            history = self.model.fit(
                train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )

            # Save the model and the tokenizer
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

        return history

    def evaluate_model(self, validation_data, texts_validation, label_names):
        input_ids_test, attention_masks_test, labels_test = validation_data.values()
        texts_test_series = pd.Series(texts_validation, name='Text')

        y_pred_logits = self.model.predict({'input_ids': input_ids_test, 'attention_mask': attention_masks_test}).logits
        y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
        y_pred_labels = tf.argmax(y_pred_scores, axis=1).numpy()

        scores_df = pd.DataFrame(y_pred_scores, columns=label_names)
        final_df = pd.concat([texts_test_series, scores_df], axis=1)
        final_df['Overall_Score'] = final_df[label_names].max(axis=1)

        report = classification_report(labels_test, y_pred_labels, target_names=label_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        return final_df, report_df

    # model_path is the directory to tf model
    def load_model(self, model_path):
        return TFRobertaForSequenceClassification.from_pretrained(model_path)

    def infer(self, model, text):

        input_ids, attention_masks = [], []
        for text in text:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='tf',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, 0)
        attention_masks = tf.concat(attention_masks, 0)

        predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_masks})
        predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

        predicted_labels = [self.label_names[label] for label in predicted_labels]

        return predicted_labels

def main():

    label_names = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    classifier = RobertaClassifier(label_names)
    input_ids, attention_masks, labels, texts = classifier.load_data_from_jsonl('hug_data.jsonl')
    classifier.create_model(num_labels=6)

    # Convert tensors to NumPy arrays
    if isinstance(input_ids, tf.Tensor):
        input_ids = input_ids.numpy()
    if isinstance(attention_masks, tf.Tensor):
        attention_masks = attention_masks.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=2018)
    train_masks, validation_masks, texts_train, texts_validation = train_test_split(attention_masks, texts, test_size=0.2, random_state=2018)

    train_inputs = tf.convert_to_tensor(train_inputs)
    validation_inputs = tf.convert_to_tensor(validation_inputs)
    train_masks = tf.convert_to_tensor(train_masks)
    validation_masks = tf.convert_to_tensor(validation_masks)
    train_labels = tf.convert_to_tensor(train_labels)
    validation_labels = tf.convert_to_tensor(validation_labels)

    train_data = {'input_ids': train_inputs, 'attention_mask': train_masks, 'labels': train_labels}
    validation_data = {'input_ids': validation_inputs, 'attention_mask': validation_masks, 'labels': validation_labels}

    train_history = classifier.train_model(train_data, validation_data)
    final_df, report_df = classifier.evaluate_model(validation_data, texts_validation, label_names)

    print("Evaluation Scores:")
    print(final_df.head())

    print("\nClassification Report:")
    print(report_df)

def from_pretrained(model_path):
    print("here")
    label_names = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    classifier = TFRobertaForSequenceClassification(label_names)

    # Load the model and the tokenizer
    classifier.model = TFRobertaForSequenceClassification.from_pretrained(model_path)
    classifier.tokenizer = RobertaTokenizer.from_pretrained(model_path)

    sentiment = classifier.infer(classifier.model, ['Please figure out the sentiment for this text. Scared if it actually works'])
    print(sentiment)

if __name__ == "__main__":
    main()

