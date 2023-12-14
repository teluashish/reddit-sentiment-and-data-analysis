import os
import re
import glob
import pandas as pd
import streamlit as st
from model_functions import *


def train_data_tabs():
    """
    Control process for selecting data when testing or training models.

    Args:
        None

    Returns:
        None
    """

    # Initalization of session_state variables
    st.session_state.butTokenizeDsabled = True
    st.session_state.labels = []
    st.session_state.text = []
    st.session_state.dataSource = pd.DataFrame()
    st.session_state.labels = []
    st.session_state.text = []
    st.session_state.sampleSize = 0
    st.session_state.butTokenizeDsabled = False
    
    
    # Select data source(Modular for optinal future addition of datasets)
    dataSource = pd.DataFrame()
    dataOption_selectbox = st.selectbox(
    'Select training data from available data sets',
    (['Hugging Face Twitter Data']),
    index=None,
    placeholder="Select data source...",)

    if dataOption_selectbox == 'Hugging Face Twitter Data':
        # Assign data to session_state variables
        dataSource = pd.read_json('hug_data.jsonl', lines=True)
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.dataSource = dataSource
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.sampleSize = 14959
        st.session_state.butTokenizeDsabled = False
    
        # Select box's for user selection of label and text from dataset
        labelColumn, textColumn = st.columns(2)
        with textColumn:
            train_text_selection = st.selectbox(
            ("Select a column as text"), st.session_state.text, index=0)
            st.session_state.choosenText = train_text_selection
        with labelColumn:
            # Allow for selection of label and text
            train_label_selection = st.selectbox(
                ("Select a column as label"), st.session_state.labels, index=1)
            st.session_state.choosenLabel = train_label_selection
        
    
    # Display choosen data source
    st.dataframe(dataSource, use_container_width=True)

    data_line_selectoin = st.number_input('Select row of sample text', value=10, step=1, format=None, key=None)
    st.session_state.line_num = data_line_selectoin

def train_model_tab():
    """
    Control process for selecting model type when testing or training models.
    As well as testing/traing that model.

    Args:
        None

    Returns:
        None
    """
    st.session_state.modelHyperParams= {}
    st.session_state.testModelHyperParams ={}
    st.session_state.modelHyperParams['staircase'] = True

    # Model tabs to either use preexisiting models or train new models
    #modelTab1, modelTab2= st.tabs(["Test already trained models", " Train New Model",])

    # Test exisitng model tab
    #with modelTab1:
    # Find current seletion of models
    train_model_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('model', x)])
    train_model_list = [f"{x}" for x in train_model_files]
    if st.button('Predict'):
        # Retreieve user selected item
        line_num = st.session_state.line_num
        print('preprocessing reddit data')
        df_twitter = pd.read_json('hug_data.jsonl', lines=True)
        item = df_twitter.iloc[line_num]
        item['text'] = preprocess_text(item['text'])
        st.write(f"Line Text: {df_twitter.iloc[line_num]['text']}")
        st.write(f"Preprocesses Text: {item['text']}\n")

        # Predict item with all models
        pred_df = pd.DataFrame(columns=['Prediction', 'Probs'])
        for i in train_model_list:
            st.session_state.model = i
            if re.search(r'roberta', st.session_state.model, re.IGNORECASE) is not None:
                st.session_state.model_name = 'roberta'
            elif re.search(r'electra', st.session_state.model, re.IGNORECASE) is not None:
                st.session_state.model_name = 'electra'
            elif re.search(r'bert', st.session_state.model, re.IGNORECASE) is not None:
                st.session_state.model_name = 'bert'

            cwd = os.getcwd()
            classifier = EmotionClassifier(model_name=st.session_state.model_name, 
                                            model_path=cwd + '/content/models/' + st.session_state.model, 
                                            tokenizer_path=''
                                            )
            prediction, probs = classifier.predict_emotions(item["text"])

            # Log collected data
            emotion_columns = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
            emo_df = pd.DataFrame()
            emo_df[emotion_columns] = probs  
            st.write(f"Model name: {i}")
            st.write(prediction, emo_df)
            pred_df = pred_df._append({'Prediction': prediction, 'Probs': probs}, ignore_index=True)
    
    # COMMENTED OUT LAST MINUTE BECAUSE OF BUG INTRODUCED WHEN CHANGING FILE STRUCTURE FOR SUBMISSION
    # Train new model tab
    # with modelTab2:
    #     trainModelOption_selectbox = st.selectbox(
    #         'Select Model from pre-loaded sources',
    #         ('Retrain TFBertForSequenceClassification', 'Model B', 'Model C'),
    #         index=None,
    #         placeholder="Select model...",)

    #     if trainModelOption_selectbox == 'Retrain TFBertForSequenceClassification':
    #         st.session_state.newModelType = "TFBERT"

    #     # Hyper param options
    #     if st.checkbox("Scheduler Staircase",value=True):
    #         st.session_state.modelHyperParams['staircase'] = False
    #     else:
    #         st.session_state.modelHyperParams['staircase'] = True
    #     trainModelcol1, trainModelcol2 = st.columns(2)
    #     with trainModelcol1:
            
    #         new_model_name =st.text_input(
    #             "Select an name for your model",
    #             "unnamed",
    #         )
    #         batch_size_input =st.text_input(
    #             "Select a batchsize",
    #             "128",
    #         )
    #         initial_learning_rate_input =st.text_input(
    #             "Select an inital learning rate",
    #             "0.0001",
    #         )
    #     with trainModelcol2:
    #         decay_steps_input =st.text_input(
    #             "Select decay steps",
    #             "1400",
    #         )
    #         decay_rate_input =st.text_input(
    #             "Select decay rate",
    #             "0.5",
    #         )
    #         epochs_input =st.text_input(
    #             "Select epoch",
    #             "15",
    #         )
        
    #     if st.button('train'):
    #         # lock in hyper params
    #         st.session_state.modelHyperParams['new_model_name'] =  new_model_name
    #         st.session_state.modelHyperParams['batch_size'] = batch_size_input
    #         st.session_state.modelHyperParams['initial_learning_rate'] = initial_learning_rate_input
    #         st.session_state.modelHyperParams['decay_steps'] = decay_steps_input
    #         st.session_state.modelHyperParams['decay_rate'] = decay_rate_input
    #         st.session_state.modelHyperParams['epochs'] = epochs_input
    #         #
    #         model, history = train_model(st.session_state.newModelType, **st.session_state.modelHyperParams)
