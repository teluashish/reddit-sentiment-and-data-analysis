import streamlit as st
from training_page import  train_data_tabs, train_model_tab
from analysis_page import analysis_data_tabs, analysis_model_tab

mainTab1, mainTab2= st.tabs([ "Sentiment Logging Training", "Sentiment Training",])


# Main function calls of the project.
# mainTab1: Core functionality of the program, allows for applying models to live data.
# mainTab2: Allows for demonstraition of models and training of new bert models.


# Main analysis/application page
with mainTab1:
    st.title("Sentiment Logging Training")
    analysis_data_tabs()
    analysis_model_tab()

# Model verification/training page
with mainTab2:
    st.title("Model Training & Analysis")
    st.session_state.dataSource = "None Selected"

    # Data Section
    train_data_tabs()
    st.divider()

    # Model Section
    train_model_tab()
    st.divider()

