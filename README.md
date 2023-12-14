# reddit-sentiment-and-data-analysis

This is a copy of the repository where I worked with a team of 3 to complete the project.

Actual Repository Link: https://github.com/jschild01/Final-Project-Group3

## Installations

##### Requirements

The only packages needed are contained in the 'general_requirements.txt' file.
To install, run the following:
**MUST BE IN /code FIRST**

```bash
pip3 install -r general_requirements.txt
```

The trained models should be downloaded before running any files using the following link to an accessible Google Drive. There are four models: BERT, BERT using Reddit data, RoBERTa, and ELECTRA. Models should be put in Code/content/models

```bash
https://drive.google.com/drive/folders/1mwFB0R89RYZ_qlLp675Z19VXZbMWdCuK?usp=sharing
```

---

## Running project

The training and test algorithms run from the DATS_6312_NLP directory. The main.py function conveniently starts a Streamlit serve on port 8888 that can be used to run our scripts and can be executed with the below. The server code that it calls can be found in streamlit_display.py.
**MUST BE IN /code FIRST**

```bash
python3 main.py
```

---

**Online Community Monitoring**

**Quantitative and Qualitative Analysis of Subreddit Sentiment Trends**


**Introduction**

Reddit is a popular social media forum that attracts significant numbers of users, content, and engagement from around the globe. Its ever-growing volume of online community discourse presents a platform that can inadvertently shape opinions, damage brands, and incite real-world action. Identifying and understanding swings in user sentiments is becoming more and more critical to combat inaccurate and inadequate information and general community divisiveness. Our application allows subreddit stake holders to monitor their online communities for significant swings in sentiment that could have broadly negative implications if left unchecked.

Our initiative leverages live data extraction from Reddit, gathering posts and comments based on either predefined datasets or user-provided criteria. We employ sophisticated transformer-based models such as BERT, RoBERTa, and Electra, which have been meticulously adjusted for the task of detecting and forecasting sentiment pertaining to six distinct emotions: sadness, joy, love, anger, fear, and surprise. We compile these sentiment evaluations to construct a detailed sentiment trajectory for each emotion over a chosen period. The outcome of this assessment is depicted in a detailed chart that illustrates the fluctuation of these emotional sentiments over time, providing critical insights into the dynamics of subreddit communities.

**Description of the data sets**





Team Members (Contributors):
1. Ashish Telukunta
2. Thomas Stanton
3. Jonathan Schild 
