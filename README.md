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

## Online Community Monitoring

## Quantitative and Qualitative Analysis of Subreddit Sentiment Trends**


### Introduction 

Reddit is a popular social media forum that attracts significant numbers of users, content, and engagement from around the globe. Its ever-growing volume of online community discourse presents a platform that can inadvertently shape opinions, damage brands, and incite real-world action. Identifying and understanding swings in user sentiments is becoming more and more critical to combat inaccurate and inadequate information and general community divisiveness. Our application allows subreddit stakeholders to monitor their online communities for significant swings in sentiment that could have broadly negative implications if left unchecked.

Our initiative leverages live data extraction from Reddit, gathering posts and comments based on either predefined datasets or user-provided criteria. We employ sophisticated transformer-based models such as BERT, RoBERTa, and Electra, which have been meticulously adjusted for the task of detecting and forecasting the sentiment of six distinct emotions: sadness, joy, love, anger, fear, and surprise. We compile these sentiment evaluations to construct a detailed sentiment trajectory for each emotion over a chosen period. The outcome of this assessment is depicted in a detailed chart that illustrates the fluctuation of these emotional sentiments over time, providing critical insights into the dynamics of subreddit communities.

### Description of the data sets

We used two datasets to train our final model(s). Our models primarily use a dataset from Hugging Face that contains approximately 90,000 labeled tweets distributed evenly across six emotions: sadness, joy, love, anger, fear, and surprise.  We also used a dataset available on Kaggle that contained almost 8,500 unlabeled Reddit comments from various subreddits.  This dataset was used to experiment with pseudo-labeling and incorporating confident test data into our initial Twitter training data (predictions with confidence score > 0.6 are included in the final merged dataset).

### Description of the NLP models

We have fine-tuned the three different pre-trained transformer networks from hugging face during our analysis out of suspicion that predictions of emotions and sentiment could vary depending on the type of model implemented. With each model we tuned hyperparameters like learning rate, dropout rate and batch size. All of them also used the Adam optimizer with the sparse categorical cross entropy as loss function and monitored validation accuracy for early stopping. 

We developed two BERT-based multi-class classifiers for emotion detection in text, as it learns contextual word representation beneficial for informal language. Model 1 (BERT1) was trained on Twitter data split into training, validation, and test sets. BERT1 achieved validation accuracy and F1 of 0.95. To improve Reddit generalization, we applied BERT1 to unlabeled Reddit comments to pseudo-label high confidence examples (threshold 0.6/label) and combined these newly labeled samples with the Twitter training set. Using this semi-supervised dataset, we trained Model 2 (BERT2) with the same BERT architecture. Incorporating Reddit data reduced performance (validation accuracy/F1: 0.91) but improved applicability to this target domain.


### Experimental Setup

Our experimental setup involves leveraging a dataset comprising 90,000 tweets, each labeled with one of six distinct emotions, along with subreddit data to train our sentiment analysis model. To evaluate the model's performance, we employ F1 scores for individual emotions, overall accuracy, and a composite F1 score that encapsulates the model's average effectiveness across all emotions. These metrics provide a quantitative assessment of the model's precision and recall capabilities. As there are six emotions to predict, we are going to use the sparse categorical cross entropy loss to minimize the error. Furthermore, we conduct both quantitative and qualitative comparisons to ensure a thorough analysis. Numerically, we measure the model's performance by comparing the calculated metrics against benchmarks of other transformer models. Qualitatively, we examine the trends in the subreddit data to verify that the model's sentiment predictions align with the actual discourse and emotional context present within the data. This dual approach allows us to validate the model's accuracy and its practical applicability in interpreting and reflecting the nuanced sentiment trends within the sub reddit communities.

### Implementation and the Execution Flow

We utilized a RoBERTa sequence classifier for our second model, motivated by its optimization of BERT's pre-training procedures and ability to comprehend complex syntactic structures prevalent in Reddit comments. Solely trained on the Twitter corpus, it achieved validation accuracy and F1 scores of 0.95. To diverge from BERT-based models, we trained an ELECTRA model capable of learning more granular and nuanced language patterns ubiquitous in social media. Unlike BERT and RoBERTa which rely on masked language modeling, ELECTRA is context-sensitive and uses token replacement to increase the performance, enabling more precise sentiment predictions suitable for the intricacies in this domain. Our ELECTRA model obtained a validation accuracy and F1 score of 0.95, comparable to our BERT and RoBERTa models.

We used Streamlit to apply our models and display visualizations that help a user identify and understand changes in sentiments of online discourse, in this case subreddits. A scraper using PRAW fetches Reddit posts in real-time based on inputs provided by the user, such as subreddit, number of posts and comments, and date-range criteria. The Reddit API limits scraping to the past day, week, month, or year, instead of providing the ability to specify a date range. to ensure we could measure changes over time, we pulled the dates for each post and comment and later grouped them according to a daily, weekly, or monthly intervals extending out from the present. This allowed us to average sentiments together over the specified interval and visualize the change over time. 

After pulling in the live data, our application displays a histogram that allows the user to gauge the number of comments and posts per interval to ensure certain time periods do not contain missing data and that they do not contain and significantly more or less data than other periods. We then provide the user the ability to select and apply the data using our four models (BERT1, BERT2, RoBERTa, ELECTRA). The application returns two charts: one showing the six emotions plotted on a line chart by time-interval averages, and a second chart showing the positive and negative sentiment over the same time-interval averages for easy evaluation.

### Hyperparameter Search 

In the fine-tuning of our models, we conducted a hyperparameter search focusing primarily on the learning rate and batch size, two crucial aspects that significantly influence model performance. For BERT1 and BERT2, we arrived at an optimal learning rate of 0.0001, while RoBERTa's configuration required a lower rate of 0.00002, reflecting its sensitivity to learning speed. ELECTRA, on the other hand, performed best with a slightly higher learning rate of 0.00005, possibly due to its distinctive pre-training approach. All models were trained using a consistent batch size of 128, which provided a balance between resource allocation and model update frequency. The Adam optimizer was employed across all models, known for its adaptive learning rate capabilities, contributing to the fine-tuning process's efficiency and effectiveness.

### Results and Observations 

#### Streamlit UI and Example Execution


 
Figure 1: Stream Lit User Input Tab 

The above figure allows users to input number of comments, posts, subreddit name, time filer, and interval to retrieving live reddit posts and comments.



Team Members (Contributors):
1. Ashish Telukunta
2. Thomas Stanton
3. Jonathan Schild 
