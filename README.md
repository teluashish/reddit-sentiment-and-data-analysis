# reddit-sentiment-and-data-analysis

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

The training and test algorithms run from the reddit-sentiment-and-data-analysis directory. The main.py function conveniently starts a Streamlit serve on port 8888 that can be used to run the scripts and can be executed with the below. The server code that it calls can be found in streamlit_display.py.
**MUST BE IN /code FIRST**

```bash
python3 main.py
```

---

## Online Community Monitoring

## Quantitative and Qualitative Analysis of Subreddit Sentiment Trends**


### Introduction 

Reddit is a popular social media forum that attracts significant numbers of users, content, and engagement from around the globe. Its ever-growing volume of online community discourse presents a platform that can inadvertently shape opinions, damage brands, and incite real-world action. Identifying and understanding swings in user sentiments is becoming more and more critical to combat inaccurate and inadequate information and general community divisiveness. This application allows subreddit stakeholders to monitor their online communities for significant swings in sentiment that could have broadly negative implications if left unchecked.

This initiative leverages live data extraction from Reddit, gathering posts and comments based on either predefined datasets or user-provided criteria. I aim to use sophisticated transformer-based models such as BERT, RoBERTa, and Electra, which have been meticulously adjusted for the task of detecting and forecasting the sentiment of six distinct emotions: sadness, joy, love, anger, fear, and surprise. The sentiment evaluations are compiled to construct a detailed sentiment trajectory for each emotion over a chosen period. The outcome of this assessment is depicted in a detailed chart that illustrates the fluctuation of these emotional sentiments over time, providing critical insights into the dynamics of subreddit communities.

### Description of the data sets

I used two datasets to train the final model(s). The models primarily use a dataset from Hugging Face that contains approximately 90,000 labeled tweets distributed evenly across six emotions: sadness, joy, love, anger, fear, and surprise. I have also used a dataset available on Kaggle that contained almost 8,500 unlabeled Reddit comments from various subreddits.  This dataset was used to experiment with pseudo-labeling and incorporating confident test data into the initial Twitter training data (predictions with confidence score > 0.6 are included in the final merged dataset).

### Description of the NLP models

Three different pre-trained transformer networks from hugging face are fine-tuned during the analysis out of suspicion that predictions of emotions and sentiment could vary depending on the type of model implemented. With each model, I have tuned hyperparameters like learning rate, dropout rate, and batch size. All of them also used the Adam optimizer with the sparse categorical cross entropy as a loss function and monitored validation accuracy for early stopping. 

Developed two BERT-based multi-class classifiers for emotion detection in text, as it learns contextual word representation beneficial for informal language. Model 1 (BERT1) was trained on Twitter data split into training, validation, and test sets. BERT1 achieved validation accuracy and F1 of 0.95. To improve Reddit generalization, BERT1 is applied to unlabeled Reddit comments to pseudo-label high confidence examples (threshold 0.6/label) and combined these newly labeled samples with the Twitter training set. Using this semi-supervised dataset, Model 2 (BERT2) is trained with the same BERT architecture. Incorporating Reddit data reduced performance (validation accuracy/F1: 0.91) but improved applicability to this target domain.


### Experimental Setup

The experimental setup involves leveraging a dataset comprising 90,000 tweets, each labeled with one of six distinct emotions, along with subreddit data to train the sentiment analysis model. To evaluate the model's performance, I chose F1 scores for individual emotions, overall accuracy, and a composite F1 score that encapsulates the model's average effectiveness across all emotions. These metrics provide a quantitative assessment of the model's precision and recall capabilities. As there are six emotions to predict, I will be using the sparse categorical cross-entropy loss to minimize the error. Furthermore, I will conduct both quantitative and qualitative comparisons to ensure a thorough analysis. Numerically, the model's performance is measured by comparing the calculated metrics against benchmarks of other transformer models. Qualitatively, we can examine the trends in the subreddit data to verify that the model's sentiment predictions align with the actual discourse and emotional context present within the data. This dual approach allows us to validate the model's accuracy and its practical applicability in interpreting and reflecting the nuanced sentiment trends within the subreddit communities.

### Implementation and the Execution Flow

I utilized a RoBERTa sequence classifier for the second model, motivated by its optimization of BERT's pre-training procedures and ability to comprehend complex syntactic structures prevalent in Reddit comments. Solely trained on the Twitter corpus, it achieved validation accuracy and F1 scores of 0.95. To diverge from BERT-based models, I have trained an ELECTRA model capable of learning more granular and nuanced language patterns ubiquitous in social media. Unlike BERT and RoBERTa which rely on masked language modeling, ELECTRA is context-sensitive and uses token replacement to increase the performance, enabling more precise sentiment predictions suitable for the intricacies in this domain. ELECTRA model obtained a validation accuracy and F1 score of 0.95, comparable to the BERT and RoBERTa models.

For UI, I used Streamlit to apply the models and display visualizations that help a user identify and understand changes in sentiments of online discourse, in this case, subreddits. A scraper using PRAW fetches Reddit posts in real time based on inputs provided by the user, such as the subreddit name, number of posts and comments, and date-range criteria. The Reddit API limits scraping to the past day, week, month, or year, instead of providing the ability to specify a date range. To ensure to measure changes over time, the dates are pulled for each post and comment and later grouped according to daily, weekly, or monthly intervals extending out from the present. This allowed me to average the sentiments together over the specified interval and visualize the change over time. 

After pulling in the live data, the application displays a histogram that allows the user to gauge the number of comments and posts per interval to ensure certain time periods do not contain missing data and that they do not contain significantly more or less data than other periods. Users will also have the ability to select and apply the data using the four models (BERT1, BERT2, RoBERTa, ELECTRA). The application returns two charts: one showing the six emotions plotted on a line chart by time-interval averages, and a second chart showing the positive and negative sentiment over the same time-interval averages for easy evaluation.

### Hyperparameter Search 

In the fine-tuning of the models, I conducted a hyperparameter search focusing primarily on the learning rate and batch size, two crucial aspects that significantly influence model performance. For BERT1 and BERT2, the optimal learning rate is 0.0001, while RoBERTa's configuration required a lower rate of 0.00002, reflecting its sensitivity to learning speed. ELECTRA, on the other hand, performed best with a slightly higher learning rate of 0.00005, possibly due to its distinctive pre-training approach. All models were trained using a consistent batch size of 128, which provided a balance between resource allocation and model update frequency. The Adam optimizer was employed across all models, known for its adaptive learning rate capabilities, contributing to the fine-tuning process's efficiency and effectiveness.

### Compute Sentiment Score Averages

compute six sentiment scores for each scraped Reddit post or comment. compute the average for each of the scores for each interval using the code 

```
combined_averages = df.groupby('Interval Number')[emotion_columns].mean()
```

This code returns data for concise analysis and plotting of emotional and sentiment trends across different time intervals in the Reddit posts and comments.


### Results and Observations 

#### Model Evaluations

F-1 Scores are consistent across all the emotions in RoBERTa and Electra and are acceptable for a good model. However, ‘Fear’ has the lowest F-1 Score of 0.92 which is still high enough to be acceptable. 

<img width="250" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture10.svg">

The validation accuracy of Electra is 0.9531 with a 0.9530 overall F-1 Score and the validation accuracy of RoBERTa is 0.9502 and F-1 Score is 0.9502. There is no significant difference between the performance metrics between the two models.

<img width="350" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture11.svg">

Figure 1: Classification Report of RoBERTa

The above figure represents different performance metrics to gauge how the fine-tuned RoBERTa model has performed across various emotion categories.

<img width="350" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture12.svg">
Figure 2: Classification Report of Electra 

The above figure represents different performance metrics to gauge how the fine-tuned Electra model has performed across various emotion categories.

#### Streamlit UI and Example Execution

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture1.svg">
 
Figure 3: Stream Lit User Input Tab 

The above figure allows users to input the number of comments, posts, subreddit name, time filter, and interval to retrieve live reddit posts and comments.

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture2.svg">

Figure 4: Transformer model Selection Tab. 

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture3.svg">

Figure 5: Selecting best_model_electra

The above figure represents the output after selecting the best_model_electra and applying the fetched live reddit data on it.

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture4.svg">

Figure 6: Final plots for sentiment trends

The above figure represents the output after selecting the best_model_electra and applying the fetched live reddit data on it.

The first graph on the top shows the trend of average scores for each interval over a period specified by the user. Similarly, the second graph shows the average positive and negative sentiment trends to help correlate the positivity and negativity with the emotions trend.

We have also implemented the feature to use pre-loaded data to plot the sentiment trends, which can be seen from the below figures.

<img width="400" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture5.svg">

Figure 7: Sample execution using Pre-loaded Data (1)


<img width="400" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture6.svg">

Figure 8: Sample execution using Pre-loaded Data (2)

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture7.svg">

Figure 9: Average Sentiment Score Plot of r/wallstreetbets

r/wallstreetbets is primarily a community focused on stock market trading, often characterized by rapid shifts in investment positions, risk-taking, and the volatile nature of gains and losses. The primary emotions involved in this context are typically related to the outcomes of financial decisions—Joy in response to profits, Anger or Sadness due to losses, Fear from market uncertainty, and Surprise from unexpected market movements.

The figure above showcases a line graph depicting the average sentiment scores for six emotions across different intervals. A higher interval number corresponds to more recent data, while lower interval numbers indicate older data. From the graph, we can observe that members of r/wallstreetbets predominantly exhibit emotions of Anger, potentially due to financial losses in stocks, or Joy, likely resulting from profits. These predominant emotions are followed by Sadness, Fear, and Surprise, which are common in the context of r/wallstreetbets. The emotion of Love is observed the least as expected. As we can see the sentiment trend is characteristic of the r/wallstreetbets community. 


Summary and Conclusions

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture8.svg">

Figure 10: Qualitative Analysis of r/funny

<img width="600" alt="image" src="https://github.com/teluashish/reddit-sentiment-and-data-analysis/blob/main/assets/images/Picture9.svg">

Figure 11: Qualitative Analysis of r/wallstreetbets

All the models identified and predicted similar trends in the changes of sentiment, i.e., they each displayed similar deltas in emotion across time. However, while the BERT-based architectures provided robust accuracy, the qualitative analysis showed ELECTRA better captured contextual emotion semantics. This was evident when predicting sentiment across different subreddits using all four of the models and comparing their visualizations. Examples with the “funny” and “wallstreetbets” subreddits show how each model identifies similar patterns, and how the ELECTRA better captures emotions that would be expected from the subreddits. 

## Summary and Conclusions

This project involved developing a sentiment analysis tool using advanced NLP models—BERT1, BERT2, RoBERTa, and ELECTRA—fine-tuned on a dataset of 90,000 Twitter posts labeled with six emotions and Reddit comments from r/wallstreetbets. Optimized the models by conducting a hyperparameter search, particularly focusing on learning rates and batch size, and utilized the Adam optimizer for training efficiency. The models' performance was quantitatively assessed using F1 scores and accuracy, and qualitatively by examining sentiment trends against real-world events. I have integrated these models into a Streamlit application that visually displays sentiment trends over time, allowing users to interact with and analyze the changing online discourse within specified intervals.

## Future Improvements

1. Custom Date Range Retrieval: Modify the Reddit scraping script to enable custom date range inputs for more personalized data extraction.
2. Dynamic Sentiment Analysis: Establish an automated system that continuously integrates, evaluates, and forecasts new data to provide up-to-the-minute sentiment analysis.
3. Enhanced Token Utilization: Expand the max_len parameter in pre-trained models from Hugging Face to include a greater number of tokens, enhancing the model's ability to discern subtleties in lengthier text segments, which could lead to more accurate sentiment interpretation.
4. Data Augmentation and Pseudo-Labeling: Enrich the training dataset by collecting additional data and employing pseudo-labeling techniques to expand the model's learning scope with unsupervised data.

