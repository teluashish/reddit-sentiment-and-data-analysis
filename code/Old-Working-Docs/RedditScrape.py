#%%
import praw
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Initialize PRAW with your API credentials
reddit = praw.Reddit(client_id="nFKOCvQQEIoW2hFeVG6kfA", 
                     client_secret="5BBB4fr-HMPtO8f4jZhle74-fYcDkQ",
                     user_agent="Icy_Process3191",)

def fetch_comments(submission, limit):
    submission.comment_sort = 'best'
    submission.comments.replace_more(limit=0)
    return [(submission.id, comment.body, comment.created_utc, comment.score) for comment in submission.comments.list()[:limit]]

def get_posts(sub_name, top_posts_count, top_comments_count):
    subreddit = reddit.subreddit(sub_name)
    top_posts = list(subreddit.top(time_filter='month', limit=top_posts_count))

    with ThreadPoolExecutor() as executor:
        comments = list(executor.map(lambda post: fetch_comments(post, top_comments_count), top_posts))

    post_data = [{'post_subreddit': post.subreddit.display_name, 
                  'post_id': post.id, 
                  'post_text': post.selftext, 
                  'post_date': post.created_utc,
                  'post_score': post.score} for post in top_posts]

    comment_data = [item for sublist in comments for item in sublist]

    post_df = pd.DataFrame(post_data)
    comment_df = pd.DataFrame(comment_data, columns=['post_id', 'comment_text', 'comment_date', 'comment_score'])

    # Sort comments within each post by score and then add a rank column
    post_df['post_rank'] = post_df['post_score'].rank(method='first', ascending=False)
    comment_df['comment_rank'] = comment_df.groupby('post_id')['comment_score'].rank(method='first', ascending=False)

    combined_df = pd.merge(post_df, comment_df, on='post_id')
    combined_df['post_date'] = pd.to_datetime(combined_df['post_date'], unit='s')
    combined_df['comment_date'] = pd.to_datetime(combined_df['comment_date'], unit='s')
    
    return combined_df

# get top 10 comments from top 30 posts in r/funny; 6-7 seconds
subreddit = 'natureporn'
num_posts = 3
num_comments = 1

reddit_comms_df = get_posts(sub_name=subreddit, top_posts_count=num_posts, top_comments_count=num_comments)
reddit_comms_df.to_csv(f'{subreddit}.csv', index=False)
reddit_comms_df.head()

# get comment text, preprocess, apply model to it
#texts = combined_df['comment_text'].tolist()
