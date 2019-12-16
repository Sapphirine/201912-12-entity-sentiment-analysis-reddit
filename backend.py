from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import string
import nltk
import re
import json
import nltk
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')

# Setup BigQuery client with credentials
## EDIT CREDENTIALS

key_path = "causal-block-257406-c3c917894932.json"

credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bqclient = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
)

## EDIT COMPANY TO INPUT RECEIVED FROM FRONT-END
company='twitter'

# Query and generate relevant posts and comments
## df_comments and df_posts to be sent for modeling sentiment analysis

query_post = (
    "SELECT id, title, selftext, num_comments "
    "FROM `homework2-255022.redditbigdata.posts` "
    "WHERE LOWER(title) LIKE LOWER('%" + company + "%');"
)

job_post = bqclient.query(
    query_post,
    location="US",
)  # API request - starts the query

df_post = (
    job_post
    .result()
    .to_dataframe()
)

comments = []
query_comments = (
    "SELECT body, link_id, ups, downs, score "
    "FROM `homework2-255022.redditbigdata.comments` "
    "WHERE SUBSTR(link_id, STRPOS(link_id, '_') + 1, LENGTH(link_id)) IN ("
    "SELECT id "
    "FROM `homework2-255022.redditbigdata.posts`"
    "WHERE LOWER(title) LIKE LOWER('%" + company + "'))"
)

job_comments = bqclient.query(
    query_comments,
    location="US",
)  # API request - starts the query

df_comments = (
    job_comments
    .result()
    .to_dataframe()
)

comments.append(df_comments)
df_comments = pd.concat(comments, ignore_index=True)


# Compute Metrics

# Remove unnecessary characters 
df_post.title = df_post.title.apply(lambda x: [x.replace("*", "").\
                                                   replace("#", "").\
                                                   replace("-", "")][0])
df_comments.body = df_comments.body.apply(lambda x: [x.replace("*", "").\
                                                   replace("#", "").\
                                                   replace("-", "")][0])

# Download and remove set of stop words
stop_words_set = set(stopwords.words('english'))
df_post.title = df_post.title.str.lower().str.split()
df_comments.body = df_comments.body.str.lower().str.split()
df_post.title = df_post.title.apply(lambda x: [item for item in x if item not in stop_words_set])
df_comments.body = df_comments.body.apply(lambda x: [item for item in x if item not in stop_words_set])

# Compute most common words
word_frequency = Counter(df_post.title.sum() + df_comments.body.sum())
most_common_words = []
for i in word_frequency.most_common(15):
    if i[0] != '[removed]' and len(most_common_words) < 10:
        most_common_words.append(i[0])

# Compute top 4 metrics
query_comment_mentions = (
    "SELECT COUNT(*)"
    "FROM `homework2-255022.redditbigdata.comments` "
    "WHERE LOWER(body) LIKE LOWER('%" + company + "%')"
)

job_comment_mentions = bqclient.query(
    query_comment_mentions,
    location="US",
)  # API request - starts the query

comment_mentions = job_comment_mentions.result().to_dataframe().iloc[0, 0]
mean_comments = df_post.num_comments.mean()

query_post_mentions = (
    "SELECT COUNT(*) "
    "FROM `homework2-255022.redditbigdata.posts` "
    "WHERE LOWER(title) LIKE LOWER('%" + company + "%')"
)

job_post_mentions = bqclient.query(
    query_post_mentions,
    location="US",
)

post_mentions = job_post_mentions.result().to_dataframe().iloc[0, 0]

query_post_score = (
    "SELECT SUM(score) as score "
    "FROM "
      "(SELECT DISTINCT id, score "
      "FROM `homework2-255022.redditbigdata.posts` "
      "WHERE LOWER(title) LIKE LOWER('%" + company + "%'))"
)

job_post_score = bqclient.query(
    query_post_score,
    location="US",
)

post_score = job_post_score.result().to_dataframe().iloc[0, 0]

# Output results to json

results = {
    "post_mentions": str(post_mentions),
    "post_score": str(post_score),
    "comment_mentions": str(comment_mentions),
    "mean_comments": str(round(mean_comments, 2)),
    "most_common_words": most_common_words
}

results_json = json.dumps(results)




