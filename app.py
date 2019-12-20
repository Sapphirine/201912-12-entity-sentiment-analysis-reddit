from google.cloud import bigquery
from operator import add, iadd
from google.oauth2 import service_account
import pandas as pd
import string
import nltk
import re
import json
import nltk
from nltk.corpus import stopwords
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import reduce
import json
# from passlib.hash import sha256_crypt
import os
from os import path
# from sqlalchemy import *
# from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response, flash, session, abort, url_for

nltk.download('stopwords')

google_key = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
with open('google_key.json','w') as fp:
    json.dump(google_key, fp)

key_path = "google_key.json"

credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bqclient = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
)

tmpl_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
success_code = json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

@app.route('/')
def main_page():
    
    context = dict(
        entity_name="", 
        post_mentions="-",
        post_scores="-",
        comment_mentions="-",
        mean_comments="-",
        most_common_words=[], 
        week_count=[],
        month_count=[],
        year_count=[],
        sentiment_score=0,
        most_positive=[],
        most_negative=[]
    )

    return render_template("index.html", **context)


@app.route('/search-entity')
def search():
    entity_name = request.args.get('entity')

    context = descriptive_analytics(entity_name)

    return render_template("index.html", **context)

import time

def descriptive_analytics(company):
    
    company = str(company).lower()

    # Query and generate relevant posts and comments
    ## df_comments and df_posts to be sent for modeling sentiment analysis

    query_post = (
        "SELECT id, title, created_utc, num_comments, score "
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
        "SELECT body, link_id, score, created_utc "
        "FROM `homework2-255022.redditbigdata.comments` "
        "WHERE SUBSTR(link_id, STRPOS(link_id, '_') + 1, LENGTH(link_id)) IN ("
        "SELECT id "
        "FROM `homework2-255022.redditbigdata.posts`"
        "WHERE LOWER(title) LIKE LOWER('%" + company + "%'))"
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
    df_comments = df_comments[(df_comments.body != '[removed]') & (df_comments.body != '[deleted]')]

    df_post = df_post.drop_duplicates()
    df_comments = df_comments.drop_duplicates()

    # Sentiment Analysis

    df_comments['scaled_score'] = df_comments.apply(lambda x: 1 if x.score >= 50 else (0 if x.score <= -10 else (x.score + 10)/60), axis=1)
    scaled_sum = sum(df_comments.scaled_score)
    df_comments['weight'] = df_comments.apply(lambda x: x.scaled_score / scaled_sum, axis = 1)

    analyzer = SentimentIntensityAnalyzer()
    def vader_score(comment):
        score = analyzer.polarity_scores(comment)["compound"]
        return score

    df_comments['vader'] = df_comments.apply(lambda x: vader_score(x.body), axis=1)
    df_comments['weighted_score'] = df_comments.apply(lambda x: x.weight * ((x.vader*50)+50), axis=1)
    sentiment_score = sum(df_comments.weighted_score)

    most_positive = []
    most_negative = []
    sorted_df = df_comments.sort_values(by=['vader'], ascending=False)
    for i in range (0, 10):
        most_negative.append([sorted_df.iloc[(i+1)*(-1),0],str(sorted_df.iloc[(i+1)*(-1),2]), str(sorted_df.iloc[(i+1)*(-1),6])])
        most_positive.append([sorted_df.iloc[i,0],str(sorted_df.iloc[i,2]), str(sorted_df.iloc[i,6])])


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
    if len(df_post) > 0:
        df_post.title = df_post.title.str.lower().str.split()
    if len(df_comments) > 0:
        df_comments.body = df_comments.body.str.lower().str.split()
    df_post.title = df_post.title.apply(lambda x: [item for item in x if item not in stop_words_set])
    df_comments.body = df_comments.body.apply(lambda x: [item for item in x if item not in stop_words_set])

    # Compute most common words
    if len(df_post) != 0 and len(df_comments) != 0:
        words = []
        df_post.title.apply(lambda x: iadd(words, x))
        df_comments.body.apply(lambda x: iadd(words, x))
        word_frequency = Counter(words)
    elif len(df_post) == 0:
        words = []
        df_comments.body.apply(lambda x: iadd(words, x))
        word_frequency = Counter(words)
    elif len(df_comments) == 0:
        words = []
        df_post.title.apply(lambda x: iadd(words, x))
        word_frequency = Counter(words)
    else:
        word_frequency = [("None", 0)]

    most_common_words = []
    for i in word_frequency.most_common(15):
        if i[0] != '[removed]' and len(most_common_words) < 10:
            most_common_words.append(i)

    # Compute top 4 metrics
    query_comment_mentions = (
        "SELECT body, link_id, ups, downs, score, created_utc "
        "FROM `homework2-255022.redditbigdata.comments` "
        "WHERE LOWER(body) LIKE LOWER('%" + company + "%')"
    )

    job_comment_mentions = bqclient.query(
        query_comment_mentions,
        location="US",
    )  # API request - starts the query

    df_comment_mentions = job_comment_mentions.result().to_dataframe()
    comment_mentions = len(df_comment_mentions)
    mean_comments = df_post.num_comments.mean()

    # Compute periodical counts
    max_time = 1564617378
    interval = 3600

    week_count_posts = []
    month_count_posts = []
    year_count_posts = []
    for i in range(7, 0, -1):
        tmp = df_post[(df_post.created_utc > max_time - (i * interval * 24)) & (df_post.created_utc <= max_time - ((i-1) * interval * 24))]
        week_count_posts.append(len(tmp))
    for i in range(4, 0, -1):
        tmp = df_post[(df_post.created_utc > max_time - (i * interval * 24 * 7)) & (df_post.created_utc <= max_time - ((i-1) * interval * 24 * 7))]
        month_count_posts.append(len(tmp))
    for i in range(7, 0, -1):
        tmp = df_post[(df_post.created_utc > max_time - (i * interval * 24 * 30)) & (df_post.created_utc <= max_time - ((i-1) * interval * 24 * 30))]
        year_count_posts.append(len(tmp))

    week_count_comments = []
    month_count_comments = []
    year_count_comments = []
    for i in range(7, 0, -1):
        tmp = df_comment_mentions[(df_comment_mentions.created_utc > max_time - (i * interval * 24)) & (df_comment_mentions.created_utc <= max_time - ((i-1) * interval * 24))]
        week_count_comments.append(len(tmp))
    for i in range(4, 0, -1):
        tmp = df_comment_mentions[(df_comment_mentions.created_utc > max_time - (i * interval * 24 * 7)) & (df_comment_mentions.created_utc <= max_time - ((i-1) * interval* 24 * 7))]
        month_count_comments.append(len(tmp))
    for i in range(7, 0, -1):
        tmp = df_comment_mentions[(df_comment_mentions.created_utc > max_time - (i * interval * 24 * 30)) & (df_comment_mentions.created_utc <= max_time - ((i-1) * interval * 24 * 30))]
        year_count_comments.append(len(tmp))

    week_count = list(map(add, week_count_posts, week_count_comments))
    month_count = list(map(add, month_count_posts, month_count_comments))
    year_count = list(map(add, year_count_posts, year_count_comments))

    post_mentions = len(df_post)
    post_score = df_post.score.sum()

    # Output results to json
    if str(post_score) == "None":
        post_score = "0"
    if str(mean_comments) == "nan":
        mean_comments = "0"
    else:
        mean_comments = round(mean_comments, 2)

    total_mentions = int(post_mentions) + int(comment_mentions)
    # Output results to json
    results = {
        "entity_name": company, 
        "total_mentions": "{0:,.0f}".format(total_mentions),
        "post_mentions": "{0:,.0f}".format(post_mentions),
        "post_scores": "{0:,.0f}".format(post_score),
        "comment_mentions": "{0:,.0f}".format(comment_mentions),
        "mean_comments": "{0:,.2f}".format(mean_comments),
        "most_common_words": most_common_words,
        "week_count": week_count,
        "month_count": month_count,
        "year_count": year_count,
        "sentiment_score": sentiment_score,
        "most_positive": most_positive,
        "most_negative": most_negative
    }
            # Output results to json
    print(results)

    return results
# results_json = json.dumps(results)






if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=8111, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using

            python server.py
        """

        # reload templates when HTML changes
        extra_dirs = [os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'templates'), ]
        extra_files = extra_dirs[:]
        for extra_dir in extra_dirs:
            for dirname, dirs, files in os.walk(extra_dir):
                for filename in files:
                    filename = path.join(dirname, filename)
                    if path.isfile(filename):
                        extra_files.append(filename)

        HOST, PORT = host, port
        print("running on %s:%d" % (HOST, PORT))

        app.run(host=HOST, port=PORT, debug=debug,
                threaded=threaded, extra_files=extra_files)

    run()