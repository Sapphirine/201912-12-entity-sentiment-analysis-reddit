# RedditFeels: A Sentiment Analysis Tool for Reddit News

**Project ID:** 201912-12

**Team Members:** 
- Timotius Kartawijaya (tak2151@columbia.edu)
- Fernando Troeman (ft2515@columbia.edu)
- Hritik Jain (hj2533@columbia.edu)

## Overview:
Our web application is capable of processing and computing public sentiment towards companies or organizational entities. Such a product or service will be valuable in terms of informing investment decisions as well as evaluating brand recognition and loyalty. We have elected to work with data from the popular social media platform, Reddit, which houses a huge repository of user opinions and discussions on numerous wide-ranging topics and events. The resulting product is a web application that, given the name of a company specified by the user, extracts associated Reddit posts/comments and computes a sentiment score as well as other useful metrics to be output in the form of an analytics dashboard.


## URL:
https://reddit-feels.herokuapp.com/

## Technology Stack
User Interface: 
- Flask
- JQuery
- D3.js / Chart.js

Backend/Database: 
- Flask 
- Spark
- VADER
- BigQuery

## Folder and File Structure:
- `static` and `templates` contain all code needed for the user interface.
- `analysis` contain jupyter notebooks that are used for validation and analysis.
- `app.py` is the entrypoint to the web app.
- `Procfile` used for Heroku deployment. 

## How to use locally:

1. `pip install -r requirements.txt` to install required packages.

2. A Google API Service Account Key is needed to access the BigQuery database containing a subset of the reddit data (more info: https://cloud.google.com/iam/docs/creating-managing-service-account-keys). To get a key, please contact one of the authors above.

3. run `python app.py` to start the application on your local server. Use `--debug` flag to use debug mode. 






