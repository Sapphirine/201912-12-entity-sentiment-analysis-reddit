# RedditFeels: A Sentiment Analysis Tool for Reddit News

**Project ID:** 201912-12
**Team Members:** 
- Timotius Kartawijaya (tak2151@columbia.edu)
- Fernando Troeman (ft2515@columbia.edu)
- Hritik Jain (hj2533@columbia.edu)

## Overview:

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

## Folder Structure:
- `static` and `templates` contain all code needed for the user interface.
- `analysis` contain jupyter notebooks that are used for validation and analysis
- `Procfile` used for Heroku deployment. 

## How to use locally:

1. `pip install -r requirements.txt` to install required packages.

2. A Google API Service Account Key is needed to access the BigQuery database containing a subset of the reddit data (more info: https://cloud.google.com/iam/docs/creating-managing-service-account-keys). To get a key, please contact one of the authors above.

3. run `python app.py` to start the application on your local server. Use `--debug` flag to use debug mode. 






