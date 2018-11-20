from disasterresponseapp import app
from flask import Flask
from flask import render_template, request, jsonify
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import plotly
from plotly.graph_objs import Bar
# tutorial suggestion did not work
#import plotly.graph_objs as go
# this worked instead
from plotly.graph_objs import *
import plotly.plotly as py
import re
#from sklearn.externals import joblib
import pickle
import sqlalchemy
from sqlalchemy import create_engine
from files.essential import tokenize
import wrangling_scripts.wrangle_data
from wrangling_scripts.wrangle_data import return_figures

# index webpage displays cool visuals and receives user input text for model

graphs, model, df_data = return_figures()

@app.route('/')
@app.route('/index')
def index():
    '''prepare plotly graphs and layout to dump to json
    for html frontend use

    Args:
      None

    Returns:
      None

    '''

    

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''save user input in query

    Arge:
      None

    Returns:
      None

    '''

    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(
        zip(df_data.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



    