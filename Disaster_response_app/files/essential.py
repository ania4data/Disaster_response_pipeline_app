from flask import Flask
from flask import render_template, request, jsonify
import json
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
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
from sklearn.externals import joblib
import sqlalchemy
from sqlalchemy import create_engine




def tokenize(text):
    '''process the text into cleaned tokens

    The text is processed by removing links,emails, ips,
    keeping only alphabet a-z in lower case, then
    test split into individual tokens, stop word is removed,
    and words lemmatized to their original stem

    Args:
      text (str): a message in text form

    Returns:
      clean_tokens (array): array of words after processing
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    emails_regex = '[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+'
    ips_regex = '(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})'
    stopword_list = stopwords.words('english')
    placeholder_list = ['urlplaceholder', 'emailplaceholder', 'ipplaceholder']

    # Remove extra paranthesis for better URL detection
    text = text.replace("(", "")
    text = text.replace(")", "")

    # get list of all urls/emails/ips using regex
    detected_urls = re.findall(url_regex, text)
    detected_emails = re.findall(emails_regex, text)
    # remove white spaces detected ar end of some urls
    detected_emails = [email.split()[0] for email in detected_emails]
    detected_ips = re.findall(ips_regex, text)

    # Remove numbers and special characters, help down vocab size
    pattern = re.compile(r'[^a-zA-Z]')
    stopword_list = stopwords.words('english')

    for url in detected_urls:
        text = re.sub(url, 'urlplaceholder', text)
    for email in detected_emails:
        text = re.sub(email, 'emailplaceholder', text)
    for ip in detected_ips:
        text = re.sub(ip, 'ipplaceholder', text)
    for stop_word in stopword_list:
        if(stop_word in text):
            text.replace(stop_word, '')

    # remove everything except letetrs
    text = re.sub(pattern, ' ', text)
    # initilize
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if((tok not in stopword_list) and (tok not in placeholder_list) and len(tok) > 2):
            clean_tok = lemmatizer.lemmatize(
                lemmatizer.lemmatize(tok.strip()), pos='v')
            # Remove Stemmer for better word recognition in app
            #clean_tok = PorterStemmer().stem(clean_tok)
            clean_tokens.append(clean_tok)

    return clean_tokens