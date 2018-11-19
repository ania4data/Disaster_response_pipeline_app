from flask import Flask
from flask import render_template, request, jsonify
import json
import nltk
nltk.download(['stopwords'])
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


app = Flask(__name__)


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


# load data
sql_engine = create_engine(
    'sqlite:///DisasterResponse_est50_vocabnone_size33_rn42.db')
connection = sql_engine.raw_connection()
df_data = pd.read_sql("SELECT * FROM '{}'".format('data'), con=connection)
df_word = pd.read_sql("SELECT * FROM '{}'".format('word'), con=connection)
category_names = list(df_data.columns[4:])

# load model
model = joblib.load("model_est50_vocabnone_size33_rn42.joblib")
print('')

# pick several category for better vialization
df_word_subset = df_word[df_word.category_name.isin(
    ['cold', 'storm', 'shelter', 'weather_related', 'clothing', 'infrastructure_related', 'buildings'])]

# get unique words with high training weight, and their categories
unique_category = list(set(df_word_subset.category_name))
unique_word = list(set(df_word_subset.important_word))

# word dictionary with categories one-hot coded
dict_category_word = {}
for category in unique_category:
    dict_category_word[category] = []
for category in unique_category:
    sub_sub = df_word_subset[df_word_subset.category_name == str(category)]
    for word in unique_word:
        if(word in list(sub_sub.important_word.values)):
            dict_category_word[category].append(np.round(
                sub_sub[sub_sub.important_word == str(word)].importance_value.values[0], 2))
        else:
            dict_category_word[category].append(float(0))

category_active = [np.round(df_data[str(category)].sum(
) * 100 / df_data.shape[0]) for category in category_names]


# get arrays for heat map plotting
heatmap_array = []
for category in unique_category:
    heatmap_array.append(dict_category_word[category])

genre_counts = df_data.groupby('genre').count()['message']
genre_names = list(genre_counts.index)


# index webpage displays cool visuals and receives user input text for model
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

    # Graph1
    graph_one = []
    graph_one.append(
            Bar(
                x=genre_names,
                y=genre_counts, marker=dict(
                color='rgba(222,45,38,0.8)'), opacity=0.6,
            )
        )

    layout_one = dict(title='Distribution of Message Genres',
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count'),
                      font=dict(size=18),
                      )

        # Graph2
    graph_two = []
    graph_two.append(
            Bar(
                x=category_names,
                y=category_active, marker=dict(
                color='rgba(222,45,38,0.8)'), opacity=0.6,
            )
        )

    layout_two = dict(title='Percent of True sample over all sample, per Message category',
                      xaxis=dict(autotick=False, tickangle=-
                      35, automargins=True),
                      yaxis=dict(title='% of True samples',
                      automargins=True, font=dict(size=18),)

                      )

        # Graph3
    trace1 = Bar(
        x=unique_word,
        y=dict_category_word['cold'],
        marker=dict(color='rgba(222,45,38,0.8)'),
        opacity=0.6,
        name='Cold',
        text='Cold',
        width=0.3
            #orientation = 'h'
    )

    trace2 = Bar(
        x=unique_word,
        y=dict_category_word['storm'],
        marker=dict(color='rgb(49,130,189)'),
        opacity=0.76,
        name='Storm',
        text='Storm',
        width=0.3
        #orientation = 'h'
    )

    trace3 = Bar(
        x=unique_word,
        y=dict_category_word['shelter'],
        marker=dict(color='rgb(204,204,204)'),
        opacity=0.9,
        name='Shelter',
        text='Shelter',
        width=0.3
        #orientation = 'h'
    )

    trace4 = Bar(
        x=unique_word,
        y=dict_category_word['weather_related'],
        marker=dict(color='rgb(244,109,67)'),
        opacity=0.4,
        name='Weather_related',
        text='Weather_related',
        width=0.3
        ##orientation = 'h'
    )

    trace5 = Bar(
        x=unique_word,
        y=dict_category_word['clothing'],
        marker=dict(color='rgb(102,205,170)'),
        opacity=0.6,
        name='Clothing',
        text='Clothing',
        width=0.3
        ##orientation = 'h'
    )

    trace6 = Bar(
        x=unique_word,
        y=dict_category_word['infrastructure_related'],
        marker=dict(color='rgb(100,149,237)'),
        opacity=0.6,
        name='Infrastructure_related',
        text='Infrastructure_related',
        width=0.3
        ##orientation = 'h'
    )

    trace7 = Bar(
        x=unique_word,
        y=dict_category_word['buildings'],
        marker=dict(color='rgb(160,82,45)'),
        opacity=0.6,
        name='Buildings',
        text='Buildings',
        width=0.3
        ##orientation = 'h'
    )

    layout_three = dict(title='Words importances per category after training (few columns)',
                        xaxis=dict(autotick=False, tickangle=-35,),
                        yaxis=dict(title='Weights', automargins=True),
                        hovermode='closest',
                        font=dict(size=18),  # barmode='group',
                        )

    graph_three = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

    # Graph4
    graph_four = [Heatmap(z=heatmap_array,
                          x=unique_word,
                          y=unique_category,
                          opacity=0.6,
                          xgap=3,
                          ygap=3,
                          colorscale='Jet')]

    layout_four = dict(
        title='Few category name vs. their most important words after training',
        xaxis=dict(showline=False, showgrid=False, zeroline=False,),
        yaxis=dict(showline=False, showgrid=False, zeroline=False),
        font=dict(size=18),
        plot_bgcolor=('#fff'), height=600
    )

    # add plots/layouts in arrays for Json dump
    graphs = []
    graphs.append(dict(data=graph_four, layout=layout_four))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_one, layout=layout_one))

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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
