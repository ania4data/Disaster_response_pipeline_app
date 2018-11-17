import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import sqlalchemy
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
sql_engine = create_engine('sqlite:///DisasterResponse_est50_vocabnone_size33_rn42.db')
connection = sql_engine.raw_connection()
df_data = pd.read_sql("SELECT * FROM '{}'".format('data'), con=connection)
df_word = pd.read_sql("SELECT * FROM '{}'".format('word'), con=connection)
#df_data = pd.read_sql_table('data', sql_engine)
#df_word = pd.read_sql_table('word', sql_engine)

# load model
#model = joblib.load("model_est50_vocabnone_size33_rn42.joblib")
model = joblib.load("best_model.joblib")
print('')

print(df_word[df_word.category_name=='cold'])

#print(df_word[df_word.category_name=='floods'])  #no

print(df_word[df_word.category_name=='storm'])

#print(df_word[df_word.category_name=='fire']) #no <>

#print(df_word[df_word.category_name=='earthquake']) #no <>

print(df_word[df_word.category_name=='shelter'])

#print(df_word[df_word.category_name=='missing_people']) #<>

print(df_word[df_word.category_name=='weather_related'])

print(df_word[df_word.category_name=='clothing'])

print(df_word[df_word.category_name=='infrastructure_related'])

#print(df_word[df_word.category_name=='other_infrastructure']) #no

#print(df_word[df_word.category_name=='transport']) #no

print(df_word[df_word.category_name=='buildings'])

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df_data.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_data.columns[4:], classification_labels))

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