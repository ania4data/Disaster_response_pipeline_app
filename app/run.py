import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import sqlalchemy
from sqlalchemy import create_engine
import plotly
# tutorial suggestion did not work
#import plotly.graph_objs as go
# this worked instead
from plotly.graph_objs import *
import plotly.plotly as py



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
category_names = list(df_data.columns[4:])
#df_data = pd.read_sql_table('data', sql_engine)
#df_word = pd.read_sql_table('word', sql_engine)

# load model
#model = joblib.load("model_est50_vocabnone_size33_rn42.joblib")
model = joblib.load("best_model.joblib")
print('')

df_word_subset = df_word[df_word.category_name.isin(['cold', 'storm', 'shelter', 'weather_related', 'clothing', 'infrastructure_related', 'buildings'])]

unique_category = list(set(df_word_subset.category_name))
unique_word = list(set(df_word_subset.important_word))

print(list(set(df_word_subset.category_name)))
print(list(set(df_word_subset.important_word)))

dict_category_word = {}


for category in unique_category:
    dict_category_word[category] = []

for category in unique_category:
    sub_sub = df_word_subset[df_word_subset.category_name == str(category)]
    for word in unique_word:
        if(word in list(sub_sub.important_word.values)):
            dict_category_word[category].append(np.round(sub_sub[sub_sub.important_word == str(word)].importance_value.values[0],2))
        else:
            dict_category_word[category].append(float(0))


print(dict_category_word)


category_active = [np.round(df_data[str(category)].sum()*100/df_data.shape[0]) for category in category_names]

print(category_active)

heatmap_array = []
for category in unique_category:
    heatmap_array.append(dict_category_word[category])

print(heatmap_array)

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

    graphs = []

    # Graph1

    graph_one = []
    graph_one.append(
      Bar(
      x = genre_names,
      y = genre_counts,marker=dict(
        color='rgba(222,45,38,0.8)'), opacity=0.6,
      )
    )

    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre'),
                yaxis = dict(title = 'Count'),
                )  

    graphs.append(dict(data=graph_one, layout=layout_one))

    # Graph2

    category_active = [np.round(df_data[str(category)].sum()*100.0/df_data.shape[0]) for category in category_names]

    graph_two = []
    graph_two.append(
      Bar(
      x = category_names,
      y = category_active ,marker=dict(
        color='rgba(222,45,38,0.8)'), opacity=0.6,
      )
    )

    layout_two = dict(title = 'Percent of True sample over all sample, per Message category',
                xaxis = dict(autotick= False, tickangle=-35, automargins=True),
                yaxis = dict(title = '% of True samples', automargins=True)
                
                )  

    graphs.append(dict(data=graph_two, layout=layout_two))

    # Graph3
#['blanket', 'well', 'clothe', 'rain', 'flood', 'shelter', 'avalanche', 'damage', 'tent', 'earthquake', 'destroy', 'hurricane', 'snow', 'snowfall', 'wind']
    
    trace1 = Bar(
      x = unique_word,
      y = dict_category_word['cold'] ,
      marker=dict(color='rgba(222,45,38,0.8)'),
      opacity=0.6,
      name = 'Cold',
      text = 'Cold',
      #width = [1.5,1.5,1.5,1.5,1.5,1.5,7.5,1.5,1.5,1.5,1.5,1.5,7.5,7.5,1.5]
      #orientation = 'h'
      )

    trace2 = Bar(
      x = unique_word,
      y = dict_category_word['storm'] ,
      marker=dict(color='rgb(49,130,189)'),
      opacity=0.76,
      name = 'Storm',
      text = 'Storm'
      #orientation = 'h'
      )

    trace3 = Bar(
      x = unique_word,
      y = dict_category_word['shelter'] ,
      marker=dict(color='rgb(204,204,204)'),
      opacity=0.9,
      name = 'Shelter',
      text = 'Shelter'
      #orientation = 'h'
      )

    trace4= Bar(
      x = unique_word,
      y = dict_category_word['weather_related'] ,
      marker=dict(color='rgb(244,109,67)'),
      opacity=0.4,
      name = 'Weather_related',
      text = 'Weather_related'
      ##orientation = 'h'
      )


    trace5= Bar(
      x = unique_word,
      y = dict_category_word['clothing'] ,
      marker=dict(color='rgb(102,205,170)'),
      opacity=0.6,
      name = 'Clothing',
      text = 'Clothing'
      ##orientation = 'h'
      )

    trace6= Bar(
      x = unique_word,
      y = dict_category_word['infrastructure_related'] ,
      marker=dict(color='rgb(100,149,237)'),
      opacity=0.6,
      name = 'Infrastructure_related',
      text = 'Infrastructure_related'
      ##orientation = 'h'
      )

    trace7= Bar(
      x = unique_word,
      y = dict_category_word['buildings'] ,
      marker=dict(color='rgb(160,82,45)'),
      opacity=0.6,
      name = 'Buildings',
      text = 'Buildings'
      ##orientation = 'h'
      )   

    #['cold', 'storm', 'shelter', 'weather_related', 'clothing', 'infrastructure_related', 'buildings']

    layout_three = dict(title = 'Words importances per category',
                xaxis = dict(autotick= False, tickangle=0, automargins=True),
                yaxis = dict(title = 'Weights', automargins=True),
                hovermode= 'closest' #, barmode='group',               
                )  

    graph_three = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

    graphs.append(dict(data=graph_three, layout=layout_three))

    # Graph4

    heatmap_array = []
    for category in unique_category:
        heatmap_array.append(dict_category_word[category])


    graph_four = [Heatmap(z=heatmap_array,
                   x=unique_word,
                   y=unique_category,
                   opacity=0.6,
                   xgap = 15,
                   ygap = 15,
                   colorscale='Jet')]
                   #[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'],
                   #[0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'],
                   #[0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
                   #[0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'],
                   #[0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
                   

    #graph_four=[trace]

    layout_four = dict(
            title='Category name vs. most important words',
            xaxis = dict(ticks='', nticks=36),
            yaxis = dict(ticks='' )
    )


    graphs.append(dict(data=graph_four, layout=layout_four))

    print(category_active)

    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],

    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]
    
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