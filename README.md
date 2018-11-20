# Disaster Response Pipeline Project

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/wordcloud_twitter_disaster.jpg", style="width:30%">
</p>

# App layout
```
├── app
│   ├── best_model.joblib
│   ├── DisasterResponse_est50_vocabnone_size33_rn42.db
│   ├── model_est50_vocabnone_size33_rn42.joblib
│   ├── run.py
│   ├── static
│   │   ├── category_selection_app.png
│   │   ├── evaluation_matrix.png
│   │   ├── front_page.png
│   │   └── wordcloud_twitter_disaster.jpg
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └──process_data.py
│
├── ETL_pipeline_prep
│   ├── categories.csv
│   ├── ETL Pipeline Preparation.ipynb
│   └── messages.csv
│
├── ML_pipeline_prep
│   ├── best_model.sav
│   ├── DisasterResponse.db
│   ├── ML Pipeline Preparation.ipynb
│   └── model_est50_vocabnone_size33_rn42_idftrue_depth1.joblib
├── models
│   ├── DisasterResponse_est50_vocabnone_size33_rn42.db
│   ├── DisasterResponse_est50_vocabnone_size33_rn42_idftrue_depth1.db
│   ├── model_est50_vocabnone_size33_rn42_idftrue_depth1.joblib
│   ├── model_est50_vocabnone_size33_rn42.joblib
│   └── train_classifier.py
│
├── LICENSE
└── README.md
```
## General repository content:

- data folder: two csv files, that will be processed using `process_data.py` into `DisasterResponse.db`
- models folder: contain `train_classifier.py` that use information from the database and create clssifier for disaster messages, while saving best model to a `*.joblib` file
- app folder: contain `run.py` where all database tables are processed into plotly plots, as well as sample of database and model created with app
- app/static folder: contain all static images in the app
- app/html folder: contain `*.html` files for running the app
- ETL_pipeline_prep and ML_pipeline_prep: Jupyter notebooks regarding the project (not needed for app)
- LICENSE file
- README.md file


## How to run the code:

0. Clone the repository use: `https://github.com/ania4data/Disaster_response_pipeline.git`
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves it
        `python models/train_classifier.py data/DisasterResponse.db models/model.joblib`

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/evaluation_matrix.png", style="width:30%">
</p>

2. After replacing the database name and model pickle file in `run.py` with ones in `step1`, Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/front_page.png", style="width:30%">
</p>

## Some discussion about the unbalanced categories:

- The original csv files after merging, contained about 20% samples that did not have labels. After investigating content of those messages, it seemed that the no-label is not part of a one-hot coding strategy. Many of those messages were related to actual events (e.g. fire, aid, ...) but were not labeled. Those samples as well as ~200 rows of data with label `2` belonging to `related` category were removed. Some of these messages also showed translation from `original` message column in English to `message` column in Spanish!, keeping those messages when no translation algorithms is not applied to them before pipeline would impact the algorithm performance considering their sample size.
- The multioutput method in the pipeline, applis Adaboost algorithm to each category. Considering its nature Adaboost puts more weight on mislabled samples during the training. This seems suitable over methods such is Random Forest that are less accurate and slower.
- The data for several categories is highly unbalanced with even less than 2% sample for positive class. Therefore using accuracy metrics is inappropriate for the optimization task. Since accuracy will not penalize the class with low sample size. For example for a very important category `missing` the positive label was less than `5%` of the data. Mis-identifying the missing people messages (high False negative rate) is extremely consequetial. Another side of the story is identifying events incorrectly, for example algorithm predict `fire` incorrectly, or have high false positive rate, which can lead to sending resource to places that are not affected. This is also costly. So a delicated balance between handling `FN` and `FP` is needed. For this analysis, recall metrics to catch the `FN` and precision to get `FP`, Fscore (F1, F_beta), or roc_auc_score are more appropriate. In order to ensure better performance algorithm is opimized using f1_score (combination of recall and precision) but also tested with roc_auc score, which showed not significant improvement over f1_score.
- Even after considering different metrics to deal with unbalanced categories, prediction is not ideal. More obvious remedy is collecting more data in those specific scenraios. Down sampling the negative class is also an option when the vocabulary integrity will not get jeopardized.

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/category_selection_app.png", style="width:30%">
</p>

## Preliminary Heroku app

The custom tokenizer function within the pipeline when model deployed on heroku is not properly called after many attempts, even though it works ok on local machine. It seems the problem is due to pickle function hierarchy. Therefore, app is deployed using trained model with scikit embedded tokenizer. The performance is not as great as cutom fucntion for this reason. One remedy can be cleaning the messages in the dabase from links/stop words, and also lemmanize and save to db and then use this arrays as input to the model, without directly calling custom tokenizer within pipeline.

https://disasterresponseapp.herokuapp.com/
