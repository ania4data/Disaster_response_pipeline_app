# Disaster Response Pipeline Project

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/wordcloud_twitter_disaster.jpg", style="width:50%">
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

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/wordcloud_twitter_disaster.jpg", style="width:50%">
</p>

## Instructions:

## How to run the code:

0. Clone the repository use: `https://github.com/ania4data/Disaster_response_pipeline.git`
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves it
        `python models/train_classifier.py data/DisasterResponse.db models/model.joblib`

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/evaluation_matrix.png", style="width:50%">
</p>

2. After replacing the database name and model pickle file in `run.py` with ones in `step1`, Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/front_page.png", style="width:50%">
</p>

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/category_selection_app.png", style="width:50%">
</p>
