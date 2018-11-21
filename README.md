# Disaster Response Pipeline Project

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/wordcloud_twitter_disaster.jpg", style="width:30%">
</p>

# Deployed Heroku app online

## https://disasterresponseapp.herokuapp.com/


# Repository layout

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
├── README.md
├── requirement.txt
│
└── Disaster_response_app

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
- requirement.txt file: list of program to be pip installed in order to run the app
- Disaster_reponse_app folder: this folder contain all the files needed to create the app online. It is important to note that while main structure of repository support app on the local host, the layout for hosting the app on a server require additional files and chnaged to `*.py` file. In case interestednecessary layout for online hosting. For more info see end of the README.md file


## How to run the code:

0. Clone the repository use: `https://github.com/ania4data/Disaster_response_pipeline.git`, and pip install `requirement.txt`
```
conda update python
python3 -m venv name_of_your_choosing
source name_of_your_choosing/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                      # install packages in requirement
```
you can also follow similar instruction as listed here:

https://github.com/ania4data/World_happiness_app/blob/master/README.md

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves it
        `python models/train_classifier.py data/DisasterResponse.db models/model.joblib`

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/evaluation_matrix.png" height="500" style="width:30%">
</p>

2. After replacing the database name and model pickle file in `run.py` with ones in `step1`, Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/front_page.png" style="width:30%">
</p>

## Some discussion about the unbalanced categories:

- The original csv files after merging, contained about 20% samples that did not have labels. After investigating content of those messages, it seemed that the no-label is not part of a one-hot coding strategy. Many of those messages were related to actual events (e.g. fire, aid, ...) but were not labeled. Those samples as well as ~200 rows of data with label `2` belonging to `related` category were removed. Some of these messages also showed translation from `original` message column in English to `message` column in Spanish!, keeping those messages when no translation algorithms is not applied to them before pipeline would impact the algorithm performance considering their sample size.
- The multioutput method in the pipeline, applis Adaboost algorithm to each category. Considering its nature Adaboost puts more weight on mislabled samples during the training. This seems suitable over methods such is Random Forest that are less accurate and slower.
- The data for several categories is highly unbalanced with even less than 2% sample for positive class. Therefore using accuracy metrics is inappropriate for the optimization task. Since accuracy will not penalize the class with low sample size. For example for a very important category `missing` the positive label was less than `5%` of the data. Mis-identifying the missing people messages (high False negative rate) is extremely consequetial. Another side of the story is identifying events incorrectly, for example algorithm predict `fire` incorrectly, or have high false positive rate, which can lead to sending resource to places that are not affected. This is also costly. So a delicated balance between handling `FN` and `FP` is needed. For this analysis, recall metrics to catch the `FN` and precision to get `FP`, Fscore (F1, F_beta), or roc_auc_score are more appropriate. In order to ensure better performance algorithm is opimized using f1_score (combination of recall and precision) but also tested with roc_auc score, which showed not significant improvement over f1_score.
- Even after considering different metrics to deal with unbalanced categories, prediction is not ideal. More obvious remedy is collecting more data in those specific scenraios. Down sampling the negative class is also an option when the vocabulary integrity will not get jeopardized.

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/category_selection_app.png" style="width:30%">
</p>


## Some Discussion on app deployment online (not local host)

After cloning the repository, use diaster_response_app folder from now on only:

- Create a virtual enviroment where `requirements.txt` is located: 
```
conda update python
python3 -m venv name_of_your_choosing
source name_of_your_choosing/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                      # install packages in requirement
```
- you can also follow similar instruction as listed here:

https://github.com/ania4data/World_happiness_app/blob/master/README.md

IMPORTANT: In addition of creating Procfile, __init__ file, and modifying several other file including run.py, disasterresponse.py, within `wrangling_scripts/wrangle_data.py` following class is added:

```
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'tokenize':
            from files.essential import tokenize
            return tokenize
        return super().find_class(module, name)

model = CustomUnpickler(open('files/best_model_serial_wtokenizer.pkl', 'rb')).load()
 ```   
 Without this class model built using train_classifer_pkl.py even though successfully generate a working model (with tokenizer) in local host, it can not be depoloyed. The reason is due to pickle function hierarchy and the fact that creating pkl file within train_classifer_pkl.py happened in the __main__ function. When model depolyed, the structure of the model pipeline is understood but not custom functions e.g.(tokenize) within them. In the app whre the pkl file is loaded again, the app has no knowledge of __main__, and this type of error is reported `Can't get attribute 'tokenize' on <module '__main__'` By using a Custompickler function before loading the app, the class overwrite initirinsic pickle `find_class` function and provide it a direct address to tokenize function (within essetial.py). So simply importing the tokenize on top of the wrangle_data.py was not enough. Need to also add that for consistency and due to sklearn warinig only pickle dump and load are used (instead of joblib). Also see https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules/27733727#27733727

