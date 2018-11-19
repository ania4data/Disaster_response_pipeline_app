import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import time

import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):

    sql_engine = create_engine('sqlite:///' + str(database_filepath), echo=False)
    #had to have this line otherwise froze
    connection = sql_engine.raw_connection() 
    table_name = str(sql_engine.table_names()[0])
    print('DB table names', sql_engine.table_names())

    df = pd.read_sql("SELECT * FROM '{}'".format(table_name), con=connection)
    category_names = list(df.columns[4:])
    #Remove rows when all categories not labled, or unknown '2' value
    df = df[(df.related!=2) & (df[category_names].sum(axis=1)!=0)]
    # if do df[['message']], later need get df.message
    X = df['message'] 
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    return X, Y, category_names

    
def tokenize(text):
    
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
        text = re.sub(url,'urlplaceholder', text)                     
    for email in detected_emails:
        text = re.sub(email,'emailplaceholder', text)           
    for ip in detected_ips:
        text = re.sub(ip,'ipplaceholder', text)             
    for stop_word in stopword_list:      
        if(stop_word in text):
             text.replace(stop_word,'')

    # remove everything except letetrs
    text = re.sub(pattern, ' ', text)
    # initilize
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if((tok not in stopword_list) and (tok not in placeholder_list) and len(tok) > 2):      
            clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok.strip()), pos='v')
            # Remove Stemmer for better word recognition in app 
            #clean_tok = PorterStemmer().stem(clean_tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
        # RF extremely slow, no benefit, overfitting train, xgboost not worked
        ('clf_ada', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier())))        
        ])

    parameters = {
        # None better overall, timing comparable
        'vect__max_features': [None, 10000],       
        # idf = False not giid features importance
        'tfidf__use_idf': [True],
        # 50 better overall          
        'clf_ada__estimator__n_estimators': [50, 100, 200],    
        # Use max_depth =2 -> more decimal in feature_importance
        'clf_ada__estimator__base_estimator__max_depth': [1, 2]   
    }

    # auc(micro) did not change results much over f1
    scorer = make_scorer(f1_score,average='micro')
    # job = -1 improve time 30%
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred_test=model.predict(X_test.values)
    for count, col in enumerate(category_names):
        tup1 = precision_recall_fscore_support(Y_test[col].values,y_pred_test[:,count],average='micro')
        tup2 = precision_recall_fscore_support(Y_test[col].values,y_pred_test[:,count],average='macro')
        tup3 = precision_recall_fscore_support(Y_test[col].values,y_pred_test[:,count],average='weighted')
        print('================================================')
        print('                ',col,'')
        print('------------------------------------------------')
        print()
        print('          %Precision     %Recall      %F1_score')
        print()
        print('Micro   ','   {0:.2f}          {0:.2f}          {0:.2f}'.format(tup1[0],tup1[1],tup1[2]))
        print('Macro   ','   {0:.2f}          {0:.2f}          {0:.2f}'.format(tup2[0],tup2[1],tup2[2]))
        print('Weighted','   {0:.2f}          {0:.2f}          {0:.2f}'.format(tup3[0],tup3[1],tup3[2]))
        print()

 
def get_feature_importance(model, category_names, database_filepath):

    # need .best_estimator to access pipeline dictionary
    best_pipeline = model.best_estimator_
    col_name = []
    imp_value = []
    imp_word = []
    # List vocabulary
    x_name = best_pipeline.named_steps['vect'].get_feature_names()
    for j, col in enumerate(category_names):
        x_imp = best_pipeline.named_steps['clf_ada'].estimators_[j].feature_importances_
        value_max = x_imp.max() / 2.0
        # only get features not lless than half max weight per column
        for i,value in enumerate(x_imp):
            if(value > value_max):
                col_name.append(col)
                imp_value.append(value)
                imp_word.append(x_name[i])

    # get columns of data
    col_name = np.array(col_name).reshape(-1, 1)
    imp_value = np.array(imp_value).reshape(-1, 1)
    imp_word = np.array(imp_word).reshape(-1, 1)
    imp_array = np.concatenate((col_name, imp_value, imp_word), axis=1)

    df_imp = pd.DataFrame(imp_array, columns=['category_name', 'importance_value', 'important_word'])  
    # need to get float after uniform str from np.concat 
    df_imp.importance_value = pd.to_numeric(df_imp.importance_value, downcast='float')

    # Create engine
    sql_engine = create_engine('sqlite:///' + str(database_filepath), echo=False)

    # Use this line to avoid freezing while process
    connection = sql_engine.raw_connection()  

    # Save dataframe to 'data' table
    df_imp.to_sql('word', connection, index=False, if_exists='replace')  
    df_imp = pd.read_sql("SELECT * FROM '{}'".format('word'), con=connection)
    print(df_imp)


def save_model(model, model_filepath):
    # save model, saving pipline itself or cv do the same
    joblib.dump(model, str(model_filepath))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('\n')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # fix test_size and random state for better feature imp words
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        print('\n')
        print('Building model...')
        model = build_model()
        
        print('\n')
        print('Training model...')
        model.fit(X_train.values, Y_train)
        print(model.best_params_)
        
        print('\n')
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('\n')
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('\n')
        print('Saving feature importance...')
        get_feature_importance(model, category_names, database_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()