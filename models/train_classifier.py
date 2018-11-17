import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlalchemy
import time
import matplotlib.pyplot as plt

import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
from sklearn.metrics import make_scorer, classification_report, precision_recall_fscore_support
from sklearn.externals import joblib


def load_data(database_filepath):

    sql_engine = create_engine('sqlite:///'+str(database_filepath), echo=False)
    #had to have this line otherwise froze
    connection = sql_engine.raw_connection() 

    table_name = str(sql_engine.table_names()[0])
    print(table_name)

    df = pd.read_sql("SELECT * FROM '{}'".format(table_name),con=connection)
    category_names = list(df.columns[4:])
    #category_names = list(set(df.columns)-set(df[['id','message','original','genre']]))
    #Remove rows when all categories not labled, or unknown '2' value
    df = df[(df.related!=2) & (df[category_names].sum(axis=1)!=0)]


    X = df[['message']]  
    Y = df.drop(columns=['id','message','original','genre'])

    return X, Y, category_names


    
def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    emails_regex = '[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+'
    ips_regex = '(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})'
    stopword_list = stopwords.words('english')
    placeholder_list = ['urlplaceholder','emailplaceholder','ipplaceholder']
    
    # get list of all urls using regex
    # Remove extra paranthesis for better URL detection
    text = text.replace("(","")
    text = text.replace(")","")  

    detected_urls = re.findall(url_regex,text) 
    detected_emails = re.findall(emails_regex,text)
    detected_emails = [email.split()[0] for email in detected_emails]
    detected_ips = re.findall(ips_regex,text)
            
    # Remove numbers and special characters
    pattern = re.compile(r'[^a-zA-Z]') 
    stopword_list = stopwords.words('english')
    

    for url in detected_urls:
        text = re.sub(url,'urlplaceholder',text)   
                 
    for email in detected_emails:
        text = re.sub(email,'emailplaceholder',text)
            
    for ip in detected_ips:
        text = re.sub(ip,'ipplaceholder',text)       
      

    for stop_word in stopword_list:
        
        if(stop_word in text):
             text.replace(stop_word,'')
    
    text = re.sub(pattern,' ',text)
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        if((tok not in stopword_list) and (tok not in placeholder_list) and len(tok)>2):      

            clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok.strip()),pos='v')
            # Remove Stemmer for better word recognition in app
            #clean_tok = PorterStemmer().stem(clean_tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
        ('clf_ada', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier())))        
        ])
# {'clf_ada__estimator__base_estimator__max_depth': 1,
#  'clf_ada__estimator__n_estimators': 50,
#  'tfidf__use_idf': False,
#  'vect__max_features': 10000}

    parameters = {
        'vect__max_features': [10000],       #(None, 10000),
        'tfidf__use_idf': [False],          #(True, False),
        'clf_ada__estimator__n_estimators': [50],     #[50, 100, 200],
        'clf_ada__estimator__base_estimator__max_depth': [1]   #[1, 2]
    }

    scorer = make_scorer(f1_score,average='micro')
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3, verbose=2)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):


    y_pred_test=model.predict(X_test.message.values)

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

    best_pipeline = model.best_estimator_

    col_name = []
    imp_value = []
    imp_word = []

    # List vocabulary
    x_name = best_pipeline.named_steps['vect'].get_feature_names()

    for j, col in enumerate(category_names):

        x_imp = best_pipeline.named_steps['clf_ada'].estimators_[j].feature_importances_
        value_max = x_imp.max()/2.0

        for i,value in enumerate(x_imp):
            if(value>value_max):
                #print(col,'{0:.3f}'.format(value), x_name[i])
                col_name.append(col)
                imp_value.append(value)
                imp_word.append(x_name[i])


    col_name = np.array(col_name).reshape(-1,1)
    imp_value = np.array(imp_value).reshape(-1,1)
    imp_word = np.array(imp_word).reshape(-1,1)


    imp_array = np.concatenate((col_name, imp_value, imp_word), axis=1)

    df_imp = pd.DataFrame(imp_array,columns=['category_name','importance_value','important_word'])   

    # Create engine
    sql_engine = create_engine('sqlite:///'+str(database_filepath), echo=False)

    # Use this line to avoid freezing while process
    connection = sql_engine.raw_connection()  

    # Save dataframe to 'data' table
    df_imp.to_sql('word', connection, index=False, if_exists='replace')  

    df_imp = pd.read_sql("SELECT * FROM '{}'".format('word'),con=connection)

    print(df_imp.head())




def save_model(model, model_filepath):
    joblib.dump(model, str(model_filepath))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('\n')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('\n')
        print('Building model...')
        model = build_model()
        
        print('\n')
        print('Training model...')
        model.fit(X_train.message.values, Y_train)
        
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