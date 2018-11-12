import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy

def load_data(messages_filepath, categories_filepath):

  '''
  This function read the file paths for two csv files and load them
  into two pandas dataframe

  Args (str): messages_filepath,categories_filepath


  return: None

  '''
    
    messages = pd.read_csv(str(messages_filepath))
    categories =pd.read_csv(str(categories_filepath))


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()