import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy

def load_data(messages_filepath, categories_filepath):

  '''
  This function read the file paths for two csv files and load them
  into two pandas dataframe, and return merged them to dataframe

  Args (str): messages_filepath,categories_filepath


  return (dataframe): df

  '''


  # Read csv files  
  messages = pd.read_csv(str(messages_filepath))
  categories =pd.read_csv(str(categories_filepath))

  # Merge datasets
  df = pd.merge(categories,messages,on='id',how='outer')

  return df


def clean_data(df):

  '''
  This function get a dataframe, create a dataframe by splitting
  "categories" column string content, and only keeping the numeric part,
  and merge thisdataframe with original dataframe, and remove dupicates

  Args (dataframe): df


  return (dataframe): df

  ''' 

  # Create dataframe from categories column

  categories = df.categories.str.split(';',expand=True)
  column_ = [col_.split('-')[0].strip() for col_ in list(categories.iloc[0])]
  categories.columns = column_

  # Convert category values to numeric

  for column in categories.columns:

    categories[column] = categories[column].astype(str)
    categories[column] = categories[column].apply(lambda x:int(x.split('-')[1].strip()))

  # Replace categories column in df with new category columns

  df.drop(columns=['categories'],inplace=True)

  df = pd.concat([df,categories],axis=1)

  # Remove duplicates

  df = df.drop_duplicates()

  return df




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