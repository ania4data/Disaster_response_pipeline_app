import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    '''loads data into dataframe for analysis

    Read the file paths for two csv files and load them
    into two pandas dataframe, and return merged them to dataframe

    Args:
      messages_filepath (str): name of csv datafile
      categories_filepath (str): name of csv datafile

    Returns:
      df (dataframe): resulting datframe

    '''

    # Read csv files
    messages = pd.read_csv(str(messages_filepath))
    categories = pd.read_csv(str(categories_filepath))

    # Merge datasets
    df = pd.merge(categories, messages, on='id', how='outer')

    return df


def clean_data(df):
    ''' Clean dataframe

    CreateS a dataframe by splitting "categories" column string content,
    and only keeping the numeric part,
    and merge this dataframe with original dataframe, and remove dupicates

    Args:
      df (dataframe): name of dataframe

    Returns:
      df (dataframe): name of dataframe

    '''

    # Create dataframe from categories column
    categories = df.categories.str.split(';', expand=True)
    column_ = [col_.split('-')[0].strip() for col_ in list(categories.iloc[0])]
    categories.columns = column_

    # Convert category values to numeric
    for column in categories.columns:

        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].apply(
            lambda x: int(x.split('-')[1].strip()))

    # Replace categories column in df with new category columns
    df.drop(columns=['categories'], inplace=True)

    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    ''' Function for saveing data in SQL database

    This function read the cleaned dataframe and load that into SQLite database
    either using sqlachemy or sqlite3

    Args:
      df (dataframe): name of dataframe

    Returns:
      None

    '''

    # SQLAlchemy
    def save_with_alchemy(df, database_filename):

        # Create engine
        sql_engine = create_engine(
            'sqlite:///' + str(database_filename), echo=False)

        # Use this line to avoid freezing while process
        connection = sql_engine.raw_connection()

        # Save dataframe to 'data' table
        df.to_sql('data', connection, index=False, if_exists='replace')

    # SQLite3
    def save_with_sqlite3(df, database_filename):

        # Create connection
        conn = sqlite3.connect(str(database_filename))

        # Save dataframe to 'data' table
        df.to_sql('data', con=conn, index=False, if_exists='replace')

    # Use alchemy
    save_with_alchemy(df, database_filename)


def test(database_filename):
    ''' Testing if output from SQL as expected

    This function tests if database was saved correctly by loading it into 
    pandas dataframe

    Args:
      database_filename (string): name of database

    Returns:
      None

    '''

    sql_engine = create_engine(
        'sqlite:///' + str(database_filename), echo=False)
    connection = sql_engine.raw_connection()
    dg = pd.read_sql("SELECT * FROM data", con=connection)
    print(' ')
    print(dg.head())


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

        print('Testing database...\n    DATABASE: {}'.format(database_filepath))
        test(database_filepath)

        print('Read from database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
