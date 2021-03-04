import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import numpy as np

# function to load the data
def load_data(messages_filepath, categories_filepath):
    '''
    input:
    path to the message and categories files
    output:
    returns a merged dataframe
    '''
    # load messages dataset
    messages = pd.read_csv('../data/disaster_messages.csv', dtype=str)
    # load categories dataset
    categories = pd.read_csv('../data/disaster_categories.csv')
    # merge datasets
    df = pd.merge(messages, categories, left_index=True, right_index=True, how='outer')
    return df


def clean_data(df):
    '''
    input:
    dataframe
    output:
    clean dataframe
    '''
    df.drop('id_y', axis=1, inplace=True)
    df.rename(columns={'id_x': 'id'}, inplace=True)
    # split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[0]
    new_row = pd.Index(map(lambda x: str(x)[:-2], row))
    # rename the columns of `categories`
    categories.columns = new_row
    # Iterate through the category columns in df to keep only the last character of each string
    digits = categories.iloc[0]
    new_digits = pd.Index(map(lambda x: str(x)[-1:], digits))
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]       
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #Replace categories column in df with new category columns.
    df.drop(['categories'], axis=1, inplace=True)
    df.drop(['original'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    #remove duplicates
    df.duplicated().any()
    df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    input:
    merged and clean dataframe
    output:
    database
    '''
    tablename = database_filename
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename[:len(database_filename) - 3], engine, index=False)
    return df


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