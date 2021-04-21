import sys
import os

import pandas as pd
from sqlalchemy import create_engine
def load_data(messages_filepath, categories_filepath):

    cwd = os.getcwd() # Get Current Working Directory.
    messages = pd.read_csv(os.path.join(cwd + '\\data\\disaster_messages.csv')) # Load Messages Dataset.
    categories = pd.read_csv(os.path.join(cwd + '\\data\\disaster_categories.csv')) # Load Categories Dataset.

    return messages, categories


def clean_data(messages, categories):
    df = pd.merge(messages, categories, on='id') # Merge Both datasets together.
    df.drop_duplicates(inplace=True) # Drop duplicate rows.
    categories = pd.Series(categories['categories']).str.split(';', expand=True) # Create DataFrame of the 36 individual category columns. They are currently all under one column.
    row = categories.iloc[0] # Select first row of the categories DataFrame. This will be used to retrieve a list of names of the categories.
    category_colnames = [x[:-2] for x in row] # Extracting a list of only the name of each category, without the value of it which is in the last two characters.
    categories.columns = category_colnames # Rename the columns of Categories.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # In line below, we drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe.
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Table', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database.')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()