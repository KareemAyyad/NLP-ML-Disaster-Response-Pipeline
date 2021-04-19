import sys
# import libraries
import pandas as pd
import numpy as np
import pickle

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = stopwords.words("english")

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import re

def load_data(path_db='Database.db', table='Table'):
    """
    Function to load dataset from SQL DB.
    
        Args: 
            path_db: Database Path
            table: Name of Table in SQL DB to load
        
        Returns: 
            Numpy Arrays: Split Dataset in the form of Data (X) and target (y), in that order.
    """
    # load data from database
    engine = create_engine('sqlite:///Database.db')
    df = pd.read_sql_table('Table', engine)
    df.drop(columns='original', inplace=True)
    
    # There are columns with missing values. We have to drop these
    # to maintain quality of data.
    # And also to be able to pass the data to the classifier as it doesn't seem to accept NaN in target labels
    df = df.dropna(axis=0, how='any')
    df.isna().sum().sum()
    print("Number of Null values in Dataset after dropping: ", df.isna().sum().sum()) 
    
    X = df['message']
    y = df.iloc[:,4:]
    
    
    return X, y





def tokenize(text):
    """
    Function to tokenize words in given text.
    
        Args: 
            Text: The text to tokenize.
        
        Returns: 
            List: List of Strings, each of which is a tokenized version of text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Function to build model using Scikit-Learn.
    
        Args: 
            None.
        
        Returns: 
            Pipeline: Scikit-learn Pipeline that can be called to fit on data, and then used to predict.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   
    ])
    #parameters = {
    #'vect__max_features': [None],
    #'tfidf__use_idf': [True],
    #'clf__estimator__min_samples_leaf': [1],
    #
    
    #cv = GridSearchCV(pipeline, param_grid=parameters)

    #return cv

    return pipeline

def eval_model(true_label, pred_label, target_names):
    """Function to iterate through columns of dataset, calling sklearn's classification scoring functions
    to evaluate the performance of the model. Multi-Class Classification.
    
    
    Args:
        true_label: The True Label; array of all true labels.
        pred_label: The Predicted Label; array of all predicted labels. Must match shape of true_label.
        columns: Columns in the dataset where each column is a class.
    
    Returns: 
        df_metrics: A table that displays the resulting metrics; Accuracy, Precision, Recall, F1, and Support.
    """
    
    results = []
    
    #Evaluate
    for i in range(len(target_names)):
        accuracy = accuracy_score(true_label[:, i], pred_label[:, i])
        precision = precision_score(true_label[:, i], pred_label[:, i])
        recall = recall_score(true_label[:, i], pred_label[:, i])
        f1 = f1_score(true_label[:, i], pred_label[:, i])
        
        results.append([accuracy, precision, recall, f1])
    
    #Store
    results = np.array(results)
    df_metrics = pd.DataFrame(data = results, index = target_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return df_metrics


def save_model(model):
    """
    Function to save the model as pickle file.
    
        Args: 
            model: cv version of the model.
        
        Returns: 
            None. Saves model as .pkl file in working directory.
    """
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y  = load_data(database_filepath)
        target_names = np.array(y.columns.values)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=40)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        df_metrics = eval_model(np.array(y_test), y_pred, target_names)
        print(df_metrics)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()