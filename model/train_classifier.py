import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    '''
    input:
    path to location of the database
    outout:
    x, features dataframe
    y, labels dataframe
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    tablename = database_filepath[8:-3]
    df = pd.read_sql_table(tablename, engine)
    X = df['message']
    Y = df.iloc[:, 3:]
    category_names = Y.columns
    return X, Y, category_names

# tokenization function to process text data
def tokenize(text):
    '''
    input:
    text - string that needs to be tokenized
    outout:
    a list of words
    '''
    #normalize the text
    text = text.lower()
    #Remove punctuation characters
    text = re.sub(r'(?:([A-Za-z0-9]+)/([A-Za-z0-9]+))|(/)', " ", text)
    text = re.sub(r'[?|$|.|!]', r'', text)
    # text = r'(?:([A-Za-z0-9]+)/([A-Za-z0-9]+))|(/)'
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [i for i in tokens if i not in stopwords.words("english")]
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    inputs: none
    output: the model
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    ])),
        ('clf', RandomForestClassifier())
    ])
    # parameters = { 
    # 'clf__estimator__n_estimators': [12]
    # # 'clf__estimator__max_features': ['auto']
    # }
    # cv = GridSearchCV(pipeline, param_grid = parameters)
    return pipeline
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input:
    model
    X_test, features of the test
    Y_test, labels of the test
    category_names, name of category columns
    output:
    prediction, an array of each of the rows predicted
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    
    return Y_pred

# function to test model
def test_model(Y_pred, Y_test, category_names):
    '''
    input: 
    Y_pred, prediction on the test data
    Y_test, test data
    category_names, category columns name
    the path where the model will be saved
    output: 
    prints precision and 
    '''
    for i, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    input: 
    the model created
    the path where the model will be saved
    output: 
    pickle file with the model created
    '''
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
              
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        Y_pred = evaluate_model(model, X_test, Y_test, category_names)

        print('Testing the model...')
        test_model(Y_pred, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()