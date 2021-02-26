#import libraries
import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd
from flask import Flask
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import flask
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import nltk
# nltk.download(['punkt', 'wordnet'])

# import statements
# import pandas as pd


# class StartingVerbExtractor(BaseEstimator, TransformerMixin):

#     def starting_verb(self, text):
#         sentence_list = nltk.sent_tokenize(text)
#         for sentence in sentence_list:
#             pos_tags = nltk.pos_tag(tokenize(sentence))
#             first_word, first_tag = pos_tags[0]
#             if first_tag in ['VB', 'VBP'] or first_word == 'RT':
#                 return True
#         return False

#     def fit(self, x, y=None):
#         return self

#     def transform(self, X):
#         X_tagged = pd.Series(X).apply(self.starting_verb)
#         return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    print(database_filepath)
    df = pd.read_sql_table('disaster-responses', engine)
    X = df['message']
    Y = df.iloc[:, 3:]
    print(Y.head(2))
    # print(" ")
    # print(y.head())
    print(" ")
   
    print(df.head(2))
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # print(text)
    #still needs to do sometohin more here? check
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    ])),
    ('clf', RandomForestClassifier())])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    # X_test_counts = vect.transform(X_test)
    # X_test_tfidf = tfidf.transform(X_test_counts)
    # y_pred = clf.predict(X_test_tfidf)




    # predict on test data
    Y_pred = model.predict(X_test)
    print(Y_pred)


    # y_test_preds = lm_model.predict(X_test)
    # y_train_preds = lm_model.predict(X_train)
    # print(y_train_preds)

    # #append the r2 value from the test set
    
    # r2_scores_test.append(r2_score(y_test, y_test_preds))
    # r2_scores_train.append(r2_score(y_train, y_train_preds))
    # results[str(cutoff)] = r2_score(y_test, y_test_preds)


def save_model(model, model_filepath):
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # print(X[5])
        print("------------------------------------------")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(Y_train)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        # train classifier
        # X_train_counts = vect.fit_transform(X_train)
        # X_train_tfidf = tfidf.fit_transform(X_train_counts)
        # clf.fit(X_train_tfidf, y_train)


        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        # print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()