#import libraries
import sys
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

# from sklearn.model_selection import train_test_split
# from sqlalchemy import create_engine
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.ensemble import RandomForestClassifier
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import nltk
# import json
# import plotly
# import pandas as pd

# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

# from flask import Flask
# from flask import render_template, request, jsonify
# from plotly.graph_objs import Bar
# # from sklearn.externals import joblib
# # import sklearn.external.joblib as extjoblib
# import joblib
# from sqlalchemy import create_engine
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
# import pickle
# import re
# import numpy as np
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import flask
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/disaster-responses.db')
df = pd.read_sql_table('disaster-responses', engine)

print("pass1")
# load model
model = joblib.load("../model/classifier.pkl")

print("pass2")


# def getnames(query):
#     print(query)
#     # use model to predict classification for query
#     classification_labels = model.predict([query])[0]
#     # classification_results = dict(zip(df.columns[4:], classification_labels))
#     print(classification_labels)




print("pass1")
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)




# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    # getnames(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        
        classification_result=classification_results
        
        
    )


def main():
    app.run(port=8000, debug=True)


if __name__ == '__main__':
    main()