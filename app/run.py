#import libraries
import sys
import json
import plotly
import pandas as pd
import plotly
import plotly.graph_objs as go
import json, plotly
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as goes


app = Flask(__name__)
#give name to your database
database = "DisasterResponse"

#method to separate text into words
def tokenize(text):
    '''
    input:
    text - string that needs to be tokenized
    outout:
    tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

engine = create_engine('sqlite:///../data/' + database + '.db')
df = pd.read_sql_table(database, engine)

# load model
model = joblib.load("../model/classifier.pkl")

def return_figures():
    """Creates four plotly visualizations
    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """
    engine = create_engine('sqlite:///../data/' + database + '.db')
    df = pd.read_sql_table(database, engine)
    
    graph_one = []

    df_new = df.iloc[:, 3:]
    category_names = df_new.columns
    df_new = df_new.mean()
    df_new.sort_values(ascending=True)

    graph_one.append(
        goes.Bar(
            x = category_names.tolist(),
            y = df_new.tolist(),
        )
    )

    layout_one = dict(title = 'Disaster Responses Popularity',
            xaxis = dict(title = 'Disaster Categories',),
            yaxis = dict(title = 'Ammount'),
            )

    # third chart plots percent of messages by types
    graph_two = []

    category_type_cols =['Natural Disasters','Supplies','People and Security','Infraestructure','Health and Aid']
    category_type = []

    nd_df = df.iloc[:, 3:]
    #natural disasters
    nd_df.drop(nd_df.columns.difference(['floods', 'storm', 'fire', 'earthquake']), 1, inplace = True)
    category_type.append(sum(nd_df.mean()))
 
    #supplies
    sdf = df.iloc[:, 3:]
    sdf.drop(sdf.columns.difference(['request','tools','transport','food','shelter','clothing' ,'money','water','medical_products']), 1, inplace=True)
    category_type.append(sum(sdf.mean()))

    #people and security
    psdf = df.iloc[:, 3:]
    psdf.drop(psdf.columns.difference(['offer ','search_and_rescue' ,'security' ,'military' ,'child_alone', 'missing_people' ,'refugees']), 1, inplace=True)
    category_type.append(sum(psdf.mean()))

    #infraestructure
    idf = df.iloc[:, 3:]
    idf.drop(idf.columns.difference(['infrastructure_related' ,'buildings' ,'electricity' ,'shops', 'other_infrastructure']), 1, inplace=True)
    category_type.append(sum(idf.mean()))

    #health and aid
    hdf = df.iloc[:, 3:]
    hdf.drop(hdf.columns.difference(['medical_help','death','aid_related','hospitals','aid_centers' ,'other_aid','direct_report'  ]), 1, inplace=True)
    category_type.append(sum(hdf.mean()))

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_two.append(
        goes.Bar(
            x = category_type_cols,
            y = category_type,
        )
    )
    layout_two = dict(title = 'Percentage of Disaster Messages by Types',
        xaxis = dict(title = 'Categories',),
        yaxis = dict(title = 'Count'),
        )

    graph_three = []

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_three.append(
        goes.Bar(
            x = list(genre_counts.index),
            y = df.groupby('genre').count()['message'],
        )
    )
    
    layout_three = dict(title = 'Distribution of Message Genres',
        xaxis = dict(title = 'Genre',),
        yaxis = dict(title = 'Count'),
        )
       
    # append all charts to the graphs list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    # figures.append(dict(data=graph_four, layout=layout_four))
    # figures.append(dict(data=graph_five, layout=layout_five))

    return figures

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    graphs = return_figures()
    
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