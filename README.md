# disaster-response-data-science

### Table of Contents

- [Summary](#Summary)
- [FilesDescription](#FilesDescription)
- [Installation](#Installation)
- [Application](#Application)
- [Acknowledgements](#Acknowledgements)


## Summary <a name="Summary"></a>
In this project we are analyzing data provided by [Figure Eight ](https://appen.com/). Thousands of messages that have been sent during natural disasters to either social media or disaster organizations. Our intention is to build a model for an application that classifies the given disaster messages.

We created a machine learning pipeline to categorize the disaster messages. When a person input a message a visual classification of the message gets generated.

## FilesDescription <a name="FilesDescription"></a>

- app:  
&nbsp;&nbsp;&nbsp; templates: contains html files  
&nbsp;&nbsp;&nbsp; run.py: file that runs the application in flask  

- data: contains the two data files that will be used to create the databse and the pythn program that executes the creation of the databse.  
&nbsp;&nbsp;&nbsp; disaster_messages.csv  
&nbsp;&nbsp;&nbsp; disaster-categories.csv  
&nbsp;&nbsp;&nbsp; process_data.py  

- models: python file that executes the creation of the model  
&nbsp;&nbsp;&nbsp; two notebooks to prepare, analyze and visualize the data.  
&nbsp;&nbsp;&nbsp; ETL Pipeline Preparation.ipynb  
&nbsp;&nbsp;&nbsp; ML Pipeline Preparation.ipynb  


## Installation <a name="Installation"></a>
Make sure you have the libraries needed.  

Anaconda Distribution of Python
nltk
pandas
plotly
flask
sklearn
sqlalchemy
json
pickle

1. create database  
(in terminal)  
virtualenv app  
cd app  
source bin/activate  
pip install Flask  
cd data  
python process_data.py (this will create the database and save it in this folder)  

2. create model  
cd ..  
cd model  
run train_classifier.py (this will create the model)  

3. run application  
cd ..  
cd app  
python run.py  

in the url type: http://localhost:8000/  


## Application <a name="Results"></a>
The main findings of the code can be found at the post available [here](https://gichellivento.medium.com/i-used-a-simple-data-file-to-get-my-boston-apartment-in-airbnb-943669d49e78).

 
## Acknowledgements <a name="Acknowledgements"></a>
Our data is provided by [Figure Eight ](https://appen.com/). 