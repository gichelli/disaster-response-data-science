# disaster-response-data-science

### Table of Contents

- [Description](#Installation)
- [FilesDescription](#FilesDescription)
- [Installation](#Installation)
- [Application](#Application)
- [Acknowledgements](#Acknowledgements)


## Description <a name="Description"></a>
In this project we are analyzing data provided by [Figure Eight ](https://appen.com/). Thousands of messages that have been sent during natural disasters to either social media or disaster organizations. Our intention is to build a model for an application that classifies the given disaster messages.

We created a machine learning pipeline to categorize the disaster messages. When a person input a message a visual classification of the message gets generated.

## FilesDescription <a name="FilesDescription"></a>
Folders:
-> app:
-> templates: contains html files
run.py: file that runs the application in flask

-> data: contains the two data files that will be used to create the databse and the pythn program that executes the creation of the databse.
-> disaster_messages.csv
-> disaster-categories.csv
-> process_data.py

-> models: python file that executes the creation of the model
-> two notebooks to prepare, analyze and visualize the data.
-> ETL Pipeline Preparation.ipynb
-> ML Pipeline Preparation.ipynb


## Installation <a name="Installation"></a>
Libraries needed:

Anaconda Distribution of Python
nltk
pandas
plotly
flask
sklearn
sqlalchemy
json
pickle


## Application <a name="Results"></a>
The main findings of the code can be found at the post available [here](https://gichellivento.medium.com/i-used-a-simple-data-file-to-get-my-boston-apartment-in-airbnb-943669d49e78).

 
## Acknowledgements <a name="Acknowledgements"></a>
Our data is provided by [Figure Eight ](https://appen.com/). 