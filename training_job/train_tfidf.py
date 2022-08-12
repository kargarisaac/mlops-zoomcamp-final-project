# ref: https://www.kaggle.com/code/granjithkumar/nlp-with-women-clothing-reviews/data

import numpy as np
import pandas as pd
import os

import nltk
import re
# nltk.download('stopwords')
os.environ["NLTK_DATA"] = "./corpora"

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer as TV
import pickle
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer-sentiment-analysis")

## data loading
data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv',index_col =[0])

## preprocess text
data = data[~data['Review Text'].isnull()]  #Dropping columns which don't have any review
X = data[['Review Text']]
X.index = np.arange(len(X))

y = data['Recommended IND']

corpus =[]
for i in range(len(X)):
    review = re.sub('[^a-zA-z]',' ',X['Review Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review =[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)

tv  = TV(ngram_range =(1,1),max_features = 3000)
X_tv = tv.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tv, y, test_size = 0.20, random_state = 0)

mlflow.sklearn.autolog()

with mlflow.start_run():

    mlflow.set_tag("developer", "Isaac")
    mlflow.set_tag("algorithm", "MultinomialNB")
    mlflow.log_param("train-data", "Womens Clothing E-Commerce Reviews")

    alpha = 1
    mlflow.log_param("alpha", alpha)

    classifier = MultinomialNB(alpha = alpha)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    print("accuracy on test data:", acc)

    model_name = "model_tfidf.bin"
    with open("models/" + model_name, 'wb') as fout:
        pickle.dump((tv, classifier), fout)

    # mlflow.sklearn.log_model(classifier, artifact_path="models")
    mlflow.log_artifact(local_path="models/" + model_name, artifact_path="models_pickle")

