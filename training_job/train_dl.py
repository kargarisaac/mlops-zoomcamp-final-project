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

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
import pickle

tracking_uri = "sqlite:///mlflow.db"
model_name = "customer-sentiment-analysis"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(model_name)
client = MlflowClient(tracking_uri=tracking_uri)

## data loading
def read_data(filename='Womens Clothing E-Commerce Reviews.csv'):
    data = pd.read_csv(filename,index_col =[0])
    print("Data loaded.\n\n")
    return data

## preprocess text
def preprocess_data(data):
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

    print("Data preprocessed.\n\n")
    return corpus, y

## tokenization and dataset creation
def create_dataset(corpus, y, test_size=0.2, random_state=0):
    tokenizer = Tokenizer(num_words = 3000)
    tokenizer.fit_on_texts(corpus)

    sequences = tokenizer.texts_to_sequences(corpus)
    padded = pad_sequences(sequences, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size = 0.20, random_state = 0)

    print("Dataset created.\n\n")
    return X_train, X_test, y_train, y_test, tokenizer

# mlflow.tensorflow.autolog()
def model_training(X_train, y_train, X_test, y_test, tokenizer):
    for embedding_dim, batch_size in zip([32, 64, 128], [32, 64, 128]):
        with mlflow.start_run():
            ## model definition
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(3000, embedding_dim),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(6, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            ## training
            num_epochs = 50
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=2,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )

            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

            mlflow.set_tag("developer", "Isaac")
            mlflow.set_tag("algorithm", "Deep Learning")
            mlflow.log_param("train-data", "Womens Clothing E-Commerce Reviews")
            mlflow.log_param("embedding-dim", embedding_dim)

            print("Fit model on training data")
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                callbacks=callback,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(X_test, y_test),
            )

            ## save model and tokenizer
            # model.save('models/model_dl.h5')
            mlflow.keras.log_model(model, 'models/model_dl')

            with open('models/tf_tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            mlflow.log_artifact(local_path="models/tf_tokenizer.pickle", artifact_path="tokenizer_pickle")

            # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results = model.evaluate(X_test, y_test, batch_size=128)
            print("test loss, test acc:", results)
            mlflow.log_metric("loss", results[0])
            mlflow.log_metric("accuracy", results[1])

    print("Model training completed.\n\n")

def get_best_model():
    model = mlflow.keras.load_model(f"models:/{model_name}/production", dst_path=None)
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if dict(mv)['current_stage'] == 'Production':
            run_id = dict(mv)['run_id']

    artifact_folder = "models_pickle" #tokenizer_pickle
    client.download_artifacts(run_id=run_id, path=artifact_folder, dst_path='.')
    with open(f"{artifact_folder}/tf_tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("Model and tokenizer loaded.\n\n")
    return model, tokenizer


def test_model(model, X_test, tokenizer):
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(X_test[:3])
    print("predictions shape:", predictions.shape)

    sample_string = "I Will tell my friends for sure"
    sample = tokenizer.texts_to_sequences(sample_string)
    padded_sample = pad_sequences(sample, padding='post').T
    sample_predict = model.predict(padded_sample)
    print(f"model prediction for input: {sample_string} \n {sample_predict}")


if __name__ == '__main__':
    data = read_data()
    corpus, y = preprocess_data(data)
    X_train, X_test, y_train, y_test, tokenizer = create_dataset(corpus, y)
    model_training(X_train, y_train, X_test, y_test, tokenizer)
    model, tokenizer = get_best_model()
    test_model(model, X_test, tokenizer)