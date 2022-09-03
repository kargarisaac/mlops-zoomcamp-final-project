# ref: https://www.kaggle.com/code/granjithkumar/nlp-with-women-clothing-reviews/data

import numpy as np
import pandas as pd
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import mlflow
import pickle
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.artifacts import DataArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.config.docker_configuration import DockerConfiguration
from zenml.steps import step, ResourceConfiguration
from typing import Type

nltk.download('stopwords')
DATA_PATH = "data/Womens Clothing E-Commerce Reviews.csv"

# docker config
docker_config = DockerConfiguration(
    requirements=[
        "nltk",
        "tensorflow==2.9.1",
        "scikit-learn"
    ],
    dockerignore="./.dockerignore"
    )

# materializer for TF tokenizer (custom) inputs and outputs
class TokenizerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Tokenizer,)
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact,)

    def handle_input(self, data_type: Type[Tokenizer]) -> Tokenizer:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.pickle'), 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def handle_return(self, tokenizer: Tokenizer) -> None:
        """Write to artifact store"""
        super().handle_return(tokenizer)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


## data loading
@step
def read_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, index_col =[0])
    print("Data loaded.\n\n")
    return data

## preprocess text
@step(resource_configuration=ResourceConfiguration(cpu_count=16, memory="16GB"))
def preprocess_data(
    data: pd.DataFrame,
    ) -> Output(corpus=np.ndarray, y=np.ndarray):
    data = data[~data['Review Text'].isnull()]  #Dropping columns which don't have any review
    X = data[['Review Text']]
    X.index = np.arange(len(X))

    y = data['Recommended IND'].values

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

    return np.array(corpus), y

# tokenization and dataset creation
@step(resource_configuration=ResourceConfiguration(cpu_count=8, memory="16GB"))
def create_dataset(
    corpus: np.ndarray, 
    y: np.ndarray
    ) -> Output(X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, tokenizer=Tokenizer):

    tokenizer = Tokenizer(num_words = 3000)
    tokenizer.fit_on_texts(corpus)

    sequences = tokenizer.texts_to_sequences(corpus)
    padded = pad_sequences(sequences, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size = 0.2, random_state = 42)

    print("Dataset created.\n\n")
    return X_train, X_test, y_train, y_test, tokenizer

@enable_mlflow
@step(resource_configuration=ResourceConfiguration(cpu_count=8, memory="16GB"), enable_cache=False)
def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    tokenizer: Tokenizer
    ) -> None:

    # mlflow.set_tracking_uri("http://34.89.202.47")
    # mlflow.set_experiment("customer-sentiment-analysis")
    
    mlflow.tensorflow.autolog()
    
    embedding_dim = 32
    batch_size = 64

    # model definition
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(3000, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ## training
    num_epochs = 2
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
    mlflow.keras.log_model(model, 'models/model_dl')

    with open('tf_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    mlflow.log_artifact(local_path="tf_tokenizer.pickle", artifact_path="tokenizer_pickle")

    print("Model training completed.\n\n")

    # Evaluate the model on the test data using `evaluate`
    # print("Evaluate on test data")
    # results = model.evaluate(X_test, y_test, batch_size=128)
    # print("test loss, test acc:", results)
    # mlflow.log_metric("loss", results[0])
    # mlflow.log_metric("accuracy", results[1])


@pipeline(docker_configuration=docker_config)
def training_pipeline(
    reading_data,
    preprocessing_data,
    creating_dataset,
    training_model,
):
    data = reading_data()
    corpus, y = preprocessing_data(data)
    X_train, X_test, y_train, y_test, tokenizer = creating_dataset(corpus, y)
    training_model(X_train, y_train, X_test, y_test, tokenizer)

    # model, tokenizer = get_best_model(model_name, client)
    # test_model(model, X_test, tokenizer)

if __name__ == '__main__': 

    training_pipeline(
        reading_data=read_data(),
        preprocessing_data=preprocess_data(),
        creating_dataset=create_dataset().with_return_materializers({"tokenizer": TokenizerMaterializer}),
        training_model=train_model(),
    ).run()
