from fastapi import FastAPI
import os
from utils import preprocess
# nltk.download('stopwords', download_dir='.')
# export NLTK_DATA=/my/path/nltk_data
os.environ["NLTK_DATA"] = "./corpora"

from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle


# fastapi app
app = FastAPI(
    title="Fashion Sentment Analysis REST API",
    description="This API analyses the review for a fashion product.",
    version="0.0.1",
    contact={
        "name": "Isaac Kargar",
        "email": "kargarisaac@yahoo.com",
    },
)


@app.get("/")
async def root():
    return {"message": "Customer review sentiment analysis"}


# @app.get("/predict/{review}/{method}")
@app.get("/predict")
async def get_predict(review: str, method: str):
    """
    Reads the list of sensors from the database
    """
    try:
        review_processed = preprocess(review)

        if len(review_processed) == 0:
            return {"message": "Please enter a valid review - It seems there is no valubale review in the text."}
            
        if method == 'tfidf':
            with open('models/model_tfidf.pickle', 'rb') as f:
                tv, clf = pickle.load(f)
            tfidf_vector = tv.transform([review_processed])
            return {
                "prediction": str(clf.predict(tfidf_vector)[0])
                }
        elif method == 'bow':
            with open('models/model_bow.pickle', 'rb') as f:
                cv, clf = pickle.load(f)
            bow_vector = cv.transform([review_processed])
            return {
                "prediction": str(clf.predict(bow_vector)[0])
                }
        elif method == "dl":
            ## load model and tokenizer
            model = tf.keras.models.load_model('models/model.h5')
            with open('models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            review_processed = tokenizer.texts_to_sequences(review_processed)
            review_processed = pad_sequences(review_processed, padding='post').T
            prediction = model.predict(review_processed)
            return {
                "prediction": str(prediction[0][0])
                }
        else:
            raise ValueError("method must be 'tfidf', 'bow' or 'dl'")
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
