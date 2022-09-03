from fastapi import FastAPI
import os
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow
import pickle
from google.cloud import storage
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
experiment_id = str(os.getenv("EXPERIMENT_ID"))
run_id = str(os.getenv("RUN_ID"))

path_on_gcs = experiment_id + "/" + run_id
stop_words = set(stopwords.words('english'))


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


def preprocess(review):
    review_processed = review.lower()
    review_processed = review_processed.split()
    ps = PorterStemmer()
    review_processed =[ps.stem(i) for i in review_processed if not i in set(stopwords.words('english'))]
    review_processed =' '.join(review_processed)
    return review_processed

def download_mlflow_artifacts():
    # create storage client
    storage_client = storage.Client.from_service_account_json('key.json')
    # storage_client = storage.Client()
    # get bucket with name
    bucket = storage_client.get_bucket('mlflow-demo-1360')
    # get bucket data as blob
    blobs = bucket.list_blobs(prefix=path_on_gcs) 

    for blob in blobs:
        blob_name = blob.name 
        file_path = "models/" + blob_name
        folder_path = os.path.dirname(file_path)
        if os.path.isdir(folder_path) == False:
            os.makedirs(folder_path)
        
        blob.download_to_filename(file_path)
    
    print("Artifacts downloaded.\n\n")

def load_model_tokenizer():
    artifact_folder = f"models/{path_on_gcs}/artifacts"
    model = mlflow.pyfunc.load_model(f"{artifact_folder}/models/model_dl")

    with open(f"{artifact_folder}/tokenizer_pickle/tf_tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Model and tokenizer loaded.\n\n")
    return model, tokenizer


@app.get("/")
async def root():
    return {"message": "Customer review sentiment analysis"}


@app.get("/predict")
async def get_predict(review: str,):
    """
    Reads the list of sensors from the database
    """
    try:
        review_processed = preprocess(review)

        if len(review_processed) == 0:
            return {"message": "Please enter a valid review - It seems there is no valubale review in the text."}
        
        download_mlflow_artifacts()
        model, tokenizer = load_model_tokenizer()
        review_processed = tokenizer.texts_to_sequences(review_processed)
        review_processed = pad_sequences(review_processed, padding='post').T
        prediction = model.predict(review_processed)
        return {
            "prediction": str(prediction[0][0])
            }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
