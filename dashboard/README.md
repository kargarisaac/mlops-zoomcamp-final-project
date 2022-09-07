# Test Locally

- build the image
```bash
docker build -t sentiment_ui:v1 .
```

- run the econtainer
```bash
docker run -it --rm -p 8085:8085 sentiment_ui:v1
```

- Then visit http://localhost:8085 to view your fastapi app.

- test sentence: 
Like it, but don't love it.


# Build and deploy on Cloud Run

- Set project ID variable:
```bash
export PROJECT_ID=$(gcloud config get-value core/project)
```

- build the image:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/sentiment_ui
```

- deploy on cloud run:
```bash
gcloud run deploy sentimentui --image gcr.io/$PROJECT_ID/sentiment_ui --platform managed --allow-unauthenticated
```

# Import model to Vertex AI and create an endpoint

- import model to Vertex AI model registry:
```bash
gcloud ai models upload \
  --region=europe-west1 \
  --display-name=sentiment2 \
  --container-image-uri=gcr.io/$PROJECT_ID/sentiment_ui
```

- create an endpoint:
```bash
gcloud ai endpoints create \
  --region=europe-west1 \
  --display-name=sentiment2
```

- get the ID for model and endpoint:

```bash
gcloud ai endpoints list --region=europe-west1
gcloud ai models list --region=europe-west1
```

- deploy the model to the endpoint using the above IDs:
```bash
gcloud ai endpoints deploy-model <ENDPOINT_ID> \
  --region=europe-west1 \
  --model=<MODEL_ID> \
  --display-name=sentiment2


gcloud ai endpoints deploy-model 3509588339302858752 \
  --region=europe-west1 \
  --model=4628381002983538688 \
  --display-name=sentiment2
```

