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


# Build and deploy

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

