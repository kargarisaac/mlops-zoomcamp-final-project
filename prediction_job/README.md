# Service Account for GCS
You need to create a service account and give it storage access admin role. Then download the json file and put it in the same directory as the script with the name of `key.json`.


# Test Locally

- build the image
```bash
docker build -t sentiment:v1 .
```

- run the econtainer
```bash
docker run -it --rm -p 8080:8080 sentiment:v1
```

- Then visit http://localhost:8080/docs to view your fastapi app.

- test sentence: 
Like it, but don't love it.


# Build and deploy

- Set project ID variable:
```bash
export PROJECT_ID=$(gcloud config get-value core/project)
```

- build the image:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/sentiment
```

- deploy on cloud run:
```bash
gcloud run deploy sentiment --image gcr.io/$PROJECT_ID/sentiment --platform managed --allow-unauthenticated
```

