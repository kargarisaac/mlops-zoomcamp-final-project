# Build and deploy

- Set project ID variable:
```bash
export PROJECT_ID=$(gcloud config get-value core/project)
```

- Configure gcloud for your chosen region:

```bash
gcloud config set run/region europe-west1
```

- build the image:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/streamlit-dashboard
```

- deploy on cloud run:
```bash
gcloud run deploy streamlit-dashboard --image gcr.io/$PROJECT_ID/streamlit-dashboard --platform managed --allow-unauthenticated
```