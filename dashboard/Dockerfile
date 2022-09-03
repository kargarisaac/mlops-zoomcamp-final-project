FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv 

EXPOSE 8080

WORKDIR /app

COPY . .

RUN pipenv install --system --deploy

CMD streamlit run --server.port 8080 --server.enableCORS=false app.py
