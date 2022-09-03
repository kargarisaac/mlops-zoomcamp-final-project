FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv 

EXPOSE 8080
ENV EXPERIMENT_ID 2
ENV RUN_ID 7e6fc298380e484ba7cac0ab029ba510

WORKDIR /app

COPY . .

RUN pipenv install --system --deploy

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

