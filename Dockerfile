# Description: Dockerfile for the metascore API
FROM python:3.10.4-buster

LABEL version="1.0"
LABEL description="Dockerfile for the Gym API"
LABEL maintainer="Omar Gonzales"

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY coach coach
COPY setup.py setup.py
COPY .env.yaml .env.yaml

RUN pip install .

CMD uvicorn coach.api.fast:app --host 0.0.0.0 --port $PORT
