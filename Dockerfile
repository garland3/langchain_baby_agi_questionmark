FROM python:3.10-slim

RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y git
RUN  apt-get install -y default-libmysqlclient-dev build-essential

# copy requirements.txt to the working directory
COPY requirements.txt .
RUN python -m pip -r requirements.txt


RUN  mkdir /root/.llminterface
COPY .secrets.toml /root/.llminterface/
