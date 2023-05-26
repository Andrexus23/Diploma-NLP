FROM python:3.9.13
WORKDIR /src
COPY ../../requirements.txt /src
RUN pip install --upgrade pip -r requirements.txt
RUN python -m nltk.downloader stopwords
COPY . /src
EXPOSE 5000