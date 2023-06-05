FROM python:3.9.13
WORKDIR /service
COPY requirements.txt /service
RUN pip install --upgrade pip -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt -d /usr/local/nltk_data
COPY . /service
EXPOSE 5000