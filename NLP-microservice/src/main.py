import logging
import time
import traceback
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import numpy as np
from flask_restful import Api, Resource, reqparse
import json
from flasgger.utils import swag_from
from urllib.parse import unquote
from flask_swagger_ui import get_swaggerui_blueprint
import nlp.ModelResearcher as nlp

SWAGGER_URL = '/api/docs'  # URL для размещения SWAGGER_UI
API_URL = '/static/swagger.json'
TRAIN_PATH = './posted/train.json'
PREPROCESSED_PATH = './nlp/data/preprocessed_documents.json'
MODELS_PATH = "nlp/models/"
ALLOWED_MODELS = ["w2v", "fast_text"]

app = Flask(__name__)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
)

app.register_blueprint(swaggerui_blueprint)

api = Api(app)


@app.route("/api/docs/train/uploadDataset", methods=['POST'])
def upload_train_data():
    if len(request.files):
        dataset = request.files.get("file")
        dataset.save(TRAIN_PATH)
        dataset.close()
        return {"Success": "File's been successfully uploaded"}

    return {"Error": "Couldn't load file"}


@app.route("/api/docs/train/<string:name>", methods=['GET'])
def train_model(name):
    res = None
    modelResearcher = nlp.ModelResearcher()
    preprocessed_texts = None
    try:
        preprocessed_texts = nlp.read_json(PREPROCESSED_PATH)
    except FileNotFoundError:
        start = time.perf_counter()
        texts = nlp.read_json(TRAIN_PATH)
        preprocessed_texts = modelResearcher.preprocess_and_save(texts, PREPROCESSED_PATH)
        end = time.perf_counter()
        res = f'Preprocessing time: {end - start:0.4f} secs'
        print(res)

    if name not in ALLOWED_MODELS:
        return jsonify({"Error": "Incorrect model name"})
    try:
        start = time.perf_counter()
        modelResearcher.train(preprocessed_texts, model=name, model_path=MODELS_PATH)
        end = time.perf_counter()
        res = f'Model training time: {end - start:0.4f} secs'
        return jsonify({"Success": res})
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Cannot train model"})


@app.route("/api/docs/match2texts/<string:name>", methods=['POST'])
def match2texts(name):
    if name not in ALLOWED_MODELS:
        print(name, ALLOWED_MODELS)
        return jsonify({"Error": "No such model in service"})
    modelResearcher = nlp.ModelResearcher()
    try:
        values = request.values.values()
        first = next(values)
        second = next(values)
        exists = modelResearcher.load(MODELS_PATH + name)
        if not exists:
            return {"Error": "No model: you should train or download it"}
        sim = modelResearcher.predict_sim_two_texts(first, second)
        print(sim)
        return jsonify({"Texts' similarity": str(sim)})
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Something went wrong"})



if __name__ == '__main__':
    app.run(port=5000, debug=True)
