import logging
import os
import time
import traceback
from io import StringIO
from flask import Flask, jsonify, request, make_response, send_from_directory, send_file
from flasgger import Swagger
import pandas as pd
import numpy as np
from flask_restful import Api, Resource, reqparse
import json
from flasgger.utils import swag_from
from urllib.parse import unquote
from flask_swagger_ui import get_swaggerui_blueprint
from sentence_transformers import SentenceTransformer

import nlp.Common as Common
import nlp.ModelResearcher as MR


SWAGGER_URL = '/api/docs'  # URL для размещения SWAGGER_UI
API_URL = '/static/swagger.json'
TRAIN_PATH = './posted/train.json'
PREPROCESSED_PATH = './nlp/data/preprocessed_documents.json'
MODELS_GENSIM_PATH = "nlp/models/"
MODELS_TRANSFORMER_PATH = "nlp/models/sentence-transformers/"
ALLOWED_MODELS_GENSIM = ["w2v",
                         "fast_text"]

ALLOWED_MODELS_TRANSFORMER = [
    "rubert-base-cased",
    "paraphrase-multilingual-MiniLM-L12-v2"]

UPLOADED = './uploaded'

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
    modelResearcher = MR.ModelResearcher()
    preprocessed_texts = None
    try:
        preprocessed_texts = Common.read_json(PREPROCESSED_PATH)
    except FileNotFoundError:
        start = time.perf_counter()
        texts = Common.read_json(TRAIN_PATH)
        preprocessed_texts = Common.preprocess_and_save(texts, PREPROCESSED_PATH)
        end = time.perf_counter()
        res = f'Preprocessing time: {end - start:0.4f} secs'
        print(res)

    if name not in ALLOWED_MODELS_GENSIM:
        return jsonify({"Error": "Incorrect model name"})
    try:
        start = time.perf_counter()
        modelResearcher.train(preprocessed_texts, model=name, model_path=MODELS_GENSIM_PATH)
        end = time.perf_counter()
        res = f'Model training time: {end - start:0.4f} secs'
        return jsonify({"Success": res})
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Cannot train model"})


@app.route("/api/docs/match2texts/<string:name>", methods=['POST'])
def match2texts(name):
    modelResearcher = None
    path = None
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})
    modelResearcher = MR.ModelResearcher()
    type_ = None
    if name in ALLOWED_MODELS_GENSIM:
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    elif name in ALLOWED_MODELS_TRANSFORMER:
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    try:
        values = request.values.values()
        first = next(values)
        second = next(values)
        exists = modelResearcher.load(path, type_)
        if not exists:
            return {"Error": "No model: you should train or download it"}
        sim = modelResearcher.predict_sim_two_texts(first, second, model_name=path,  model_type=type_)
        if sim < 0:
            sim = 0
        print(sim)
        return jsonify({"Texts' similarity": str(sim)})
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Something went wrong"})


@app.route("/api/docs/maximize-f1-score/<string:name>", methods=['POST'])
def maximize_f1_score(name):
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})

    if name in ALLOWED_MODELS_GENSIM:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    else:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    exists = modelResearcher.load(path, model_type=type_)

    if not exists:
        return {"Error": "No model: you should train or download it"}
    dataset = request.files['file']
    filename = UPLOADED + "/" + dataset.filename
    dataset.save(filename)
    dataset.close()
    try:
        df = pd.read_json(filename)
        if name in ALLOWED_MODELS_GENSIM:
            df = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
            res = modelResearcher.maximize_f1_score(df["preprocessed_text_rp"], df["preprocessed_text_proj"], df,
                                                    model_name=path,
                                                    model_type="gensim",
                                                    LOO=True,
                                                    step=0.02)
        else:
            res = modelResearcher.maximize_f1_score(df["text_rp"], df["text_proj"], df,
                                                    model_name=path,
                                                    model_type="transformer",
                                                    LOO=True,
                                                    step=0.02)
        return res
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Something went wrong"})


@app.route("/api/docs/get-list-of-allowed-models", methods=['GET'])
def get_list_models():
    return jsonify(ALLOWED_MODELS_GENSIM + ALLOWED_MODELS_TRANSFORMER)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
